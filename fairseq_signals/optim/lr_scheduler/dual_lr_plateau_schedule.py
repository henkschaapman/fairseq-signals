import warnings
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import List

from omegaconf import II

from fairseq_signals.dataclass import Dataclass
from fairseq_signals.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class DualLRPlateauConfig(Dataclass):
    warmup_updates: int = field(
        default=0,
        metadata={
            "help": (
                "Linear warmup steps for the backbone LR (ramps from min_lr to optimization.lr). "
                "The head skips warmup entirely and starts at head_lr from step 0."
            )
        },
    )
    head_lr: float = field(
        default=-1.0,
        metadata={
            "help": (
                "Peak LR for the classification head. The head is randomly initialized and can "
                "tolerate a higher LR than the pre-trained backbone. -1 means use optimization.lr "
                "(same as backbone). Requires head parameters to be tagged with p._is_head = True "
                "(done automatically by ECGTransformerClassificationModel)."
            )
        },
    )
    patience_epochs: int = field(
        default=5,
        metadata={"help": "Validation epochs with no sufficient improvement before reducing both LRs"},
    )
    lr_shrink: float = field(
        default=0.5,
        metadata={"help": "Factor to multiply both LRs by when a plateau is detected (e.g. 0.5 halves both)"},
    )
    min_lr: float = field(
        default=1e-8,
        metadata={"help": "LR floor — neither backbone nor head LR is ever reduced below this"},
    )
    maximize: bool = field(
        default=False,
        metadata={"help": "True if the monitored metric should increase (e.g. AUC); False for loss"},
    )
    threshold: float = field(
        default=0.001,
        metadata={"help": "Minimum relative improvement required to count as progress (0.001 = 0.1%)"},
    )
    lr: List[float] = field(
        default=II("optimization.lr"),
        metadata={"help": "Peak backbone LR (interpolated from optimization.lr)"},
    )


@register_lr_scheduler("dual_lr_plateau", dataclass=DualLRPlateauConfig)
class DualLRPlateauSchedule(FairseqLRScheduler):
    """Dual-LR reduce-on-plateau scheduler for simultaneous backbone+head fine-tuning.

    Trains backbone and head from step 0 with no freezing. The backbone
    (pre-trained) optionally warms up from min_lr to optimization.lr over
    warmup_updates steps. The head (randomly initialized) skips warmup and
    starts at head_lr immediately — it can tolerate the higher LR from the start.

    Both groups share a single plateau detector: same patience counter, same
    trigger, same shrink factor. When a plateau is detected, both LRs are
    multiplied by lr_shrink (floored at min_lr).

    Head parameters must be tagged with ``p._is_head = True``.
    ECGTransformerClassificationModel does this automatically for self.proj.
    If no tagged head params are found, a warning is issued and a single group
    is used with a uniform LR (backbone LR).

    The optimizer param groups are split on the FIRST call to step_update (or
    on the first call after checkpoint resume). This is intentional: Adam
    optimizer state is keyed by param tensor identity, so in-place splitting
    of param_groups preserves accumulated first/second moments.
    ``groups_split`` is never saved to state_dict for this reason.

    Example YAML config::

        optimization:
          lr: [0.00001]             # backbone peak LR

        lr_scheduler:
          _name: dual_lr_plateau
          warmup_updates: 500       # backbone ramps min_lr → 1e-5 over 500 steps
          head_lr: 0.0001           # 10x backbone LR — head is randomly initialized
          patience_epochs: 5
          lr_shrink: 0.5
          min_lr: 0.000001
          maximize: false
          threshold: 0.001

        model:
          freeze_finetune_updates: 0   # no freezing
    """

    def __init__(self, cfg: DualLRPlateauConfig, optimizer):
        super().__init__(cfg, optimizer)

        # Backbone peak LR from optimization.lr
        self.peak_backbone_lr = cfg.lr[0] if isinstance(cfg.lr, Collection) else cfg.lr
        # Head peak LR; -1 sentinel means use backbone LR
        self.peak_head_lr = cfg.head_lr if cfg.head_lr > 0 else self.peak_backbone_lr

        assert 0 < cfg.lr_shrink <= 1, "lr_shrink must be in (0, 1]"
        assert cfg.min_lr >= 0, "min_lr must be non-negative"
        assert self.peak_backbone_lr > cfg.min_lr, "optimization.lr must be greater than min_lr"
        assert self.peak_head_lr > cfg.min_lr, "head_lr must be greater than min_lr"
        assert cfg.patience_epochs > 0, "patience_epochs must be positive"
        assert 0 <= cfg.threshold < 1, "threshold must be in [0, 1)"

        # Warmup ramp per step for backbone (0 if no warmup)
        if cfg.warmup_updates > 0:
            self.warmup_lr_step = (self.peak_backbone_lr - cfg.min_lr) / cfg.warmup_updates
        else:
            self.warmup_lr_step = 0.0

        # Backbone LR: starts at min_lr if warmup requested, else peak
        self.lr = cfg.min_lr if cfg.warmup_updates > 0 else self.peak_backbone_lr
        # Head LR: always starts at peak (no warmup for the head)
        self.head_lr = self.peak_head_lr

        self.num_updates = 0
        self.patience_count = 0
        # self.best is inherited from FairseqLRScheduler, initialised to None

        # Whether optimizer groups have been split into backbone (0) + head (1).
        # NOT saved to state_dict — always reset to False on load so step_update
        # re-splits the groups after a checkpoint resume.
        self.groups_split = False

        # Set initial LR on the single pre-split group.
        # Head LR is applied after the split in the first step_update call.
        self.optimizer.set_lr(self.lr)

    # ------------------------------------------------------------------
    # Param-group splitting
    # ------------------------------------------------------------------

    def _split_param_groups(self):
        """Split the optimizer's single param group into backbone (group 0) + head (group 1).

        Head parameters are identified by the ``_is_head = True`` attribute.
        Adam optimizer state is keyed by param tensor identity, so splitting
        the param_groups list in-place preserves all accumulated first/second moments.
        """
        if self.groups_split:
            return

        pg = self.optimizer.param_groups
        all_params = pg[0]["params"]

        backbone_params = [p for p in all_params if not getattr(p, "_is_head", False)]
        head_params = [p for p in all_params if getattr(p, "_is_head", False)]

        if not head_params:
            warnings.warn(
                "DualLRPlateauSchedule: no head parameters found (expected p._is_head=True on "
                "classification head params). Will use a single optimizer group with uniform LR. "
                "head_lr setting will have no effect."
            )
            self.groups_split = True
            return

        # Rewrite group 0 to contain only backbone params
        pg[0]["params"] = backbone_params

        # Append a new group for head params, copying all optimizer hyperparams from group 0
        head_group = {k: v for k, v in pg[0].items() if k != "params"}
        head_group["params"] = head_params
        pg.append(head_group)

        self.groups_split = True

    def _set_backbone_lr(self, lr):
        """Set LR on backbone param group (group 0)."""
        self.optimizer.param_groups[0]["lr"] = lr

    def _set_head_lr(self, lr):
        """Set LR on head param group (group 1). Falls back to group 0 if not split."""
        if self.groups_split and len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]["lr"] = lr
        else:
            self.optimizer.param_groups[0]["lr"] = lr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_better(self, current, best):
        """Return True if `current` is a meaningful improvement over `best`."""
        if best == 0:
            return current != 0
        relative_change = (current - best) / abs(best)
        if self.cfg.maximize:
            return relative_change > self.cfg.threshold
        else:
            return relative_change < -self.cfg.threshold

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def state_dict(self):
        return {
            "lr": self.lr,
            "head_lr": self.head_lr,
            "num_updates": self.num_updates,
            "patience_count": self.patience_count,
            "best": self.best,
            # groups_split intentionally omitted — always reset to False on load
            # so step_update re-splits the optimizer groups when training resumes
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict.get("lr", self.lr)
        self.head_lr = state_dict.get("head_lr", self.peak_head_lr)
        self.num_updates = state_dict.get("num_updates", 0)
        self.patience_count = state_dict.get("patience_count", 0)
        self.best = state_dict.get("best", None)
        # Always reset so step_update re-splits the param groups on resume
        self.groups_split = False
        # Apply loaded backbone LR to the single unsplit group as a safe default
        self.optimizer.set_lr(self.lr)

    # ------------------------------------------------------------------
    # LR hooks
    # ------------------------------------------------------------------

    def step_update(self, num_updates):
        """Called after every gradient update — handles group splitting and backbone warmup."""
        self.num_updates = num_updates

        # Split param groups on first call (or first call after checkpoint resume).
        # Apply correct LRs to each group immediately after splitting.
        if not self.groups_split:
            self._split_param_groups()
            self._set_backbone_lr(self.lr)
            self._set_head_lr(self.head_lr)

        cfg = self.cfg

        # Backbone warmup ramp. Head LR is never touched here.
        if cfg.warmup_updates > 0 and num_updates < cfg.warmup_updates:
            self.lr = cfg.min_lr + num_updates * self.warmup_lr_step
            self._set_backbone_lr(self.lr)
        elif cfg.warmup_updates > 0 and num_updates == cfg.warmup_updates:
            # Snap exactly to backbone peak at end of warmup to avoid float drift
            self.lr = self.peak_backbone_lr
            self._set_backbone_lr(self.lr)
        # else: hold current self.lr (may have been shrunk by a plateau event)

        return self.optimizer.get_lr()

    def step(self, epoch, val_loss=None):
        """Called at the end of each validation epoch — handles plateau detection."""
        # Don't start plateau counting until backbone warmup is complete
        if self.num_updates < self.cfg.warmup_updates:
            return self.optimizer.get_lr()

        if val_loss is None:
            return self.optimizer.get_lr()

        if self.best is None or self._is_better(val_loss, self.best):
            self.best = val_loss
            self.patience_count = 0
        else:
            self.patience_count += 1
            if self.patience_count >= self.cfg.patience_epochs:
                # Shrink both LRs together
                self.lr = max(self.lr * self.cfg.lr_shrink, self.cfg.min_lr)
                self.head_lr = max(self.head_lr * self.cfg.lr_shrink, self.cfg.min_lr)
                self._set_backbone_lr(self.lr)
                self._set_head_lr(self.head_lr)
                self.patience_count = 0

        return self.optimizer.get_lr()
