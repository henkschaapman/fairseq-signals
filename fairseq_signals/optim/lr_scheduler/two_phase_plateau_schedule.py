from collections.abc import Collection
from dataclasses import dataclass, field
from typing import List

from omegaconf import II

from fairseq_signals.dataclass import Dataclass
from fairseq_signals.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class TwoPhaseConfig(Dataclass):
    phase1_updates: int = field(
        default=5000,
        metadata={
            "help": (
                "Total number of gradient updates in Phase 1 (head-only training). "
                "IMPORTANT: set model.freeze_finetune_updates to the same value so the "
                "backbone is unfrozen at exactly the same step."
            )
        },
    )
    phase1_warmup_updates: int = field(
        default=0,
        metadata={"help": "Linear warmup steps within Phase 1 (from min_lr to phase1_lr)"},
    )
    phase1_lr: float = field(
        default=-1.0,
        metadata={
            "help": (
                "Peak LR during Phase 1 (head-only). Can be higher than backbone LR because "
                "the head is randomly initialised. -1 means use optimization.lr (same as Phase 2)."
            )
        },
    )
    phase1_patience: int = field(
        default=5,
        metadata={"help": "Validation epochs with no improvement before reducing LR in Phase 1"},
    )
    phase2_warmup_updates: int = field(
        default=0,
        metadata={
            "help": (
                "Linear warmup steps at the start of Phase 2. LR ramps from min_lr back up to "
                "the peak (optimization.lr). Recommended: set this to avoid a sudden large LR "
                "hitting the just-unfrozen backbone."
            )
        },
    )
    phase2_patience: int = field(
        default=10,
        metadata={"help": "Validation epochs with no improvement before reducing LR in Phase 2"},
    )
    lr_shrink: float = field(
        default=0.5,
        metadata={"help": "Factor to multiply LR by when a plateau is detected (applied in both phases)"},
    )
    min_lr: float = field(
        default=1e-8,
        metadata={"help": "LR floor — never reduce below this; also used as warmup start in both phases"},
    )
    maximize: bool = field(
        default=False,
        metadata={"help": "True if monitored metric should increase (e.g. AUC); False for loss"},
    )
    threshold: float = field(
        default=0.001,
        metadata={"help": "Minimum relative improvement to count as progress (0.001 = 0.1%)"},
    )
    lr: List[float] = field(
        default=II("optimization.lr"),
        metadata={"help": "Peak LR for Phase 2 (backbone fine-tuning). Also used for Phase 1 if phase1_lr=-1."},
    )


@register_lr_scheduler("two_phase_plateau", dataclass=TwoPhaseConfig)
class TwoPhaseSchedule(FairseqLRScheduler):
    """Two-phase LR scheduler for gradual backbone unfreezing.

    **Phase 1** (updates 0 → phase1_updates): only the head is trained (backbone
    frozen via model.freeze_finetune_updates). LR warms up to phase1_lr, then
    follows reduce-on-plateau.

    **Phase 2** (updates phase1_updates → end): backbone unfreezes. LR drops to
    min_lr and warms back up to optimization.lr (the backbone LR, typically lower
    than phase1_lr), then follows reduce-on-plateau again.

    IMPORTANT — you must coordinate two config values::

        model:
          freeze_finetune_updates: 5000   # <-- must equal lr_scheduler.phase1_updates

    Example YAML config::

        optimization:
          lr: [0.00001]          # Phase 2 backbone LR (low — protects pre-trained weights)

        lr_scheduler:
          _name: two_phase_plateau
          phase1_updates: 5000         # ~10 epochs; must match model.freeze_finetune_updates
          phase1_warmup_updates: 500
          phase1_lr: 0.0001            # 10x backbone LR — head starts random so can handle more
          phase1_patience: 5
          phase2_warmup_updates: 1000
          phase2_patience: 10
          lr_shrink: 0.5
          min_lr: 0.000001
          maximize: false
          threshold: 0.001

        model:
          freeze_finetune_updates: 5000
    """

    def __init__(self, cfg: TwoPhaseConfig, optimizer):
        super().__init__(cfg, optimizer)

        # Phase 2 peak LR comes from optimization.lr
        self.phase2_lr = cfg.lr[0] if isinstance(cfg.lr, Collection) else cfg.lr
        # Phase 1 peak LR — falls back to phase2_lr if not set
        self.phase1_peak_lr = cfg.phase1_lr if cfg.phase1_lr > 0 else self.phase2_lr

        assert 0 < cfg.lr_shrink <= 1, "lr_shrink must be in (0, 1]"
        assert cfg.min_lr >= 0, "min_lr must be non-negative"
        assert self.phase1_peak_lr > cfg.min_lr, "phase1_lr must be greater than min_lr"
        assert self.phase2_lr > cfg.min_lr, "optimization.lr must be greater than min_lr"
        assert cfg.phase1_patience > 0, "phase1_patience must be positive"
        assert cfg.phase2_patience > 0, "phase2_patience must be positive"

        # Warmup step sizes
        if cfg.phase1_warmup_updates > 0:
            self.phase1_warmup_step = (self.phase1_peak_lr - cfg.min_lr) / cfg.phase1_warmup_updates
        else:
            self.phase1_warmup_step = 0.0

        if cfg.phase2_warmup_updates > 0:
            self.phase2_warmup_step = (self.phase2_lr - cfg.min_lr) / cfg.phase2_warmup_updates
        else:
            self.phase2_warmup_step = 0.0

        # Starting LR
        self.lr = cfg.min_lr if cfg.phase1_warmup_updates > 0 else self.phase1_peak_lr
        self.num_updates = 0
        self.patience_count = 0
        self.phase1_best = None   # best val metric seen during Phase 1
        self.phase2_best = None   # best val metric seen during Phase 2

        self.optimizer.set_lr(self.lr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _in_phase2(self):
        return self.num_updates >= self.cfg.phase1_updates

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
            "num_updates": self.num_updates,
            "patience_count": self.patience_count,
            "phase1_best": self.phase1_best,
            "phase2_best": self.phase2_best,
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict.get("lr", self.lr)
        self.num_updates = state_dict.get("num_updates", 0)
        self.patience_count = state_dict.get("patience_count", 0)
        self.phase1_best = state_dict.get("phase1_best", None)
        self.phase2_best = state_dict.get("phase2_best", None)
        self.optimizer.set_lr(self.lr)

    # ------------------------------------------------------------------
    # LR hooks
    # ------------------------------------------------------------------

    def step_update(self, num_updates):
        """Called after every gradient update — handles warmup and phase transition."""
        prev_updates = self.num_updates
        self.num_updates = num_updates

        if not self._in_phase2():
            # ---- Phase 1 ----
            cfg = self.cfg
            if cfg.phase1_warmup_updates > 0 and num_updates < cfg.phase1_warmup_updates:
                self.lr = cfg.min_lr + num_updates * self.phase1_warmup_step
                self.optimizer.set_lr(self.lr)
            elif cfg.phase1_warmup_updates > 0 and num_updates == cfg.phase1_warmup_updates:
                # Snap exactly to peak at end of warmup
                self.lr = self.phase1_peak_lr
                self.optimizer.set_lr(self.lr)
            # else: hold current lr (may have been reduced by plateau)

        else:
            # ---- Phase 2 ----
            cfg = self.cfg
            phase2_elapsed = num_updates - cfg.phase1_updates

            if phase2_elapsed == 0:
                # First update of Phase 2: reset LR to min_lr for fresh warmup and
                # reset patience (the plateau tracker for Phase 2 starts clean)
                self.lr = cfg.min_lr
                self.patience_count = 0
                self.optimizer.set_lr(self.lr)

            elif cfg.phase2_warmup_updates > 0 and phase2_elapsed < cfg.phase2_warmup_updates:
                self.lr = cfg.min_lr + phase2_elapsed * self.phase2_warmup_step
                self.optimizer.set_lr(self.lr)

            elif cfg.phase2_warmup_updates > 0 and phase2_elapsed == cfg.phase2_warmup_updates:
                # Snap exactly to phase 2 peak at end of warmup
                self.lr = self.phase2_lr
                self.optimizer.set_lr(self.lr)
            # else: hold current lr (may have been reduced by plateau)

        return self.optimizer.get_lr()

    def step(self, epoch, val_loss=None):
        """Called at the end of each validation epoch — handles plateau detection."""
        if val_loss is None:
            return self.optimizer.get_lr()

        cfg = self.cfg

        if not self._in_phase2():
            # ---- Phase 1 plateau ----
            # Don't start counting until Phase 1 warmup is done
            if self.num_updates < cfg.phase1_warmup_updates:
                return self.optimizer.get_lr()

            if self.phase1_best is None or self._is_better(val_loss, self.phase1_best):
                self.phase1_best = val_loss
                self.patience_count = 0
            else:
                self.patience_count += 1
                if self.patience_count >= cfg.phase1_patience:
                    self.lr = max(self.lr * cfg.lr_shrink, cfg.min_lr)
                    self.optimizer.set_lr(self.lr)
                    self.patience_count = 0

        else:
            # ---- Phase 2 plateau ----
            # Don't start counting until Phase 2 warmup is done
            phase2_elapsed = self.num_updates - cfg.phase1_updates
            if phase2_elapsed < cfg.phase2_warmup_updates:
                return self.optimizer.get_lr()

            if self.phase2_best is None or self._is_better(val_loss, self.phase2_best):
                self.phase2_best = val_loss
                self.patience_count = 0
            else:
                self.patience_count += 1
                if self.patience_count >= cfg.phase2_patience:
                    self.lr = max(self.lr * cfg.lr_shrink, cfg.min_lr)
                    self.optimizer.set_lr(self.lr)
                    self.patience_count = 0

        return self.optimizer.get_lr()
