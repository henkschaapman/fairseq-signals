from collections.abc import Collection
from dataclasses import dataclass, field
from typing import List

from omegaconf import II

from fairseq_signals.dataclass import Dataclass
from fairseq_signals.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class ReduceOnPlateauConfig(Dataclass):
    warmup_updates: int = field(
        default=0,
        metadata={"help": "linearly warm up LR from warmup_init_lr to peak LR over this many gradient updates"},
    )
    warmup_init_lr: float = field(
        default=-1,
        metadata={"help": "initial LR at start of warmup; -1 means use min_lr"},
    )
    patience_epochs: int = field(
        default=5,
        metadata={"help": "number of validation epochs with no sufficient improvement before reducing LR"},
    )
    lr_shrink: float = field(
        default=0.5,
        metadata={"help": "factor to multiply LR by when a plateau is detected (e.g. 0.5 halves the LR)"},
    )
    min_lr: float = field(
        default=1e-8,
        metadata={"help": "LR floor — never reduce below this value"},
    )
    maximize: bool = field(
        default=False,
        metadata={"help": "set True if the monitored metric should increase (e.g. AUC); False if it should decrease (e.g. loss)"},
    )
    threshold: float = field(
        default=0.001,
        metadata={"help": "minimum relative improvement required to count as progress (e.g. 0.001 = 0.1%); prevents reacting to noise"},
    )
    lr: List[float] = field(
        default=II("optimization.lr"),
        metadata={"help": "peak learning rate (interpolated from optimization.lr)"},
    )


@register_lr_scheduler("reduce_on_plateau", dataclass=ReduceOnPlateauConfig)
class ReduceOnPlateauSchedule(FairseqLRScheduler):
    """Reduce LR on validation plateau, with optional linear warmup.

    Holds LR at peak after warmup. After `patience_epochs` consecutive
    validation epochs with no improvement exceeding `threshold` (relative),
    the LR is multiplied by `lr_shrink`. LR is never reduced below `min_lr`.

    Example YAML config::

        lr_scheduler:
          _name: reduce_on_plateau
          warmup_updates: 1000   # ~2 epochs of linear warmup
          warmup_init_lr: -1     # start from min_lr
          patience_epochs: 10
          lr_shrink: 0.5
          min_lr: 0.000001
          maximize: false        # monitoring val loss (lower is better)
          threshold: 0.001       # 0.1% relative improvement required
    """

    def __init__(self, cfg: ReduceOnPlateauConfig, optimizer):
        super().__init__(cfg, optimizer)

        self.peak_lr = cfg.lr[0] if isinstance(cfg.lr, Collection) else cfg.lr
        self.min_lr = cfg.min_lr
        self.warmup_updates = cfg.warmup_updates
        self.warmup_init_lr = cfg.warmup_init_lr if cfg.warmup_init_lr >= 0 else self.min_lr

        assert 0 < cfg.lr_shrink <= 1, "lr_shrink must be in (0, 1]"
        assert self.min_lr >= 0, "min_lr must be non-negative"
        assert self.peak_lr > self.min_lr, "peak lr must be greater than min_lr"
        assert cfg.patience_epochs > 0, "patience_epochs must be positive"
        assert 0 <= cfg.threshold < 1, "threshold must be in [0, 1)"

        if self.warmup_updates > 0:
            self.warmup_lr_step = (self.peak_lr - self.warmup_init_lr) / self.warmup_updates
        else:
            self.warmup_lr_step = 0

        # current learning rate — starts at warmup_init_lr (or peak if no warmup)
        self.lr = self.warmup_init_lr if self.warmup_updates > 0 else self.peak_lr
        self.num_updates = 0
        self.patience_count = 0
        # self.best is inherited from FairseqLRScheduler (initialised to None)

        self.optimizer.set_lr(self.lr)

    def _is_better(self, current, best):
        """Return True if `current` is a meaningful improvement over `best`."""
        if best == 0:
            return current != 0
        relative_change = (current - best) / abs(best)
        if self.cfg.maximize:
            return relative_change > self.cfg.threshold
        else:
            return relative_change < -self.cfg.threshold

    def state_dict(self):
        return {
            "best": self.best,
            "lr": self.lr,
            "patience_count": self.patience_count,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict):
        self.best = state_dict.get("best", None)
        self.lr = state_dict.get("lr", self.lr)
        self.patience_count = state_dict.get("patience_count", 0)
        self.num_updates = state_dict.get("num_updates", 0)
        self.optimizer.set_lr(self.lr)

    def step(self, epoch, val_loss=None):
        """Called at the end of each validation epoch."""
        # Don't start plateau tracking until warmup is complete
        if self.num_updates < self.warmup_updates:
            return self.optimizer.get_lr()

        if val_loss is None:
            return self.optimizer.get_lr()

        if self.best is None or self._is_better(val_loss, self.best):
            self.best = val_loss
            self.patience_count = 0
        else:
            self.patience_count += 1
            if self.patience_count >= self.cfg.patience_epochs:
                self.lr = max(self.lr * self.cfg.lr_shrink, self.min_lr)
                self.optimizer.set_lr(self.lr)
                self.patience_count = 0

        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Called after every gradient update — handles warmup ramp."""
        self.num_updates = num_updates

        if self.warmup_updates > 0 and num_updates < self.warmup_updates:
            self.lr = self.warmup_init_lr + num_updates * self.warmup_lr_step
            self.optimizer.set_lr(self.lr)
        elif num_updates == self.warmup_updates and self.warmup_updates > 0:
            # Exactly at the end of warmup — snap to peak
            self.lr = self.peak_lr
            self.optimizer.set_lr(self.lr)

        return self.optimizer.get_lr()
