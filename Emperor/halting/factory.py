from Emperor.base.utils import Module
from Emperor.halting.config import HaltingConfig
from Emperor.halting.options import HaltingOptions
from Emperor.halting.options.soft_halting import SoftHalting
from Emperor.halting.options.stick_breaking import StickBreaking

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class HaltingFactory(Module):
    def __init__(
        self,
        cfg: "HaltingConfig | ModelConfig",
    ):
        super().__init__()
        self.cfg: "HaltingConfig" = getattr(cfg, "halting_config", cfg)
        self.halting_option: HaltingOptions = self.cfg.halting_option

    def build(self) -> Module:
        match self.halting_option:
            case HaltingOptions.SOFT_HALTING:
                return SoftHalting(self.cfg)
            case HaltingOptions.STICK_BREAKING:
                return StickBreaking(self.cfg)
            case _:
                raise ValueError(
                    "If the `halting_option` is set to `DISABLED`, this class should not be initialized"
                )
