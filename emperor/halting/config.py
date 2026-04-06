from dataclasses import dataclass, field
from emperor.base.layer.config import LayerStackConfig
from emperor.base.utils import ConfigBase
from emperor.halting.options import HaltingHiddenStateModeOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.base.utils import Module


@dataclass
class HaltingConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Hidden dimension used to build the halting gate network"},
    )
    threshold: float | None = field(
        default=None,
        metadata={
            "help": "Halting probability threshold; tokens above this stop computing"
        },
    )
    halting_dropout: float | None = field(
        default=None,
        metadata={
            "help": "Dropout probability applied inside the soft halting gate network"
        },
    )
    hidden_state_mode: HaltingHiddenStateModeOptions | None = field(
        default=None,
        metadata={
            "help": "Controls whether each step returns the raw hidden state or the current accumulated weighted representation"
        },
    )
    halting_gate_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": "Config used to build the model module within the layer"},
    )

    def build(self, input_dim: int) -> "Module":
        raise NotImplementedError


@dataclass
class StickBreakingConfig(HaltingConfig):
    def build(self, input_dim: int) -> "Module":
        from emperor.halting.utils.options.stick_breaking import StickBreaking

        overrides = StickBreakingConfig(input_dim=input_dim)
        return StickBreaking(self, overrides)


@dataclass
class SoftHaltingConfig(HaltingConfig):
    def build(self, input_dim: int) -> "Module":
        from emperor.halting.utils.options.soft_halting import SoftHalting

        overrides = SoftHaltingConfig(input_dim=input_dim)
        return SoftHalting(self, overrides)
