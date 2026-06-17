from dataclasses import dataclass
from emperor.base.layer.config import LayerStackConfig
from emperor.base.utils import ConfigBase, optional_field
from emperor.halting.options import HaltingHiddenStateModeOptions


@dataclass
class HaltingConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Hidden dimension used to build the halting gate network"
    )
    threshold: float | None = optional_field(
        "Halting probability threshold; tokens above this stop computing. "
        "Recommended: use a high value such as 0.99 so tokens halt only after "
        "most of their probability mass has been assigned. Smaller values stop "
        "earlier, but can produce less stable accumulated representations."
    )
    halting_dropout: float | None = optional_field(
        "Dropout probability applied inside the soft halting gate network"
    )
    hidden_state_mode: HaltingHiddenStateModeOptions | None = optional_field(
        "Controls whether each step returns the raw hidden state or the current accumulated weighted representation"
    )
    halting_gate_config: "LayerStackConfig | None" = optional_field(
        "Config used to build the model module within the layer"
    )


@dataclass
class StickBreakingConfig(HaltingConfig):
    def _registry_owner(self) -> type:
        from emperor.halting.core.variants import StickBreaking

        return StickBreaking


@dataclass
class SoftHaltingConfig(HaltingConfig):
    def _registry_owner(self) -> type:
        from emperor.halting.core.variants import SoftHalting

        return SoftHalting
