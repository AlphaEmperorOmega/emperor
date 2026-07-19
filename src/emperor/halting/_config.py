import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from emperor.config import BaseOptions, ConfigBase, optional_field

if TYPE_CHECKING:
    from emperor.halting._interface import HaltingInterface
    from emperor.layers import LayerStackConfig


class HaltingHiddenStateModeOptions(BaseOptions):
    RAW = 0
    ACCUMULATED = 1


@dataclass
class HaltingConfig(ConfigBase):
    DEFAULT_THRESHOLD: ClassVar[float | None] = None

    input_dim: int | None = optional_field(
        "Hidden dimension used to build the halting gate network"
    )
    threshold: float | None = optional_field(
        "Halting probability threshold; tokens above this stop computing. "
        "Recommended: use a high value such as 0.999 so tokens halt only after "
        "most of their probability mass has been assigned. Smaller values stop "
        "earlier, but can produce less stable accumulated representations."
    )
    dropout_probability: float | None = optional_field(
        "Dedicated dropout probability applied immediately before the final "
        "soft-halting gate projection. None or 0.0 disables it. This option is "
        "not used by StickBreaking, whose configured gate stack controls its "
        "own dropout."
    )
    hidden_state_mode: HaltingHiddenStateModeOptions | None = optional_field(
        "Controls whether each step returns the raw hidden state or the "
        "current accumulated weighted representation"
    )
    halting_gate_config: "LayerStackConfig | None" = optional_field(
        "Config used to build the model module within the layer"
    )

    def build(self, overrides: "HaltingConfig | None" = None):
        configured_threshold = None if overrides is None else overrides.threshold
        if (
            self.threshold is not None
            or configured_threshold is not None
            or self.DEFAULT_THRESHOLD is None
        ):
            return super().build(overrides)

        resolved_overrides = (
            type(self)() if overrides is None else copy.deepcopy(overrides)
        )
        resolved_overrides.threshold = self.DEFAULT_THRESHOLD
        return super().build(resolved_overrides)


@dataclass
class StickBreakingConfig(HaltingConfig):
    DEFAULT_THRESHOLD: ClassVar[float] = 0.999

    def _registry_owner(self) -> "type[HaltingInterface]":
        from emperor.halting._strategies.stick_breaking import StickBreaking

        return StickBreaking


@dataclass
class SoftHaltingConfig(HaltingConfig):
    DEFAULT_THRESHOLD: ClassVar[float] = 0.999

    def _registry_owner(self) -> "type[HaltingInterface]":
        from emperor.halting._strategies.soft import SoftHalting

        return SoftHalting
