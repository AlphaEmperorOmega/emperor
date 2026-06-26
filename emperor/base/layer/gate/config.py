from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.base.options import ActivationOptions
from emperor.base.utils import ConfigBase, optional_field

from .options import LayerGateOptions

if TYPE_CHECKING:
    from emperor.base.layer.config import LayerStackConfig


@dataclass
class GateConfig(ConfigBase):
    gate_dim: int | None = optional_field(
        "Gate feature dimension. Gate model input and output dimensions are both "
        "set to this value."
    )
    option: LayerGateOptions | None = optional_field(
        "Gate composition mode. Use MULTIPLIER to scale the current value or "
        "ADDITION to add gate values to the current value."
    )
    activation: ActivationOptions | None = optional_field(
        "Optional activation applied to gate logits before composition. "
        "Set to None for raw gate values."
    )
    model_config: "LayerStackConfig | None" = optional_field(
        "Gate stack model config. Required when GateConfig is provided."
    )

    def _registry_owner(self) -> type:
        from .model import LayerGate

        return LayerGate
