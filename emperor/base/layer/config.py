from dataclasses import dataclass
from emperor.base.utils import ConfigBase, optional_field
from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Sequential
    from emperor.base.layer.layer import Layer
    from emperor.halting.config import HaltingConfig
    from emperor.memory.config import DynamicMemoryConfig


@dataclass
class LayerConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    activation: ActivationOptions | None = optional_field(
        "Activation applied to the layer output."
    )
    residual_flag: bool | None = optional_field(
        "Adds the input back to the output. Requires input_dim == output_dim."
    )
    dropout_probability: float | None = optional_field(
        "Dropout probability. Use 0.0 to disable."
    )
    layer_norm_position: LayerNormPositionOptions | None = optional_field(
        "Where layer normalization is applied."
    )
    gate_config: "LayerStackConfig | None" = optional_field(
        "Optional gate stack that scales the layer output. Set to None to disable."
    )
    halting_config: "HaltingConfig | None" = optional_field(
        "Optional adaptive computation module. Set to None to disable halting."
    )
    memory_config: "DynamicMemoryConfig | None" = optional_field(
        "Optional dynamic memory module. Set to None to disable memory."
    )
    shared_halting_flag: bool | None = optional_field(
        "Reserved for shared halting. Not supported yet; keep False."
    )
    layer_model_config: ConfigBase | None = optional_field(
        "Config for the wrapped module, such as LinearLayerConfig."
    )

    def _registry_owner(self) -> type:
        from emperor.base.layer.layer import Layer

        return Layer


@dataclass
class LayerStackConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input dimensionality of the first layer in the stack."
    )
    hidden_dim: int | None = optional_field(
        "Hidden dimensionality used by intermediate layers in the stack."
    )
    output_dim: int | None = optional_field(
        "Output dimensionality of the final layer in the stack."
    )
    num_layers: int | None = optional_field("Total number of layers in the stack.")
    last_layer_bias_option: "LastLayerBiasOptions | None" = optional_field(
        "Controls whether the final layer uses its default bias behavior or an explicit override."
    )
    apply_output_pipeline_flag: bool | None = optional_field(
        "When enabled, the final layer applies the full wrapper pipeline instead of returning the raw module output."
    )
    layer_config: LayerConfig | None = optional_field(
        "Configuration that defines the per-layer wrapper pipeline applied throughout the stack."
    )

    def build(
        self, overrides: "LayerStackConfig | None" = None
    ) -> "Layer | Sequential":
        from emperor.base.layer.stack import LayerStack

        return LayerStack(self, overrides).build()
