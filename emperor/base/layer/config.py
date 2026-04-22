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


@dataclass
class LayerConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input dimensionality of the wrapped module."
    )
    output_dim: int | None = optional_field(
        "Output dimensionality of the wrapped module."
    )
    activation: ActivationOptions | None = optional_field(
        "Activation function applied to the wrapped module output."
    )
    residual_flag: bool | None = optional_field(
        "Enables a residual connection from input to output. Requires input_dim == output_dim."
    )
    dropout_probability: float | None = optional_field(
        "Dropout probability applied after the layer output."
    )
    layer_norm_position: LayerNormPositionOptions | None = optional_field(
        "Specifies where layer normalization is applied within the layer pipeline."
    )
    gate_config: "LayerStackConfig | None" = optional_field(
        "Configuration for the optional gating stack. Set to None to disable gating."
    )
    halting_config: "HaltingConfig | None" = optional_field(
        "Configuration for the optional adaptive halting module. Set to None to disable halting."
    )
    shared_halting_flag: bool | None = optional_field(
        "When enabled, a single halting module is shared across layers instead of creating one per layer."
    )
    layer_model_config: ConfigBase | None = optional_field(
        "Configuration for the wrapped inner module, such as LinearLayerConfig."
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
    num_layers: int | None = optional_field(
        "Total number of layers in the stack."
    )
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
