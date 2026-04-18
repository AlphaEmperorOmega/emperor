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
    from emperor.base.utils import Module
    from emperor.base.layer.layer import Layer
    from emperor.halting.config import HaltingConfig


@dataclass
class LayerConfig(ConfigBase):
    input_dim: int | None = optional_field("Input dimension of the first `Linear` layer")
    output_dim: int | None = optional_field("Output dimension of the output `Linear` layer")
    activation: ActivationOptions | None = optional_field("Activation function or layer to use")
    residual_flag: bool | None = optional_field(
        "When True, adds a residual connection from layer input to output. Requires input_dim == output_dim."
    )
    dropout_probability: float | None = optional_field(
        "Probability for dropout applied after the layer output. Must be in [0.0, 1.0]; dropout is skipped when <= 0."
    )
    layer_norm_position: LayerNormPositionOptions | None = optional_field(
        "Where LayerNorm is applied: BEFORE (pre-norm on input), DEFAULT (after model output, before activation), AFTER (post-activation on final output), DISABLED (no normalization)."
    )
    gate_config: "LayerStackConfig | None" = optional_field(
        "LayerStack config for the gating mechanism; if None gates are skipped"
    )
    halting_config: "HaltingConfig | None" = optional_field(
        "Optional halting config for adaptive computation per layer"
    )
    shared_halting_flag: bool | None = optional_field(
        "If True, one halting module is shared across all layers; if False, each layer gets its own"
    )
    layer_model_config: ConfigBase | None = optional_field(
        "Config used to build the model module within the layer"
    )

    def build(self, overrides: "LayerConfig | None" = None) -> "Module":
        from emperor.base.layer.layer import Layer

        return Layer(self, overrides)


@dataclass
class LayerStackConfig(ConfigBase):
    input_dim: int | None = optional_field("Input dimension of the first layer in the stack")
    hidden_dim: int | None = optional_field(
        "Dimension used for all hidden layers between input and output"
    )
    output_dim: int | None = optional_field("Output dimension of the last layer in the stack")
    num_layers: int | None = optional_field("Total number of layers in the stack")
    layer_type: "Layer | None" = optional_field(
        "Layer subclass to use for each layer; defaults to Layer if None"
    )
    last_layer_bias_option: "LastLayerBiasOptions | None" = optional_field(
        "Override bias on the last layer: DEFAULT keeps model_config value, DISABLED removes bias, ENABLED adds bias"
    )
    apply_output_pipeline_flag: bool | None = optional_field(
        "If True, the output layer applies the full pipeline (activation, dropout, layer norm, residual, gate); if False, returns clean model output"
    )
    layer_config: LayerConfig | None = optional_field(
        "LayerConfig shared across all layers in the stack; per-layer overrides are applied on top"
    )

    def build(self, overrides: "LayerStackConfig | None" = None) -> "Layer | Sequential":
        from emperor.base.layer.stack import LayerStack

        return LayerStack(self, overrides).build()
