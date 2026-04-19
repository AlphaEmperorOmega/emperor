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
    input_dim: int | None = optional_field("Input dimension of the wrapped module")
    output_dim: int | None = optional_field("Output dimension of the wrapped module")
    activation: ActivationOptions | None = optional_field(
        "Activation applied to the wrapped module's output"
    )
    residual_flag: bool | None = optional_field(
        "Add residual from input to output; requires input_dim == output_dim"
    )
    dropout_probability: float | None = optional_field(
        "Dropout after the layer output; range 0-1"
    )
    layer_norm_position: LayerNormPositionOptions | None = optional_field(
        "LayerNorm position: BEFORE (pre-norm), DEFAULT (post-model, pre-activation), AFTER (post-activation), DISABLED"
    )
    gate_config: "LayerStackConfig | None" = optional_field(
        "Gating LayerStack config; None skips gating"
    )
    halting_config: "HaltingConfig | None" = optional_field(
        "Halting config for adaptive computation; None skips halting"
    )
    shared_halting_flag: bool | None = optional_field(
        "When True, one halting module shared across layers; else per-layer"
    )
    layer_model_config: ConfigBase | None = optional_field(
        "Config for the wrapped inner module (e.g., LinearLayerConfig)"
    )

    def _registry_owner(self) -> type:
        from emperor.base.layer.layer import Layer

        return Layer


@dataclass
class LayerStackConfig(ConfigBase):
    input_dim: int | None = optional_field("Input dimension of the first layer")
    hidden_dim: int | None = optional_field("Hidden layers dimension")
    output_dim: int | None = optional_field("Output dimension of the last layer")
    num_layers: int | None = optional_field("Total layers in the stack")
    last_layer_bias_option: "LastLayerBiasOptions | None" = optional_field(
        "Bias override for the final layer: DEFAULT inherits, DISABLED off, ENABLED on"
    )
    apply_output_pipeline_flag: bool | None = optional_field(
        "When True, final layer runs the wrapper pipeline (activation, dropout, norm, residual, gate); else returns raw module output"
    )
    layer_config: LayerConfig | None = optional_field(
        "Layer config that determines the pipeline to be applied to the wrapped model"
    )

    def build(
        self, overrides: "LayerStackConfig | None" = None
    ) -> "Layer | Sequential":
        from emperor.base.layer.stack import LayerStack

        return LayerStack(self, overrides).build()
