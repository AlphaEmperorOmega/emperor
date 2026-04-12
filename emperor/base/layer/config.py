from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
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
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the first `Linear` layer"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the output `Linear` layer"},
    )
    activation: ActivationOptions | None = field(
        default=None,
        metadata={"help": "Activation function or layer to use"},
    )
    residual_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    dropout_probability: float | None = field(
        default=None,
        metadata={"help": ""},
    )
    layer_norm_position: LayerNormPositionOptions | None = field(
        default=None,
        metadata={"help": ""},
    )
    gate_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={
            "help": "LayerStack config for the gating mechanism; if None gates are skipped"
        },
    )
    halting_config: "HaltingConfig | None" = field(
        default=None,
        metadata={"help": "Optional halting config for adaptive computation per layer"},
    )
    shared_halting_flag: bool | None = field(
        default=None,
        metadata={
            "help": "If True, one halting module is shared across all layers; if False, each layer gets its own"
        },
    )
    layer_model_config: ConfigBase | None = field(
        default=None,
        metadata={"help": "Config used to build the model module within the layer"},
    )

    def build(self, overrides: "LayerConfig | None" = None) -> "Module":
        from emperor.base.layer.layer import Layer

        return Layer(self, overrides)


@dataclass
class LayerStackConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the first layer in the stack"},
    )
    hidden_dim: int | None = field(
        default=None,
        metadata={
            "help": "Dimension used for all hidden layers between input and output"
        },
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the last layer in the stack"},
    )
    num_layers: int | None = field(
        default=None,
        metadata={"help": "Total number of layers in the stack"},
    )
    layer_type: "Layer | None" = field(
        default=None,
        metadata={
            "help": "Layer subclass to use for each layer; defaults to Layer if None"
        },
    )
    last_layer_bias_option: "LastLayerBiasOptions | None" = field(
        default=None,
        metadata={
            "help": "Override bias on the last layer: DEFAULT keeps model_config value, DISABLED removes bias, ENABLED adds bias"
        },
    )
    apply_output_pipeline_flag: bool | None = field(
        default=None,
        metadata={
            "help": "If True, the output layer applies the full pipeline (activation, dropout, layer norm, residual, gate); if False, returns clean model output"
        },
    )
    layer_config: LayerConfig | None = field(
        default=None,
        metadata={
            "help": "LayerConfig shared across all layers in the stack; per-layer overrides are applied on top"
        },
    )

    def build(self, overrides: "LayerStackConfig | None" = None) -> "Layer | Sequential":
        from emperor.base.layer.stack import LayerStack

        return LayerStack(self, overrides).build()
