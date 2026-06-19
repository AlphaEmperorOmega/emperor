from .gate import GateConfig
from dataclasses import dataclass
from emperor.base.utils import ConfigBase, optional_field
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from .residual import ResidualConnectionOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.halting.config import HaltingConfig
    from emperor.memory.config import DynamicMemoryConfig


@dataclass
class LayerConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    activation: ActivationOptions | None = optional_field(
        "Activation applied to the layer output."
    )
    residual_connection_option: ResidualConnectionOptions | None = optional_field(
        "Residual connection behavior. Enabled options require input_dim == output_dim."
    )
    dropout_probability: float | None = optional_field(
        "Dropout probability. Use 0.0 to disable."
    )
    layer_norm_position: LayerNormPositionOptions | None = optional_field(
        "Where layer normalization is applied."
    )
    gate_config: "GateConfig | None" = optional_field(
        "Optional layer gate config. Set to None to disable."
    )
    halting_config: "HaltingConfig | None" = optional_field(
        "Optional adaptive computation module. Set to None to disable halting."
    )
    memory_config: "DynamicMemoryConfig | None" = optional_field(
        "Optional dynamic memory module. Set to None to disable memory."
    )
    layer_model_config: ConfigBase | None = optional_field(
        "Config for the wrapped module, such as LinearLayerConfig."
    )

    def _registry_owner(self) -> type:
        from .layer import Layer

        return Layer


@dataclass
class LayerStackConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    hidden_dim: int | None = optional_field("Hidden feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    num_layers: int | None = optional_field("Total number of layers in the stack.")
    apply_output_pipeline_flag: bool | None = optional_field(
        "Apply the full layer pipeline to the final layer."
    )
    last_layer_bias_option: "LastLayerBiasOptions | None" = optional_field(
        "Bias behavior for the final layer."
    )
    shared_gate_config: "GateConfig | None" = optional_field(
        "Optional shared gate config used by all stack layers."
    )
    shared_halting_config: "HaltingConfig | None" = optional_field(
        "Optional halting config used to build one shared halting module for all stack layers."
    )
    shared_memory_config: "DynamicMemoryConfig | None" = optional_field(
        "Optional memory config used to build one shared dynamic memory module for all stack layers."
    )
    layer_config: LayerConfig | None = optional_field(
        "Base layer config used to build each stack layer."
    )

    def _registry_owner(self) -> type:
        from .stack import LayerStack

        return LayerStack


@dataclass
class RecurrentLayerConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    max_steps: int | None = optional_field("Maximum recurrent applications.")
    recurrent_layer_norm_position: LayerNormPositionOptions | None = optional_field(
        "Where layer normalization is applied within each recurrent step."
    )
    block_config: ConfigBase | None = optional_field(
        "ConfigBase block reused at every recurrent step. The built module must consume "
        "and return LayerState-compatible values and declare input_dim and output_dim fields."
    )
    gate_config: "GateConfig | None" = optional_field(
        "Optional recurrent gate config. Set to None to disable."
    )
    residual_connection_option: ResidualConnectionOptions = optional_field(
        "Residual connection behavior between recurrent steps. "
        "Use DISABLED to disable recurrent residuals."
    )
    halting_config: "HaltingConfig | None" = optional_field(
        "Optional recurrent adaptive computation module. Set to None to disable."
    )
    memory_config: "DynamicMemoryConfig | None" = optional_field(
        "Optional dynamic memory module applied around the recurrent block. "
        "Set to None to disable memory."
    )

    def _registry_owner(self) -> type:
        from .recurrent import RecurrentLayer

        return RecurrentLayer
