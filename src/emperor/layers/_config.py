from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field
from emperor.layers._options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    ResidualConnectionOptions,
)

if TYPE_CHECKING:
    from emperor.halting import HaltingConfig
    from emperor.linears import LinearLayerConfig
    from emperor.memory import DynamicMemoryConfig


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
        from emperor.layers._composition.gate import LayerGate

        return LayerGate


@dataclass
class AttentionResidualConfig(ConfigBase):
    residual_dim: int | None = optional_field(
        "Residual feature dimension. ResidualConnection owners override this value "
        "from ResidualConfig.residual_dim."
    )
    block_size: int | None = optional_field(
        "Number of consecutive raw transformation outputs combined into one depth "
        "source. Use 1 for Full Attention Residuals and values greater than 1 for "
        "Block Attention Residuals. Defaults to 1."
    )
    rms_norm_epsilon: float | None = optional_field(
        "Numerical stability epsilon used to RMS-normalize routing keys. Defaults "
        "to 1e-6."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.attention_residual import AttentionResidual

        return AttentionResidual


@dataclass
class ResidualConfig(ConfigBase):
    residual_dim: int | None = optional_field(
        "Residual feature dimension. Layer owners override this value from their "
        "output dimension. Required for direct construction when model_config is "
        "provided or option is ATTENTION_RESIDUAL."
    )
    option: ResidualConnectionOptions | None = optional_field(
        "Residual composition mode. Providing ResidualConfig enables the residual "
        "connection; use residual_config=None to disable it."
    )
    model_config: "LinearLayerConfig | None" = optional_field(
        "Optional data-dependent coefficient model for weighted residual modes. "
        "When provided, the model receives the concatenated current and previous "
        "values and produces one raw mixing coefficient per feature. When omitted, "
        "weighted modes use a learned scalar nn.Parameter. RESIDUAL performs direct "
        "addition and ATTENTION_RESIDUAL performs depth mixing; neither accepts a "
        "coefficient model."
    )
    attention_config: "AttentionResidualConfig | None" = optional_field(
        "Optional depth-mixing config for ATTENTION_RESIDUAL. When omitted, Full "
        "Attention Residuals use block_size=1 and rms_norm_epsilon=1e-6. Other "
        "residual modes do not accept this config."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._composition.residual import ResidualConnection

        return ResidualConnection


@dataclass
class LayerConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    activation: ActivationOptions | None = optional_field(
        "Activation applied to the layer output."
    )
    residual_config: "ResidualConfig | None" = optional_field(
        "Optional residual connection config. Set to None to disable. Enabled options "
        "require input_dim == output_dim."
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
        from emperor.layers._layer import Layer

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
        "Optional halting config used to build one shared halting module for all "
        "stack layers."
    )
    shared_memory_config: "DynamicMemoryConfig | None" = optional_field(
        "Optional memory config used to build one shared dynamic memory module for "
        "all stack layers."
    )
    layer_config: LayerConfig | None = optional_field(
        "Base layer config used to build each stack layer."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._stack import LayerStack

        return LayerStack


@dataclass
class MirroredLayerStackConfig(LayerStackConfig):
    def _registry_owner(self) -> type:
        from emperor.layers._mirrored import MirroredLayerStack

        return MirroredLayerStack


@dataclass
class RecurrentLayerConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    max_steps: int | None = optional_field("Maximum recurrent applications.")
    recurrent_layer_norm_position: LayerNormPositionOptions | None = optional_field(
        "Where layer normalization is applied within each recurrent step."
    )
    block_config: ConfigBase | None = optional_field(
        "ConfigBase block reused at every recurrent step. The built module must "
        "consume and return LayerState-compatible values and declare input_dim and "
        "output_dim fields."
    )
    gate_config: "GateConfig | None" = optional_field(
        "Optional recurrent gate config. Set to None to disable."
    )
    residual_config: "ResidualConfig | None" = optional_field(
        "Optional residual connection config applied between recurrent steps. Set to "
        "None to disable recurrent residuals."
    )
    halting_config: "HaltingConfig | None" = optional_field(
        "Optional recurrent adaptive computation module. Set to None to disable."
    )
    memory_config: "DynamicMemoryConfig | None" = optional_field(
        "Optional dynamic memory module applied around the recurrent block. "
        "Set to None to disable memory."
    )

    def _registry_owner(self) -> type:
        from emperor.layers._recurrent import RecurrentLayer

        return RecurrentLayer
