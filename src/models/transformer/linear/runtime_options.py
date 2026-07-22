from __future__ import annotations

from dataclasses import dataclass

from emperor.embedding.absolute import (
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.halting import (
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    ResidualConnectionOptions,
)
from emperor.memory import (
    DynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
)


@dataclass(frozen=True)
class TransformerStackOptions:
    num_layers: int = 3
    layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
    stack_gate_flag: bool = False
    stack_halting_flag: bool = False
    halting_option: type[HaltingConfig] = StickBreakingConfig
    halting_threshold: float | None = None
    memory_flag: bool = False
    recurrent_flag: bool = False
    recurrent_stack_gate_flag: bool = False
    recurrent_stack_halting_flag: bool = False
    recurrent_halting_option: type[HaltingConfig] = StickBreakingConfig
    recurrent_halting_threshold: float | None = None
    recurrent_max_steps: int = 2
    stack_residual_connection_option: ResidualConnectionOptions | None = None
    recurrent_residual_connection_option: ResidualConnectionOptions | None = None


@dataclass(frozen=True)
class SubmoduleStackOptions:
    hidden_dim: int = 128
    num_layers: int = 1
    last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
    apply_output_pipeline_flag: bool = False
    activation: ActivationOptions = ActivationOptions.DISABLED
    layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED
    residual_connection_option: ResidualConnectionOptions | None = None
    dropout_probability: float = 0.0
    bias_flag: bool = True


@dataclass(frozen=True)
class ControllerStackOptions:
    independent_flag: bool = False
    hidden_dim: int | None = None
    num_layers: int | None = None
    last_layer_bias_option: LastLayerBiasOptions | None = None
    apply_output_pipeline_flag: bool | None = None
    activation: ActivationOptions | None = None
    layer_norm_position: LayerNormPositionOptions | None = None
    residual_connection_option: ResidualConnectionOptions | None = None
    dropout_probability: float | None = None
    bias_flag: bool | None = None


def resolve_controller_stack_options(
    source: ControllerStackOptions,
    defaults: SubmoduleStackOptions,
) -> SubmoduleStackOptions:
    if not source.independent_flag:
        return defaults
    return SubmoduleStackOptions(
        hidden_dim=defaults.hidden_dim
        if source.hidden_dim is None
        else source.hidden_dim,
        num_layers=defaults.num_layers
        if source.num_layers is None
        else source.num_layers,
        last_layer_bias_option=(
            defaults.last_layer_bias_option
            if source.last_layer_bias_option is None
            else source.last_layer_bias_option
        ),
        apply_output_pipeline_flag=(
            defaults.apply_output_pipeline_flag
            if source.apply_output_pipeline_flag is None
            else source.apply_output_pipeline_flag
        ),
        activation=(
            defaults.activation if source.activation is None else source.activation
        ),
        layer_norm_position=(
            defaults.layer_norm_position
            if source.layer_norm_position is None
            else source.layer_norm_position
        ),
        residual_connection_option=(
            defaults.residual_connection_option
            if source.residual_connection_option is None
            else source.residual_connection_option
        ),
        dropout_probability=(
            defaults.dropout_probability
            if source.dropout_probability is None
            else source.dropout_probability
        ),
        bias_flag=defaults.bias_flag if source.bias_flag is None else source.bias_flag,
    )


@dataclass(frozen=True)
class LayerControllerOptions:
    stack_gate_flag: bool = False
    gate_option: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
    gate_activation: ActivationOptions | None = ActivationOptions.SIGMOID
    gate_stack_options: ControllerStackOptions = ControllerStackOptions()
    stack_halting_flag: bool = False
    halting_option: type[HaltingConfig] = StickBreakingConfig
    halting_threshold: float | None = None
    halting_dropout: float = 0.0
    halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        HaltingHiddenStateModeOptions.RAW
    )
    halting_stack_options: ControllerStackOptions = ControllerStackOptions()
    shared_gate_config: GateConfig | None = None


@dataclass(frozen=True)
class DynamicMemoryOptions:
    memory_flag: bool = False
    memory_option: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig
    memory_position_option: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE
    memory_test_time_training_learning_rate: float | None = None
    memory_test_time_training_num_inner_steps: int | None = None
    memory_stack_options: ControllerStackOptions = ControllerStackOptions()


@dataclass(frozen=True)
class RecurrentControllerOptions:
    recurrent_flag: bool = False
    recurrent_max_steps: int = 2
    recurrent_layer_norm_position: LayerNormPositionOptions = (
        LayerNormPositionOptions.DISABLED
    )
    recurrent_stack_gate_flag: bool = False
    recurrent_gate_option: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
    recurrent_gate_activation: ActivationOptions | None = ActivationOptions.SIGMOID
    recurrent_gate_stack_options: ControllerStackOptions = ControllerStackOptions()
    recurrent_stack_halting_flag: bool = False
    recurrent_halting_option: type[HaltingConfig] = StickBreakingConfig
    recurrent_halting_threshold: float | None = None
    recurrent_halting_dropout: float = 0.0
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
        HaltingHiddenStateModeOptions.RAW
    )
    recurrent_halting_stack_options: ControllerStackOptions = ControllerStackOptions()


@dataclass(frozen=True)
class TransformerAttentionOptions:
    num_heads: int = 4
    add_key_value_bias_flag: bool = False
    zero_attention_flag: bool = False
    stack_options: SubmoduleStackOptions = SubmoduleStackOptions()
    layer_controller_options: LayerControllerOptions = LayerControllerOptions()
    dynamic_memory_options: DynamicMemoryOptions = DynamicMemoryOptions()
    recurrent_controller_options: RecurrentControllerOptions = (
        RecurrentControllerOptions()
    )


@dataclass(frozen=True)
class TransformerFeedForwardOptions:
    stack_options: SubmoduleStackOptions = SubmoduleStackOptions(
        hidden_dim=512,
        num_layers=2,
        activation=ActivationOptions.RELU,
        dropout_probability=0.1,
    )
    layer_controller_options: LayerControllerOptions = LayerControllerOptions()
    dynamic_memory_options: DynamicMemoryOptions = DynamicMemoryOptions()
    recurrent_controller_options: RecurrentControllerOptions = (
        RecurrentControllerOptions()
    )


@dataclass(frozen=True)
class RuntimeOptions:
    batch_size: int = 64
    learning_rate: float = 1.0
    vocab_size: int = 8192
    model_dim: int = 128
    source_sequence_length: int = 64
    target_sequence_length: int = 64
    dropout_probability: float = 0.1
    positional_embedding_option: type = TextSinusoidalPositionalEmbeddingConfig
    encoder_options: TransformerStackOptions = TransformerStackOptions()
    decoder_options: TransformerStackOptions = TransformerStackOptions()
    encoder_attention_options: TransformerAttentionOptions = (
        TransformerAttentionOptions()
    )
    decoder_self_attention_options: TransformerAttentionOptions = (
        TransformerAttentionOptions()
    )
    decoder_cross_attention_options: TransformerAttentionOptions = (
        TransformerAttentionOptions()
    )
    encoder_feed_forward_options: TransformerFeedForwardOptions = (
        TransformerFeedForwardOptions()
    )
    decoder_feed_forward_options: TransformerFeedForwardOptions = (
        TransformerFeedForwardOptions()
    )


__all__ = [
    "ControllerStackOptions",
    "DynamicMemoryOptions",
    "LayerControllerOptions",
    "RecurrentControllerOptions",
    "RuntimeOptions",
    "SubmoduleStackOptions",
    "TransformerAttentionOptions",
    "TransformerFeedForwardOptions",
    "TransformerStackOptions",
]
