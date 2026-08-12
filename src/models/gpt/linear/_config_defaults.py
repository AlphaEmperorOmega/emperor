from dataclasses import dataclass
from types import ModuleType
from typing import Literal, Protocol

from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from models.gpt.linear.runtime_options import (
    DynamicMemoryOptions,
    GptEmbeddingOptions,
    GptLmHeadOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    SubmoduleStackSource,
    TransformerAttentionOptions,
    TransformerDecoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)


class _ConfigDefaults(Protocol):
    EMBEDDING_LAYER_NORM_FLAG: bool
    EMBEDDING_DROPOUT_PROBABILITY: float
    LM_HEAD_WEIGHT_TYING_FLAG: bool
    LM_HEAD_BIAS_FLAG: bool
    POSITIONAL_EMBEDDING_OPTION: type[AbsolutePositionalEmbeddingConfig]
    POSITIONAL_EMBEDDING_PADDING_IDX: int | None
    POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG: bool
    HIDDEN_DIM: int
    STACK_NUM_LAYERS: int
    STACK_ACTIVATION: ActivationOptions
    STACK_DROPOUT_PROBABILITY: float
    LAYER_NORM_POSITION: LayerNormPositionOptions
    ATTN_NUM_HEADS: int
    ATTN_NUM_LAYERS: int
    ATTN_BIAS_FLAG: bool
    ATTN_ADD_KEY_VALUE_BIAS_FLAG: bool
    FF_NUM_LAYERS: int
    FF_BIAS_FLAG: bool
    FF_STACK_HIDDEN_DIM: int
    STACK_BIAS_FLAG: bool
    STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None
    STACK_RESIDUAL_MODEL_FLAG: bool
    STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions
    STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool


def gpt_embedding_options(config: _ConfigDefaults) -> GptEmbeddingOptions:
    return GptEmbeddingOptions(
        layer_norm_flag=config.EMBEDDING_LAYER_NORM_FLAG,
        dropout_probability=config.EMBEDDING_DROPOUT_PROBABILITY,
    )


def gpt_lm_head_options(config: _ConfigDefaults) -> GptLmHeadOptions:
    return GptLmHeadOptions(
        weight_tying_flag=config.LM_HEAD_WEIGHT_TYING_FLAG,
        bias_flag=config.LM_HEAD_BIAS_FLAG,
    )


def gpt_positional_embedding_options(
    config: _ConfigDefaults,
) -> TransformerPositionalEmbeddingOptions:
    return TransformerPositionalEmbeddingOptions(
        option=config.POSITIONAL_EMBEDDING_OPTION,
        padding_idx=config.POSITIONAL_EMBEDDING_PADDING_IDX,
        auto_expand_flag=config.POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG,
    )


def gpt_decoder_options(config: _ConfigDefaults) -> TransformerDecoderOptions:
    return TransformerDecoderOptions(
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.STACK_NUM_LAYERS,
        activation=config.STACK_ACTIVATION,
        dropout_probability=config.STACK_DROPOUT_PROBABILITY,
        layer_norm_position=config.LAYER_NORM_POSITION,
    )


def gpt_attention_options(config: _ConfigDefaults) -> TransformerAttentionOptions:
    return TransformerAttentionOptions(
        num_heads=config.ATTN_NUM_HEADS,
        num_layers=config.ATTN_NUM_LAYERS,
        bias_flag=config.ATTN_BIAS_FLAG,
        add_key_value_bias_flag=config.ATTN_ADD_KEY_VALUE_BIAS_FLAG,
    )


def gpt_feed_forward_options(
    config: _ConfigDefaults,
) -> TransformerFeedForwardOptions:
    return TransformerFeedForwardOptions(
        num_layers=config.FF_NUM_LAYERS,
        bias_flag=config.FF_BIAS_FLAG,
    )


def scaled_feed_forward_hidden_dim(config: _ConfigDefaults, hidden_dim: int) -> int:
    if config.HIDDEN_DIM > 0 and config.FF_STACK_HIDDEN_DIM % config.HIDDEN_DIM == 0:
        return hidden_dim * (config.FF_STACK_HIDDEN_DIM // config.HIDDEN_DIM)
    return config.FF_STACK_HIDDEN_DIM


def main_layer_stack_options(config: _ConfigDefaults) -> MainLayerStackOptions:
    return MainLayerStackOptions(
        bias_flag=config.STACK_BIAS_FLAG,
        layer_norm_position=config.LAYER_NORM_POSITION,
        num_layers=config.STACK_NUM_LAYERS,
        activation=config.STACK_ACTIVATION,
        residual_connection_option=config.STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.STACK_DROPOUT_PROBABILITY,
        last_layer_bias_option=config.STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
    )


@dataclass(frozen=True)
class _SubmoduleStackDefaults:
    hidden_dim: int
    num_layers: int
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    dropout_probability: float
    bias_flag: bool


def _submodule_stack_options(
    defaults: _SubmoduleStackDefaults,
) -> SubmoduleStackOptions:
    return SubmoduleStackOptions(
        hidden_dim=defaults.hidden_dim,
        num_layers=defaults.num_layers,
        last_layer_bias_option=defaults.last_layer_bias_option,
        apply_output_pipeline_flag=defaults.apply_output_pipeline_flag,
        activation=defaults.activation,
        layer_norm_position=defaults.layer_norm_position,
        residual_connection_option=defaults.residual_connection_option,
        residual_model_flag=defaults.residual_model_flag,
        dropout_probability=defaults.dropout_probability,
        bias_flag=defaults.bias_flag,
    )


def submodule_stack_options(
    config: ModuleType,
    stack_options: MainLayerStackOptions,
) -> SubmoduleStackOptions:
    return _submodule_stack_options(
        _SubmoduleStackDefaults(
            hidden_dim=config.SUBMODULE_STACK_HIDDEN_DIM,
            num_layers=config.SUBMODULE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=(
                config.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.SUBMODULE_STACK_ACTIVATION,
            layer_norm_position=config.SUBMODULE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.SUBMODULE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
            bias_flag=stack_options.bias_flag,
        )
    )


def attention_projection_stack_options(
    config: ModuleType,
    decoder_options: TransformerDecoderOptions,
    attention_options: TransformerAttentionOptions,
) -> SubmoduleStackOptions:
    return _submodule_stack_options(
        _SubmoduleStackDefaults(
            hidden_dim=decoder_options.hidden_dim,
            num_layers=attention_options.num_layers,
            last_layer_bias_option=config.ATTN_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            activation=decoder_options.activation,
            layer_norm_position=config.ATTN_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.ATTN_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ATTN_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ATTN_STACK_DROPOUT_PROBABILITY,
            bias_flag=attention_options.bias_flag,
        )
    )


def feed_forward_stack_options(
    config: ModuleType,
    decoder_options: TransformerDecoderOptions,
    feed_forward_options: TransformerFeedForwardOptions,
) -> SubmoduleStackOptions:
    return _submodule_stack_options(
        _SubmoduleStackDefaults(
            hidden_dim=scaled_feed_forward_hidden_dim(
                config, decoder_options.hidden_dim
            ),
            num_layers=feed_forward_options.num_layers,
            last_layer_bias_option=config.FF_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            activation=decoder_options.activation,
            layer_norm_position=config.FF_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.FF_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.FF_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=decoder_options.dropout_probability,
            bias_flag=feed_forward_options.bias_flag,
        )
    )


_LinearControlRole = Literal["main", "attention", "feed_forward"]
_ControllerStackRole = Literal[
    "gate", "halting", "memory", "recurrent_gate", "recurrent_halting"
]


@dataclass(frozen=True)
class _ControllerStackDefaults:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_pipeline_flag: bool | None
    activation: ActivationOptions | None
    layer_norm_position: LayerNormPositionOptions | None
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    dropout_probability: float | None
    bias_flag: bool | None


@dataclass(frozen=True)
class _ControllerStackGroups:
    gate: _ControllerStackDefaults
    halting: _ControllerStackDefaults
    memory: _ControllerStackDefaults
    recurrent_gate: _ControllerStackDefaults
    recurrent_halting: _ControllerStackDefaults


def _main_controller_stack_defaults(config: ModuleType) -> _ControllerStackGroups:
    return _ControllerStackGroups(
        gate=_ControllerStackDefaults(
            independent_flag=config.GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.GATE_STACK_HIDDEN_DIM,
            num_layers=config.GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            activation=config.GATE_STACK_ACTIVATION,
            layer_norm_position=config.GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.GATE_STACK_BIAS_FLAG,
        ),
        halting=_ControllerStackDefaults(
            independent_flag=config.HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.HALTING_STACK_HIDDEN_DIM,
            num_layers=config.HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            activation=config.HALTING_STACK_ACTIVATION,
            layer_norm_position=config.HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.HALTING_STACK_BIAS_FLAG,
        ),
        memory=_ControllerStackDefaults(
            independent_flag=config.MEMORY_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.MEMORY_STACK_HIDDEN_DIM,
            num_layers=config.MEMORY_STACK_NUM_LAYERS,
            last_layer_bias_option=config.MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            activation=config.MEMORY_STACK_ACTIVATION,
            layer_norm_position=config.MEMORY_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.MEMORY_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.MEMORY_STACK_BIAS_FLAG,
        ),
        recurrent_gate=_ControllerStackDefaults(
            independent_flag=config.RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.RECURRENT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.RECURRENT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=(config.RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION),
            apply_output_pipeline_flag=(
                config.RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.RECURRENT_GATE_STACK_ACTIVATION,
            layer_norm_position=config.RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.RECURRENT_GATE_STACK_BIAS_FLAG,
        ),
        recurrent_halting=_ControllerStackDefaults(
            independent_flag=config.RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.RECURRENT_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.RECURRENT_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config.RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_pipeline_flag=(
                config.RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.RECURRENT_HALTING_STACK_ACTIVATION,
            layer_norm_position=config.RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.RECURRENT_HALTING_STACK_BIAS_FLAG,
        ),
    )


def _attention_controller_stack_defaults(
    config: ModuleType,
) -> _ControllerStackGroups:
    return _ControllerStackGroups(
        gate=_ControllerStackDefaults(
            independent_flag=config.ATTN_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ATTN_GATE_STACK_HIDDEN_DIM,
            num_layers=config.ATTN_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.ATTN_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=(
                config.ATTN_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.ATTN_GATE_STACK_ACTIVATION,
            layer_norm_position=config.ATTN_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.ATTN_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.ATTN_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ATTN_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ATTN_GATE_STACK_BIAS_FLAG,
        ),
        halting=_ControllerStackDefaults(
            independent_flag=config.ATTN_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ATTN_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.ATTN_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=(config.ATTN_HALTING_STACK_LAST_LAYER_BIAS_OPTION),
            apply_output_pipeline_flag=(
                config.ATTN_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.ATTN_HALTING_STACK_ACTIVATION,
            layer_norm_position=config.ATTN_HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.ATTN_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.ATTN_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ATTN_HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ATTN_HALTING_STACK_BIAS_FLAG,
        ),
        memory=_ControllerStackDefaults(
            independent_flag=config.ATTN_MEMORY_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ATTN_MEMORY_STACK_HIDDEN_DIM,
            num_layers=config.ATTN_MEMORY_STACK_NUM_LAYERS,
            last_layer_bias_option=config.ATTN_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=(
                config.ATTN_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.ATTN_MEMORY_STACK_ACTIVATION,
            layer_norm_position=config.ATTN_MEMORY_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.ATTN_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.ATTN_MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ATTN_MEMORY_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ATTN_MEMORY_STACK_BIAS_FLAG,
        ),
        recurrent_gate=_ControllerStackDefaults(
            independent_flag=config.ATTN_RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ATTN_RECURRENT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.ATTN_RECURRENT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config.ATTN_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_pipeline_flag=(
                config.ATTN_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.ATTN_RECURRENT_GATE_STACK_ACTIVATION,
            layer_norm_position=(config.ATTN_RECURRENT_GATE_STACK_LAYER_NORM_POSITION),
            residual_connection_option=(
                config.ATTN_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=(config.ATTN_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG),
            dropout_probability=(config.ATTN_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY),
            bias_flag=config.ATTN_RECURRENT_GATE_STACK_BIAS_FLAG,
        ),
        recurrent_halting=_ControllerStackDefaults(
            independent_flag=config.ATTN_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ATTN_RECURRENT_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.ATTN_RECURRENT_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config.ATTN_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_pipeline_flag=(
                config.ATTN_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.ATTN_RECURRENT_HALTING_STACK_ACTIVATION,
            layer_norm_position=(
                config.ATTN_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION
            ),
            residual_connection_option=(
                config.ATTN_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=(
                config.ATTN_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config.ATTN_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY
            ),
            bias_flag=config.ATTN_RECURRENT_HALTING_STACK_BIAS_FLAG,
        ),
    )


def _feed_forward_controller_stack_defaults(
    config: ModuleType,
) -> _ControllerStackGroups:
    return _ControllerStackGroups(
        gate=_ControllerStackDefaults(
            independent_flag=config.FF_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.FF_GATE_STACK_HIDDEN_DIM,
            num_layers=config.FF_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.FF_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=(
                config.FF_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.FF_GATE_STACK_ACTIVATION,
            layer_norm_position=config.FF_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.FF_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.FF_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.FF_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.FF_GATE_STACK_BIAS_FLAG,
        ),
        halting=_ControllerStackDefaults(
            independent_flag=config.FF_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.FF_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.FF_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=config.FF_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=(
                config.FF_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.FF_HALTING_STACK_ACTIVATION,
            layer_norm_position=config.FF_HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.FF_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.FF_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.FF_HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.FF_HALTING_STACK_BIAS_FLAG,
        ),
        memory=_ControllerStackDefaults(
            independent_flag=config.FF_MEMORY_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.FF_MEMORY_STACK_HIDDEN_DIM,
            num_layers=config.FF_MEMORY_STACK_NUM_LAYERS,
            last_layer_bias_option=config.FF_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=(
                config.FF_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.FF_MEMORY_STACK_ACTIVATION,
            layer_norm_position=config.FF_MEMORY_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.FF_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.FF_MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.FF_MEMORY_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.FF_MEMORY_STACK_BIAS_FLAG,
        ),
        recurrent_gate=_ControllerStackDefaults(
            independent_flag=config.FF_RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.FF_RECURRENT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.FF_RECURRENT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config.FF_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_pipeline_flag=(
                config.FF_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.FF_RECURRENT_GATE_STACK_ACTIVATION,
            layer_norm_position=(config.FF_RECURRENT_GATE_STACK_LAYER_NORM_POSITION),
            residual_connection_option=(
                config.FF_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.FF_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=(config.FF_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY),
            bias_flag=config.FF_RECURRENT_GATE_STACK_BIAS_FLAG,
        ),
        recurrent_halting=_ControllerStackDefaults(
            independent_flag=config.FF_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.FF_RECURRENT_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.FF_RECURRENT_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config.FF_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_pipeline_flag=(
                config.FF_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.FF_RECURRENT_HALTING_STACK_ACTIVATION,
            layer_norm_position=(config.FF_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION),
            residual_connection_option=(
                config.FF_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=(config.FF_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG),
            dropout_probability=(config.FF_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY),
            bias_flag=config.FF_RECURRENT_HALTING_STACK_BIAS_FLAG,
        ),
    )


def _controller_stack_source(
    config: ModuleType,
    role: _LinearControlRole,
    stack: _ControllerStackRole,
) -> SubmoduleStackSource:
    if role == "main":
        groups = _main_controller_stack_defaults(config)
    elif role == "attention":
        groups = _attention_controller_stack_defaults(config)
    else:
        groups = _feed_forward_controller_stack_defaults(config)

    if stack == "gate":
        defaults = groups.gate
    elif stack == "halting":
        defaults = groups.halting
    elif stack == "memory":
        defaults = groups.memory
    elif stack == "recurrent_gate":
        defaults = groups.recurrent_gate
    else:
        defaults = groups.recurrent_halting

    return SubmoduleStackSource(
        independent_flag=defaults.independent_flag,
        hidden_dim=defaults.hidden_dim,
        num_layers=defaults.num_layers,
        last_layer_bias_option=defaults.last_layer_bias_option,
        apply_output_pipeline_flag=defaults.apply_output_pipeline_flag,
        activation=defaults.activation,
        layer_norm_position=defaults.layer_norm_position,
        residual_connection_option=defaults.residual_connection_option,
        residual_model_flag=defaults.residual_model_flag,
        dropout_probability=defaults.dropout_probability,
        bias_flag=defaults.bias_flag,
    )


def linears_layer_controller_options(
    config: ModuleType,
    role: _LinearControlRole,
) -> LayerControllerOptions:
    if role == "main":
        stack_gate_flag = config.STACK_GATE_FLAG
        gate_option = config.GATE_OPTION
        gate_activation = config.GATE_ACTIVATION
        stack_halting_flag = config.STACK_HALTING_FLAG
        halting_option = config.HALTING_OPTION
        halting_threshold = config.HALTING_THRESHOLD
        halting_dropout = config.HALTING_DROPOUT
        halting_hidden_state_mode = config.HALTING_HIDDEN_STATE_MODE
    elif role == "attention":
        stack_gate_flag = config.ATTN_STACK_GATE_FLAG
        gate_option = config.ATTN_GATE_OPTION
        gate_activation = config.ATTN_GATE_ACTIVATION
        stack_halting_flag = config.ATTN_STACK_HALTING_FLAG
        halting_option = config.ATTN_HALTING_OPTION
        halting_threshold = config.ATTN_HALTING_THRESHOLD
        halting_dropout = config.ATTN_HALTING_DROPOUT
        halting_hidden_state_mode = config.ATTN_HALTING_HIDDEN_STATE_MODE
    else:
        stack_gate_flag = config.FF_STACK_GATE_FLAG
        gate_option = config.FF_GATE_OPTION
        gate_activation = config.FF_GATE_ACTIVATION
        stack_halting_flag = config.FF_STACK_HALTING_FLAG
        halting_option = config.FF_HALTING_OPTION
        halting_threshold = config.FF_HALTING_THRESHOLD
        halting_dropout = config.FF_HALTING_DROPOUT
        halting_hidden_state_mode = config.FF_HALTING_HIDDEN_STATE_MODE

    return LayerControllerOptions(
        stack_gate_flag=stack_gate_flag,
        gate_option=gate_option,
        gate_activation=gate_activation,
        gate_stack_source=_controller_stack_source(config, role, "gate"),
        stack_halting_flag=stack_halting_flag,
        halting_option=halting_option,
        halting_threshold=halting_threshold,
        halting_dropout=halting_dropout,
        halting_hidden_state_mode=halting_hidden_state_mode,
        halting_stack_source=_controller_stack_source(config, role, "halting"),
    )


def linears_dynamic_memory_options(
    config: ModuleType,
    role: _LinearControlRole,
) -> DynamicMemoryOptions:
    if role == "main":
        memory_flag = config.MEMORY_FLAG
        memory_option: type[DynamicMemoryConfig] = config.MEMORY_OPTION
        memory_position_option: MemoryPositionOptions = config.MEMORY_POSITION_OPTION
        learning_rate = config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        num_inner_steps = config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
    elif role == "attention":
        memory_flag = config.ATTN_MEMORY_FLAG
        memory_option = config.ATTN_MEMORY_OPTION
        memory_position_option = config.ATTN_MEMORY_POSITION_OPTION
        learning_rate = config.ATTN_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        num_inner_steps = config.ATTN_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
    else:
        memory_flag = config.FF_MEMORY_FLAG
        memory_option = config.FF_MEMORY_OPTION
        memory_position_option = config.FF_MEMORY_POSITION_OPTION
        learning_rate = config.FF_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        num_inner_steps = config.FF_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS

    return DynamicMemoryOptions(
        memory_flag=memory_flag,
        memory_option=memory_option,
        memory_position_option=memory_position_option,
        memory_test_time_training_learning_rate=learning_rate,
        memory_test_time_training_num_inner_steps=num_inner_steps,
        memory_stack_source=_controller_stack_source(config, role, "memory"),
    )


def linears_recurrent_controller_options(
    config: ModuleType,
    role: _LinearControlRole,
) -> RecurrentControllerOptions:
    if role == "main":
        recurrent_flag = config.RECURRENT_FLAG
        recurrent_max_steps = config.RECURRENT_MAX_STEPS
        recurrent_initial_iterations = config.RECURRENT_INITIAL_ITERATIONS
        recurrent_gradient_transition_count = config.RECURRENT_GRADIENT_TRANSITION_COUNT
        recurrent_iteration_increment = config.RECURRENT_ITERATION_INCREMENT
        recurrent_forward_calls_before_iteration_increment = (
            config.RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT
        )
        recurrent_layer_norm_position = config.RECURRENT_LAYER_NORM_POSITION
        recurrent_stack_gate_flag = config.RECURRENT_STACK_GATE_FLAG
        recurrent_gate_option = config.RECURRENT_GATE_OPTION
        recurrent_gate_activation = config.RECURRENT_GATE_ACTIVATION
        recurrent_stack_halting_flag = config.RECURRENT_STACK_HALTING_FLAG
        recurrent_halting_option: type[HaltingConfig] = config.RECURRENT_HALTING_OPTION
        recurrent_halting_threshold = config.RECURRENT_HALTING_THRESHOLD
        recurrent_halting_dropout = config.RECURRENT_HALTING_DROPOUT
        recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
            config.RECURRENT_HALTING_HIDDEN_STATE_MODE
        )
    elif role == "attention":
        recurrent_flag = config.ATTN_RECURRENT_FLAG
        recurrent_max_steps = config.ATTN_RECURRENT_MAX_STEPS
        recurrent_initial_iterations = 2
        recurrent_gradient_transition_count = None
        recurrent_iteration_increment = 1
        recurrent_forward_calls_before_iteration_increment = 1
        recurrent_layer_norm_position = config.ATTN_RECURRENT_LAYER_NORM_POSITION
        recurrent_stack_gate_flag = config.ATTN_RECURRENT_STACK_GATE_FLAG
        recurrent_gate_option = config.ATTN_RECURRENT_GATE_OPTION
        recurrent_gate_activation = config.ATTN_RECURRENT_GATE_ACTIVATION
        recurrent_stack_halting_flag = config.ATTN_RECURRENT_STACK_HALTING_FLAG
        recurrent_halting_option = config.ATTN_RECURRENT_HALTING_OPTION
        recurrent_halting_threshold = config.ATTN_RECURRENT_HALTING_THRESHOLD
        recurrent_halting_dropout = config.ATTN_RECURRENT_HALTING_DROPOUT
        recurrent_halting_hidden_state_mode = (
            config.ATTN_RECURRENT_HALTING_HIDDEN_STATE_MODE
        )
    else:
        recurrent_flag = config.FF_RECURRENT_FLAG
        recurrent_max_steps = config.FF_RECURRENT_MAX_STEPS
        recurrent_initial_iterations = 2
        recurrent_gradient_transition_count = None
        recurrent_iteration_increment = 1
        recurrent_forward_calls_before_iteration_increment = 1
        recurrent_layer_norm_position = config.FF_RECURRENT_LAYER_NORM_POSITION
        recurrent_stack_gate_flag = config.FF_RECURRENT_STACK_GATE_FLAG
        recurrent_gate_option = config.FF_RECURRENT_GATE_OPTION
        recurrent_gate_activation = config.FF_RECURRENT_GATE_ACTIVATION
        recurrent_stack_halting_flag = config.FF_RECURRENT_STACK_HALTING_FLAG
        recurrent_halting_option = config.FF_RECURRENT_HALTING_OPTION
        recurrent_halting_threshold = config.FF_RECURRENT_HALTING_THRESHOLD
        recurrent_halting_dropout = config.FF_RECURRENT_HALTING_DROPOUT
        recurrent_halting_hidden_state_mode = (
            config.FF_RECURRENT_HALTING_HIDDEN_STATE_MODE
        )

    return RecurrentControllerOptions(
        recurrent_flag=recurrent_flag,
        recurrent_max_steps=recurrent_max_steps,
        recurrent_initial_iterations=recurrent_initial_iterations,
        recurrent_gradient_transition_count=recurrent_gradient_transition_count,
        recurrent_iteration_increment=recurrent_iteration_increment,
        recurrent_forward_calls_before_iteration_increment=(
            recurrent_forward_calls_before_iteration_increment
        ),
        recurrent_layer_norm_position=recurrent_layer_norm_position,
        recurrent_stack_gate_flag=recurrent_stack_gate_flag,
        recurrent_gate_option=recurrent_gate_option,
        recurrent_gate_activation=recurrent_gate_activation,
        recurrent_gate_stack_source=_controller_stack_source(
            config, role, "recurrent_gate"
        ),
        recurrent_stack_halting_flag=recurrent_stack_halting_flag,
        recurrent_halting_option=recurrent_halting_option,
        recurrent_halting_threshold=recurrent_halting_threshold,
        recurrent_halting_dropout=recurrent_halting_dropout,
        recurrent_halting_hidden_state_mode=recurrent_halting_hidden_state_mode,
        recurrent_halting_stack_source=_controller_stack_source(
            config, role, "recurrent_halting"
        ),
    )
