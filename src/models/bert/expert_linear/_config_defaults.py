from dataclasses import dataclass
from types import ModuleType
from typing import Literal

from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from models.bert.expert_linear.runtime_options import (
    BertEmbeddingOptions,
    BertMlmHeadOptions,
    BertNspHeadOptions,
    DynamicMemoryOptions,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    SubmoduleStackSource,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
    resolve_experts_submodule_stack_options,
)


def bert_embedding_options(config: ModuleType) -> BertEmbeddingOptions:
    return BertEmbeddingOptions(
        token_type_vocab_size=config.TOKEN_TYPE_VOCAB_SIZE,
        layer_norm_flag=config.EMBEDDING_LAYER_NORM_FLAG,
        dropout_probability=config.EMBEDDING_DROPOUT_PROBABILITY,
    )


def bert_positional_embedding_options(
    config: ModuleType,
) -> TransformerPositionalEmbeddingOptions:
    return TransformerPositionalEmbeddingOptions(
        option=config.POSITIONAL_EMBEDDING_OPTION,
        padding_idx=config.POSITIONAL_EMBEDDING_PADDING_IDX,
        auto_expand_flag=config.POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG,
    )


def bert_encoder_options(config: ModuleType) -> TransformerEncoderOptions:
    return TransformerEncoderOptions(
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.STACK_NUM_LAYERS,
        activation=config.STACK_ACTIVATION,
        dropout_probability=config.STACK_DROPOUT_PROBABILITY,
        layer_norm_position=config.LAYER_NORM_POSITION,
        causal_attention_mask_flag=config.CAUSAL_ATTENTION_MASK_FLAG,
    )


def bert_attention_options(config: ModuleType) -> TransformerAttentionOptions:
    return TransformerAttentionOptions(
        num_heads=config.ATTN_NUM_HEADS,
        num_layers=config.ATTN_NUM_LAYERS,
        bias_flag=config.ATTN_BIAS_FLAG,
        add_key_value_bias_flag=config.ATTN_ADD_KEY_VALUE_BIAS_FLAG,
    )


def bert_feed_forward_options(config: ModuleType) -> TransformerFeedForwardOptions:
    return TransformerFeedForwardOptions(
        num_layers=config.FF_NUM_LAYERS,
        bias_flag=config.FF_BIAS_FLAG,
    )


def scaled_feed_forward_hidden_dim(config: ModuleType, hidden_dim: int) -> int:
    if config.HIDDEN_DIM > 0 and config.FF_STACK_HIDDEN_DIM % config.HIDDEN_DIM == 0:
        return hidden_dim * (config.FF_STACK_HIDDEN_DIM // config.HIDDEN_DIM)
    return config.FF_STACK_HIDDEN_DIM


def bert_mlm_head_options(config: ModuleType) -> BertMlmHeadOptions:
    return BertMlmHeadOptions(
        activation=config.MLM_ACTIVATION,
        dense_bias_flag=config.MLM_DENSE_BIAS_FLAG,
        layer_norm_flag=config.MLM_LAYER_NORM_FLAG,
        decoder_bias_flag=config.MLM_DECODER_BIAS_FLAG,
        decoder_weight_tying_flag=config.MLM_DECODER_WEIGHT_TYING_FLAG,
    )


def bert_nsp_head_options(config: ModuleType) -> BertNspHeadOptions:
    return BertNspHeadOptions(
        pooler_activation=config.NSP_POOLER_ACTIVATION,
        pooler_bias_flag=config.NSP_POOLER_BIAS_FLAG,
        output_dim=config.NSP_OUTPUT_DIM,
        head_bias_flag=config.NSP_HEAD_BIAS_FLAG,
    )


def main_layer_stack_options(config: ModuleType) -> MainLayerStackOptions:
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


def _base_submodule_stack_defaults(
    config: ModuleType,
    *,
    bias_flag: bool,
) -> _SubmoduleStackDefaults:
    return _SubmoduleStackDefaults(
        hidden_dim=config.SUBMODULE_STACK_HIDDEN_DIM,
        num_layers=config.SUBMODULE_STACK_NUM_LAYERS,
        last_layer_bias_option=config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.SUBMODULE_STACK_ACTIVATION,
        layer_norm_position=config.SUBMODULE_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.SUBMODULE_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
        bias_flag=bias_flag,
    )


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
        _base_submodule_stack_defaults(
            config,
            bias_flag=stack_options.bias_flag,
        )
    )


def attention_projection_stack_options(
    config: ModuleType,
    encoder_options: TransformerEncoderOptions,
    attention_options: TransformerAttentionOptions,
) -> SubmoduleStackOptions:
    return _submodule_stack_options(
        _SubmoduleStackDefaults(
            hidden_dim=encoder_options.hidden_dim,
            num_layers=attention_options.num_layers,
            last_layer_bias_option=config.ATTN_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            activation=encoder_options.activation,
            layer_norm_position=config.ATTN_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.ATTN_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ATTN_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ATTN_STACK_DROPOUT_PROBABILITY,
            bias_flag=attention_options.bias_flag,
        )
    )


def feed_forward_stack_options(
    config: ModuleType,
    encoder_options: TransformerEncoderOptions,
    feed_forward_options: TransformerFeedForwardOptions,
) -> SubmoduleStackOptions:
    return _submodule_stack_options(
        _SubmoduleStackDefaults(
            hidden_dim=scaled_feed_forward_hidden_dim(
                config, encoder_options.hidden_dim
            ),
            num_layers=feed_forward_options.num_layers,
            last_layer_bias_option=config.FF_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            activation=encoder_options.activation,
            layer_norm_position=config.FF_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.FF_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.FF_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=encoder_options.dropout_probability,
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


_ExpertsStackRole = Literal["base", "expert", "router"]
_ExpertsControlRole = Literal["main", "expert", "router"]


def _experts_stack_defaults(
    config: ModuleType,
    role: _ExpertsStackRole,
) -> _SubmoduleStackDefaults:
    if role == "base":
        return _base_submodule_stack_defaults(
            config,
            bias_flag=config.SUBMODULE_STACK_BIAS_FLAG,
        )
    if role == "expert":
        return _SubmoduleStackDefaults(
            hidden_dim=config.EXPERT_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_STACK_NUM_LAYERS,
            last_layer_bias_option=config.EXPERT_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.EXPERT_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            activation=config.EXPERT_STACK_ACTIVATION,
            layer_norm_position=config.EXPERT_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.EXPERT_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.EXPERT_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.EXPERT_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.EXPERT_BIAS_FLAG,
        )
    return _SubmoduleStackDefaults(
        hidden_dim=config.ROUTER_STACK_HIDDEN_DIM,
        num_layers=config.ROUTER_STACK_NUM_LAYERS,
        last_layer_bias_option=config.ROUTER_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.ROUTER_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.ROUTER_STACK_ACTIVATION,
        layer_norm_position=config.ROUTER_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.ROUTER_STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.ROUTER_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.ROUTER_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.ROUTER_BIAS_FLAG,
    )


def experts_submodule_stack_options(
    config: ModuleType,
    role: _ExpertsStackRole,
) -> ExpertsSubmoduleStackOptions:
    defaults = _experts_stack_defaults(config, role)
    return ExpertsSubmoduleStackOptions(
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


def experts_stack_options(
    config: ModuleType,
    hidden_dim: int,
) -> ExpertsStackOptions:
    defaults = main_layer_stack_options(config)
    return ExpertsStackOptions(
        hidden_dim=hidden_dim,
        bias_flag=defaults.bias_flag,
        layer_norm_position=defaults.layer_norm_position,
        num_layers=defaults.num_layers,
        activation=defaults.activation,
        residual_connection_option=defaults.residual_connection_option,
        residual_model_flag=defaults.residual_model_flag,
        dropout_probability=defaults.dropout_probability,
        last_layer_bias_option=defaults.last_layer_bias_option,
        apply_output_pipeline_flag=defaults.apply_output_pipeline_flag,
    )


def experts_role_stack_options(
    config: ModuleType,
    role: Literal["expert", "router"],
    defaults: ExpertsSubmoduleStackOptions,
) -> ExpertsSubmoduleStackOptions:
    if role == "expert":
        layer_norm_position = config.EXPERT_STACK_LAYER_NORM_POSITION
        apply_output_pipeline_flag = config.EXPERT_STACK_APPLY_OUTPUT_PIPELINE_FLAG
    else:
        layer_norm_position = config.ROUTER_STACK_LAYER_NORM_POSITION
        apply_output_pipeline_flag = config.ROUTER_STACK_APPLY_OUTPUT_PIPELINE_FLAG
    return resolve_experts_submodule_stack_options(
        defaults,
        layer_norm_position=layer_norm_position,
        apply_output_pipeline_flag=apply_output_pipeline_flag,
    )


def _expert_controller_stack_defaults(
    config: ModuleType,
) -> _ControllerStackGroups:
    return _ControllerStackGroups(
        gate=_ControllerStackDefaults(
            independent_flag=config.EXPERT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.EXPERT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.EXPERT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=(
                config.EXPERT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.EXPERT_GATE_STACK_ACTIVATION,
            layer_norm_position=config.EXPERT_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.EXPERT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.EXPERT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.EXPERT_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.EXPERT_GATE_STACK_BIAS_FLAG,
        ),
        halting=_ControllerStackDefaults(
            independent_flag=config.EXPERT_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.EXPERT_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=(config.EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION),
            apply_output_pipeline_flag=(
                config.EXPERT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.EXPERT_HALTING_STACK_ACTIVATION,
            layer_norm_position=config.EXPERT_HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.EXPERT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.EXPERT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.EXPERT_HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.EXPERT_HALTING_STACK_BIAS_FLAG,
        ),
        memory=_ControllerStackDefaults(
            independent_flag=config.EXPERT_MEMORY_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.EXPERT_MEMORY_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_MEMORY_STACK_NUM_LAYERS,
            last_layer_bias_option=config.EXPERT_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=(
                config.EXPERT_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.EXPERT_MEMORY_STACK_ACTIVATION,
            layer_norm_position=config.EXPERT_MEMORY_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.EXPERT_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.EXPERT_MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.EXPERT_MEMORY_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.EXPERT_MEMORY_STACK_BIAS_FLAG,
        ),
        recurrent_gate=_ControllerStackDefaults(
            independent_flag=config.EXPERT_RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.EXPERT_RECURRENT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_RECURRENT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config.EXPERT_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_pipeline_flag=(
                config.EXPERT_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.EXPERT_RECURRENT_GATE_STACK_ACTIVATION,
            layer_norm_position=(
                config.EXPERT_RECURRENT_GATE_STACK_LAYER_NORM_POSITION
            ),
            residual_connection_option=(
                config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=(
                config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config.EXPERT_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY
            ),
            bias_flag=config.EXPERT_RECURRENT_GATE_STACK_BIAS_FLAG,
        ),
        recurrent_halting=_ControllerStackDefaults(
            independent_flag=config.EXPERT_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.EXPERT_RECURRENT_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_RECURRENT_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config.EXPERT_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_pipeline_flag=(
                config.EXPERT_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.EXPERT_RECURRENT_HALTING_STACK_ACTIVATION,
            layer_norm_position=(
                config.EXPERT_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION
            ),
            residual_connection_option=(
                config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=(
                config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config.EXPERT_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY
            ),
            bias_flag=config.EXPERT_RECURRENT_HALTING_STACK_BIAS_FLAG,
        ),
    )


def _router_controller_stack_defaults(
    config: ModuleType,
) -> _ControllerStackGroups:
    return _ControllerStackGroups(
        gate=_ControllerStackDefaults(
            independent_flag=config.ROUTER_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_GATE_STACK_HIDDEN_DIM,
            num_layers=config.ROUTER_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.ROUTER_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=(
                config.ROUTER_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.ROUTER_GATE_STACK_ACTIVATION,
            layer_norm_position=config.ROUTER_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.ROUTER_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.ROUTER_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ROUTER_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ROUTER_GATE_STACK_BIAS_FLAG,
        ),
        halting=_ControllerStackDefaults(
            independent_flag=config.ROUTER_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.ROUTER_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=(config.ROUTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION),
            apply_output_pipeline_flag=(
                config.ROUTER_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.ROUTER_HALTING_STACK_ACTIVATION,
            layer_norm_position=config.ROUTER_HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.ROUTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.ROUTER_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ROUTER_HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ROUTER_HALTING_STACK_BIAS_FLAG,
        ),
        memory=_ControllerStackDefaults(
            independent_flag=config.ROUTER_MEMORY_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_MEMORY_STACK_HIDDEN_DIM,
            num_layers=config.ROUTER_MEMORY_STACK_NUM_LAYERS,
            last_layer_bias_option=config.ROUTER_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=(
                config.ROUTER_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.ROUTER_MEMORY_STACK_ACTIVATION,
            layer_norm_position=config.ROUTER_MEMORY_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config.ROUTER_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config.ROUTER_MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ROUTER_MEMORY_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ROUTER_MEMORY_STACK_BIAS_FLAG,
        ),
        recurrent_gate=_ControllerStackDefaults(
            independent_flag=config.ROUTER_RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_RECURRENT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.ROUTER_RECURRENT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config.ROUTER_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_pipeline_flag=(
                config.ROUTER_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.ROUTER_RECURRENT_GATE_STACK_ACTIVATION,
            layer_norm_position=(
                config.ROUTER_RECURRENT_GATE_STACK_LAYER_NORM_POSITION
            ),
            residual_connection_option=(
                config.ROUTER_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=(
                config.ROUTER_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config.ROUTER_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY
            ),
            bias_flag=config.ROUTER_RECURRENT_GATE_STACK_BIAS_FLAG,
        ),
        recurrent_halting=_ControllerStackDefaults(
            independent_flag=config.ROUTER_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_RECURRENT_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.ROUTER_RECURRENT_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config.ROUTER_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_pipeline_flag=(
                config.ROUTER_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config.ROUTER_RECURRENT_HALTING_STACK_ACTIVATION,
            layer_norm_position=(
                config.ROUTER_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION
            ),
            residual_connection_option=(
                config.ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=(
                config.ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config.ROUTER_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY
            ),
            bias_flag=config.ROUTER_RECURRENT_HALTING_STACK_BIAS_FLAG,
        ),
    )


def _experts_controller_stack_source(
    config: ModuleType,
    role: _ExpertsControlRole,
    stack: _ControllerStackRole,
) -> ExpertsSubmoduleStackSource:
    if role == "main":
        groups = _main_controller_stack_defaults(config)
    elif role == "expert":
        groups = _expert_controller_stack_defaults(config)
    else:
        groups = _router_controller_stack_defaults(config)

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

    return ExpertsSubmoduleStackSource(
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


def experts_layer_controller_options(
    config: ModuleType,
    role: _ExpertsControlRole,
) -> ExpertsLayerControllerOptions:
    if role == "main":
        stack_gate_flag = config.STACK_GATE_FLAG
        gate_option = config.GATE_OPTION
        gate_activation = config.GATE_ACTIVATION
        stack_halting_flag = config.STACK_HALTING_FLAG
        halting_option = config.HALTING_OPTION
        halting_threshold = config.HALTING_THRESHOLD
        halting_dropout = config.HALTING_DROPOUT
        halting_hidden_state_mode = config.HALTING_HIDDEN_STATE_MODE
        halting_output_dim = config.HALTING_OUTPUT_DIM
    elif role == "expert":
        stack_gate_flag = config.EXPERT_STACK_GATE_FLAG
        gate_option = config.EXPERT_GATE_OPTION
        gate_activation = config.EXPERT_GATE_ACTIVATION
        stack_halting_flag = config.EXPERT_STACK_HALTING_FLAG
        halting_option = config.EXPERT_HALTING_OPTION
        halting_threshold = config.EXPERT_HALTING_THRESHOLD
        halting_dropout = config.EXPERT_HALTING_DROPOUT
        halting_hidden_state_mode = config.EXPERT_HALTING_HIDDEN_STATE_MODE
        halting_output_dim = config.EXPERT_HALTING_OUTPUT_DIM
    else:
        stack_gate_flag = config.ROUTER_STACK_GATE_FLAG
        gate_option = config.ROUTER_GATE_OPTION
        gate_activation = config.ROUTER_GATE_ACTIVATION
        stack_halting_flag = config.ROUTER_STACK_HALTING_FLAG
        halting_option = config.ROUTER_HALTING_OPTION
        halting_threshold = config.ROUTER_HALTING_THRESHOLD
        halting_dropout = config.ROUTER_HALTING_DROPOUT
        halting_hidden_state_mode = config.ROUTER_HALTING_HIDDEN_STATE_MODE
        halting_output_dim = config.ROUTER_HALTING_OUTPUT_DIM

    return ExpertsLayerControllerOptions(
        stack_gate_flag=stack_gate_flag,
        gate_option=gate_option,
        gate_activation=gate_activation,
        gate_stack_source=_experts_controller_stack_source(config, role, "gate"),
        stack_halting_flag=stack_halting_flag,
        halting_option=halting_option,
        halting_threshold=halting_threshold,
        halting_dropout=halting_dropout,
        halting_hidden_state_mode=halting_hidden_state_mode,
        halting_stack_source=_experts_controller_stack_source(config, role, "halting"),
        halting_output_dim=halting_output_dim,
    )


def experts_dynamic_memory_options(
    config: ModuleType,
    role: _ExpertsControlRole,
) -> ExpertsDynamicMemoryOptions:
    if role == "main":
        memory_flag = config.MEMORY_FLAG
        memory_option = config.MEMORY_OPTION
        memory_position_option = config.MEMORY_POSITION_OPTION
        learning_rate = config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        num_inner_steps = config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
    elif role == "expert":
        memory_flag = config.EXPERT_MEMORY_FLAG
        memory_option = config.EXPERT_MEMORY_OPTION
        memory_position_option = config.EXPERT_MEMORY_POSITION_OPTION
        learning_rate = config.EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        num_inner_steps = config.EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
    else:
        memory_flag = config.ROUTER_MEMORY_FLAG
        memory_option = config.ROUTER_MEMORY_OPTION
        memory_position_option = config.ROUTER_MEMORY_POSITION_OPTION
        learning_rate = config.ROUTER_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        num_inner_steps = config.ROUTER_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS

    return ExpertsDynamicMemoryOptions(
        memory_flag=memory_flag,
        memory_option=memory_option,
        memory_position_option=memory_position_option,
        memory_test_time_training_learning_rate=learning_rate,
        memory_test_time_training_num_inner_steps=num_inner_steps,
        memory_stack_source=_experts_controller_stack_source(config, role, "memory"),
    )


def experts_recurrent_controller_options(
    config: ModuleType,
    role: _ExpertsControlRole,
) -> ExpertsRecurrentControllerOptions:
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
        recurrent_halting_option = (
            ExpertsRecurrentControllerOptions.recurrent_halting_option
        )
        recurrent_halting_threshold = config.RECURRENT_HALTING_THRESHOLD
        recurrent_halting_dropout = config.RECURRENT_HALTING_DROPOUT
        recurrent_halting_hidden_state_mode = config.RECURRENT_HALTING_HIDDEN_STATE_MODE
    elif role == "expert":
        recurrent_flag = config.EXPERT_RECURRENT_FLAG
        recurrent_max_steps = config.EXPERT_RECURRENT_MAX_STEPS
        recurrent_initial_iterations = 2
        recurrent_gradient_transition_count = None
        recurrent_iteration_increment = 1
        recurrent_forward_calls_before_iteration_increment = 1
        recurrent_layer_norm_position = config.EXPERT_RECURRENT_LAYER_NORM_POSITION
        recurrent_stack_gate_flag = config.EXPERT_RECURRENT_STACK_GATE_FLAG
        recurrent_gate_option = config.EXPERT_RECURRENT_GATE_OPTION
        recurrent_gate_activation = config.EXPERT_RECURRENT_GATE_ACTIVATION
        recurrent_stack_halting_flag = config.EXPERT_RECURRENT_STACK_HALTING_FLAG
        recurrent_halting_option = config.EXPERT_RECURRENT_HALTING_OPTION
        recurrent_halting_threshold = config.EXPERT_RECURRENT_HALTING_THRESHOLD
        recurrent_halting_dropout = config.EXPERT_RECURRENT_HALTING_DROPOUT
        recurrent_halting_hidden_state_mode = (
            config.EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE
        )
    else:
        recurrent_flag = config.ROUTER_RECURRENT_FLAG
        recurrent_max_steps = config.ROUTER_RECURRENT_MAX_STEPS
        recurrent_initial_iterations = 2
        recurrent_gradient_transition_count = None
        recurrent_iteration_increment = 1
        recurrent_forward_calls_before_iteration_increment = 1
        recurrent_layer_norm_position = config.ROUTER_RECURRENT_LAYER_NORM_POSITION
        recurrent_stack_gate_flag = config.ROUTER_RECURRENT_STACK_GATE_FLAG
        recurrent_gate_option = config.ROUTER_RECURRENT_GATE_OPTION
        recurrent_gate_activation = config.ROUTER_RECURRENT_GATE_ACTIVATION
        recurrent_stack_halting_flag = config.ROUTER_RECURRENT_STACK_HALTING_FLAG
        recurrent_halting_option = config.ROUTER_RECURRENT_HALTING_OPTION
        recurrent_halting_threshold = config.ROUTER_RECURRENT_HALTING_THRESHOLD
        recurrent_halting_dropout = config.ROUTER_RECURRENT_HALTING_DROPOUT
        recurrent_halting_hidden_state_mode = (
            config.ROUTER_RECURRENT_HALTING_HIDDEN_STATE_MODE
        )

    return ExpertsRecurrentControllerOptions(
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
        recurrent_gate_stack_source=_experts_controller_stack_source(
            config, role, "recurrent_gate"
        ),
        recurrent_stack_halting_flag=recurrent_stack_halting_flag,
        recurrent_halting_option=recurrent_halting_option,
        recurrent_halting_threshold=recurrent_halting_threshold,
        recurrent_halting_dropout=recurrent_halting_dropout,
        recurrent_halting_hidden_state_mode=recurrent_halting_hidden_state_mode,
        recurrent_halting_stack_source=_experts_controller_stack_source(
            config, role, "recurrent_halting"
        ),
    )


def expert_layer_controller_options(
    config: ModuleType,
    provided: ExpertsLayerControllerOptions | None,
) -> ExpertsLayerControllerOptions:
    return (
        provided
        if provided is not None
        else experts_layer_controller_options(config, "expert")
    )


def expert_dynamic_memory_options(
    config: ModuleType,
    provided: ExpertsDynamicMemoryOptions | None,
) -> ExpertsDynamicMemoryOptions:
    return (
        provided
        if provided is not None
        else experts_dynamic_memory_options(config, "expert")
    )


def expert_recurrent_controller_options(
    config: ModuleType,
    provided: ExpertsRecurrentControllerOptions | None,
) -> ExpertsRecurrentControllerOptions:
    return (
        provided
        if provided is not None
        else experts_recurrent_controller_options(config, "expert")
    )


def experts_mixture_options(config: ModuleType) -> ExpertsMixtureOptions:
    return ExpertsMixtureOptions(
        top_k=config.TOP_K,
        num_experts=config.NUM_EXPERTS,
        capacity_factor=config.CAPACITY_FACTOR,
        dropped_token_behavior=config.DROPPED_TOKEN_BEHAVIOR,
        compute_expert_mixture_flag=config.COMPUTE_EXPERT_MIXTURE_FLAG,
        weighted_parameters_flag=config.WEIGHTED_PARAMETERS_FLAG,
        weighting_position_option=config.WEIGHTING_POSITION_OPTION,
        routing_initialization_mode=config.ROUTING_INITIALIZATION_MODE,
    )


def experts_sampler_options(config: ModuleType) -> ExpertsSamplerOptions:
    return ExpertsSamplerOptions(
        threshold=config.SAMPLER_THRESHOLD,
        filter_above_threshold=config.SAMPLER_FILTER_ABOVE_THRESHOLD,
        num_topk_samples=config.SAMPLER_NUM_TOPK_SAMPLES,
        normalize_probabilities_flag=config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
        noisy_topk_flag=config.SAMPLER_NOISY_TOPK_FLAG,
        coefficient_of_variation_loss_weight=config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
        switch_loss_weight=config.SAMPLER_SWITCH_LOSS_WEIGHT,
        zero_centred_loss_weight=config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT,
        mutual_information_loss_weight=config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT,
    )


def experts_router_options(config: ModuleType) -> ExpertsRouterOptions:
    return ExpertsRouterOptions(noisy_topk_flag=config.ROUTER_NOISY_TOPK_FLAG)
