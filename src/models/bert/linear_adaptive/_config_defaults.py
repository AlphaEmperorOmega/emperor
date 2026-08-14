from enum import Enum, auto
from types import ModuleType

from emperor.augmentations.adaptive_parameters import (
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDepthOptions,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from models.bert.linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
    BertEmbeddingOptions,
    BertMlmHeadOptions,
    BertNspHeadOptions,
    DynamicMemoryOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    SubmoduleStackSource,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)


class LinearRole(Enum):
    MAIN = auto()
    ATTENTION = auto()
    FEED_FORWARD = auto()


class _ControllerStackRole(Enum):
    MAIN_GATE = auto()
    MAIN_HALTING = auto()
    MAIN_MEMORY = auto()
    MAIN_RECURRENT_GATE = auto()
    MAIN_RECURRENT_HALTING = auto()
    ATTENTION_GATE = auto()
    ATTENTION_HALTING = auto()
    ATTENTION_MEMORY = auto()
    ATTENTION_RECURRENT_GATE = auto()
    ATTENTION_RECURRENT_HALTING = auto()
    FEED_FORWARD_GATE = auto()
    FEED_FORWARD_HALTING = auto()
    FEED_FORWARD_MEMORY = auto()
    FEED_FORWARD_RECURRENT_GATE = auto()
    FEED_FORWARD_RECURRENT_HALTING = auto()


class _AdaptiveParameter(Enum):
    WEIGHT = auto()
    BIAS = auto()
    DIAGONAL = auto()
    MASK = auto()


def bert_embedding_options(config: ModuleType) -> BertEmbeddingOptions:
    return BertEmbeddingOptions(
        token_type_vocab_size=config.TOKEN_TYPE_VOCAB_SIZE,
        layer_norm_flag=config.EMBEDDING_LAYER_NORM_FLAG,
        dropout_probability=config.EMBEDDING_DROPOUT_PROBABILITY,
    )


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


def linears_submodule_stack_options(
    config: ModuleType,
    role: LinearRole,
) -> SubmoduleStackOptions:
    if role is LinearRole.MAIN:
        return _submodule_stack_options(
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
            bias_flag=config.SUBMODULE_STACK_BIAS_FLAG,
        )
    if role is LinearRole.ATTENTION:
        return _submodule_stack_options(
            hidden_dim=config.ATTN_STACK_HIDDEN_DIM,
            num_layers=config.ATTN_NUM_LAYERS,
            last_layer_bias_option=config.ATTN_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            activation=config.ATTN_STACK_ACTIVATION,
            layer_norm_position=config.ATTN_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.ATTN_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ATTN_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ATTN_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ATTN_BIAS_FLAG,
        )
    return _submodule_stack_options(
        hidden_dim=config.FF_STACK_HIDDEN_DIM,
        num_layers=config.FF_NUM_LAYERS,
        last_layer_bias_option=config.FF_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.FF_STACK_ACTIVATION,
        layer_norm_position=config.FF_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.FF_STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.FF_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.FF_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.FF_BIAS_FLAG,
    )


def _submodule_stack_options(
    *,
    hidden_dim: int,
    num_layers: int,
    last_layer_bias_option: LastLayerBiasOptions,
    apply_output_pipeline_flag: bool,
    activation: ActivationOptions,
    layer_norm_position: LayerNormPositionOptions,
    residual_connection_option: type[ResidualConfig] | None,
    residual_model_flag: bool,
    dropout_probability: float,
    bias_flag: bool,
) -> SubmoduleStackOptions:
    return SubmoduleStackOptions(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=last_layer_bias_option,
        apply_output_pipeline_flag=apply_output_pipeline_flag,
        activation=activation,
        layer_norm_position=layer_norm_position,
        residual_connection_option=residual_connection_option,
        residual_model_flag=residual_model_flag,
        dropout_probability=dropout_probability,
        bias_flag=bias_flag,
    )


def linears_controller_stack_source(
    config: ModuleType,
    role: _ControllerStackRole,
) -> SubmoduleStackSource:
    if role in (
        _ControllerStackRole.MAIN_GATE,
        _ControllerStackRole.MAIN_HALTING,
        _ControllerStackRole.MAIN_MEMORY,
        _ControllerStackRole.MAIN_RECURRENT_GATE,
        _ControllerStackRole.MAIN_RECURRENT_HALTING,
    ):
        return _main_controller_stack_source(config, role)
    if role in (
        _ControllerStackRole.ATTENTION_GATE,
        _ControllerStackRole.ATTENTION_HALTING,
        _ControllerStackRole.ATTENTION_MEMORY,
        _ControllerStackRole.ATTENTION_RECURRENT_GATE,
        _ControllerStackRole.ATTENTION_RECURRENT_HALTING,
    ):
        return _attention_controller_stack_source(config, role)
    return _feed_forward_controller_stack_source(config, role)


def _main_controller_stack_source(
    config: ModuleType,
    role: _ControllerStackRole,
) -> SubmoduleStackSource:
    if role is _ControllerStackRole.MAIN_GATE:
        return _controller_stack_source(
            config.GATE_STACK_INDEPENDENT_FLAG,
            config.GATE_STACK_HIDDEN_DIM,
            config.GATE_STACK_NUM_LAYERS,
            config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
            config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.GATE_STACK_ACTIVATION,
            config.GATE_STACK_LAYER_NORM_POSITION,
            config.GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            config.GATE_STACK_RESIDUAL_MODEL_FLAG,
            config.GATE_STACK_DROPOUT_PROBABILITY,
            config.GATE_STACK_BIAS_FLAG,
        )
    if role is _ControllerStackRole.MAIN_HALTING:
        return _controller_stack_source(
            config.HALTING_STACK_INDEPENDENT_FLAG,
            config.HALTING_STACK_HIDDEN_DIM,
            config.HALTING_STACK_NUM_LAYERS,
            config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            config.HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.HALTING_STACK_ACTIVATION,
            config.HALTING_STACK_LAYER_NORM_POSITION,
            config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
            config.HALTING_STACK_RESIDUAL_MODEL_FLAG,
            config.HALTING_STACK_DROPOUT_PROBABILITY,
            config.HALTING_STACK_BIAS_FLAG,
        )
    if role is _ControllerStackRole.MAIN_MEMORY:
        return _controller_stack_source(
            config.MEMORY_STACK_INDEPENDENT_FLAG,
            config.MEMORY_STACK_HIDDEN_DIM,
            config.MEMORY_STACK_NUM_LAYERS,
            config.MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            config.MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.MEMORY_STACK_ACTIVATION,
            config.MEMORY_STACK_LAYER_NORM_POSITION,
            config.MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
            config.MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            config.MEMORY_STACK_DROPOUT_PROBABILITY,
            config.MEMORY_STACK_BIAS_FLAG,
        )
    if role is _ControllerStackRole.MAIN_RECURRENT_GATE:
        return _controller_stack_source(
            config.RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            config.RECURRENT_GATE_STACK_HIDDEN_DIM,
            config.RECURRENT_GATE_STACK_NUM_LAYERS,
            config.RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            config.RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.RECURRENT_GATE_STACK_ACTIVATION,
            config.RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
            config.RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            config.RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            config.RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
            config.RECURRENT_GATE_STACK_BIAS_FLAG,
        )
    if role is _ControllerStackRole.MAIN_RECURRENT_HALTING:
        return _controller_stack_source(
            config.RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
            config.RECURRENT_HALTING_STACK_HIDDEN_DIM,
            config.RECURRENT_HALTING_STACK_NUM_LAYERS,
            config.RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            config.RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.RECURRENT_HALTING_STACK_ACTIVATION,
            config.RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
            config.RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
            config.RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            config.RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
            config.RECURRENT_HALTING_STACK_BIAS_FLAG,
        )


def _attention_controller_stack_source(
    config: ModuleType,
    role: _ControllerStackRole,
) -> SubmoduleStackSource:
    if role is _ControllerStackRole.ATTENTION_GATE:
        return _controller_stack_source(
            config.ATTN_GATE_STACK_INDEPENDENT_FLAG,
            config.ATTN_GATE_STACK_HIDDEN_DIM,
            config.ATTN_GATE_STACK_NUM_LAYERS,
            config.ATTN_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            config.ATTN_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.ATTN_GATE_STACK_ACTIVATION,
            config.ATTN_GATE_STACK_LAYER_NORM_POSITION,
            config.ATTN_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            config.ATTN_GATE_STACK_RESIDUAL_MODEL_FLAG,
            config.ATTN_GATE_STACK_DROPOUT_PROBABILITY,
            config.ATTN_GATE_STACK_BIAS_FLAG,
        )
    if role is _ControllerStackRole.ATTENTION_HALTING:
        return _controller_stack_source(
            config.ATTN_HALTING_STACK_INDEPENDENT_FLAG,
            config.ATTN_HALTING_STACK_HIDDEN_DIM,
            config.ATTN_HALTING_STACK_NUM_LAYERS,
            config.ATTN_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            config.ATTN_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.ATTN_HALTING_STACK_ACTIVATION,
            config.ATTN_HALTING_STACK_LAYER_NORM_POSITION,
            config.ATTN_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
            config.ATTN_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            config.ATTN_HALTING_STACK_DROPOUT_PROBABILITY,
            config.ATTN_HALTING_STACK_BIAS_FLAG,
        )
    if role is _ControllerStackRole.ATTENTION_MEMORY:
        return _controller_stack_source(
            config.ATTN_MEMORY_STACK_INDEPENDENT_FLAG,
            config.ATTN_MEMORY_STACK_HIDDEN_DIM,
            config.ATTN_MEMORY_STACK_NUM_LAYERS,
            config.ATTN_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            config.ATTN_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.ATTN_MEMORY_STACK_ACTIVATION,
            config.ATTN_MEMORY_STACK_LAYER_NORM_POSITION,
            config.ATTN_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
            config.ATTN_MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            config.ATTN_MEMORY_STACK_DROPOUT_PROBABILITY,
            config.ATTN_MEMORY_STACK_BIAS_FLAG,
        )
    if role is _ControllerStackRole.ATTENTION_RECURRENT_GATE:
        return _controller_stack_source(
            config.ATTN_RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            config.ATTN_RECURRENT_GATE_STACK_HIDDEN_DIM,
            config.ATTN_RECURRENT_GATE_STACK_NUM_LAYERS,
            config.ATTN_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            config.ATTN_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.ATTN_RECURRENT_GATE_STACK_ACTIVATION,
            config.ATTN_RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
            config.ATTN_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            config.ATTN_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            config.ATTN_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
            config.ATTN_RECURRENT_GATE_STACK_BIAS_FLAG,
        )
    if role is _ControllerStackRole.ATTENTION_RECURRENT_HALTING:
        return _controller_stack_source(
            config.ATTN_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
            config.ATTN_RECURRENT_HALTING_STACK_HIDDEN_DIM,
            config.ATTN_RECURRENT_HALTING_STACK_NUM_LAYERS,
            config.ATTN_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            config.ATTN_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.ATTN_RECURRENT_HALTING_STACK_ACTIVATION,
            config.ATTN_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
            config.ATTN_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
            config.ATTN_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            config.ATTN_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
            config.ATTN_RECURRENT_HALTING_STACK_BIAS_FLAG,
        )


def _feed_forward_controller_stack_source(
    config: ModuleType,
    role: _ControllerStackRole,
) -> SubmoduleStackSource:
    if role is _ControllerStackRole.FEED_FORWARD_GATE:
        return _controller_stack_source(
            config.FF_GATE_STACK_INDEPENDENT_FLAG,
            config.FF_GATE_STACK_HIDDEN_DIM,
            config.FF_GATE_STACK_NUM_LAYERS,
            config.FF_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            config.FF_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.FF_GATE_STACK_ACTIVATION,
            config.FF_GATE_STACK_LAYER_NORM_POSITION,
            config.FF_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            config.FF_GATE_STACK_RESIDUAL_MODEL_FLAG,
            config.FF_GATE_STACK_DROPOUT_PROBABILITY,
            config.FF_GATE_STACK_BIAS_FLAG,
        )
    if role is _ControllerStackRole.FEED_FORWARD_HALTING:
        return _controller_stack_source(
            config.FF_HALTING_STACK_INDEPENDENT_FLAG,
            config.FF_HALTING_STACK_HIDDEN_DIM,
            config.FF_HALTING_STACK_NUM_LAYERS,
            config.FF_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            config.FF_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.FF_HALTING_STACK_ACTIVATION,
            config.FF_HALTING_STACK_LAYER_NORM_POSITION,
            config.FF_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
            config.FF_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            config.FF_HALTING_STACK_DROPOUT_PROBABILITY,
            config.FF_HALTING_STACK_BIAS_FLAG,
        )
    if role is _ControllerStackRole.FEED_FORWARD_MEMORY:
        return _controller_stack_source(
            config.FF_MEMORY_STACK_INDEPENDENT_FLAG,
            config.FF_MEMORY_STACK_HIDDEN_DIM,
            config.FF_MEMORY_STACK_NUM_LAYERS,
            config.FF_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            config.FF_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.FF_MEMORY_STACK_ACTIVATION,
            config.FF_MEMORY_STACK_LAYER_NORM_POSITION,
            config.FF_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
            config.FF_MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            config.FF_MEMORY_STACK_DROPOUT_PROBABILITY,
            config.FF_MEMORY_STACK_BIAS_FLAG,
        )
    if role is _ControllerStackRole.FEED_FORWARD_RECURRENT_GATE:
        return _controller_stack_source(
            config.FF_RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            config.FF_RECURRENT_GATE_STACK_HIDDEN_DIM,
            config.FF_RECURRENT_GATE_STACK_NUM_LAYERS,
            config.FF_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            config.FF_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.FF_RECURRENT_GATE_STACK_ACTIVATION,
            config.FF_RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
            config.FF_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            config.FF_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            config.FF_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
            config.FF_RECURRENT_GATE_STACK_BIAS_FLAG,
        )
    return _controller_stack_source(
        config.FF_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        config.FF_RECURRENT_HALTING_STACK_HIDDEN_DIM,
        config.FF_RECURRENT_HALTING_STACK_NUM_LAYERS,
        config.FF_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        config.FF_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        config.FF_RECURRENT_HALTING_STACK_ACTIVATION,
        config.FF_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        config.FF_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        config.FF_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
        config.FF_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
        config.FF_RECURRENT_HALTING_STACK_BIAS_FLAG,
    )


def _controller_stack_source(
    independent_flag: bool,
    hidden_dim: int | None,
    num_layers: int | None,
    last_layer_bias_option: LastLayerBiasOptions | None,
    apply_output_pipeline_flag: bool | None,
    activation: ActivationOptions | None,
    layer_norm_position: LayerNormPositionOptions | None,
    residual_connection_option: type[ResidualConfig] | None,
    residual_model_flag: bool,
    dropout_probability: float | None,
    bias_flag: bool | None,
) -> SubmoduleStackSource:
    return SubmoduleStackSource(
        independent_flag=independent_flag,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=last_layer_bias_option,
        apply_output_pipeline_flag=apply_output_pipeline_flag,
        activation=activation,
        layer_norm_position=layer_norm_position,
        residual_connection_option=residual_connection_option,
        residual_model_flag=residual_model_flag,
        dropout_probability=dropout_probability,
        bias_flag=bias_flag,
    )


def linears_layer_controller_options(
    config: ModuleType,
    role: LinearRole,
) -> LayerControllerOptions:
    if role is LinearRole.MAIN:
        return _layer_controller_options(
            config.STACK_GATE_FLAG,
            config.GATE_OPTION,
            config.GATE_ACTIVATION,
            linears_controller_stack_source(config, _ControllerStackRole.MAIN_GATE),
            config.STACK_HALTING_FLAG,
            config.HALTING_OPTION,
            config.HALTING_THRESHOLD,
            config.HALTING_DROPOUT,
            config.HALTING_HIDDEN_STATE_MODE,
            linears_controller_stack_source(config, _ControllerStackRole.MAIN_HALTING),
        )
    if role is LinearRole.ATTENTION:
        return _layer_controller_options(
            config.ATTN_STACK_GATE_FLAG,
            config.ATTN_GATE_OPTION,
            config.ATTN_GATE_ACTIVATION,
            linears_controller_stack_source(
                config, _ControllerStackRole.ATTENTION_GATE
            ),
            config.ATTN_STACK_HALTING_FLAG,
            config.ATTN_HALTING_OPTION,
            config.ATTN_HALTING_THRESHOLD,
            config.ATTN_HALTING_DROPOUT,
            config.ATTN_HALTING_HIDDEN_STATE_MODE,
            linears_controller_stack_source(
                config, _ControllerStackRole.ATTENTION_HALTING
            ),
        )
    return _layer_controller_options(
        config.FF_STACK_GATE_FLAG,
        config.FF_GATE_OPTION,
        config.FF_GATE_ACTIVATION,
        linears_controller_stack_source(config, _ControllerStackRole.FEED_FORWARD_GATE),
        config.FF_STACK_HALTING_FLAG,
        config.FF_HALTING_OPTION,
        config.FF_HALTING_THRESHOLD,
        config.FF_HALTING_DROPOUT,
        config.FF_HALTING_HIDDEN_STATE_MODE,
        linears_controller_stack_source(
            config, _ControllerStackRole.FEED_FORWARD_HALTING
        ),
    )


def _layer_controller_options(
    stack_gate_flag: bool,
    gate_option: LayerGateOptions | None,
    gate_activation: ActivationOptions | None,
    gate_stack_source: SubmoduleStackSource,
    stack_halting_flag: bool,
    halting_option: type[HaltingConfig],
    halting_threshold: float,
    halting_dropout: float,
    halting_hidden_state_mode: HaltingHiddenStateModeOptions,
    halting_stack_source: SubmoduleStackSource,
) -> LayerControllerOptions:
    return LayerControllerOptions(
        stack_gate_flag=stack_gate_flag,
        gate_option=gate_option,
        gate_activation=gate_activation,
        gate_stack_source=gate_stack_source,
        stack_halting_flag=stack_halting_flag,
        halting_option=halting_option,
        halting_threshold=halting_threshold,
        halting_dropout=halting_dropout,
        halting_hidden_state_mode=halting_hidden_state_mode,
        halting_stack_source=halting_stack_source,
    )


def linears_dynamic_memory_options(
    config: ModuleType,
    role: LinearRole,
) -> DynamicMemoryOptions:
    if role is LinearRole.MAIN:
        return _dynamic_memory_options(
            config.MEMORY_FLAG,
            config.MEMORY_OPTION,
            config.MEMORY_POSITION_OPTION,
            config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
            config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
            linears_controller_stack_source(config, _ControllerStackRole.MAIN_MEMORY),
        )
    if role is LinearRole.ATTENTION:
        return _dynamic_memory_options(
            config.ATTN_MEMORY_FLAG,
            config.ATTN_MEMORY_OPTION,
            config.ATTN_MEMORY_POSITION_OPTION,
            config.ATTN_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
            config.ATTN_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
            linears_controller_stack_source(
                config, _ControllerStackRole.ATTENTION_MEMORY
            ),
        )
    return _dynamic_memory_options(
        config.FF_MEMORY_FLAG,
        config.FF_MEMORY_OPTION,
        config.FF_MEMORY_POSITION_OPTION,
        config.FF_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
        config.FF_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
        linears_controller_stack_source(
            config, _ControllerStackRole.FEED_FORWARD_MEMORY
        ),
    )


def _dynamic_memory_options(
    memory_flag: bool,
    memory_option: type[DynamicMemoryConfig],
    memory_position_option: MemoryPositionOptions,
    memory_test_time_training_learning_rate: float | None,
    memory_test_time_training_num_inner_steps: int | None,
    memory_stack_source: SubmoduleStackSource,
) -> DynamicMemoryOptions:
    return DynamicMemoryOptions(
        memory_flag=memory_flag,
        memory_option=memory_option,
        memory_position_option=memory_position_option,
        memory_test_time_training_learning_rate=(
            memory_test_time_training_learning_rate
        ),
        memory_test_time_training_num_inner_steps=(
            memory_test_time_training_num_inner_steps
        ),
        memory_stack_source=memory_stack_source,
    )


def linears_recurrent_controller_options(
    config: ModuleType,
    role: LinearRole,
) -> RecurrentControllerOptions:
    if role is LinearRole.MAIN:
        return _recurrent_controller_options(
            config.RECURRENT_FLAG,
            config.RECURRENT_MAX_STEPS,
            config.RECURRENT_INITIAL_ITERATIONS,
            config.RECURRENT_GRADIENT_TRANSITION_COUNT,
            config.RECURRENT_ITERATION_INCREMENT,
            config.RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT,
            config.RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG,
            config.RECURRENT_LAYER_NORM_POSITION,
            config.RECURRENT_STACK_GATE_FLAG,
            config.RECURRENT_GATE_OPTION,
            config.RECURRENT_GATE_ACTIVATION,
            linears_controller_stack_source(
                config, _ControllerStackRole.MAIN_RECURRENT_GATE
            ),
            config.RECURRENT_STACK_HALTING_FLAG,
            config.RECURRENT_HALTING_OPTION,
            config.RECURRENT_HALTING_THRESHOLD,
            config.RECURRENT_HALTING_DROPOUT,
            config.RECURRENT_HALTING_HIDDEN_STATE_MODE,
            linears_controller_stack_source(
                config, _ControllerStackRole.MAIN_RECURRENT_HALTING
            ),
        )
    if role is LinearRole.ATTENTION:
        return _recurrent_controller_options(
            config.ATTN_RECURRENT_FLAG,
            config.ATTN_RECURRENT_MAX_STEPS,
            2,
            None,
            1,
            1,
            False,
            config.ATTN_RECURRENT_LAYER_NORM_POSITION,
            config.ATTN_RECURRENT_STACK_GATE_FLAG,
            config.ATTN_RECURRENT_GATE_OPTION,
            config.ATTN_RECURRENT_GATE_ACTIVATION,
            linears_controller_stack_source(
                config, _ControllerStackRole.ATTENTION_RECURRENT_GATE
            ),
            config.ATTN_RECURRENT_STACK_HALTING_FLAG,
            config.ATTN_RECURRENT_HALTING_OPTION,
            config.ATTN_RECURRENT_HALTING_THRESHOLD,
            config.ATTN_RECURRENT_HALTING_DROPOUT,
            config.ATTN_RECURRENT_HALTING_HIDDEN_STATE_MODE,
            linears_controller_stack_source(
                config, _ControllerStackRole.ATTENTION_RECURRENT_HALTING
            ),
        )
    return _recurrent_controller_options(
        config.FF_RECURRENT_FLAG,
        config.FF_RECURRENT_MAX_STEPS,
        2,
        None,
        1,
        1,
        False,
        config.FF_RECURRENT_LAYER_NORM_POSITION,
        config.FF_RECURRENT_STACK_GATE_FLAG,
        config.FF_RECURRENT_GATE_OPTION,
        config.FF_RECURRENT_GATE_ACTIVATION,
        linears_controller_stack_source(
            config, _ControllerStackRole.FEED_FORWARD_RECURRENT_GATE
        ),
        config.FF_RECURRENT_STACK_HALTING_FLAG,
        config.FF_RECURRENT_HALTING_OPTION,
        config.FF_RECURRENT_HALTING_THRESHOLD,
        config.FF_RECURRENT_HALTING_DROPOUT,
        config.FF_RECURRENT_HALTING_HIDDEN_STATE_MODE,
        linears_controller_stack_source(
            config, _ControllerStackRole.FEED_FORWARD_RECURRENT_HALTING
        ),
    )


def _recurrent_controller_options(
    recurrent_flag: bool,
    recurrent_max_steps: int,
    recurrent_initial_iterations: int,
    recurrent_gradient_transition_count: int | None,
    recurrent_iteration_increment: int,
    recurrent_forward_calls_before_iteration_increment: int,
    recurrent_smooth_iteration_growth_flag: bool,
    recurrent_layer_norm_position: LayerNormPositionOptions,
    recurrent_stack_gate_flag: bool,
    recurrent_gate_option: LayerGateOptions | None,
    recurrent_gate_activation: ActivationOptions | None,
    recurrent_gate_stack_source: SubmoduleStackSource,
    recurrent_stack_halting_flag: bool,
    recurrent_halting_option: type[HaltingConfig],
    recurrent_halting_threshold: float,
    recurrent_halting_dropout: float,
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions,
    recurrent_halting_stack_source: SubmoduleStackSource,
) -> RecurrentControllerOptions:
    return RecurrentControllerOptions(
        recurrent_flag=recurrent_flag,
        recurrent_max_steps=recurrent_max_steps,
        recurrent_initial_iterations=recurrent_initial_iterations,
        recurrent_gradient_transition_count=recurrent_gradient_transition_count,
        recurrent_iteration_increment=recurrent_iteration_increment,
        recurrent_forward_calls_before_iteration_increment=(
            recurrent_forward_calls_before_iteration_increment
        ),
        recurrent_smooth_iteration_growth_flag=(recurrent_smooth_iteration_growth_flag),
        recurrent_layer_norm_position=recurrent_layer_norm_position,
        recurrent_stack_gate_flag=recurrent_stack_gate_flag,
        recurrent_gate_option=recurrent_gate_option,
        recurrent_gate_activation=recurrent_gate_activation,
        recurrent_gate_stack_source=recurrent_gate_stack_source,
        recurrent_stack_halting_flag=recurrent_stack_halting_flag,
        recurrent_halting_option=recurrent_halting_option,
        recurrent_halting_threshold=recurrent_halting_threshold,
        recurrent_halting_dropout=recurrent_halting_dropout,
        recurrent_halting_hidden_state_mode=recurrent_halting_hidden_state_mode,
        recurrent_halting_stack_source=recurrent_halting_stack_source,
    )


def adaptive_generator_stack_options(
    config: ModuleType,
) -> AdaptiveGeneratorStackOptions:
    return AdaptiveGeneratorStackOptions(
        hidden_dim=config.ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM,
        layer_norm_position=config.ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION,
        num_layers=config.ADAPTIVE_GENERATOR_STACK_NUM_LAYERS,
        activation=config.ADAPTIVE_GENERATOR_STACK_ACTIVATION,
        residual_connection_option=(
            config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_model_flag=config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY,
        last_layer_bias_option=config.ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=(
            config.ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        bias_flag=config.ADAPTIVE_GENERATOR_STACK_BIAS_FLAG,
    )


def adaptive_generator_stack_source(
    config: ModuleType,
    role: LinearRole,
    parameter: _AdaptiveParameter,
) -> AdaptiveGeneratorStackSource:
    if role is LinearRole.MAIN:
        return _main_adaptive_generator_stack_source(config, parameter)
    if role is LinearRole.ATTENTION:
        return _attention_adaptive_generator_stack_source(config, parameter)
    return _feed_forward_adaptive_generator_stack_source(config, parameter)


def _main_adaptive_generator_stack_source(
    config: ModuleType,
    parameter: _AdaptiveParameter,
) -> AdaptiveGeneratorStackSource:
    if parameter is _AdaptiveParameter.WEIGHT:
        return _adaptive_stack_source(
            config.WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG,
            config.WEIGHT_GENERATOR_STACK_HIDDEN_DIM,
            config.WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION,
            config.WEIGHT_GENERATOR_STACK_NUM_LAYERS,
            config.WEIGHT_GENERATOR_STACK_ACTIVATION,
            config.WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            config.WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            config.WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY,
            config.WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            config.WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.WEIGHT_GENERATOR_STACK_BIAS_FLAG,
        )
    if parameter is _AdaptiveParameter.BIAS:
        return _adaptive_stack_source(
            config.BIAS_GENERATOR_STACK_INDEPENDENT_FLAG,
            config.BIAS_GENERATOR_STACK_HIDDEN_DIM,
            config.BIAS_GENERATOR_STACK_LAYER_NORM_POSITION,
            config.BIAS_GENERATOR_STACK_NUM_LAYERS,
            config.BIAS_GENERATOR_STACK_ACTIVATION,
            config.BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            config.BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            config.BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY,
            config.BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            config.BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.BIAS_GENERATOR_STACK_BIAS_FLAG,
        )
    if parameter is _AdaptiveParameter.DIAGONAL:
        return _adaptive_stack_source(
            config.DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG,
            config.DIAGONAL_GENERATOR_STACK_HIDDEN_DIM,
            config.DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION,
            config.DIAGONAL_GENERATOR_STACK_NUM_LAYERS,
            config.DIAGONAL_GENERATOR_STACK_ACTIVATION,
            config.DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            config.DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            config.DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY,
            config.DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            config.DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.DIAGONAL_GENERATOR_STACK_BIAS_FLAG,
        )
    return _adaptive_stack_source(
        config.MASK_GENERATOR_STACK_INDEPENDENT_FLAG,
        config.MASK_GENERATOR_STACK_HIDDEN_DIM,
        config.MASK_GENERATOR_STACK_LAYER_NORM_POSITION,
        config.MASK_GENERATOR_STACK_NUM_LAYERS,
        config.MASK_GENERATOR_STACK_ACTIVATION,
        config.MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        config.MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
        config.MASK_GENERATOR_STACK_DROPOUT_PROBABILITY,
        config.MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        config.MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        config.MASK_GENERATOR_STACK_BIAS_FLAG,
    )


def _attention_adaptive_generator_stack_source(
    config: ModuleType,
    parameter: _AdaptiveParameter,
) -> AdaptiveGeneratorStackSource:
    if parameter is _AdaptiveParameter.WEIGHT:
        return _adaptive_stack_source(
            config.ATTN_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG,
            config.ATTN_WEIGHT_GENERATOR_STACK_HIDDEN_DIM,
            config.ATTN_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION,
            config.ATTN_WEIGHT_GENERATOR_STACK_NUM_LAYERS,
            config.ATTN_WEIGHT_GENERATOR_STACK_ACTIVATION,
            config.ATTN_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            config.ATTN_WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            config.ATTN_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY,
            config.ATTN_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            config.ATTN_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.ATTN_WEIGHT_GENERATOR_STACK_BIAS_FLAG,
        )
    if parameter is _AdaptiveParameter.BIAS:
        return _adaptive_stack_source(
            config.ATTN_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG,
            config.ATTN_BIAS_GENERATOR_STACK_HIDDEN_DIM,
            config.ATTN_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION,
            config.ATTN_BIAS_GENERATOR_STACK_NUM_LAYERS,
            config.ATTN_BIAS_GENERATOR_STACK_ACTIVATION,
            config.ATTN_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            config.ATTN_BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            config.ATTN_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY,
            config.ATTN_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            config.ATTN_BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.ATTN_BIAS_GENERATOR_STACK_BIAS_FLAG,
        )
    if parameter is _AdaptiveParameter.DIAGONAL:
        return _adaptive_stack_source(
            config.ATTN_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG,
            config.ATTN_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM,
            config.ATTN_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION,
            config.ATTN_DIAGONAL_GENERATOR_STACK_NUM_LAYERS,
            config.ATTN_DIAGONAL_GENERATOR_STACK_ACTIVATION,
            config.ATTN_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            config.ATTN_DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            config.ATTN_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY,
            config.ATTN_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            config.ATTN_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.ATTN_DIAGONAL_GENERATOR_STACK_BIAS_FLAG,
        )
    return _adaptive_stack_source(
        config.ATTN_MASK_GENERATOR_STACK_INDEPENDENT_FLAG,
        config.ATTN_MASK_GENERATOR_STACK_HIDDEN_DIM,
        config.ATTN_MASK_GENERATOR_STACK_LAYER_NORM_POSITION,
        config.ATTN_MASK_GENERATOR_STACK_NUM_LAYERS,
        config.ATTN_MASK_GENERATOR_STACK_ACTIVATION,
        config.ATTN_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        config.ATTN_MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
        config.ATTN_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY,
        config.ATTN_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        config.ATTN_MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        config.ATTN_MASK_GENERATOR_STACK_BIAS_FLAG,
    )


def _feed_forward_adaptive_generator_stack_source(
    config: ModuleType,
    parameter: _AdaptiveParameter,
) -> AdaptiveGeneratorStackSource:
    if parameter is _AdaptiveParameter.WEIGHT:
        return _adaptive_stack_source(
            config.FF_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG,
            config.FF_WEIGHT_GENERATOR_STACK_HIDDEN_DIM,
            config.FF_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION,
            config.FF_WEIGHT_GENERATOR_STACK_NUM_LAYERS,
            config.FF_WEIGHT_GENERATOR_STACK_ACTIVATION,
            config.FF_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            config.FF_WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            config.FF_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY,
            config.FF_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            config.FF_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.FF_WEIGHT_GENERATOR_STACK_BIAS_FLAG,
        )
    if parameter is _AdaptiveParameter.BIAS:
        return _adaptive_stack_source(
            config.FF_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG,
            config.FF_BIAS_GENERATOR_STACK_HIDDEN_DIM,
            config.FF_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION,
            config.FF_BIAS_GENERATOR_STACK_NUM_LAYERS,
            config.FF_BIAS_GENERATOR_STACK_ACTIVATION,
            config.FF_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            config.FF_BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            config.FF_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY,
            config.FF_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            config.FF_BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.FF_BIAS_GENERATOR_STACK_BIAS_FLAG,
        )
    if parameter is _AdaptiveParameter.DIAGONAL:
        return _adaptive_stack_source(
            config.FF_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG,
            config.FF_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM,
            config.FF_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION,
            config.FF_DIAGONAL_GENERATOR_STACK_NUM_LAYERS,
            config.FF_DIAGONAL_GENERATOR_STACK_ACTIVATION,
            config.FF_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            config.FF_DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            config.FF_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY,
            config.FF_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            config.FF_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            config.FF_DIAGONAL_GENERATOR_STACK_BIAS_FLAG,
        )
    return _adaptive_stack_source(
        config.FF_MASK_GENERATOR_STACK_INDEPENDENT_FLAG,
        config.FF_MASK_GENERATOR_STACK_HIDDEN_DIM,
        config.FF_MASK_GENERATOR_STACK_LAYER_NORM_POSITION,
        config.FF_MASK_GENERATOR_STACK_NUM_LAYERS,
        config.FF_MASK_GENERATOR_STACK_ACTIVATION,
        config.FF_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        config.FF_MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
        config.FF_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY,
        config.FF_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        config.FF_MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        config.FF_MASK_GENERATOR_STACK_BIAS_FLAG,
    )


def _adaptive_stack_source(
    independent_flag: bool,
    hidden_dim: int | None,
    layer_norm_position: LayerNormPositionOptions | None,
    num_layers: int | None,
    activation: ActivationOptions | None,
    residual_connection_option: type[ResidualConfig] | None,
    residual_model_flag: bool,
    dropout_probability: float | None,
    last_layer_bias_option: LastLayerBiasOptions | None,
    apply_output_pipeline_flag: bool | None,
    bias_flag: bool | None,
) -> AdaptiveGeneratorStackSource:
    return AdaptiveGeneratorStackSource(
        independent_flag=independent_flag,
        hidden_dim=hidden_dim,
        layer_norm_position=layer_norm_position,
        num_layers=num_layers,
        activation=activation,
        residual_connection_option=residual_connection_option,
        residual_model_flag=residual_model_flag,
        dropout_probability=dropout_probability,
        last_layer_bias_option=last_layer_bias_option,
        apply_output_pipeline_flag=apply_output_pipeline_flag,
        bias_flag=bias_flag,
    )


def hidden_adaptive_weight_options(
    config: ModuleType,
    role: LinearRole = LinearRole.MAIN,
) -> HiddenAdaptiveWeightOptions:
    if role is LinearRole.MAIN:
        return _adaptive_weight_options(
            config.GENERATOR_DEPTH,
            config.WEIGHT_OPTION_FLAG,
            config.WEIGHT_OPTION,
            config.WEIGHT_NORMALIZATION_OPTION,
            config.WEIGHT_NORMALIZATION_POSITION_OPTION,
            config.WEIGHT_DECAY_SCHEDULE,
            config.WEIGHT_DECAY_RATE,
            config.WEIGHT_DECAY_WARMUP_BATCHES,
            config.WEIGHT_BANK_EXPANSION_FACTOR,
            adaptive_generator_stack_source(
                config, LinearRole.MAIN, _AdaptiveParameter.WEIGHT
            ),
        )
    if role is LinearRole.ATTENTION:
        return _adaptive_weight_options(
            config.ATTN_GENERATOR_DEPTH,
            config.ATTN_WEIGHT_OPTION_FLAG,
            config.ATTN_WEIGHT_OPTION,
            config.ATTN_WEIGHT_NORMALIZATION_OPTION,
            config.ATTN_WEIGHT_NORMALIZATION_POSITION_OPTION,
            config.ATTN_WEIGHT_DECAY_SCHEDULE,
            config.ATTN_WEIGHT_DECAY_RATE,
            config.ATTN_WEIGHT_DECAY_WARMUP_BATCHES,
            config.ATTN_WEIGHT_BANK_EXPANSION_FACTOR,
            adaptive_generator_stack_source(
                config, LinearRole.ATTENTION, _AdaptiveParameter.WEIGHT
            ),
        )
    return _adaptive_weight_options(
        config.FF_GENERATOR_DEPTH,
        config.FF_WEIGHT_OPTION_FLAG,
        config.FF_WEIGHT_OPTION,
        config.FF_WEIGHT_NORMALIZATION_OPTION,
        config.FF_WEIGHT_NORMALIZATION_POSITION_OPTION,
        config.FF_WEIGHT_DECAY_SCHEDULE,
        config.FF_WEIGHT_DECAY_RATE,
        config.FF_WEIGHT_DECAY_WARMUP_BATCHES,
        config.FF_WEIGHT_BANK_EXPANSION_FACTOR,
        adaptive_generator_stack_source(
            config, LinearRole.FEED_FORWARD, _AdaptiveParameter.WEIGHT
        ),
    )


def _adaptive_weight_options(
    generator_depth: DynamicDepthOptions,
    option_flag: bool,
    option: type[DynamicWeightConfig] | None,
    normalization_option: WeightNormalizationOptions,
    normalization_position_option: WeightNormalizationPositionOptions,
    decay_schedule: WeightDecayScheduleOptions,
    decay_rate: float,
    decay_warmup_batches: int,
    bank_expansion_factor: BankExpansionFactorOptions,
    generator_stack_source: AdaptiveGeneratorStackSource,
) -> HiddenAdaptiveWeightOptions:
    return HiddenAdaptiveWeightOptions(
        generator_depth=generator_depth,
        option_flag=option_flag,
        option=option,
        normalization_option=normalization_option,
        normalization_position_option=normalization_position_option,
        decay_schedule=decay_schedule,
        decay_rate=decay_rate,
        decay_warmup_batches=decay_warmup_batches,
        bank_expansion_factor=bank_expansion_factor,
        generator_stack_source=generator_stack_source,
    )


def hidden_adaptive_bias_options(
    config: ModuleType,
    role: LinearRole = LinearRole.MAIN,
) -> HiddenAdaptiveBiasOptions:
    if role is LinearRole.MAIN:
        return _adaptive_bias_options(
            config.BIAS_OPTION_FLAG,
            config.BIAS_OPTION,
            config.BIAS_DECAY_SCHEDULE,
            config.BIAS_DECAY_RATE,
            config.BIAS_DECAY_WARMUP_BATCHES,
            config.BIAS_BANK_EXPANSION_FACTOR,
            adaptive_generator_stack_source(
                config, LinearRole.MAIN, _AdaptiveParameter.BIAS
            ),
        )
    if role is LinearRole.ATTENTION:
        return _adaptive_bias_options(
            config.ATTN_BIAS_OPTION_FLAG,
            config.ATTN_BIAS_OPTION,
            config.ATTN_BIAS_DECAY_SCHEDULE,
            config.ATTN_BIAS_DECAY_RATE,
            config.ATTN_BIAS_DECAY_WARMUP_BATCHES,
            config.ATTN_BIAS_BANK_EXPANSION_FACTOR,
            adaptive_generator_stack_source(
                config, LinearRole.ATTENTION, _AdaptiveParameter.BIAS
            ),
        )
    return _adaptive_bias_options(
        config.FF_BIAS_OPTION_FLAG,
        config.FF_BIAS_OPTION,
        config.FF_BIAS_DECAY_SCHEDULE,
        config.FF_BIAS_DECAY_RATE,
        config.FF_BIAS_DECAY_WARMUP_BATCHES,
        config.FF_BIAS_BANK_EXPANSION_FACTOR,
        adaptive_generator_stack_source(
            config, LinearRole.FEED_FORWARD, _AdaptiveParameter.BIAS
        ),
    )


def _adaptive_bias_options(
    option_flag: bool,
    option: type[DynamicBiasConfig] | None,
    decay_schedule: WeightDecayScheduleOptions,
    decay_rate: float,
    decay_warmup_batches: int,
    bank_expansion_factor: BankExpansionFactorOptions,
    generator_stack_source: AdaptiveGeneratorStackSource,
) -> HiddenAdaptiveBiasOptions:
    return HiddenAdaptiveBiasOptions(
        option_flag=option_flag,
        option=option,
        decay_schedule=decay_schedule,
        decay_rate=decay_rate,
        decay_warmup_batches=decay_warmup_batches,
        bank_expansion_factor=bank_expansion_factor,
        generator_stack_source=generator_stack_source,
    )


def hidden_adaptive_diagonal_options(
    config: ModuleType,
    role: LinearRole = LinearRole.MAIN,
) -> HiddenAdaptiveDiagonalOptions:
    if role is LinearRole.MAIN:
        return _adaptive_diagonal_options(
            config.DIAGONAL_OPTION_FLAG,
            config.DIAGONAL_OPTION,
            adaptive_generator_stack_source(
                config, LinearRole.MAIN, _AdaptiveParameter.DIAGONAL
            ),
        )
    if role is LinearRole.ATTENTION:
        return _adaptive_diagonal_options(
            config.ATTN_DIAGONAL_OPTION_FLAG,
            config.ATTN_DIAGONAL_OPTION,
            adaptive_generator_stack_source(
                config, LinearRole.ATTENTION, _AdaptiveParameter.DIAGONAL
            ),
        )
    return _adaptive_diagonal_options(
        config.FF_DIAGONAL_OPTION_FLAG,
        config.FF_DIAGONAL_OPTION,
        adaptive_generator_stack_source(
            config, LinearRole.FEED_FORWARD, _AdaptiveParameter.DIAGONAL
        ),
    )


def _adaptive_diagonal_options(
    option_flag: bool,
    option: type[DynamicDiagonalConfig] | None,
    generator_stack_source: AdaptiveGeneratorStackSource,
) -> HiddenAdaptiveDiagonalOptions:
    return HiddenAdaptiveDiagonalOptions(
        option_flag=option_flag,
        option=option,
        generator_stack_source=generator_stack_source,
    )


def hidden_adaptive_mask_options(
    config: ModuleType,
    role: LinearRole = LinearRole.MAIN,
) -> HiddenAdaptiveMaskOptions:
    if role is LinearRole.MAIN:
        return _adaptive_mask_options(
            config.MASK_OPTION_FLAG,
            config.ROW_MASK_OPTION,
            config.MASK_DIMENSION_OPTION,
            config.MASK_THRESHOLD,
            config.MASK_SURROGATE_SCALE,
            config.MASK_FLOOR,
            config.MASK_TRANSITION_WIDTH,
            adaptive_generator_stack_source(
                config, LinearRole.MAIN, _AdaptiveParameter.MASK
            ),
        )
    if role is LinearRole.ATTENTION:
        return _adaptive_mask_options(
            config.ATTN_MASK_OPTION_FLAG,
            config.ATTN_ROW_MASK_OPTION,
            config.ATTN_MASK_DIMENSION_OPTION,
            config.ATTN_MASK_THRESHOLD,
            config.ATTN_MASK_SURROGATE_SCALE,
            config.ATTN_MASK_FLOOR,
            config.ATTN_MASK_TRANSITION_WIDTH,
            adaptive_generator_stack_source(
                config, LinearRole.ATTENTION, _AdaptiveParameter.MASK
            ),
        )
    return _adaptive_mask_options(
        config.FF_MASK_OPTION_FLAG,
        config.FF_ROW_MASK_OPTION,
        config.FF_MASK_DIMENSION_OPTION,
        config.FF_MASK_THRESHOLD,
        config.FF_MASK_SURROGATE_SCALE,
        config.FF_MASK_FLOOR,
        config.FF_MASK_TRANSITION_WIDTH,
        adaptive_generator_stack_source(
            config, LinearRole.FEED_FORWARD, _AdaptiveParameter.MASK
        ),
    )


def _adaptive_mask_options(
    option_flag: bool,
    row_mask_option: type[AxisMaskConfig] | None,
    mask_dimension_option: MaskDimensionOptions,
    mask_threshold: float,
    mask_surrogate_scale: float,
    mask_floor: float,
    mask_transition_width: float,
    generator_stack_source: AdaptiveGeneratorStackSource,
) -> HiddenAdaptiveMaskOptions:
    return HiddenAdaptiveMaskOptions(
        option_flag=option_flag,
        row_mask_option=row_mask_option,
        mask_dimension_option=mask_dimension_option,
        mask_threshold=mask_threshold,
        mask_surrogate_scale=mask_surrogate_scale,
        mask_floor=mask_floor,
        mask_transition_width=mask_transition_width,
        generator_stack_source=generator_stack_source,
    )
