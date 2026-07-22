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


def bert_embedding_options(config: object) -> BertEmbeddingOptions:
    return BertEmbeddingOptions(
        token_type_vocab_size=config.TOKEN_TYPE_VOCAB_SIZE,
        layer_norm_flag=config.EMBEDDING_LAYER_NORM_FLAG,
        dropout_probability=config.EMBEDDING_DROPOUT_PROBABILITY,
    )


def bert_mlm_head_options(config: object) -> BertMlmHeadOptions:
    return BertMlmHeadOptions(
        activation=config.MLM_ACTIVATION,
        dense_bias_flag=config.MLM_DENSE_BIAS_FLAG,
        layer_norm_flag=config.MLM_LAYER_NORM_FLAG,
        decoder_bias_flag=config.MLM_DECODER_BIAS_FLAG,
        decoder_weight_tying_flag=config.MLM_DECODER_WEIGHT_TYING_FLAG,
    )


def bert_nsp_head_options(config: object) -> BertNspHeadOptions:
    return BertNspHeadOptions(
        pooler_activation=config.NSP_POOLER_ACTIVATION,
        pooler_bias_flag=config.NSP_POOLER_BIAS_FLAG,
        output_dim=config.NSP_OUTPUT_DIM,
        head_bias_flag=config.NSP_HEAD_BIAS_FLAG,
    )


def bert_positional_embedding_options(
    config: object,
) -> TransformerPositionalEmbeddingOptions:
    return TransformerPositionalEmbeddingOptions(
        option=config.POSITIONAL_EMBEDDING_OPTION,
        padding_idx=config.POSITIONAL_EMBEDDING_PADDING_IDX,
        auto_expand_flag=config.POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG,
    )


def bert_encoder_options(config: object) -> TransformerEncoderOptions:
    return TransformerEncoderOptions(
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.STACK_NUM_LAYERS,
        activation=config.STACK_ACTIVATION,
        dropout_probability=config.STACK_DROPOUT_PROBABILITY,
        layer_norm_position=config.LAYER_NORM_POSITION,
        causal_attention_mask_flag=config.CAUSAL_ATTENTION_MASK_FLAG,
    )


def bert_attention_options(config: object) -> TransformerAttentionOptions:
    return TransformerAttentionOptions(
        num_heads=config.ATTN_NUM_HEADS,
        num_layers=config.ATTN_NUM_LAYERS,
        bias_flag=config.ATTN_BIAS_FLAG,
        add_key_value_bias_flag=config.ATTN_ADD_KEY_VALUE_BIAS_FLAG,
    )


def bert_feed_forward_options(config: object) -> TransformerFeedForwardOptions:
    return TransformerFeedForwardOptions(
        num_layers=config.FF_NUM_LAYERS,
        bias_flag=config.FF_BIAS_FLAG,
    )


def main_layer_stack_options(config: object) -> MainLayerStackOptions:
    return MainLayerStackOptions(
        bias_flag=config.STACK_BIAS_FLAG,
        layer_norm_position=config.LAYER_NORM_POSITION,
        num_layers=config.STACK_NUM_LAYERS,
        activation=config.STACK_ACTIVATION,
        residual_connection_option=config.STACK_RESIDUAL_CONNECTION_OPTION,
        dropout_probability=config.STACK_DROPOUT_PROBABILITY,
        last_layer_bias_option=config.STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
    )


def linears_submodule_stack_options(
    config: object,
    prefix: str,
    *,
    num_layers_key: str | None = None,
    bias_key: str | None = None,
) -> SubmoduleStackOptions:
    return SubmoduleStackOptions(
        hidden_dim=getattr(config, f"{prefix}_HIDDEN_DIM"),
        num_layers=getattr(config, num_layers_key or f"{prefix}_NUM_LAYERS"),
        last_layer_bias_option=getattr(config, f"{prefix}_LAST_LAYER_BIAS_OPTION"),
        apply_output_pipeline_flag=getattr(
            config, f"{prefix}_APPLY_OUTPUT_PIPELINE_FLAG"
        ),
        activation=getattr(config, f"{prefix}_ACTIVATION"),
        layer_norm_position=getattr(config, f"{prefix}_LAYER_NORM_POSITION"),
        residual_connection_option=getattr(
            config, f"{prefix}_RESIDUAL_CONNECTION_OPTION"
        ),
        dropout_probability=getattr(config, f"{prefix}_DROPOUT_PROBABILITY"),
        bias_flag=getattr(config, bias_key or f"{prefix}_BIAS_FLAG"),
    )


def linears_controller_stack_source(
    config: object,
    prefix: str,
) -> SubmoduleStackSource:
    return SubmoduleStackSource(
        independent_flag=getattr(config, f"{prefix}_INDEPENDENT_FLAG"),
        hidden_dim=getattr(config, f"{prefix}_HIDDEN_DIM"),
        num_layers=getattr(config, f"{prefix}_NUM_LAYERS"),
        last_layer_bias_option=getattr(config, f"{prefix}_LAST_LAYER_BIAS_OPTION"),
        apply_output_pipeline_flag=getattr(
            config, f"{prefix}_APPLY_OUTPUT_PIPELINE_FLAG"
        ),
        activation=getattr(config, f"{prefix}_ACTIVATION"),
        layer_norm_position=getattr(config, f"{prefix}_LAYER_NORM_POSITION"),
        residual_connection_option=getattr(
            config, f"{prefix}_RESIDUAL_CONNECTION_OPTION"
        ),
        dropout_probability=getattr(config, f"{prefix}_DROPOUT_PROBABILITY"),
        bias_flag=getattr(config, f"{prefix}_BIAS_FLAG"),
    )


def _stack_control_flag_name(prefix: str, control: str) -> str:
    scope = "" if prefix == control else f"{prefix.removesuffix(f'_{control}')}_"
    return f"{scope}STACK_{control}_FLAG"


def linears_layer_controller_options(
    config: object,
    *,
    gate_prefix: str,
    gate_stack_prefix: str,
    halting_prefix: str,
    halting_stack_prefix: str,
) -> LayerControllerOptions:
    return LayerControllerOptions(
        stack_gate_flag=getattr(config, _stack_control_flag_name(gate_prefix, "GATE")),
        gate_option=getattr(config, f"{gate_prefix}_OPTION"),
        gate_activation=getattr(config, f"{gate_prefix}_ACTIVATION"),
        gate_stack_source=linears_controller_stack_source(config, gate_stack_prefix),
        stack_halting_flag=getattr(
            config, _stack_control_flag_name(halting_prefix, "HALTING")
        ),
        halting_option=getattr(
            config,
            f"{halting_prefix}_OPTION",
            LayerControllerOptions.halting_option,
        ),
        halting_threshold=getattr(config, f"{halting_prefix}_THRESHOLD"),
        halting_dropout=getattr(config, f"{halting_prefix}_DROPOUT"),
        halting_hidden_state_mode=getattr(
            config, f"{halting_prefix}_HIDDEN_STATE_MODE"
        ),
        halting_stack_source=linears_controller_stack_source(
            config, halting_stack_prefix
        ),
    )


def linears_dynamic_memory_options(
    config: object,
    *,
    memory_prefix: str,
    memory_stack_prefix: str,
) -> DynamicMemoryOptions:
    return DynamicMemoryOptions(
        memory_flag=getattr(config, f"{memory_prefix}_FLAG"),
        memory_option=getattr(config, f"{memory_prefix}_OPTION"),
        memory_position_option=getattr(config, f"{memory_prefix}_POSITION_OPTION"),
        memory_test_time_training_learning_rate=getattr(
            config, f"{memory_prefix}_TEST_TIME_TRAINING_LEARNING_RATE"
        ),
        memory_test_time_training_num_inner_steps=getattr(
            config, f"{memory_prefix}_TEST_TIME_TRAINING_NUM_INNER_STEPS"
        ),
        memory_stack_source=linears_controller_stack_source(
            config, memory_stack_prefix
        ),
    )


def linears_recurrent_controller_options(
    config: object,
    *,
    recurrent_prefix: str,
    gate_stack_prefix: str,
    halting_stack_prefix: str,
) -> RecurrentControllerOptions:
    return RecurrentControllerOptions(
        recurrent_flag=getattr(config, f"{recurrent_prefix}_FLAG"),
        recurrent_max_steps=getattr(config, f"{recurrent_prefix}_MAX_STEPS"),
        recurrent_layer_norm_position=getattr(
            config, f"{recurrent_prefix}_LAYER_NORM_POSITION"
        ),
        recurrent_stack_gate_flag=getattr(
            config, f"{recurrent_prefix}_STACK_GATE_FLAG"
        ),
        recurrent_gate_option=getattr(config, f"{recurrent_prefix}_GATE_OPTION"),
        recurrent_gate_activation=getattr(
            config, f"{recurrent_prefix}_GATE_ACTIVATION"
        ),
        recurrent_gate_stack_source=linears_controller_stack_source(
            config, gate_stack_prefix
        ),
        recurrent_stack_halting_flag=getattr(
            config, f"{recurrent_prefix}_STACK_HALTING_FLAG"
        ),
        recurrent_halting_option=getattr(
            config,
            f"{recurrent_prefix}_HALTING_OPTION",
            RecurrentControllerOptions.recurrent_halting_option,
        ),
        recurrent_halting_threshold=getattr(
            config, f"{recurrent_prefix}_HALTING_THRESHOLD"
        ),
        recurrent_halting_dropout=getattr(
            config, f"{recurrent_prefix}_HALTING_DROPOUT"
        ),
        recurrent_halting_hidden_state_mode=getattr(
            config, f"{recurrent_prefix}_HALTING_HIDDEN_STATE_MODE"
        ),
        recurrent_halting_stack_source=linears_controller_stack_source(
            config, halting_stack_prefix
        ),
    )


def adaptive_generator_stack_options(config: object) -> AdaptiveGeneratorStackOptions:
    return AdaptiveGeneratorStackOptions(
        hidden_dim=config.ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM,
        layer_norm_position=config.ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION,
        num_layers=config.ADAPTIVE_GENERATOR_STACK_NUM_LAYERS,
        activation=config.ADAPTIVE_GENERATOR_STACK_ACTIVATION,
        residual_connection_option=(
            config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        dropout_probability=config.ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY,
        last_layer_bias_option=config.ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=(
            config.ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        bias_flag=config.ADAPTIVE_GENERATOR_STACK_BIAS_FLAG,
    )


def adaptive_generator_stack_source(
    config: object,
    prefix: str,
) -> AdaptiveGeneratorStackSource:
    return AdaptiveGeneratorStackSource(
        independent_flag=getattr(config, f"{prefix}_INDEPENDENT_FLAG"),
        hidden_dim=getattr(config, f"{prefix}_HIDDEN_DIM"),
        layer_norm_position=getattr(config, f"{prefix}_LAYER_NORM_POSITION"),
        num_layers=getattr(config, f"{prefix}_NUM_LAYERS"),
        activation=getattr(config, f"{prefix}_ACTIVATION"),
        residual_connection_option=getattr(
            config,
            f"{prefix}_RESIDUAL_CONNECTION_OPTION",
        ),
        dropout_probability=getattr(config, f"{prefix}_DROPOUT_PROBABILITY"),
        last_layer_bias_option=getattr(config, f"{prefix}_LAST_LAYER_BIAS_OPTION"),
        apply_output_pipeline_flag=getattr(
            config,
            f"{prefix}_APPLY_OUTPUT_PIPELINE_FLAG",
        ),
        bias_flag=getattr(config, f"{prefix}_BIAS_FLAG"),
    )


def hidden_adaptive_weight_options(
    config: object,
    *,
    prefix: str = "",
    stack_prefix: str = "WEIGHT_GENERATOR_STACK",
) -> HiddenAdaptiveWeightOptions:
    return HiddenAdaptiveWeightOptions(
        generator_depth=getattr(config, f"{prefix}GENERATOR_DEPTH"),
        option_flag=getattr(config, f"{prefix}WEIGHT_OPTION_FLAG"),
        option=getattr(config, f"{prefix}WEIGHT_OPTION"),
        normalization_option=getattr(
            config,
            f"{prefix}WEIGHT_NORMALIZATION_OPTION",
        ),
        normalization_position_option=getattr(
            config,
            f"{prefix}WEIGHT_NORMALIZATION_POSITION_OPTION",
        ),
        decay_schedule=getattr(config, f"{prefix}WEIGHT_DECAY_SCHEDULE"),
        decay_rate=getattr(config, f"{prefix}WEIGHT_DECAY_RATE"),
        decay_warmup_batches=getattr(
            config,
            f"{prefix}WEIGHT_DECAY_WARMUP_BATCHES",
        ),
        bank_expansion_factor=getattr(
            config,
            f"{prefix}WEIGHT_BANK_EXPANSION_FACTOR",
        ),
        generator_stack_source=adaptive_generator_stack_source(
            config,
            stack_prefix,
        ),
    )


def hidden_adaptive_bias_options(
    config: object,
    *,
    prefix: str = "",
    stack_prefix: str = "BIAS_GENERATOR_STACK",
) -> HiddenAdaptiveBiasOptions:
    return HiddenAdaptiveBiasOptions(
        option_flag=getattr(config, f"{prefix}BIAS_OPTION_FLAG"),
        option=getattr(config, f"{prefix}BIAS_OPTION"),
        decay_schedule=getattr(config, f"{prefix}BIAS_DECAY_SCHEDULE"),
        decay_rate=getattr(config, f"{prefix}BIAS_DECAY_RATE"),
        decay_warmup_batches=getattr(
            config,
            f"{prefix}BIAS_DECAY_WARMUP_BATCHES",
        ),
        bank_expansion_factor=getattr(
            config,
            f"{prefix}BIAS_BANK_EXPANSION_FACTOR",
        ),
        generator_stack_source=adaptive_generator_stack_source(
            config,
            stack_prefix,
        ),
    )


def hidden_adaptive_diagonal_options(
    config: object,
    *,
    prefix: str = "",
    stack_prefix: str = "DIAGONAL_GENERATOR_STACK",
) -> HiddenAdaptiveDiagonalOptions:
    return HiddenAdaptiveDiagonalOptions(
        option_flag=getattr(config, f"{prefix}DIAGONAL_OPTION_FLAG"),
        option=getattr(config, f"{prefix}DIAGONAL_OPTION"),
        generator_stack_source=adaptive_generator_stack_source(
            config,
            stack_prefix,
        ),
    )


def hidden_adaptive_mask_options(
    config: object,
    *,
    prefix: str = "",
    stack_prefix: str = "MASK_GENERATOR_STACK",
) -> HiddenAdaptiveMaskOptions:
    return HiddenAdaptiveMaskOptions(
        option_flag=getattr(config, f"{prefix}MASK_OPTION_FLAG"),
        row_mask_option=getattr(config, f"{prefix}ROW_MASK_OPTION"),
        mask_dimension_option=getattr(config, f"{prefix}MASK_DIMENSION_OPTION"),
        mask_threshold=getattr(config, f"{prefix}MASK_THRESHOLD"),
        mask_surrogate_scale=getattr(config, f"{prefix}MASK_SURROGATE_SCALE"),
        mask_floor=getattr(config, f"{prefix}MASK_FLOOR"),
        mask_transition_width=getattr(config, f"{prefix}MASK_TRANSITION_WIDTH"),
        generator_stack_source=adaptive_generator_stack_source(
            config,
            stack_prefix,
        ),
    )
