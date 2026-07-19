from models.vit.expert_linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
    DynamicMemoryOptions,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
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
    VitOutputOptions,
    VitPatchOptions,
)


def vit_patch_options(config: object) -> VitPatchOptions:
    return VitPatchOptions(
        patch_size=config.IMAGE_PATCH_SIZE,
        input_channels=config.INPUT_CHANNELS,
        image_height=config.IMAGE_HEIGHT,
        dropout_probability=config.PATCH_DROPOUT_PROBABILITY,
        bias_flag=config.PATCH_BIAS_FLAG,
    )


def vit_positional_embedding_options(
    config: object,
) -> TransformerPositionalEmbeddingOptions:
    return TransformerPositionalEmbeddingOptions(
        option=config.POSITIONAL_EMBEDDING_OPTION,
        padding_idx=config.POSITIONAL_EMBEDDING_PADDING_IDX,
        auto_expand_flag=config.POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG,
    )


def vit_encoder_options(config: object) -> TransformerEncoderOptions:
    return TransformerEncoderOptions(
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.STACK_NUM_LAYERS,
        activation=config.STACK_ACTIVATION,
        dropout_probability=config.STACK_DROPOUT_PROBABILITY,
        layer_norm_position=config.STACK_LAYER_NORM_POSITION,
        causal_attention_mask_flag=False,
    )


def vit_attention_options(config: object) -> TransformerAttentionOptions:
    return TransformerAttentionOptions(
        num_heads=config.ATTN_NUM_HEADS,
        num_layers=config.ATTN_NUM_LAYERS,
        bias_flag=config.ATTN_BIAS_FLAG,
        add_key_value_bias_flag=config.ATTN_ADD_KEY_VALUE_BIAS_FLAG,
    )


def vit_feed_forward_options(config: object) -> TransformerFeedForwardOptions:
    return TransformerFeedForwardOptions(
        num_layers=config.FF_NUM_LAYERS, bias_flag=config.FF_BIAS_FLAG
    )


def vit_output_options(config: object) -> VitOutputOptions:
    return VitOutputOptions(bias_flag=config.OUTPUT_BIAS_FLAG)


def main_layer_stack_options(config: object) -> MainLayerStackOptions:
    return MainLayerStackOptions(
        bias_flag=config.STACK_BIAS_FLAG,
        layer_norm_position=config.STACK_LAYER_NORM_POSITION,
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
    config: object, prefix: str
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


def linears_layer_controller_options(
    config: object,
    *,
    gate_prefix: str,
    gate_stack_prefix: str,
    halting_prefix: str,
    halting_stack_prefix: str,
) -> LayerControllerOptions:
    return LayerControllerOptions(
        stack_gate_flag=getattr(config, f"{gate_prefix}_FLAG"),
        gate_option=getattr(config, f"{gate_prefix}_OPTION"),
        gate_activation=getattr(config, f"{gate_prefix}_ACTIVATION"),
        gate_stack_source=linears_controller_stack_source(config, gate_stack_prefix),
        stack_halting_flag=getattr(config, f"{halting_prefix}_FLAG"),
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
    config: object, *, memory_prefix: str, memory_stack_prefix: str
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
        recurrent_gate_flag=getattr(config, f"{recurrent_prefix}_GATE_FLAG"),
        recurrent_gate_option=getattr(config, f"{recurrent_prefix}_GATE_OPTION"),
        recurrent_gate_activation=getattr(
            config, f"{recurrent_prefix}_GATE_ACTIVATION"
        ),
        recurrent_gate_stack_source=linears_controller_stack_source(
            config, gate_stack_prefix
        ),
        recurrent_halting_flag=getattr(config, f"{recurrent_prefix}_HALTING_FLAG"),
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
        residual_connection_option=config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
        dropout_probability=config.ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY,
        last_layer_bias_option=config.ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        bias_flag=config.ADAPTIVE_GENERATOR_STACK_BIAS_FLAG,
    )


def adaptive_generator_stack_source(
    config: object, prefix: str
) -> AdaptiveGeneratorStackSource:
    return AdaptiveGeneratorStackSource(
        independent_flag=getattr(config, f"{prefix}_INDEPENDENT_FLAG"),
        hidden_dim=getattr(config, f"{prefix}_HIDDEN_DIM"),
        layer_norm_position=getattr(config, f"{prefix}_LAYER_NORM_POSITION"),
        num_layers=getattr(config, f"{prefix}_NUM_LAYERS"),
        activation=getattr(config, f"{prefix}_ACTIVATION"),
        residual_connection_option=getattr(
            config, f"{prefix}_RESIDUAL_CONNECTION_OPTION"
        ),
        dropout_probability=getattr(config, f"{prefix}_DROPOUT_PROBABILITY"),
        last_layer_bias_option=getattr(config, f"{prefix}_LAST_LAYER_BIAS_OPTION"),
        apply_output_pipeline_flag=getattr(
            config, f"{prefix}_APPLY_OUTPUT_PIPELINE_FLAG"
        ),
        bias_flag=getattr(config, f"{prefix}_BIAS_FLAG"),
    )


def hidden_adaptive_weight_options(
    config: object, *, prefix: str = "", stack_prefix: str = "WEIGHT_GENERATOR_STACK"
) -> HiddenAdaptiveWeightOptions:
    return HiddenAdaptiveWeightOptions(
        generator_depth=getattr(config, f"{prefix}WEIGHT_GENERATOR_DEPTH"),
        option_flag=getattr(config, f"{prefix}WEIGHT_OPTION_FLAG"),
        option=getattr(config, f"{prefix}WEIGHT_OPTION"),
        normalization_option=getattr(config, f"{prefix}WEIGHT_NORMALIZATION_OPTION"),
        normalization_position_option=getattr(
            config, f"{prefix}WEIGHT_NORMALIZATION_POSITION_OPTION"
        ),
        decay_schedule=getattr(config, f"{prefix}WEIGHT_DECAY_SCHEDULE"),
        decay_rate=getattr(config, f"{prefix}WEIGHT_DECAY_RATE"),
        decay_warmup_batches=getattr(config, f"{prefix}WEIGHT_DECAY_WARMUP_BATCHES"),
        bank_expansion_factor=getattr(config, f"{prefix}WEIGHT_BANK_EXPANSION_FACTOR"),
        generator_stack_source=adaptive_generator_stack_source(config, stack_prefix),
    )


def hidden_adaptive_bias_options(
    config: object, *, prefix: str = "", stack_prefix: str = "BIAS_GENERATOR_STACK"
) -> HiddenAdaptiveBiasOptions:
    return HiddenAdaptiveBiasOptions(
        option_flag=getattr(config, f"{prefix}BIAS_OPTION_FLAG"),
        option=getattr(config, f"{prefix}BIAS_OPTION"),
        decay_schedule=getattr(config, f"{prefix}BIAS_DECAY_SCHEDULE"),
        decay_rate=getattr(config, f"{prefix}BIAS_DECAY_RATE"),
        decay_warmup_batches=getattr(config, f"{prefix}BIAS_DECAY_WARMUP_BATCHES"),
        bank_expansion_factor=getattr(config, f"{prefix}BIAS_BANK_EXPANSION_FACTOR"),
        generator_stack_source=adaptive_generator_stack_source(config, stack_prefix),
    )


def hidden_adaptive_diagonal_options(
    config: object, *, prefix: str = "", stack_prefix: str = "DIAGONAL_GENERATOR_STACK"
) -> HiddenAdaptiveDiagonalOptions:
    return HiddenAdaptiveDiagonalOptions(
        option_flag=getattr(config, f"{prefix}DIAGONAL_OPTION_FLAG"),
        option=getattr(config, f"{prefix}DIAGONAL_OPTION"),
        generator_stack_source=adaptive_generator_stack_source(config, stack_prefix),
    )


def hidden_adaptive_mask_options(
    config: object, *, prefix: str = "", stack_prefix: str = "MASK_GENERATOR_STACK"
) -> HiddenAdaptiveMaskOptions:
    return HiddenAdaptiveMaskOptions(
        option_flag=getattr(config, f"{prefix}MASK_OPTION_FLAG"),
        row_mask_option=getattr(config, f"{prefix}ROW_MASK_OPTION"),
        mask_dimension_option=getattr(config, f"{prefix}MASK_DIMENSION_OPTION"),
        mask_threshold=getattr(config, f"{prefix}MASK_THRESHOLD"),
        mask_surrogate_scale=getattr(config, f"{prefix}MASK_SURROGATE_SCALE"),
        mask_floor=getattr(config, f"{prefix}MASK_FLOOR"),
        mask_transition_width=getattr(config, f"{prefix}MASK_TRANSITION_WIDTH"),
        generator_stack_source=adaptive_generator_stack_source(config, stack_prefix),
    )


def experts_submodule_stack_options(
    config: object, prefix: str, *, bias_key: str | None = None
) -> ExpertsSubmoduleStackOptions:
    return ExpertsSubmoduleStackOptions(
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
        bias_flag=getattr(config, bias_key or f"{prefix}_BIAS_FLAG"),
    )


def experts_submodule_stack_source(
    config: object, prefix: str
) -> ExpertsSubmoduleStackSource:
    return ExpertsSubmoduleStackSource(
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


def experts_layer_controller_options(
    config: object,
    *,
    gate_prefix: str,
    gate_stack_prefix: str,
    halting_prefix: str,
    halting_stack_prefix: str,
) -> ExpertsLayerControllerOptions:
    return ExpertsLayerControllerOptions(
        stack_gate_flag=getattr(config, f"{gate_prefix}_FLAG"),
        gate_option=getattr(config, f"{gate_prefix}_OPTION"),
        gate_activation=getattr(config, f"{gate_prefix}_ACTIVATION"),
        gate_stack_source=experts_submodule_stack_source(config, gate_stack_prefix),
        stack_halting_flag=getattr(config, f"{halting_prefix}_FLAG"),
        halting_option=getattr(
            config,
            f"{halting_prefix}_OPTION",
            ExpertsLayerControllerOptions.halting_option,
        ),
        halting_threshold=getattr(config, f"{halting_prefix}_THRESHOLD"),
        halting_dropout=getattr(config, f"{halting_prefix}_DROPOUT"),
        halting_hidden_state_mode=getattr(
            config, f"{halting_prefix}_HIDDEN_STATE_MODE"
        ),
        halting_stack_source=experts_submodule_stack_source(
            config, halting_stack_prefix
        ),
        halting_output_dim=getattr(config, f"{halting_prefix}_OUTPUT_DIM"),
    )


def experts_dynamic_memory_options(
    config: object, *, memory_prefix: str, memory_stack_prefix: str
) -> ExpertsDynamicMemoryOptions:
    return ExpertsDynamicMemoryOptions(
        memory_flag=getattr(config, f"{memory_prefix}_FLAG"),
        memory_option=getattr(config, f"{memory_prefix}_OPTION"),
        memory_position_option=getattr(config, f"{memory_prefix}_POSITION_OPTION"),
        memory_test_time_training_learning_rate=getattr(
            config, f"{memory_prefix}_TEST_TIME_TRAINING_LEARNING_RATE"
        ),
        memory_test_time_training_num_inner_steps=getattr(
            config, f"{memory_prefix}_TEST_TIME_TRAINING_NUM_INNER_STEPS"
        ),
        memory_stack_source=experts_submodule_stack_source(config, memory_stack_prefix),
    )


def experts_recurrent_controller_options(
    config: object,
    *,
    recurrent_prefix: str,
    gate_stack_prefix: str,
    halting_stack_prefix: str,
) -> ExpertsRecurrentControllerOptions:
    return ExpertsRecurrentControllerOptions(
        recurrent_flag=getattr(config, f"{recurrent_prefix}_FLAG"),
        recurrent_max_steps=getattr(config, f"{recurrent_prefix}_MAX_STEPS"),
        recurrent_layer_norm_position=getattr(
            config, f"{recurrent_prefix}_LAYER_NORM_POSITION"
        ),
        recurrent_gate_flag=getattr(config, f"{recurrent_prefix}_GATE_FLAG"),
        recurrent_gate_option=getattr(config, f"{recurrent_prefix}_GATE_OPTION"),
        recurrent_gate_activation=getattr(
            config, f"{recurrent_prefix}_GATE_ACTIVATION"
        ),
        recurrent_gate_stack_source=experts_submodule_stack_source(
            config, gate_stack_prefix
        ),
        recurrent_halting_flag=getattr(config, f"{recurrent_prefix}_HALTING_FLAG"),
        recurrent_halting_option=getattr(
            config,
            f"{recurrent_prefix}_HALTING_OPTION",
            ExpertsRecurrentControllerOptions.recurrent_halting_option,
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
        recurrent_halting_stack_source=experts_submodule_stack_source(
            config, halting_stack_prefix
        ),
    )


def experts_mixture_options(config: object) -> ExpertsMixtureOptions:
    return ExpertsMixtureOptions(
        top_k=config.EXPERT_TOP_K,
        num_experts=config.EXPERT_NUM_EXPERTS,
        capacity_factor=config.EXPERT_CAPACITY_FACTOR,
        dropped_token_behavior=config.EXPERT_DROPPED_TOKEN_BEHAVIOR,
        compute_expert_mixture_flag=config.EXPERT_COMPUTE_EXPERT_MIXTURE_FLAG,
        weighted_parameters_flag=config.EXPERT_WEIGHTED_PARAMETERS_FLAG,
        weighting_position_option=config.EXPERT_WEIGHTING_POSITION_OPTION,
        routing_initialization_mode=config.EXPERT_ROUTING_INITIALIZATION_MODE,
    )


def experts_sampler_options(config: object) -> ExpertsSamplerOptions:
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


def experts_router_options(config: object) -> ExpertsRouterOptions:
    return ExpertsRouterOptions(noisy_topk_flag=config.ROUTER_NOISY_TOPK_FLAG)
