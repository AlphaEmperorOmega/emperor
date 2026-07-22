from models.gpt.expert_linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
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


__all__ = [
    "adaptive_generator_stack_options",
    "adaptive_generator_stack_source",
    "hidden_adaptive_bias_options",
    "hidden_adaptive_diagonal_options",
    "hidden_adaptive_mask_options",
    "hidden_adaptive_weight_options",
]
