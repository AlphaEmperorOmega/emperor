from dataclasses import dataclass, field, replace
from types import ModuleType

from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from models.gpt.expert_linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
)


def adaptive_generator_stack_options(
    config: ModuleType,
) -> AdaptiveGeneratorStackOptions:
    return AdaptiveGeneratorStackOptions(
        hidden_dim=config.ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM,
        layer_norm_position=config.ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION,
        normalization=config.ADAPTIVE_GENERATOR_STACK_NORMALIZATION,
        num_layers=config.ADAPTIVE_GENERATOR_STACK_NUM_LAYERS,
        activation=config.ADAPTIVE_GENERATOR_STACK_ACTIVATION,
        residual_connection_option=(
            config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_block_size=config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
        residual_rms_norm_epsilon=config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
        residual_model_flag=config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY,
        last_layer_bias_option=config.ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_postprocessing_flag=(
            config.ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
        ),
        bias_flag=config.ADAPTIVE_GENERATOR_STACK_BIAS_FLAG,
    )


@dataclass(frozen=True, slots=True)
class _AdaptiveGeneratorStackDefaults:
    independent_flag: bool
    hidden_dim: int | None
    layer_norm_position: LayerNormPositionOptions | None
    normalization: NormalizationOptions | None = field(default=None, kw_only=True)
    num_layers: int | None
    activation: ActivationOptions | None
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    bias_flag: bool | None


def _adaptive_generator_stack_source(
    defaults: _AdaptiveGeneratorStackDefaults,
) -> AdaptiveGeneratorStackSource:
    return AdaptiveGeneratorStackSource(
        independent_flag=defaults.independent_flag,
        hidden_dim=defaults.hidden_dim,
        layer_norm_position=defaults.layer_norm_position,
        normalization=defaults.normalization,
        num_layers=defaults.num_layers,
        activation=defaults.activation,
        residual_connection_option=defaults.residual_connection_option,
        residual_block_size=defaults.residual_block_size,
        residual_rms_norm_epsilon=defaults.residual_rms_norm_epsilon,
        residual_model_flag=defaults.residual_model_flag,
        dropout_probability=defaults.dropout_probability,
        last_layer_bias_option=defaults.last_layer_bias_option,
        apply_output_postprocessing_flag=defaults.apply_output_postprocessing_flag,
        bias_flag=defaults.bias_flag,
    )


def weight_generator_stack_source(config: ModuleType) -> AdaptiveGeneratorStackSource:
    return _adaptive_generator_stack_source(
        _AdaptiveGeneratorStackDefaults(
            independent_flag=config.WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.WEIGHT_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION,
            normalization=config.WEIGHT_GENERATOR_STACK_NORMALIZATION,
            num_layers=config.WEIGHT_GENERATOR_STACK_NUM_LAYERS,
            activation=config.WEIGHT_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=(
                config.WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_block_size=config.WEIGHT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
            residual_rms_norm_epsilon=config.WEIGHT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
            residual_model_flag=config.WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=(
                config.WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config.WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            bias_flag=config.WEIGHT_GENERATOR_STACK_BIAS_FLAG,
        )
    )


def bias_generator_stack_source(config: ModuleType) -> AdaptiveGeneratorStackSource:
    return _adaptive_generator_stack_source(
        _AdaptiveGeneratorStackDefaults(
            independent_flag=config.BIAS_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.BIAS_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.BIAS_GENERATOR_STACK_LAYER_NORM_POSITION,
            normalization=config.BIAS_GENERATOR_STACK_NORMALIZATION,
            num_layers=config.BIAS_GENERATOR_STACK_NUM_LAYERS,
            activation=config.BIAS_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=(
                config.BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_block_size=config.BIAS_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
            residual_rms_norm_epsilon=config.BIAS_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
            residual_model_flag=config.BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=(config.BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION),
            apply_output_postprocessing_flag=(
                config.BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            bias_flag=config.BIAS_GENERATOR_STACK_BIAS_FLAG,
        )
    )


def diagonal_generator_stack_source(
    config: ModuleType,
) -> AdaptiveGeneratorStackSource:
    return _adaptive_generator_stack_source(
        _AdaptiveGeneratorStackDefaults(
            independent_flag=config.DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.DIAGONAL_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION,
            normalization=config.DIAGONAL_GENERATOR_STACK_NORMALIZATION,
            num_layers=config.DIAGONAL_GENERATOR_STACK_NUM_LAYERS,
            activation=config.DIAGONAL_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=(
                config.DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_block_size=config.DIAGONAL_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
            residual_rms_norm_epsilon=config.DIAGONAL_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
            residual_model_flag=config.DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=(
                config.DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config.DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            bias_flag=config.DIAGONAL_GENERATOR_STACK_BIAS_FLAG,
        )
    )


def mask_generator_stack_source(config: ModuleType) -> AdaptiveGeneratorStackSource:
    return _adaptive_generator_stack_source(
        _AdaptiveGeneratorStackDefaults(
            independent_flag=config.MASK_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.MASK_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.MASK_GENERATOR_STACK_LAYER_NORM_POSITION,
            normalization=config.MASK_GENERATOR_STACK_NORMALIZATION,
            num_layers=config.MASK_GENERATOR_STACK_NUM_LAYERS,
            activation=config.MASK_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=(
                config.MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_block_size=config.MASK_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
            residual_rms_norm_epsilon=config.MASK_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
            residual_model_flag=config.MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.MASK_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=(config.MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION),
            apply_output_postprocessing_flag=(
                config.MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            bias_flag=config.MASK_GENERATOR_STACK_BIAS_FLAG,
        )
    )


def router_weight_generator_stack_source(
    config: ModuleType,
) -> AdaptiveGeneratorStackSource:
    return _adaptive_generator_stack_source(
        _AdaptiveGeneratorStackDefaults(
            independent_flag=config.ROUTER_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_WEIGHT_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=(
                config.ROUTER_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION
            ),
            normalization=(config.ROUTER_WEIGHT_GENERATOR_STACK_NORMALIZATION),
            num_layers=config.ROUTER_WEIGHT_GENERATOR_STACK_NUM_LAYERS,
            activation=config.ROUTER_WEIGHT_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=(
                config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_block_size=config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
            residual_rms_norm_epsilon=config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
            residual_model_flag=(
                config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config.ROUTER_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY
            ),
            last_layer_bias_option=(
                config.ROUTER_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config.ROUTER_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            bias_flag=config.ROUTER_WEIGHT_GENERATOR_STACK_BIAS_FLAG,
        )
    )


def router_bias_generator_stack_source(
    config: ModuleType,
) -> AdaptiveGeneratorStackSource:
    return _adaptive_generator_stack_source(
        _AdaptiveGeneratorStackDefaults(
            independent_flag=config.ROUTER_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_BIAS_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=(
                config.ROUTER_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION
            ),
            normalization=(config.ROUTER_BIAS_GENERATOR_STACK_NORMALIZATION),
            num_layers=config.ROUTER_BIAS_GENERATOR_STACK_NUM_LAYERS,
            activation=config.ROUTER_BIAS_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=(
                config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_block_size=config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
            residual_rms_norm_epsilon=config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
            residual_model_flag=(
                config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config.ROUTER_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY
            ),
            last_layer_bias_option=(
                config.ROUTER_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config.ROUTER_BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            bias_flag=config.ROUTER_BIAS_GENERATOR_STACK_BIAS_FLAG,
        )
    )


def router_diagonal_generator_stack_source(
    config: ModuleType,
) -> AdaptiveGeneratorStackSource:
    return _adaptive_generator_stack_source(
        _AdaptiveGeneratorStackDefaults(
            independent_flag=config.ROUTER_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=(
                config.ROUTER_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION
            ),
            normalization=(config.ROUTER_DIAGONAL_GENERATOR_STACK_NORMALIZATION),
            num_layers=config.ROUTER_DIAGONAL_GENERATOR_STACK_NUM_LAYERS,
            activation=config.ROUTER_DIAGONAL_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=(
                config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_block_size=config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
            residual_rms_norm_epsilon=config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
            residual_model_flag=(
                config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config.ROUTER_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY
            ),
            last_layer_bias_option=(
                config.ROUTER_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config.ROUTER_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            bias_flag=config.ROUTER_DIAGONAL_GENERATOR_STACK_BIAS_FLAG,
        )
    )


def router_mask_generator_stack_source(
    config: ModuleType,
) -> AdaptiveGeneratorStackSource:
    return _adaptive_generator_stack_source(
        _AdaptiveGeneratorStackDefaults(
            independent_flag=config.ROUTER_MASK_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_MASK_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=(
                config.ROUTER_MASK_GENERATOR_STACK_LAYER_NORM_POSITION
            ),
            normalization=(config.ROUTER_MASK_GENERATOR_STACK_NORMALIZATION),
            num_layers=config.ROUTER_MASK_GENERATOR_STACK_NUM_LAYERS,
            activation=config.ROUTER_MASK_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=(
                config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_block_size=config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE,
            residual_rms_norm_epsilon=config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON,
            residual_model_flag=(
                config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config.ROUTER_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY
            ),
            last_layer_bias_option=(
                config.ROUTER_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config.ROUTER_MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            bias_flag=config.ROUTER_MASK_GENERATOR_STACK_BIAS_FLAG,
        )
    )


def adaptive_generator_stack_source(
    config: ModuleType,
    prefix: str,
) -> AdaptiveGeneratorStackSource:
    if prefix == "WEIGHT_GENERATOR_STACK":
        return weight_generator_stack_source(config)
    if prefix == "BIAS_GENERATOR_STACK":
        return bias_generator_stack_source(config)
    if prefix == "DIAGONAL_GENERATOR_STACK":
        return diagonal_generator_stack_source(config)
    if prefix == "MASK_GENERATOR_STACK":
        return mask_generator_stack_source(config)
    if prefix == "ROUTER_WEIGHT_GENERATOR_STACK":
        return router_weight_generator_stack_source(config)
    if prefix == "ROUTER_BIAS_GENERATOR_STACK":
        return router_bias_generator_stack_source(config)
    if prefix == "ROUTER_DIAGONAL_GENERATOR_STACK":
        return router_diagonal_generator_stack_source(config)
    if prefix == "ROUTER_MASK_GENERATOR_STACK":
        return router_mask_generator_stack_source(config)
    missing_name = f"{prefix}_INDEPENDENT_FLAG"
    raise AttributeError(
        f"module {config.__name__!r} has no attribute {missing_name!r}"
    )


def hidden_adaptive_weight_options(
    config: ModuleType,
    *,
    prefix: str = "",
    stack_prefix: str = "WEIGHT_GENERATOR_STACK",
) -> HiddenAdaptiveWeightOptions:
    if prefix == "":
        options = HiddenAdaptiveWeightOptions(
            generator_depth=config.GENERATOR_DEPTH,
            option_flag=config.WEIGHT_OPTION_FLAG,
            option=config.WEIGHT_OPTION,
            normalization_option=config.WEIGHT_NORMALIZATION_OPTION,
            normalization_position_option=config.WEIGHT_NORMALIZATION_POSITION_OPTION,
            decay_schedule=config.WEIGHT_DECAY_SCHEDULE,
            decay_rate=config.WEIGHT_DECAY_RATE,
            decay_warmup_batches=config.WEIGHT_DECAY_WARMUP_BATCHES,
            bank_expansion_factor=config.WEIGHT_BANK_EXPANSION_FACTOR,
            generator_stack_source=weight_generator_stack_source(config),
        )
    elif prefix == "ROUTER_":
        options = router_adaptive_weight_options(config)
    else:
        missing_name = f"{prefix}GENERATOR_DEPTH"
        raise AttributeError(
            f"module {config.__name__!r} has no attribute {missing_name!r}"
        )
    return replace(
        options,
        generator_stack_source=adaptive_generator_stack_source(config, stack_prefix),
    )


def hidden_adaptive_bias_options(
    config: ModuleType,
    *,
    prefix: str = "",
    stack_prefix: str = "BIAS_GENERATOR_STACK",
) -> HiddenAdaptiveBiasOptions:
    if prefix == "":
        options = HiddenAdaptiveBiasOptions(
            option_flag=config.BIAS_OPTION_FLAG,
            option=config.BIAS_OPTION,
            decay_schedule=config.BIAS_DECAY_SCHEDULE,
            decay_rate=config.BIAS_DECAY_RATE,
            decay_warmup_batches=config.BIAS_DECAY_WARMUP_BATCHES,
            bank_expansion_factor=config.BIAS_BANK_EXPANSION_FACTOR,
            generator_stack_source=bias_generator_stack_source(config),
        )
    elif prefix == "ROUTER_":
        options = router_adaptive_bias_options(config)
    else:
        missing_name = f"{prefix}BIAS_OPTION_FLAG"
        raise AttributeError(
            f"module {config.__name__!r} has no attribute {missing_name!r}"
        )
    return replace(
        options,
        generator_stack_source=adaptive_generator_stack_source(config, stack_prefix),
    )


def hidden_adaptive_diagonal_options(
    config: ModuleType,
    *,
    prefix: str = "",
    stack_prefix: str = "DIAGONAL_GENERATOR_STACK",
) -> HiddenAdaptiveDiagonalOptions:
    if prefix == "":
        options = HiddenAdaptiveDiagonalOptions(
            option_flag=config.DIAGONAL_OPTION_FLAG,
            option=config.DIAGONAL_OPTION,
            generator_stack_source=diagonal_generator_stack_source(config),
        )
    elif prefix == "ROUTER_":
        options = router_adaptive_diagonal_options(config)
    else:
        missing_name = f"{prefix}DIAGONAL_OPTION_FLAG"
        raise AttributeError(
            f"module {config.__name__!r} has no attribute {missing_name!r}"
        )
    return replace(
        options,
        generator_stack_source=adaptive_generator_stack_source(config, stack_prefix),
    )


def hidden_adaptive_mask_options(
    config: ModuleType,
    *,
    prefix: str = "",
    stack_prefix: str = "MASK_GENERATOR_STACK",
) -> HiddenAdaptiveMaskOptions:
    if prefix == "":
        options = HiddenAdaptiveMaskOptions(
            option_flag=config.MASK_OPTION_FLAG,
            row_mask_option=config.ROW_MASK_OPTION,
            mask_dimension_option=config.MASK_DIMENSION_OPTION,
            mask_threshold=config.MASK_THRESHOLD,
            mask_surrogate_scale=config.MASK_SURROGATE_SCALE,
            mask_floor=config.MASK_FLOOR,
            mask_transition_width=config.MASK_TRANSITION_WIDTH,
            generator_stack_source=mask_generator_stack_source(config),
        )
    elif prefix == "ROUTER_":
        options = router_adaptive_mask_options(config)
    else:
        missing_name = f"{prefix}MASK_OPTION_FLAG"
        raise AttributeError(
            f"module {config.__name__!r} has no attribute {missing_name!r}"
        )
    return replace(
        options,
        generator_stack_source=adaptive_generator_stack_source(config, stack_prefix),
    )


def router_adaptive_weight_options(
    config: ModuleType,
) -> HiddenAdaptiveWeightOptions:
    return HiddenAdaptiveWeightOptions(
        generator_depth=config.ROUTER_GENERATOR_DEPTH,
        option_flag=config.ROUTER_WEIGHT_OPTION_FLAG,
        option=config.ROUTER_WEIGHT_OPTION,
        normalization_option=config.ROUTER_WEIGHT_NORMALIZATION_OPTION,
        normalization_position_option=(
            config.ROUTER_WEIGHT_NORMALIZATION_POSITION_OPTION
        ),
        decay_schedule=config.ROUTER_WEIGHT_DECAY_SCHEDULE,
        decay_rate=config.ROUTER_WEIGHT_DECAY_RATE,
        decay_warmup_batches=config.ROUTER_WEIGHT_DECAY_WARMUP_BATCHES,
        bank_expansion_factor=config.ROUTER_WEIGHT_BANK_EXPANSION_FACTOR,
        generator_stack_source=router_weight_generator_stack_source(config),
    )


def router_adaptive_bias_options(config: ModuleType) -> HiddenAdaptiveBiasOptions:
    return HiddenAdaptiveBiasOptions(
        option_flag=config.ROUTER_BIAS_OPTION_FLAG,
        option=config.ROUTER_BIAS_OPTION,
        decay_schedule=config.ROUTER_BIAS_DECAY_SCHEDULE,
        decay_rate=config.ROUTER_BIAS_DECAY_RATE,
        decay_warmup_batches=config.ROUTER_BIAS_DECAY_WARMUP_BATCHES,
        bank_expansion_factor=config.ROUTER_BIAS_BANK_EXPANSION_FACTOR,
        generator_stack_source=router_bias_generator_stack_source(config),
    )


def router_adaptive_diagonal_options(
    config: ModuleType,
) -> HiddenAdaptiveDiagonalOptions:
    return HiddenAdaptiveDiagonalOptions(
        option_flag=config.ROUTER_DIAGONAL_OPTION_FLAG,
        option=config.ROUTER_DIAGONAL_OPTION,
        generator_stack_source=router_diagonal_generator_stack_source(config),
    )


def router_adaptive_mask_options(config: ModuleType) -> HiddenAdaptiveMaskOptions:
    return HiddenAdaptiveMaskOptions(
        option_flag=config.ROUTER_MASK_OPTION_FLAG,
        row_mask_option=config.ROUTER_ROW_MASK_OPTION,
        mask_dimension_option=config.ROUTER_MASK_DIMENSION_OPTION,
        mask_threshold=config.ROUTER_MASK_THRESHOLD,
        mask_surrogate_scale=config.ROUTER_MASK_SURROGATE_SCALE,
        mask_floor=config.ROUTER_MASK_FLOOR,
        mask_transition_width=config.ROUTER_MASK_TRANSITION_WIDTH,
        generator_stack_source=router_mask_generator_stack_source(config),
    )


__all__ = [
    "adaptive_generator_stack_options",
    "adaptive_generator_stack_source",
    "hidden_adaptive_bias_options",
    "hidden_adaptive_diagonal_options",
    "hidden_adaptive_mask_options",
    "hidden_adaptive_weight_options",
]
