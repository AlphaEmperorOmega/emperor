from collections.abc import Callable, Mapping
from enum import Enum, auto
from types import MappingProxyType, ModuleType

from models.vit.expert_linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
)


class AdaptiveRole(Enum):
    MAIN = auto()
    ATTENTION = auto()
    FEED_FORWARD = auto()
    ROUTER = auto()


class AdaptiveParameter(Enum):
    WEIGHT = auto()
    BIAS = auto()
    DIAGONAL = auto()
    MASK = auto()


_GeneratorSourceFactory = Callable[[ModuleType], AdaptiveGeneratorStackSource]
_WeightOptionsFactory = Callable[[ModuleType], HiddenAdaptiveWeightOptions]
_BiasOptionsFactory = Callable[[ModuleType], HiddenAdaptiveBiasOptions]
_DiagonalOptionsFactory = Callable[[ModuleType], HiddenAdaptiveDiagonalOptions]
_MaskOptionsFactory = Callable[[ModuleType], HiddenAdaptiveMaskOptions]


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
        last_layer_bias_option=(config.ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION),
        apply_output_pipeline_flag=(
            config.ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        bias_flag=config.ADAPTIVE_GENERATOR_STACK_BIAS_FLAG,
    )


_GENERATOR_SOURCE_FACTORIES: Mapping[
    tuple[AdaptiveRole, AdaptiveParameter], _GeneratorSourceFactory
] = MappingProxyType(
    {
        (
            AdaptiveRole.MAIN,
            AdaptiveParameter.WEIGHT,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.WEIGHT_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.WEIGHT_GENERATOR_STACK_NUM_LAYERS,
            activation=config.WEIGHT_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.WEIGHT_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.MAIN,
            AdaptiveParameter.BIAS,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.BIAS_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.BIAS_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.BIAS_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.BIAS_GENERATOR_STACK_NUM_LAYERS,
            activation=config.BIAS_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.BIAS_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.MAIN,
            AdaptiveParameter.DIAGONAL,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.DIAGONAL_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.DIAGONAL_GENERATOR_STACK_NUM_LAYERS,
            activation=config.DIAGONAL_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.DIAGONAL_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.MAIN,
            AdaptiveParameter.MASK,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.MASK_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.MASK_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.MASK_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.MASK_GENERATOR_STACK_NUM_LAYERS,
            activation=config.MASK_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.MASK_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.MASK_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.ATTENTION,
            AdaptiveParameter.WEIGHT,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.ATTN_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ATTN_WEIGHT_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.ATTN_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.ATTN_WEIGHT_GENERATOR_STACK_NUM_LAYERS,
            activation=config.ATTN_WEIGHT_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.ATTN_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ATTN_WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ATTN_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.ATTN_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.ATTN_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.ATTN_WEIGHT_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.ATTENTION,
            AdaptiveParameter.BIAS,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.ATTN_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ATTN_BIAS_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.ATTN_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.ATTN_BIAS_GENERATOR_STACK_NUM_LAYERS,
            activation=config.ATTN_BIAS_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.ATTN_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ATTN_BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ATTN_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.ATTN_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.ATTN_BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.ATTN_BIAS_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.ATTENTION,
            AdaptiveParameter.DIAGONAL,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.ATTN_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ATTN_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.ATTN_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.ATTN_DIAGONAL_GENERATOR_STACK_NUM_LAYERS,
            activation=config.ATTN_DIAGONAL_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.ATTN_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ATTN_DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ATTN_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.ATTN_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.ATTN_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.ATTN_DIAGONAL_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.ATTENTION,
            AdaptiveParameter.MASK,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.ATTN_MASK_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ATTN_MASK_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.ATTN_MASK_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.ATTN_MASK_GENERATOR_STACK_NUM_LAYERS,
            activation=config.ATTN_MASK_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.ATTN_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ATTN_MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ATTN_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.ATTN_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.ATTN_MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.ATTN_MASK_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.FEED_FORWARD,
            AdaptiveParameter.WEIGHT,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.FF_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.FF_WEIGHT_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.FF_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.FF_WEIGHT_GENERATOR_STACK_NUM_LAYERS,
            activation=config.FF_WEIGHT_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.FF_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.FF_WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.FF_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.FF_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.FF_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.FF_WEIGHT_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.FEED_FORWARD,
            AdaptiveParameter.BIAS,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.FF_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.FF_BIAS_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.FF_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.FF_BIAS_GENERATOR_STACK_NUM_LAYERS,
            activation=config.FF_BIAS_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.FF_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.FF_BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.FF_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.FF_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.FF_BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.FF_BIAS_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.FEED_FORWARD,
            AdaptiveParameter.DIAGONAL,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.FF_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.FF_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.FF_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.FF_DIAGONAL_GENERATOR_STACK_NUM_LAYERS,
            activation=config.FF_DIAGONAL_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.FF_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.FF_DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.FF_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.FF_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.FF_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.FF_DIAGONAL_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.FEED_FORWARD,
            AdaptiveParameter.MASK,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.FF_MASK_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.FF_MASK_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.FF_MASK_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.FF_MASK_GENERATOR_STACK_NUM_LAYERS,
            activation=config.FF_MASK_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.FF_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.FF_MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.FF_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.FF_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.FF_MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.FF_MASK_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.ROUTER,
            AdaptiveParameter.WEIGHT,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.ROUTER_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_WEIGHT_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.ROUTER_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.ROUTER_WEIGHT_GENERATOR_STACK_NUM_LAYERS,
            activation=config.ROUTER_WEIGHT_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ROUTER_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.ROUTER_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.ROUTER_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.ROUTER_WEIGHT_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.ROUTER,
            AdaptiveParameter.BIAS,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.ROUTER_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_BIAS_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.ROUTER_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.ROUTER_BIAS_GENERATOR_STACK_NUM_LAYERS,
            activation=config.ROUTER_BIAS_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ROUTER_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.ROUTER_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.ROUTER_BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.ROUTER_BIAS_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.ROUTER,
            AdaptiveParameter.DIAGONAL,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.ROUTER_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.ROUTER_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.ROUTER_DIAGONAL_GENERATOR_STACK_NUM_LAYERS,
            activation=config.ROUTER_DIAGONAL_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ROUTER_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.ROUTER_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.ROUTER_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.ROUTER_DIAGONAL_GENERATOR_STACK_BIAS_FLAG,
        ),
        (
            AdaptiveRole.ROUTER,
            AdaptiveParameter.MASK,
        ): lambda config: AdaptiveGeneratorStackSource(
            independent_flag=config.ROUTER_MASK_GENERATOR_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_MASK_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=config.ROUTER_MASK_GENERATOR_STACK_LAYER_NORM_POSITION,
            num_layers=config.ROUTER_MASK_GENERATOR_STACK_NUM_LAYERS,
            activation=config.ROUTER_MASK_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ROUTER_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.ROUTER_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.ROUTER_MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            bias_flag=config.ROUTER_MASK_GENERATOR_STACK_BIAS_FLAG,
        ),
    }
)


def adaptive_generator_stack_source(
    config: ModuleType,
    role: AdaptiveRole,
    parameter: AdaptiveParameter,
) -> AdaptiveGeneratorStackSource:
    return _GENERATOR_SOURCE_FACTORIES[role, parameter](config)


_WEIGHT_OPTIONS_FACTORIES: Mapping[AdaptiveRole, _WeightOptionsFactory] = (
    MappingProxyType(
        {
            AdaptiveRole.MAIN: lambda config: HiddenAdaptiveWeightOptions(
                generator_depth=config.GENERATOR_DEPTH,
                option_flag=config.WEIGHT_OPTION_FLAG,
                option=config.WEIGHT_OPTION,
                normalization_option=config.WEIGHT_NORMALIZATION_OPTION,
                normalization_position_option=config.WEIGHT_NORMALIZATION_POSITION_OPTION,
                decay_schedule=config.WEIGHT_DECAY_SCHEDULE,
                decay_rate=config.WEIGHT_DECAY_RATE,
                decay_warmup_batches=config.WEIGHT_DECAY_WARMUP_BATCHES,
                bank_expansion_factor=config.WEIGHT_BANK_EXPANSION_FACTOR,
                generator_stack_source=adaptive_generator_stack_source(
                    config, AdaptiveRole.MAIN, AdaptiveParameter.WEIGHT
                ),
            ),
            AdaptiveRole.ATTENTION: lambda config: HiddenAdaptiveWeightOptions(
                generator_depth=config.ATTN_GENERATOR_DEPTH,
                option_flag=config.ATTN_WEIGHT_OPTION_FLAG,
                option=config.ATTN_WEIGHT_OPTION,
                normalization_option=config.ATTN_WEIGHT_NORMALIZATION_OPTION,
                normalization_position_option=config.ATTN_WEIGHT_NORMALIZATION_POSITION_OPTION,
                decay_schedule=config.ATTN_WEIGHT_DECAY_SCHEDULE,
                decay_rate=config.ATTN_WEIGHT_DECAY_RATE,
                decay_warmup_batches=config.ATTN_WEIGHT_DECAY_WARMUP_BATCHES,
                bank_expansion_factor=config.ATTN_WEIGHT_BANK_EXPANSION_FACTOR,
                generator_stack_source=adaptive_generator_stack_source(
                    config, AdaptiveRole.ATTENTION, AdaptiveParameter.WEIGHT
                ),
            ),
            AdaptiveRole.FEED_FORWARD: lambda config: HiddenAdaptiveWeightOptions(
                generator_depth=config.FF_GENERATOR_DEPTH,
                option_flag=config.FF_WEIGHT_OPTION_FLAG,
                option=config.FF_WEIGHT_OPTION,
                normalization_option=config.FF_WEIGHT_NORMALIZATION_OPTION,
                normalization_position_option=config.FF_WEIGHT_NORMALIZATION_POSITION_OPTION,
                decay_schedule=config.FF_WEIGHT_DECAY_SCHEDULE,
                decay_rate=config.FF_WEIGHT_DECAY_RATE,
                decay_warmup_batches=config.FF_WEIGHT_DECAY_WARMUP_BATCHES,
                bank_expansion_factor=config.FF_WEIGHT_BANK_EXPANSION_FACTOR,
                generator_stack_source=adaptive_generator_stack_source(
                    config, AdaptiveRole.FEED_FORWARD, AdaptiveParameter.WEIGHT
                ),
            ),
            AdaptiveRole.ROUTER: lambda config: HiddenAdaptiveWeightOptions(
                generator_depth=config.ROUTER_GENERATOR_DEPTH,
                option_flag=config.ROUTER_WEIGHT_OPTION_FLAG,
                option=config.ROUTER_WEIGHT_OPTION,
                normalization_option=config.ROUTER_WEIGHT_NORMALIZATION_OPTION,
                normalization_position_option=config.ROUTER_WEIGHT_NORMALIZATION_POSITION_OPTION,
                decay_schedule=config.ROUTER_WEIGHT_DECAY_SCHEDULE,
                decay_rate=config.ROUTER_WEIGHT_DECAY_RATE,
                decay_warmup_batches=config.ROUTER_WEIGHT_DECAY_WARMUP_BATCHES,
                bank_expansion_factor=config.ROUTER_WEIGHT_BANK_EXPANSION_FACTOR,
                generator_stack_source=adaptive_generator_stack_source(
                    config, AdaptiveRole.ROUTER, AdaptiveParameter.WEIGHT
                ),
            ),
        }
    )
)


_BIAS_OPTIONS_FACTORIES: Mapping[AdaptiveRole, _BiasOptionsFactory] = MappingProxyType(
    {
        AdaptiveRole.MAIN: lambda config: HiddenAdaptiveBiasOptions(
            option_flag=config.BIAS_OPTION_FLAG,
            option=config.BIAS_OPTION,
            decay_schedule=config.BIAS_DECAY_SCHEDULE,
            decay_rate=config.BIAS_DECAY_RATE,
            decay_warmup_batches=config.BIAS_DECAY_WARMUP_BATCHES,
            bank_expansion_factor=config.BIAS_BANK_EXPANSION_FACTOR,
            generator_stack_source=adaptive_generator_stack_source(
                config, AdaptiveRole.MAIN, AdaptiveParameter.BIAS
            ),
        ),
        AdaptiveRole.ATTENTION: lambda config: HiddenAdaptiveBiasOptions(
            option_flag=config.ATTN_BIAS_OPTION_FLAG,
            option=config.ATTN_BIAS_OPTION,
            decay_schedule=config.ATTN_BIAS_DECAY_SCHEDULE,
            decay_rate=config.ATTN_BIAS_DECAY_RATE,
            decay_warmup_batches=config.ATTN_BIAS_DECAY_WARMUP_BATCHES,
            bank_expansion_factor=config.ATTN_BIAS_BANK_EXPANSION_FACTOR,
            generator_stack_source=adaptive_generator_stack_source(
                config, AdaptiveRole.ATTENTION, AdaptiveParameter.BIAS
            ),
        ),
        AdaptiveRole.FEED_FORWARD: lambda config: HiddenAdaptiveBiasOptions(
            option_flag=config.FF_BIAS_OPTION_FLAG,
            option=config.FF_BIAS_OPTION,
            decay_schedule=config.FF_BIAS_DECAY_SCHEDULE,
            decay_rate=config.FF_BIAS_DECAY_RATE,
            decay_warmup_batches=config.FF_BIAS_DECAY_WARMUP_BATCHES,
            bank_expansion_factor=config.FF_BIAS_BANK_EXPANSION_FACTOR,
            generator_stack_source=adaptive_generator_stack_source(
                config, AdaptiveRole.FEED_FORWARD, AdaptiveParameter.BIAS
            ),
        ),
        AdaptiveRole.ROUTER: lambda config: HiddenAdaptiveBiasOptions(
            option_flag=config.ROUTER_BIAS_OPTION_FLAG,
            option=config.ROUTER_BIAS_OPTION,
            decay_schedule=config.ROUTER_BIAS_DECAY_SCHEDULE,
            decay_rate=config.ROUTER_BIAS_DECAY_RATE,
            decay_warmup_batches=config.ROUTER_BIAS_DECAY_WARMUP_BATCHES,
            bank_expansion_factor=config.ROUTER_BIAS_BANK_EXPANSION_FACTOR,
            generator_stack_source=adaptive_generator_stack_source(
                config, AdaptiveRole.ROUTER, AdaptiveParameter.BIAS
            ),
        ),
    }
)


_DIAGONAL_OPTIONS_FACTORIES: Mapping[AdaptiveRole, _DiagonalOptionsFactory] = (
    MappingProxyType(
        {
            AdaptiveRole.MAIN: lambda config: HiddenAdaptiveDiagonalOptions(
                option_flag=config.DIAGONAL_OPTION_FLAG,
                option=config.DIAGONAL_OPTION,
                generator_stack_source=adaptive_generator_stack_source(
                    config, AdaptiveRole.MAIN, AdaptiveParameter.DIAGONAL
                ),
            ),
            AdaptiveRole.ATTENTION: lambda config: HiddenAdaptiveDiagonalOptions(
                option_flag=config.ATTN_DIAGONAL_OPTION_FLAG,
                option=config.ATTN_DIAGONAL_OPTION,
                generator_stack_source=adaptive_generator_stack_source(
                    config, AdaptiveRole.ATTENTION, AdaptiveParameter.DIAGONAL
                ),
            ),
            AdaptiveRole.FEED_FORWARD: lambda config: HiddenAdaptiveDiagonalOptions(
                option_flag=config.FF_DIAGONAL_OPTION_FLAG,
                option=config.FF_DIAGONAL_OPTION,
                generator_stack_source=adaptive_generator_stack_source(
                    config, AdaptiveRole.FEED_FORWARD, AdaptiveParameter.DIAGONAL
                ),
            ),
            AdaptiveRole.ROUTER: lambda config: HiddenAdaptiveDiagonalOptions(
                option_flag=config.ROUTER_DIAGONAL_OPTION_FLAG,
                option=config.ROUTER_DIAGONAL_OPTION,
                generator_stack_source=adaptive_generator_stack_source(
                    config, AdaptiveRole.ROUTER, AdaptiveParameter.DIAGONAL
                ),
            ),
        }
    )
)


_MASK_OPTIONS_FACTORIES: Mapping[AdaptiveRole, _MaskOptionsFactory] = MappingProxyType(
    {
        AdaptiveRole.MAIN: lambda config: HiddenAdaptiveMaskOptions(
            option_flag=config.MASK_OPTION_FLAG,
            row_mask_option=config.ROW_MASK_OPTION,
            mask_dimension_option=config.MASK_DIMENSION_OPTION,
            mask_threshold=config.MASK_THRESHOLD,
            mask_surrogate_scale=config.MASK_SURROGATE_SCALE,
            mask_floor=config.MASK_FLOOR,
            mask_transition_width=config.MASK_TRANSITION_WIDTH,
            generator_stack_source=adaptive_generator_stack_source(
                config, AdaptiveRole.MAIN, AdaptiveParameter.MASK
            ),
        ),
        AdaptiveRole.ATTENTION: lambda config: HiddenAdaptiveMaskOptions(
            option_flag=config.ATTN_MASK_OPTION_FLAG,
            row_mask_option=config.ATTN_ROW_MASK_OPTION,
            mask_dimension_option=config.ATTN_MASK_DIMENSION_OPTION,
            mask_threshold=config.ATTN_MASK_THRESHOLD,
            mask_surrogate_scale=config.ATTN_MASK_SURROGATE_SCALE,
            mask_floor=config.ATTN_MASK_FLOOR,
            mask_transition_width=config.ATTN_MASK_TRANSITION_WIDTH,
            generator_stack_source=adaptive_generator_stack_source(
                config, AdaptiveRole.ATTENTION, AdaptiveParameter.MASK
            ),
        ),
        AdaptiveRole.FEED_FORWARD: lambda config: HiddenAdaptiveMaskOptions(
            option_flag=config.FF_MASK_OPTION_FLAG,
            row_mask_option=config.FF_ROW_MASK_OPTION,
            mask_dimension_option=config.FF_MASK_DIMENSION_OPTION,
            mask_threshold=config.FF_MASK_THRESHOLD,
            mask_surrogate_scale=config.FF_MASK_SURROGATE_SCALE,
            mask_floor=config.FF_MASK_FLOOR,
            mask_transition_width=config.FF_MASK_TRANSITION_WIDTH,
            generator_stack_source=adaptive_generator_stack_source(
                config, AdaptiveRole.FEED_FORWARD, AdaptiveParameter.MASK
            ),
        ),
        AdaptiveRole.ROUTER: lambda config: HiddenAdaptiveMaskOptions(
            option_flag=config.ROUTER_MASK_OPTION_FLAG,
            row_mask_option=config.ROUTER_ROW_MASK_OPTION,
            mask_dimension_option=config.ROUTER_MASK_DIMENSION_OPTION,
            mask_threshold=config.ROUTER_MASK_THRESHOLD,
            mask_surrogate_scale=config.ROUTER_MASK_SURROGATE_SCALE,
            mask_floor=config.ROUTER_MASK_FLOOR,
            mask_transition_width=config.ROUTER_MASK_TRANSITION_WIDTH,
            generator_stack_source=adaptive_generator_stack_source(
                config, AdaptiveRole.ROUTER, AdaptiveParameter.MASK
            ),
        ),
    }
)


def hidden_adaptive_weight_options(
    config: ModuleType, role: AdaptiveRole = AdaptiveRole.MAIN
) -> HiddenAdaptiveWeightOptions:
    return _WEIGHT_OPTIONS_FACTORIES[role](config)


def hidden_adaptive_bias_options(
    config: ModuleType, role: AdaptiveRole = AdaptiveRole.MAIN
) -> HiddenAdaptiveBiasOptions:
    return _BIAS_OPTIONS_FACTORIES[role](config)


def hidden_adaptive_diagonal_options(
    config: ModuleType, role: AdaptiveRole = AdaptiveRole.MAIN
) -> HiddenAdaptiveDiagonalOptions:
    return _DIAGONAL_OPTIONS_FACTORIES[role](config)


def hidden_adaptive_mask_options(
    config: ModuleType, role: AdaptiveRole = AdaptiveRole.MAIN
) -> HiddenAdaptiveMaskOptions:
    return _MASK_OPTIONS_FACTORIES[role](config)
