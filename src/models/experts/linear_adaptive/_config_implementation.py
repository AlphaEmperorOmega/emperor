from dataclasses import dataclass, field, replace
from typing import cast

import models.experts.linear_adaptive.config as config
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from models.experts.linear_adaptive._adaptive_defaults import (
    AdaptiveDefaults,
    AdaptiveDefaultValues,
    resolve_adaptive_defaults,
)
from models.experts.linear_adaptive._control_defaults import (
    ControlDefaults,
    ControlDefaultValues,
    resolve_control_defaults,
)
from models.experts.linear_adaptive._grouping import GroupingDefaultValues
from models.experts.linear_adaptive.runtime_options import (
    ExpertsMixtureOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
    resolve_experts_submodule_stack_options,
)


@dataclass(frozen=True, kw_only=True)
class CoreDefaultValues:
    batch_size: int = config.BATCH_SIZE
    learning_rate: float = config.LEARNING_RATE
    input_dim: int = config.INPUT_DIM
    hidden_dim: int = config.HIDDEN_DIM
    output_dim: int = config.OUTPUT_DIM
    stack_bias_flag: bool = config.STACK_BIAS_FLAG
    layer_norm_position: LayerNormPositionOptions = config.LAYER_NORM_POSITION
    normalization: NormalizationOptions = field(
        default=config.NORMALIZATION, kw_only=True
    )
    stack_num_layers: int = config.STACK_NUM_LAYERS
    stack_activation: ActivationOptions = config.STACK_ACTIVATION
    stack_residual_connection_option: type[ResidualConfig] | None = (
        config.STACK_RESIDUAL_CONNECTION_OPTION
    )
    stack_residual_model_flag: bool = config.STACK_RESIDUAL_MODEL_FLAG
    stack_residual_block_size: int | None = config.STACK_RESIDUAL_BLOCK_SIZE
    stack_residual_rms_norm_epsilon: float | None = (
        config.STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY
    stack_last_layer_bias_option: LastLayerBiasOptions = (
        config.STACK_LAST_LAYER_BIAS_OPTION
    )
    stack_apply_output_postprocessing_flag: bool = (
        config.STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    submodule_stack_hidden_dim: int = config.SUBMODULE_STACK_HIDDEN_DIM
    submodule_stack_num_layers: int = config.SUBMODULE_STACK_NUM_LAYERS
    submodule_stack_activation: ActivationOptions = config.SUBMODULE_STACK_ACTIVATION
    submodule_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
    )
    submodule_stack_residual_model_flag: bool = (
        config.SUBMODULE_STACK_RESIDUAL_MODEL_FLAG
    )
    submodule_stack_residual_block_size: int | None = (
        config.SUBMODULE_STACK_RESIDUAL_BLOCK_SIZE
    )
    submodule_stack_residual_rms_norm_epsilon: float | None = (
        config.SUBMODULE_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    submodule_stack_dropout_probability: float = (
        config.SUBMODULE_STACK_DROPOUT_PROBABILITY
    )
    submodule_stack_layer_norm_position: LayerNormPositionOptions = (
        config.SUBMODULE_STACK_LAYER_NORM_POSITION
    )
    submodule_stack_normalization: NormalizationOptions = field(
        default=(config.SUBMODULE_STACK_NORMALIZATION), kw_only=True
    )
    submodule_stack_last_layer_bias_option: LastLayerBiasOptions = (
        config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION
    )
    submodule_stack_apply_output_postprocessing_flag: bool = (
        config.SUBMODULE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    submodule_stack_bias_flag: bool = config.SUBMODULE_STACK_BIAS_FLAG
    top_k: int = config.TOP_K
    num_experts: int = config.NUM_EXPERTS
    capacity_factor: float = config.CAPACITY_FACTOR
    dropped_token_behavior: DroppedTokenOptions = config.DROPPED_TOKEN_BEHAVIOR
    compute_expert_mixture_flag: bool = config.COMPUTE_EXPERT_MIXTURE_FLAG
    weighted_parameters_flag: bool = config.WEIGHTED_PARAMETERS_FLAG
    weighting_position_option: ExpertWeightingPositionOptions = (
        config.WEIGHTING_POSITION_OPTION
    )
    routing_initialization_mode: RoutingInitializationMode = (
        config.ROUTING_INITIALIZATION_MODE
    )
    expert_stack_hidden_dim: int | None = None
    expert_stack_num_layers: int | None = None
    expert_stack_activation: ActivationOptions | None = None
    expert_stack_residual_connection_option: type[ResidualConfig] | None = None
    expert_stack_residual_model_flag: bool = False
    expert_stack_residual_block_size: int | None = field(default=None, kw_only=True)
    expert_stack_residual_rms_norm_epsilon: float | None = field(
        default=None, kw_only=True
    )
    expert_stack_dropout_probability: float | None = None
    expert_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.EXPERT_STACK_LAYER_NORM_POSITION
    )
    expert_stack_normalization: NormalizationOptions | None = field(
        default=(config.EXPERT_STACK_NORMALIZATION), kw_only=True
    )
    expert_stack_last_layer_bias_option: LastLayerBiasOptions | None = None
    expert_stack_apply_output_postprocessing_flag: bool | None = (
        config.EXPERT_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    expert_bias_flag: bool | None = None
    sampler_threshold: float = config.SAMPLER_THRESHOLD
    sampler_filter_above_threshold: bool = config.SAMPLER_FILTER_ABOVE_THRESHOLD
    sampler_num_topk_samples: int = config.SAMPLER_NUM_TOPK_SAMPLES
    sampler_normalize_probabilities_flag: bool = (
        config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG
    )
    sampler_noisy_topk_flag: bool = config.SAMPLER_NOISY_TOPK_FLAG
    sampler_coefficient_of_variation_loss_weight: float = (
        config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT
    )
    sampler_switch_loss_weight: float = config.SAMPLER_SWITCH_LOSS_WEIGHT
    sampler_zero_centred_loss_weight: float = config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT
    sampler_mutual_information_loss_weight: float = (
        config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT
    )
    stack_options: ExpertsStackOptions | None = None
    submodule_stack_options: ExpertsSubmoduleStackOptions | None = None
    mixture_options: ExpertsMixtureOptions | None = None
    expert_stack_options: ExpertsSubmoduleStackOptions | None = None
    sampler_options: ExpertsSamplerOptions | None = None


@dataclass(frozen=True, kw_only=True)
class _RuntimeDefaultValues(
    GroupingDefaultValues,
    ControlDefaultValues,
    AdaptiveDefaultValues,
    CoreDefaultValues,
):
    pass


@dataclass(frozen=True, slots=True)
class CoreDefaults:
    batch_size: int
    learning_rate: float
    input_dim: int
    output_dim: int
    stack_options: ExpertsStackOptions
    submodule_stack_options: ExpertsSubmoduleStackOptions
    mixture_options: ExpertsMixtureOptions
    expert_stack_options: ExpertsSubmoduleStackOptions
    sampler_options: ExpertsSamplerOptions


def resolve_core_defaults(values: CoreDefaultValues) -> CoreDefaults:
    stack_options = cast(ExpertsStackOptions | None, values.stack_options)
    submodule_stack_options = cast(
        ExpertsSubmoduleStackOptions | None, values.submodule_stack_options
    )
    stack_options = stack_options or ExpertsStackOptions(
        hidden_dim=values.hidden_dim,
        bias_flag=values.stack_bias_flag,
        layer_norm_position=values.layer_norm_position,
        normalization=values.normalization,
        num_layers=values.stack_num_layers,
        activation=values.stack_activation,
        residual_connection_option=values.stack_residual_connection_option,
        residual_block_size=values.stack_residual_block_size,
        residual_rms_norm_epsilon=values.stack_residual_rms_norm_epsilon,
        residual_model_flag=values.stack_residual_model_flag,
        dropout_probability=values.stack_dropout_probability,
        last_layer_bias_option=values.stack_last_layer_bias_option,
        apply_output_postprocessing_flag=values.stack_apply_output_postprocessing_flag,
    )
    submodule_stack_options = submodule_stack_options or ExpertsSubmoduleStackOptions(
        hidden_dim=values.submodule_stack_hidden_dim,
        num_layers=values.submodule_stack_num_layers,
        last_layer_bias_option=values.submodule_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=values.submodule_stack_apply_output_postprocessing_flag,
        activation=values.submodule_stack_activation,
        layer_norm_position=values.submodule_stack_layer_norm_position,
        normalization=values.submodule_stack_normalization,
        residual_connection_option=(values.submodule_stack_residual_connection_option),
        residual_block_size=values.submodule_stack_residual_block_size,
        residual_rms_norm_epsilon=values.submodule_stack_residual_rms_norm_epsilon,
        residual_model_flag=values.submodule_stack_residual_model_flag,
        dropout_probability=values.submodule_stack_dropout_probability,
        bias_flag=values.submodule_stack_bias_flag,
    )
    mixture_options = cast(ExpertsMixtureOptions | None, values.mixture_options)
    expert_stack_options = cast(
        ExpertsSubmoduleStackOptions | None, values.expert_stack_options
    )
    sampler_options = cast(ExpertsSamplerOptions | None, values.sampler_options)
    mixture_options = mixture_options or ExpertsMixtureOptions(
        top_k=values.top_k,
        num_experts=values.num_experts,
        capacity_factor=values.capacity_factor,
        dropped_token_behavior=values.dropped_token_behavior,
        compute_expert_mixture_flag=values.compute_expert_mixture_flag,
        weighted_parameters_flag=values.weighted_parameters_flag,
        weighting_position_option=values.weighting_position_option,
        routing_initialization_mode=values.routing_initialization_mode,
    )
    expert_stack_options = (
        expert_stack_options
        or resolve_experts_submodule_stack_options(
            submodule_stack_options,
            hidden_dim=values.expert_stack_hidden_dim,
            num_layers=values.expert_stack_num_layers,
            last_layer_bias_option=values.expert_stack_last_layer_bias_option,
            apply_output_postprocessing_flag=values.expert_stack_apply_output_postprocessing_flag,
            activation=values.expert_stack_activation,
            layer_norm_position=values.expert_stack_layer_norm_position,
            normalization=values.expert_stack_normalization,
            residual_connection_option=values.expert_stack_residual_connection_option,
            residual_block_size=values.expert_stack_residual_block_size,
            residual_rms_norm_epsilon=values.expert_stack_residual_rms_norm_epsilon,
            residual_model_flag=values.expert_stack_residual_model_flag,
            dropout_probability=values.expert_stack_dropout_probability,
            bias_flag=values.expert_bias_flag,
        )
    )
    sampler_options = sampler_options or ExpertsSamplerOptions(
        threshold=values.sampler_threshold,
        filter_above_threshold=values.sampler_filter_above_threshold,
        num_topk_samples=values.sampler_num_topk_samples,
        normalize_probabilities_flag=values.sampler_normalize_probabilities_flag,
        noisy_topk_flag=values.sampler_noisy_topk_flag,
        coefficient_of_variation_loss_weight=(
            values.sampler_coefficient_of_variation_loss_weight
        ),
        switch_loss_weight=values.sampler_switch_loss_weight,
        zero_centred_loss_weight=values.sampler_zero_centred_loss_weight,
        mutual_information_loss_weight=values.sampler_mutual_information_loss_weight,
    )
    return CoreDefaults(
        batch_size=values.batch_size,
        learning_rate=values.learning_rate,
        input_dim=values.input_dim,
        output_dim=values.output_dim,
        stack_options=stack_options,
        submodule_stack_options=submodule_stack_options,
        mixture_options=mixture_options,
        expert_stack_options=expert_stack_options,
        sampler_options=sampler_options,
    )


@dataclass(frozen=True, slots=True)
class _RuntimeDefaultsResolver:
    core: CoreDefaults
    control: ControlDefaults
    adaptive: AdaptiveDefaults


def resolve_runtime_defaults(values: _RuntimeDefaultValues) -> _RuntimeDefaultsResolver:
    core = resolve_core_defaults(values)
    control = resolve_control_defaults(values, core.submodule_stack_options)
    adaptive = resolve_adaptive_defaults(values)
    residual_stack_options = control.residual_stack_options
    core = replace(
        core,
        stack_options=replace(
            core.stack_options, residual_stack_options=residual_stack_options
        ),
        submodule_stack_options=replace(
            core.submodule_stack_options,
            residual_stack_options=residual_stack_options,
        ),
        expert_stack_options=replace(
            core.expert_stack_options,
            residual_stack_options=residual_stack_options,
        ),
    )
    control = replace(
        control,
        router_stack_options=replace(
            control.router_stack_options,
            residual_stack_options=residual_stack_options,
        ),
    )
    adaptive = replace(
        adaptive,
        adaptive_generator_stack_options=replace(
            adaptive.adaptive_generator_stack_options,
            residual_stack_options=residual_stack_options,
        ),
    )
    return _RuntimeDefaultsResolver(core=core, control=control, adaptive=adaptive)
