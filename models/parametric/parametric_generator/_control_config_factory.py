from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.base.layer import LayerStackConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import ActivationOptions, LastLayerBiasOptions
from emperor.base.options import LayerNormPositionOptions
from emperor.experts.core.config import MixtureOfExpertsConfig
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.parametric import (
    AdaptiveRouterOptions,
    GeneratorBiasMixtureConfig,
    GeneratorWeightsMixtureConfig,
    ParametricLayerConfig,
    ParametricLayerHandlerConfig,
)
from emperor.parametric.core.mixtures.options import ClipParameterOptions
from models.parametric._shared_stack_factory import (
    ParametricGeneratorStackOptions,
    ParametricMixtureOptions,
    ParametricRouterOptions,
    ParametricSamplerOptions,
    ParametricStackOptions,
    build_linear_stack_config,
    build_router_config,
    build_sampler_config,
)


def build_parametric_stack_config(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    stack_options: ParametricStackOptions,
    mixture_options: ParametricMixtureOptions,
    sampler_options: ParametricSamplerOptions,
    router_options: ParametricRouterOptions,
    adaptive_bias_option: type[GeneratorBiasMixtureConfig] | None,
    generator_stack_options: ParametricGeneratorStackOptions,
) -> LayerStackConfig:
    generator_config = build_generator_config(
        input_dim=input_dim,
        output_dim=output_dim,
        mixture_options=mixture_options,
        generator_stack_options=generator_stack_options,
    )
    weight_mixture_config = GeneratorWeightsMixtureConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=mixture_options.top_k,
        num_experts=mixture_options.num_experts,
        weighted_parameters_flag=mixture_options.weighted_parameters_flag,
        clip_parameter_option=mixture_options.clip_parameter_option,
        clip_range=mixture_options.clip_range,
        generator_config=generator_config,
    )
    bias_mixture_config = build_generator_bias_config(
        adaptive_bias_option,
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=mixture_options.top_k,
        num_experts=mixture_options.num_experts,
        weighted_parameters_flag=mixture_options.weighted_parameters_flag,
        clip_parameter_option=mixture_options.clip_parameter_option,
        clip_range=mixture_options.clip_range,
        generator_config=generator_config,
    )
    parametric_layer_config = ParametricLayerConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        weight_mixture_config=weight_mixture_config,
        bias_mixture_config=bias_mixture_config,
        routing_initialization_mode=AdaptiveRouterOptions.SHARED_ROUTER,
        router_config=build_router_config(
            input_dim=input_dim,
            mixture_options=mixture_options,
            router_options=router_options,
        ),
        sampler_config=build_sampler_config(
            mixture_options=mixture_options,
            sampler_options=sampler_options,
        ),
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            weight_config=None,
            diagonal_config=None,
            bias_config=None,
            mask_config=None,
            model_config=None,
        ),
    )
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=stack_options.num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=True,
        layer_config=ParametricLayerHandlerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=stack_options.activation,
            residual_connection_option=stack_options.residual_connection_option,
            dropout_probability=stack_options.dropout_probability,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=parametric_layer_config,
        ),
    )


def build_generator_config(
    *,
    input_dim: int,
    output_dim: int,
    mixture_options: ParametricMixtureOptions,
    generator_stack_options: ParametricGeneratorStackOptions,
) -> MixtureOfExpertsConfig:
    return MixtureOfExpertsConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=mixture_options.top_k,
        num_experts=mixture_options.num_experts,
        capacity_factor=0.0,
        dropped_token_behavior=DroppedTokenOptions.ZEROS,
        compute_expert_mixture_flag=False,
        weighted_parameters_flag=False,
        weighting_position_option=ExpertWeightingPositionOptions.BEFORE_EXPERTS,
        routing_initialization_mode=RoutingInitializationMode.DISABLED,
        sampler_config=None,
        expert_model_config=build_linear_stack_config(
            input_dim=input_dim,
            hidden_dim=generator_stack_options.hidden_dim,
            output_dim=output_dim,
            num_layers=generator_stack_options.num_layers,
            activation=generator_stack_options.activation,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=generator_stack_options.dropout_probability,
            apply_output_pipeline_flag=False,
        ),
    )


def build_generator_bias_config(
    bias_config_cls: type[GeneratorBiasMixtureConfig] | None,
    **mixture_kwargs,
) -> GeneratorBiasMixtureConfig | None:
    if bias_config_cls is None:
        return None
    return bias_config_cls(**mixture_kwargs)
