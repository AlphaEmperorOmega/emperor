from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import ActivationOptions, LastLayerBiasOptions
from emperor.base.options import LayerNormPositionOptions
from emperor.experts.core.config import MixtureOfExpertsConfig
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.linears.core.config import LinearLayerConfig
from emperor.parametric import (
    AdaptiveRouterOptions,
    GeneratorBiasMixtureConfig,
    GeneratorWeightsMixtureConfig,
    ParametricLayerConfig,
    ParametricLayerHandlerConfig,
)
from emperor.parametric.core.mixtures.options import ClipParameterOptions
from emperor.sampler.core.config import RouterConfig, SamplerConfig
from models.parametric.parametric_generator._controller_stack import router_hidden_dim


def build_parametric_stack_config(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    activation: ActivationOptions,
    residual_connection_option: ResidualConnectionOptions,
    dropout_probability: float,
    adaptive_mixture_top_k: int,
    adaptive_mixture_num_experts: int,
    adaptive_mixture_weighted_parameters_flag: bool,
    adaptive_mixture_clip_parameter_option: ClipParameterOptions,
    adaptive_mixture_clip_range: float,
    adaptive_bias_option: type[GeneratorBiasMixtureConfig] | None,
    sampler_threshold: float,
    sampler_filter_above_threshold: bool,
    sampler_num_topk_samples: int,
    sampler_normalize_probabilities_flag: bool,
    sampler_noisy_topk_flag: bool,
    sampler_coefficient_of_variation_loss_weight: float,
    sampler_switch_loss_weight: float,
    sampler_zero_centred_loss_weight: float,
    sampler_mutual_information_loss_weight: float,
    generator_stack_num_layers: int,
    generator_stack_hidden_dim: int,
    generator_stack_activation: ActivationOptions,
    generator_stack_dropout_probability: float,
) -> LayerStackConfig:
    generator_config = build_generator_config(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=adaptive_mixture_top_k,
        num_experts=adaptive_mixture_num_experts,
        stack_hidden_dim=generator_stack_hidden_dim,
        stack_num_layers=generator_stack_num_layers,
        stack_activation=generator_stack_activation,
        stack_dropout_probability=generator_stack_dropout_probability,
    )
    weight_mixture_config = GeneratorWeightsMixtureConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=adaptive_mixture_top_k,
        num_experts=adaptive_mixture_num_experts,
        weighted_parameters_flag=adaptive_mixture_weighted_parameters_flag,
        clip_parameter_option=adaptive_mixture_clip_parameter_option,
        clip_range=adaptive_mixture_clip_range,
        generator_config=generator_config,
    )
    bias_mixture_config = build_generator_bias_config(
        adaptive_bias_option,
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=adaptive_mixture_top_k,
        num_experts=adaptive_mixture_num_experts,
        weighted_parameters_flag=adaptive_mixture_weighted_parameters_flag,
        clip_parameter_option=adaptive_mixture_clip_parameter_option,
        clip_range=adaptive_mixture_clip_range,
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
            num_experts=adaptive_mixture_num_experts,
            activation=activation,
        ),
        sampler_config=build_sampler_config(
            top_k=adaptive_mixture_top_k,
            num_experts=adaptive_mixture_num_experts,
            threshold=sampler_threshold,
            filter_above_threshold=sampler_filter_above_threshold,
            num_topk_samples=sampler_num_topk_samples,
            normalize_probabilities_flag=sampler_normalize_probabilities_flag,
            noisy_topk_flag=sampler_noisy_topk_flag,
            coefficient_of_variation_loss_weight=(
                sampler_coefficient_of_variation_loss_weight
            ),
            switch_loss_weight=sampler_switch_loss_weight,
            zero_centred_loss_weight=sampler_zero_centred_loss_weight,
            mutual_information_loss_weight=sampler_mutual_information_loss_weight,
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
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=True,
        layer_config=ParametricLayerHandlerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            residual_connection_option=residual_connection_option,
            dropout_probability=dropout_probability,
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
    top_k: int,
    num_experts: int,
    stack_hidden_dim: int,
    stack_num_layers: int,
    stack_activation: ActivationOptions,
    stack_dropout_probability: float,
) -> MixtureOfExpertsConfig:
    return MixtureOfExpertsConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=top_k,
        num_experts=num_experts,
        capacity_factor=0.0,
        dropped_token_behavior=DroppedTokenOptions.ZEROS,
        compute_expert_mixture_flag=False,
        weighted_parameters_flag=False,
        weighting_position_option=ExpertWeightingPositionOptions.BEFORE_EXPERTS,
        routing_initialization_mode=RoutingInitializationMode.DISABLED,
        sampler_config=None,
        expert_model_config=build_linear_stack_config(
            input_dim=input_dim,
            hidden_dim=stack_hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            activation=stack_activation,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=stack_dropout_probability,
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


def build_linear_stack_config(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    activation: ActivationOptions,
    residual_connection_option: ResidualConnectionOptions,
    dropout_probability: float,
    apply_output_pipeline_flag: bool,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=apply_output_pipeline_flag,
        layer_config=LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            residual_connection_option=residual_connection_option,
            dropout_probability=dropout_probability,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=True,
            ),
        ),
    )


def build_router_config(
    *,
    input_dim: int,
    num_experts: int,
    activation: ActivationOptions,
) -> RouterConfig:
    return RouterConfig(
        input_dim=input_dim,
        num_experts=num_experts,
        noisy_topk_flag=False,
        model_config=build_linear_stack_config(
            input_dim=input_dim,
            hidden_dim=router_hidden_dim(input_dim),
            output_dim=num_experts,
            num_layers=1,
            activation=activation,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            apply_output_pipeline_flag=False,
        ),
    )


def build_sampler_config(
    *,
    top_k: int,
    num_experts: int,
    threshold: float,
    filter_above_threshold: bool,
    num_topk_samples: int,
    normalize_probabilities_flag: bool,
    noisy_topk_flag: bool,
    coefficient_of_variation_loss_weight: float,
    switch_loss_weight: float,
    zero_centred_loss_weight: float,
    mutual_information_loss_weight: float,
) -> SamplerConfig:
    return SamplerConfig(
        top_k=top_k,
        threshold=threshold,
        filter_above_threshold=filter_above_threshold,
        num_topk_samples=num_topk_samples,
        normalize_probabilities_flag=normalize_probabilities_flag,
        noisy_topk_flag=noisy_topk_flag,
        num_experts=num_experts,
        coefficient_of_variation_loss_weight=coefficient_of_variation_loss_weight,
        switch_loss_weight=switch_loss_weight,
        zero_centred_loss_weight=zero_centred_loss_weight,
        mutual_information_loss_weight=mutual_information_loss_weight,
        router_config=None,
    )
