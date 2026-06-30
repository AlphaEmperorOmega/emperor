from dataclasses import dataclass

from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import ActivationOptions, LastLayerBiasOptions
from emperor.base.options import LayerNormPositionOptions
from emperor.linears.core.config import LinearLayerConfig
from emperor.parametric.core.mixtures.options import ClipParameterOptions
from emperor.sampler.core.config import RouterConfig, SamplerConfig


@dataclass(frozen=True)
class ParametricStackOptions:
    hidden_dim: int
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: ResidualConnectionOptions
    dropout_probability: float


@dataclass(frozen=True)
class ParametricMixtureOptions:
    top_k: int
    num_experts: int
    weighted_parameters_flag: bool
    clip_parameter_option: ClipParameterOptions
    clip_range: float


@dataclass(frozen=True)
class ParametricSamplerOptions:
    threshold: float
    filter_above_threshold: bool
    num_topk_samples: int
    normalize_probabilities_flag: bool
    noisy_topk_flag: bool
    coefficient_of_variation_loss_weight: float
    switch_loss_weight: float
    zero_centred_loss_weight: float
    mutual_information_loss_weight: float


@dataclass(frozen=True)
class ParametricRouterOptions:
    activation: ActivationOptions
    noisy_topk_flag: bool = False


@dataclass(frozen=True)
class ParametricGeneratorStackOptions:
    hidden_dim: int
    num_layers: int
    activation: ActivationOptions
    dropout_probability: float


def router_hidden_dim(input_dim: int) -> int:
    return max(4, min(input_dim, 32))


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
    mixture_options: ParametricMixtureOptions,
    router_options: ParametricRouterOptions,
) -> RouterConfig:
    return RouterConfig(
        input_dim=input_dim,
        num_experts=mixture_options.num_experts,
        noisy_topk_flag=router_options.noisy_topk_flag,
        model_config=build_linear_stack_config(
            input_dim=input_dim,
            hidden_dim=router_hidden_dim(input_dim),
            output_dim=mixture_options.num_experts,
            num_layers=1,
            activation=router_options.activation,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            apply_output_pipeline_flag=False,
        ),
    )


def build_sampler_config(
    *,
    mixture_options: ParametricMixtureOptions,
    sampler_options: ParametricSamplerOptions,
) -> SamplerConfig:
    return SamplerConfig(
        top_k=mixture_options.top_k,
        threshold=sampler_options.threshold,
        filter_above_threshold=sampler_options.filter_above_threshold,
        num_topk_samples=sampler_options.num_topk_samples,
        normalize_probabilities_flag=(
            sampler_options.normalize_probabilities_flag
        ),
        noisy_topk_flag=sampler_options.noisy_topk_flag,
        num_experts=mixture_options.num_experts,
        coefficient_of_variation_loss_weight=(
            sampler_options.coefficient_of_variation_loss_weight
        ),
        switch_loss_weight=sampler_options.switch_loss_weight,
        zero_centred_loss_weight=sampler_options.zero_centred_loss_weight,
        mutual_information_loss_weight=(
            sampler_options.mutual_information_loss_weight
        ),
        router_config=None,
    )
