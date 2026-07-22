from dataclasses import dataclass

from emperor.layers import ActivationOptions, ResidualConnectionOptions
from emperor.parametric import ClipParameterOptions
from model_runtime.packages.runtime_values import ResolvedRuntimeOptions


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


@dataclass(frozen=True, slots=True)
class RuntimeOptions(ResolvedRuntimeOptions):
    pass
