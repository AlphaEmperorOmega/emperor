from dataclasses import dataclass

from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import ActivationOptions
from emperor.parametric.core.mixtures.options import ClipParameterOptions


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


# Preserve historical class paths for serialization compatibility.
ParametricMixtureOptions.__module__ = (
    "models.parametric.parametric_vector._stack_config_factory"
)
ParametricRouterOptions.__module__ = (
    "models.parametric.parametric_vector._stack_config_factory"
)
ParametricSamplerOptions.__module__ = (
    "models.parametric.parametric_vector._stack_config_factory"
)
ParametricStackOptions.__module__ = (
    "models.parametric.parametric_vector._stack_config_factory"
)
