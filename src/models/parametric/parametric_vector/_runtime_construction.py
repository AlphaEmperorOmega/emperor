from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

from models.parametric.parametric_vector import config
from models.parametric.parametric_vector._residual import (
    ResidualStackOptions,
    ResidualStackSource,
    resolve_residual_stack_options,
)
from models.parametric.parametric_vector.runtime_options import (
    ParametricMixtureOptions,
    ParametricRouterOptions,
    ParametricSamplerOptions,
    ParametricStackOptions,
)

if TYPE_CHECKING:
    from models.parametric.parametric_vector.runtime_options import RuntimeOptions

_PACKAGE_NAME = "models.parametric.parametric_vector"
_ValueT = TypeVar("_ValueT")


@dataclass(frozen=True, slots=True)
class ParametricVectorConstructionOptions:
    """Cohesive, fully resolved inputs to vector model construction."""

    batch_size: int
    learning_rate: float
    input_dim: int
    output_dim: int
    stack: ParametricStackOptions
    residual_stack: ResidualStackOptions
    mixture: ParametricMixtureOptions
    sampler: ParametricSamplerOptions
    router: ParametricRouterOptions

    @property
    def hidden_dim(self) -> int:
        return self.stack.hidden_dim


def _pop(values: dict[str, object], key: str, default: _ValueT) -> _ValueT:
    return cast(_ValueT, values.pop(key, default))


def _positive_integer(key: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{_PACKAGE_NAME}: {key!r} must be positive; got {value!r}")


def _positive_number(key: str, value: int | float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{_PACKAGE_NAME}: {key!r} must be positive; got {value!r}")


def _probability(
    key: str,
    value: float,
    *,
    include_one: bool,
) -> None:
    upper_bound_is_valid = value <= 1.0 if include_one else value < 1.0
    if not math.isfinite(value) or value < 0.0 or not upper_bound_is_valid:
        interval = "[0.0, 1.0]" if include_one else "[0.0, 1.0)"
        raise ValueError(
            f"{_PACKAGE_NAME}: {key!r} must be in {interval}; got {value!r}"
        )


def _non_negative(key: str, value: int | float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"{_PACKAGE_NAME}: {key!r} must be finite and non-negative; got {value!r}"
        )


def _non_negative_integer(key: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(
            f"{_PACKAGE_NAME}: runtime key {key!r} has type "
            f"{type(value).__name__}; expected int"
        )
    if value < 0:
        raise ValueError(
            f"{_PACKAGE_NAME}: {key!r} must be non-negative; got {value!r}"
        )


def _validate(options: ParametricVectorConstructionOptions) -> None:
    _positive_integer("batch_size", options.batch_size)
    _positive_number("learning_rate", options.learning_rate)
    _positive_integer("input_dim", options.input_dim)
    _positive_integer("hidden_dim", options.hidden_dim)
    _positive_integer("output_dim", options.output_dim)
    _positive_integer("stack_num_layers", options.stack.num_layers)
    _probability(
        "stack_dropout_probability",
        options.stack.dropout_probability,
        include_one=False,
    )
    _positive_integer("residual_stack_hidden_dim", options.residual_stack.hidden_dim)
    _positive_integer("residual_stack_num_layers", options.residual_stack.num_layers)
    _probability(
        "residual_stack_dropout_probability",
        options.residual_stack.dropout_probability,
        include_one=False,
    )
    _positive_integer("adaptive_mixture_top_k", options.mixture.top_k)
    _positive_integer("adaptive_mixture_num_experts", options.mixture.num_experts)
    if options.mixture.top_k > options.mixture.num_experts:
        raise ValueError(
            f"{_PACKAGE_NAME}: adaptive mixture top_k cannot exceed num_experts; "
            f"got {options.mixture.top_k} and {options.mixture.num_experts}"
        )
    _non_negative("adaptive_mixture_clip_range", options.mixture.clip_range)
    _probability(
        "sampler_threshold",
        options.sampler.threshold,
        include_one=True,
    )
    _non_negative_integer(
        "sampler_num_topk_samples",
        options.sampler.num_topk_samples,
    )
    if options.sampler.num_topk_samples > options.mixture.top_k:
        raise ValueError(
            f"{_PACKAGE_NAME}: sampler num_topk_samples cannot exceed top_k; "
            f"got {options.sampler.num_topk_samples} and {options.mixture.top_k}"
        )
    for key, value in (
        (
            "sampler_coefficient_of_variation_loss_weight",
            options.sampler.coefficient_of_variation_loss_weight,
        ),
        ("sampler_switch_loss_weight", options.sampler.switch_loss_weight),
        (
            "sampler_zero_centred_loss_weight",
            options.sampler.zero_centred_loss_weight,
        ),
        (
            "sampler_mutual_information_loss_weight",
            options.sampler.mutual_information_loss_weight,
        ),
    ):
        _non_negative(key, value)


def resolve_runtime_construction(
    runtime: RuntimeOptions,
) -> ParametricVectorConstructionOptions:
    """Translate flat Runtime Defaults once into package-owned role objects."""

    values = runtime._as_construction_kwargs()
    supplied_stack = _pop(values, "stack_options", None)
    supplied_residual_stack = _pop(values, "residual_stack_options", None)
    supplied_mixture = _pop(values, "mixture_options", None)
    supplied_sampler = _pop(values, "sampler_options", None)
    supplied_router = _pop(values, "router_options", None)

    batch_size = _pop(values, "batch_size", config.BATCH_SIZE)
    learning_rate = _pop(values, "learning_rate", config.LEARNING_RATE)
    input_dim = _pop(values, "input_dim", config.INPUT_DIM)
    hidden_dim = _pop(values, "hidden_dim", config.HIDDEN_DIM)
    output_dim = _pop(values, "output_dim", config.OUTPUT_DIM)
    stack = supplied_stack or ParametricStackOptions(
        hidden_dim=hidden_dim,
        num_layers=_pop(values, "stack_num_layers", config.STACK_NUM_LAYERS),
        activation=_pop(values, "stack_activation", config.STACK_ACTIVATION),
        residual_connection_option=_pop(
            values,
            "stack_residual_connection_option",
            config.STACK_RESIDUAL_CONNECTION_OPTION,
        ),
        residual_model_flag=_pop(
            values,
            "stack_residual_model_flag",
            config.STACK_RESIDUAL_MODEL_FLAG,
        ),
        dropout_probability=_pop(
            values,
            "stack_dropout_probability",
            config.STACK_DROPOUT_PROBABILITY,
        ),
    )
    residual_source = ResidualStackSource(
        independent_flag=_pop(
            values,
            "residual_stack_independent_flag",
            config.RESIDUAL_STACK_INDEPENDENT_FLAG,
        ),
        hidden_dim=_pop(
            values,
            "residual_stack_hidden_dim",
            config.RESIDUAL_STACK_HIDDEN_DIM,
        ),
        layer_norm_position=_pop(
            values,
            "residual_stack_layer_norm_position",
            config.RESIDUAL_STACK_LAYER_NORM_POSITION,
        ),
        num_layers=_pop(
            values,
            "residual_stack_num_layers",
            config.RESIDUAL_STACK_NUM_LAYERS,
        ),
        activation=_pop(
            values,
            "residual_stack_activation",
            config.RESIDUAL_STACK_ACTIVATION,
        ),
        residual_connection_option=_pop(
            values,
            "residual_stack_residual_connection_option",
            config.RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION,
        ),
        residual_model_flag=_pop(
            values,
            "residual_stack_residual_model_flag",
            config.RESIDUAL_STACK_RESIDUAL_MODEL_FLAG,
        ),
        dropout_probability=_pop(
            values,
            "residual_stack_dropout_probability",
            config.RESIDUAL_STACK_DROPOUT_PROBABILITY,
        ),
        last_layer_bias_option=_pop(
            values,
            "residual_stack_last_layer_bias_option",
            config.RESIDUAL_STACK_LAST_LAYER_BIAS_OPTION,
        ),
        apply_output_pipeline_flag=_pop(
            values,
            "residual_stack_apply_output_pipeline_flag",
            config.RESIDUAL_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        ),
        bias_flag=_pop(
            values,
            "residual_stack_bias_flag",
            config.RESIDUAL_STACK_BIAS_FLAG,
        ),
    )
    residual_stack = supplied_residual_stack or resolve_residual_stack_options(
        residual_source,
        stack,
    )
    mixture = supplied_mixture or ParametricMixtureOptions(
        top_k=_pop(
            values,
            "adaptive_mixture_top_k",
            config.ADAPTIVE_MIXTURE_TOP_K,
        ),
        num_experts=_pop(
            values,
            "adaptive_mixture_num_experts",
            config.ADAPTIVE_MIXTURE_NUM_EXPERTS,
        ),
        weighted_parameters_flag=_pop(
            values,
            "adaptive_mixture_weighted_parameters_flag",
            config.ADAPTIVE_MIXTURE_WEIGHTED_PARAMETERS_FLAG,
        ),
        clip_parameter_option=_pop(
            values,
            "adaptive_mixture_clip_parameter_option",
            config.ADAPTIVE_MIXTURE_CLIP_PARAMETER_OPTION,
        ),
        clip_range=_pop(
            values,
            "adaptive_mixture_clip_range",
            config.ADAPTIVE_MIXTURE_CLIP_RANGE,
        ),
    )
    sampler = supplied_sampler or ParametricSamplerOptions(
        threshold=_pop(values, "sampler_threshold", config.SAMPLER_THRESHOLD),
        filter_above_threshold=_pop(
            values,
            "sampler_filter_above_threshold",
            config.SAMPLER_FILTER_ABOVE_THRESHOLD,
        ),
        num_topk_samples=_pop(
            values,
            "sampler_num_topk_samples",
            config.SAMPLER_NUM_TOPK_SAMPLES,
        ),
        normalize_probabilities_flag=_pop(
            values,
            "sampler_normalize_probabilities_flag",
            config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
        ),
        noisy_topk_flag=_pop(
            values,
            "sampler_noisy_topk_flag",
            config.SAMPLER_NOISY_TOPK_FLAG,
        ),
        coefficient_of_variation_loss_weight=_pop(
            values,
            "sampler_coefficient_of_variation_loss_weight",
            config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
        ),
        switch_loss_weight=_pop(
            values,
            "sampler_switch_loss_weight",
            config.SAMPLER_SWITCH_LOSS_WEIGHT,
        ),
        zero_centred_loss_weight=_pop(
            values,
            "sampler_zero_centred_loss_weight",
            config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT,
        ),
        mutual_information_loss_weight=_pop(
            values,
            "sampler_mutual_information_loss_weight",
            config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT,
        ),
    )
    router = supplied_router or ParametricRouterOptions(activation=stack.activation)
    if values:
        name = sorted(values)[0]
        raise TypeError(
            "ParametricVectorConfigBuilder.__init__() got an unexpected keyword "
            f"argument {name!r}"
        )
    options = ParametricVectorConstructionOptions(
        batch_size=batch_size,
        learning_rate=learning_rate,
        input_dim=input_dim,
        output_dim=output_dim,
        stack=stack,
        residual_stack=residual_stack,
        mixture=mixture,
        sampler=sampler,
        router=router,
    )
    _validate(options)
    return options


__all__ = ["ParametricVectorConstructionOptions", "resolve_runtime_construction"]
