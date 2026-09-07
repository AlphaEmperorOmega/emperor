"""Package-owned options for adaptive parameter generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import ModuleType
from typing import TYPE_CHECKING, Any

from emperor.augmentations.adaptive_parameters import LowRankFactorSourceOptions
from emperor.sampler import RouterConfig, SamplerConfig

if TYPE_CHECKING:
    from models.gpt.expert_linear_adaptive._adaptive_generator_stack_config_factory import (
        AdaptiveGeneratorStackConfigFactory,
    )
    from models.gpt.expert_linear_adaptive.runtime_options import (
        AdaptiveGeneratorStackSource,
    )


@dataclass(frozen=True)
class WeightGenerationOptions:
    input_factor_source: LowRankFactorSourceOptions | None = None
    output_factor_source: LowRankFactorSourceOptions | None = None
    mixture_num_experts: int | None = None
    mixture_top_k: int | None = None
    mixture_normalize_probabilities_flag: bool | None = None
    input_factor_generator_stack: AdaptiveGeneratorStackSource | None = None
    output_factor_generator_stack: AdaptiveGeneratorStackSource | None = None
    coefficient_generator_stack: AdaptiveGeneratorStackSource | None = None
    mixture_router_generator_stack: AdaptiveGeneratorStackSource | None = None


@dataclass(frozen=True)
class BiasGenerationOptions:
    mixture_num_experts: int | None = None
    mixture_top_k: int | None = None
    mixture_normalize_probabilities_flag: bool | None = None
    mixture_router_generator_stack: AdaptiveGeneratorStackSource | None = None


_GENERATION_PREFIXES = (
    "weight_input_factor_",
    "weight_output_factor_",
    "weight_coefficient_",
    "weight_mixture_",
    "bias_mixture_",
)
_STACK_FIELDS = (
    "independent_flag",
    "hidden_dim",
    "layer_norm_position",
    "num_layers",
    "activation",
    "residual_connection_option",
    "residual_model_flag",
    "dropout_probability",
    "last_layer_bias_option",
    "apply_output_postprocessing_flag",
    "bias_flag",
)


def generation_keys(config_module: ModuleType) -> set[str]:
    return {
        name.lower()
        for name in vars(config_module)
        if name.isupper() and any(part in name.lower() for part in _GENERATION_PREFIXES)
    }


def _generation_value(values, config_module, prefix, name):
    key = prefix + name
    if key in values:
        return values[key]
    if prefix in ("attn_", "ff_") and name in values:
        return values[name]
    return getattr(config_module, key.upper(), None)


def _stack_source(values, config_module, prefix, name):
    from models.gpt.expert_linear_adaptive.runtime_options import (
        AdaptiveGeneratorStackSource,
    )

    return AdaptiveGeneratorStackSource(
        **{
            field: _generation_value(values, config_module, prefix, name + "_" + field)
            for field in _STACK_FIELDS
        }
    )


def weight_generation_from_fields(values, config_module, prefix=""):
    scalar_fields = (
        "input_factor_source",
        "output_factor_source",
        "mixture_num_experts",
        "mixture_top_k",
        "mixture_normalize_probabilities_flag",
    )
    stack_fields = (
        "input_factor_generator_stack",
        "output_factor_generator_stack",
        "coefficient_generator_stack",
        "mixture_router_generator_stack",
    )
    return WeightGenerationOptions(
        **{
            name: _generation_value(values, config_module, prefix, "weight_" + name)
            for name in scalar_fields
        },
        **{
            name: _stack_source(values, config_module, prefix, "weight_" + name)
            for name in stack_fields
        },
    )


def bias_generation_from_fields(values, config_module, prefix=""):
    scalar_fields = (
        "mixture_num_experts",
        "mixture_top_k",
        "mixture_normalize_probabilities_flag",
    )
    return BiasGenerationOptions(
        **{
            name: _generation_value(values, config_module, prefix, "bias_" + name)
            for name in scalar_fields
        },
        mixture_router_generator_stack=_stack_source(
            values, config_module, prefix, "bias_mixture_router_generator_stack"
        ),
    )


def apply_generation_options(
    options: dict[str, Any], values, config_module
) -> dict[str, Any]:
    result = dict(options)
    for role, prefix in (
        ("hidden_adaptive", ""),
        ("attention_hidden_adaptive", "attn_"),
        ("feed_forward_hidden_adaptive", "ff_"),
        ("router_adaptive", "router_"),
    ):
        weight_key = role + "_weight_options"
        bias_key = role + "_bias_options"
        if result.get(weight_key) is not None:
            result[weight_key] = replace(
                result[weight_key],
                generation=weight_generation_from_fields(values, config_module, prefix),
            )
        if result.get(bias_key) is not None:
            result[bias_key] = replace(
                result[bias_key],
                generation=bias_generation_from_fields(values, config_module, prefix),
            )
    for role, prefix in (
        ("input_boundary_options", "input_layer_"),
        ("output_boundary_options", "output_layer_"),
    ):
        if result.get(role) is not None:
            result[role] = replace(
                result[role],
                weight_generation=weight_generation_from_fields(
                    values, config_module, prefix
                ),
                bias_generation=bias_generation_from_fields(
                    values, config_module, prefix
                ),
            )
    return result


def _generator_config(source, factory):
    if source is None:
        return None
    return factory.build_config_from_source(source)


def weight_generation_fields(
    options: WeightGenerationOptions,
    factory: AdaptiveGeneratorStackConfigFactory,
    weight_source: AdaptiveGeneratorStackSource | None = None,
) -> dict:
    factor_factory = factory
    if weight_source is not None and weight_source.independent_flag:
        updates = {
            name: getattr(weight_source, name)
            for name in _STACK_FIELDS
            if name != "independent_flag" and getattr(weight_source, name) is not None
        }
        factor_defaults = replace(factory.shared_options, **updates)
        factor_factory = type(factory)(factor_defaults)
    return {
        "input_factor_source": options.input_factor_source,
        "output_factor_source": options.output_factor_source,
        "input_factor_model_config": _generator_config(
            options.input_factor_generator_stack, factor_factory
        ),
        "output_factor_model_config": _generator_config(
            options.output_factor_generator_stack, factor_factory
        ),
        "coefficient_model_config": _generator_config(
            options.coefficient_generator_stack, factor_factory
        ),
        **mixture_generation_fields(options, factory),
    }


def mixture_generation_fields(
    options: WeightGenerationOptions | BiasGenerationOptions,
    factory: AdaptiveGeneratorStackConfigFactory,
) -> dict:
    router_model_config = _generator_config(
        options.mixture_router_generator_stack, factory
    )
    return {
        "num_experts": options.mixture_num_experts,
        "top_k": options.mixture_top_k,
        "sampler_config": SamplerConfig(
            normalize_probabilities_flag=options.mixture_normalize_probabilities_flag,
            router_config=RouterConfig(model_config=router_model_config),
        ),
    }
