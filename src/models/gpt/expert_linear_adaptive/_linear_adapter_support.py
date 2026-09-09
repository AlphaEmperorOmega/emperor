# ruff: noqa: E501

from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from typing import Any

from models.gpt.expert_linear_adaptive import _config_defaults as config_defaults
from models.gpt.expert_linear_adaptive._flat_updates import (
    pop_updates as _pop_updates,
)
from models.gpt.expert_linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
)

_ADAPTIVE_GENERATOR_SOURCE_FIELD_MAP = {
    "independent_flag": "independent_flag",
    "hidden_dim": "hidden_dim",
    "layer_norm_position": "layer_norm_position",
    "normalization": "normalization",
    "num_layers": "num_layers",
    "activation": "activation",
    "residual_connection_option": "residual_connection_option",
    "residual_model_flag": "residual_model_flag",
    "dropout_probability": "dropout_probability",
    "last_layer_bias_option": "last_layer_bias_option",
    "apply_output_postprocessing_flag": "apply_output_postprocessing_flag",
    "bias_flag": "bias_flag",
}
_HIDDEN_ADAPTIVE_WEIGHT_FIELD_MAP = {
    "generator_depth": "generator_depth",
    "weight_option_flag": "option_flag",
    "weight_option": "option",
    "weight_normalization_option": "normalization_option",
    "weight_normalization_position_option": "normalization_position_option",
    "weight_decay_schedule": "decay_schedule",
    "weight_decay_rate": "decay_rate",
    "weight_decay_warmup_batches": "decay_warmup_batches",
    "weight_bank_expansion_factor": "bank_expansion_factor",
}
_HIDDEN_ADAPTIVE_BIAS_FIELD_MAP = {
    "bias_option_flag": "option_flag",
    "bias_option": "option",
    "bias_decay_schedule": "decay_schedule",
    "bias_decay_rate": "decay_rate",
    "bias_decay_warmup_batches": "decay_warmup_batches",
    "bias_bank_expansion_factor": "bank_expansion_factor",
}
_HIDDEN_ADAPTIVE_DIAGONAL_FIELD_MAP = {
    "diagonal_option_flag": "option_flag",
    "diagonal_option": "option",
}
_HIDDEN_ADAPTIVE_MASK_FIELD_MAP = {
    "mask_option_flag": "option_flag",
    "row_mask_option": "row_mask_option",
    "mask_dimension_option": "mask_dimension_option",
    "mask_threshold": "mask_threshold",
    "mask_surrogate_scale": "mask_surrogate_scale",
    "mask_floor": "mask_floor",
    "mask_transition_width": "mask_transition_width",
}


def _adaptive_generator_stack_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: AdaptiveGeneratorStackOptions | None,
) -> AdaptiveGeneratorStackOptions:
    options = provided or config_defaults.adaptive_generator_stack_options(
        config_module
    )
    updates = _pop_updates(
        kwargs,
        {
            "adaptive_generator_stack_hidden_dim": "hidden_dim",
            "adaptive_generator_stack_layer_norm_position": "layer_norm_position",
            "adaptive_generator_stack_normalization": "normalization",
            "adaptive_generator_stack_num_layers": "num_layers",
            "adaptive_generator_stack_activation": "activation",
            "adaptive_generator_stack_residual_connection_option": "residual_connection_option",
            "adaptive_generator_stack_residual_model_flag": "residual_model_flag",
            "adaptive_generator_stack_dropout_probability": "dropout_probability",
            "adaptive_generator_stack_last_layer_bias_option": "last_layer_bias_option",
            "adaptive_generator_stack_apply_output_postprocessing_flag": "apply_output_postprocessing_flag",
            "adaptive_generator_stack_bias_flag": "bias_flag",
        },
    )
    return replace(options, **updates) if updates else options


def _adaptive_generator_stack_source_from_kwargs(
    kwargs: dict[str, Any],
    prefix: str,
    *,
    source: AdaptiveGeneratorStackSource,
) -> AdaptiveGeneratorStackSource:
    updates = _pop_updates(
        kwargs,
        {
            f"{prefix}_{flat_field}": dataclass_field
            for flat_field, dataclass_field in _ADAPTIVE_GENERATOR_SOURCE_FIELD_MAP.items()
        },
    )
    return replace(source, **updates) if updates else source


def _hidden_adaptive_weight_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveWeightOptions | None,
) -> HiddenAdaptiveWeightOptions:
    return _adaptive_weight_options_from_kwargs(
        kwargs,
        options=provided
        or config_defaults.hidden_adaptive_weight_options(config_module),
        flat_prefix="",
        stack_prefix="weight_generator_stack",
    )


def _router_adaptive_weight_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveWeightOptions | None,
) -> HiddenAdaptiveWeightOptions:
    return _adaptive_weight_options_from_kwargs(
        kwargs,
        options=provided
        or config_defaults.router_adaptive_weight_options(config_module),
        flat_prefix="router_",
        stack_prefix="router_weight_generator_stack",
    )


def _adaptive_weight_options_from_kwargs(
    kwargs: dict[str, Any],
    *,
    options: HiddenAdaptiveWeightOptions,
    flat_prefix: str,
    stack_prefix: str,
) -> HiddenAdaptiveWeightOptions:
    updates = _pop_updates(
        kwargs, _prefixed_field_map(_HIDDEN_ADAPTIVE_WEIGHT_FIELD_MAP, flat_prefix)
    )
    updates["generator_stack_source"] = _adaptive_generator_stack_source_from_kwargs(
        kwargs,
        stack_prefix,
        source=kwargs.pop(
            f"{flat_prefix}weight_generator_stack_source",
            options.generator_stack_source,
        ),
    )
    return replace(options, **updates)


def _hidden_adaptive_bias_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveBiasOptions | None,
) -> HiddenAdaptiveBiasOptions:
    return _adaptive_bias_options_from_kwargs(
        kwargs,
        options=provided or config_defaults.hidden_adaptive_bias_options(config_module),
        flat_prefix="",
        stack_prefix="bias_generator_stack",
    )


def _router_adaptive_bias_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveBiasOptions | None,
) -> HiddenAdaptiveBiasOptions:
    return _adaptive_bias_options_from_kwargs(
        kwargs,
        options=provided or config_defaults.router_adaptive_bias_options(config_module),
        flat_prefix="router_",
        stack_prefix="router_bias_generator_stack",
    )


def _adaptive_bias_options_from_kwargs(
    kwargs: dict[str, Any],
    *,
    options: HiddenAdaptiveBiasOptions,
    flat_prefix: str,
    stack_prefix: str,
) -> HiddenAdaptiveBiasOptions:
    updates = _pop_updates(
        kwargs, _prefixed_field_map(_HIDDEN_ADAPTIVE_BIAS_FIELD_MAP, flat_prefix)
    )
    updates["generator_stack_source"] = _adaptive_generator_stack_source_from_kwargs(
        kwargs,
        stack_prefix,
        source=kwargs.pop(
            f"{flat_prefix}bias_generator_stack_source", options.generator_stack_source
        ),
    )
    return replace(options, **updates)


def _hidden_adaptive_diagonal_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveDiagonalOptions | None,
) -> HiddenAdaptiveDiagonalOptions:
    return _adaptive_diagonal_options_from_kwargs(
        kwargs,
        options=provided
        or config_defaults.hidden_adaptive_diagonal_options(config_module),
        flat_prefix="",
        stack_prefix="diagonal_generator_stack",
    )


def _router_adaptive_diagonal_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveDiagonalOptions | None,
) -> HiddenAdaptiveDiagonalOptions:
    return _adaptive_diagonal_options_from_kwargs(
        kwargs,
        options=provided
        or config_defaults.router_adaptive_diagonal_options(config_module),
        flat_prefix="router_",
        stack_prefix="router_diagonal_generator_stack",
    )


def _adaptive_diagonal_options_from_kwargs(
    kwargs: dict[str, Any],
    *,
    options: HiddenAdaptiveDiagonalOptions,
    flat_prefix: str,
    stack_prefix: str,
) -> HiddenAdaptiveDiagonalOptions:
    updates = _pop_updates(
        kwargs, _prefixed_field_map(_HIDDEN_ADAPTIVE_DIAGONAL_FIELD_MAP, flat_prefix)
    )
    updates["generator_stack_source"] = _adaptive_generator_stack_source_from_kwargs(
        kwargs,
        stack_prefix,
        source=kwargs.pop(
            f"{flat_prefix}diagonal_generator_stack_source",
            options.generator_stack_source,
        ),
    )
    return replace(options, **updates)


def _hidden_adaptive_mask_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveMaskOptions | None,
) -> HiddenAdaptiveMaskOptions:
    return _adaptive_mask_options_from_kwargs(
        kwargs,
        options=provided or config_defaults.hidden_adaptive_mask_options(config_module),
        flat_prefix="",
        stack_prefix="mask_generator_stack",
    )


def _router_adaptive_mask_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveMaskOptions | None,
) -> HiddenAdaptiveMaskOptions:
    return _adaptive_mask_options_from_kwargs(
        kwargs,
        options=provided or config_defaults.router_adaptive_mask_options(config_module),
        flat_prefix="router_",
        stack_prefix="router_mask_generator_stack",
    )


def _adaptive_mask_options_from_kwargs(
    kwargs: dict[str, Any],
    *,
    options: HiddenAdaptiveMaskOptions,
    flat_prefix: str,
    stack_prefix: str,
) -> HiddenAdaptiveMaskOptions:
    updates = _pop_updates(
        kwargs, _prefixed_field_map(_HIDDEN_ADAPTIVE_MASK_FIELD_MAP, flat_prefix)
    )
    updates["generator_stack_source"] = _adaptive_generator_stack_source_from_kwargs(
        kwargs,
        stack_prefix,
        source=kwargs.pop(
            f"{flat_prefix}mask_generator_stack_source", options.generator_stack_source
        ),
    )
    return replace(options, **updates)


def _prefixed_field_map(mapping: dict[str, str], prefix: str) -> dict[str, str]:
    if not prefix:
        return mapping
    return {f"{prefix}{flat_key}": field for flat_key, field in mapping.items()}
