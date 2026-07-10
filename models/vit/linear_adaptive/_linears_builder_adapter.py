# ruff: noqa: E501

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from types import ModuleType
from typing import Any

_ADAPTIVE_GENERATOR_SOURCE_FIELD_MAP = {
    "independent_flag": "independent_flag",
    "hidden_dim": "hidden_dim",
    "layer_norm_position": "layer_norm_position",
    "num_layers": "num_layers",
    "activation": "activation",
    "residual_connection_option": "residual_connection_option",
    "dropout_probability": "dropout_probability",
    "last_layer_bias_option": "last_layer_bias_option",
    "apply_output_pipeline_flag": "apply_output_pipeline_flag",
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


def _default_adaptive_generator_stack_options(config_module: ModuleType) -> Any:
    adaptive_options = _adaptive_options()
    config_prefix = "ADAPTIVE_GENERATOR_STACK"
    return adaptive_options.AdaptiveGeneratorStackOptions(
        hidden_dim=getattr(config_module, f"{config_prefix}_HIDDEN_DIM"),
        layer_norm_position=getattr(
            config_module, f"{config_prefix}_LAYER_NORM_POSITION"
        ),
        num_layers=getattr(config_module, f"{config_prefix}_NUM_LAYERS"),
        activation=getattr(config_module, f"{config_prefix}_ACTIVATION"),
        residual_connection_option=getattr(
            config_module, f"{config_prefix}_RESIDUAL_CONNECTION_OPTION"
        ),
        dropout_probability=getattr(
            config_module, f"{config_prefix}_DROPOUT_PROBABILITY"
        ),
        last_layer_bias_option=getattr(
            config_module, f"{config_prefix}_LAST_LAYER_BIAS_OPTION"
        ),
        apply_output_pipeline_flag=getattr(
            config_module, f"{config_prefix}_APPLY_OUTPUT_PIPELINE_FLAG"
        ),
        bias_flag=getattr(config_module, f"{config_prefix}_BIAS_FLAG"),
    )


def _default_adaptive_generator_stack_source(
    config_module: ModuleType, prefix: str
) -> Any:
    adaptive_options = _adaptive_options()
    config_prefix = prefix.upper()
    return adaptive_options.AdaptiveGeneratorStackSource(
        independent_flag=getattr(config_module, f"{config_prefix}_INDEPENDENT_FLAG"),
        hidden_dim=getattr(config_module, f"{config_prefix}_HIDDEN_DIM"),
        layer_norm_position=getattr(
            config_module, f"{config_prefix}_LAYER_NORM_POSITION"
        ),
        num_layers=getattr(config_module, f"{config_prefix}_NUM_LAYERS"),
        activation=getattr(config_module, f"{config_prefix}_ACTIVATION"),
        residual_connection_option=getattr(
            config_module, f"{config_prefix}_RESIDUAL_CONNECTION_OPTION"
        ),
        dropout_probability=getattr(
            config_module, f"{config_prefix}_DROPOUT_PROBABILITY"
        ),
        last_layer_bias_option=getattr(
            config_module, f"{config_prefix}_LAST_LAYER_BIAS_OPTION"
        ),
        apply_output_pipeline_flag=getattr(
            config_module, f"{config_prefix}_APPLY_OUTPUT_PIPELINE_FLAG"
        ),
        bias_flag=getattr(config_module, f"{config_prefix}_BIAS_FLAG"),
    )


def _config_key(prefix: str, suffix: str) -> str:
    return f"{prefix.upper()}{suffix}" if prefix else suffix


def _default_hidden_adaptive_weight_options(
    config_module: ModuleType,
    *,
    config_prefix: str = "",
    stack_prefix: str = "weight_generator_stack",
) -> Any:
    adaptive_options = _adaptive_options()
    return adaptive_options.HiddenAdaptiveWeightOptions(
        generator_depth=getattr(
            config_module, _config_key(config_prefix, "WEIGHT_GENERATOR_DEPTH")
        ),
        option_flag=getattr(
            config_module, _config_key(config_prefix, "WEIGHT_OPTION_FLAG")
        ),
        option=getattr(config_module, _config_key(config_prefix, "WEIGHT_OPTION")),
        normalization_option=getattr(
            config_module, _config_key(config_prefix, "WEIGHT_NORMALIZATION_OPTION")
        ),
        normalization_position_option=getattr(
            config_module,
            _config_key(config_prefix, "WEIGHT_NORMALIZATION_POSITION_OPTION"),
        ),
        decay_schedule=getattr(
            config_module, _config_key(config_prefix, "WEIGHT_DECAY_SCHEDULE")
        ),
        decay_rate=getattr(
            config_module, _config_key(config_prefix, "WEIGHT_DECAY_RATE")
        ),
        decay_warmup_batches=getattr(
            config_module, _config_key(config_prefix, "WEIGHT_DECAY_WARMUP_BATCHES")
        ),
        bank_expansion_factor=getattr(
            config_module, _config_key(config_prefix, "WEIGHT_BANK_EXPANSION_FACTOR")
        ),
        generator_stack_source=_default_adaptive_generator_stack_source(
            config_module, stack_prefix
        ),
    )


def _default_hidden_adaptive_bias_options(
    config_module: ModuleType,
    *,
    config_prefix: str = "",
    stack_prefix: str = "bias_generator_stack",
) -> Any:
    adaptive_options = _adaptive_options()
    return adaptive_options.HiddenAdaptiveBiasOptions(
        option_flag=getattr(
            config_module, _config_key(config_prefix, "BIAS_OPTION_FLAG")
        ),
        option=getattr(config_module, _config_key(config_prefix, "BIAS_OPTION")),
        decay_schedule=getattr(
            config_module, _config_key(config_prefix, "BIAS_DECAY_SCHEDULE")
        ),
        decay_rate=getattr(
            config_module, _config_key(config_prefix, "BIAS_DECAY_RATE")
        ),
        decay_warmup_batches=getattr(
            config_module, _config_key(config_prefix, "BIAS_DECAY_WARMUP_BATCHES")
        ),
        bank_expansion_factor=getattr(
            config_module, _config_key(config_prefix, "BIAS_BANK_EXPANSION_FACTOR")
        ),
        generator_stack_source=_default_adaptive_generator_stack_source(
            config_module, stack_prefix
        ),
    )


def _default_hidden_adaptive_diagonal_options(
    config_module: ModuleType,
    *,
    config_prefix: str = "",
    stack_prefix: str = "diagonal_generator_stack",
) -> Any:
    adaptive_options = _adaptive_options()
    return adaptive_options.HiddenAdaptiveDiagonalOptions(
        option_flag=getattr(
            config_module, _config_key(config_prefix, "DIAGONAL_OPTION_FLAG")
        ),
        option=getattr(config_module, _config_key(config_prefix, "DIAGONAL_OPTION")),
        generator_stack_source=_default_adaptive_generator_stack_source(
            config_module, stack_prefix
        ),
    )


def _default_hidden_adaptive_mask_options(
    config_module: ModuleType,
    *,
    config_prefix: str = "",
    stack_prefix: str = "mask_generator_stack",
) -> Any:
    adaptive_options = _adaptive_options()
    return adaptive_options.HiddenAdaptiveMaskOptions(
        option_flag=getattr(
            config_module, _config_key(config_prefix, "MASK_OPTION_FLAG")
        ),
        row_mask_option=getattr(
            config_module, _config_key(config_prefix, "ROW_MASK_OPTION")
        ),
        mask_dimension_option=getattr(
            config_module, _config_key(config_prefix, "MASK_DIMENSION_OPTION")
        ),
        mask_threshold=getattr(
            config_module, _config_key(config_prefix, "MASK_THRESHOLD")
        ),
        mask_surrogate_scale=getattr(
            config_module, _config_key(config_prefix, "MASK_SURROGATE_SCALE")
        ),
        mask_floor=getattr(config_module, _config_key(config_prefix, "MASK_FLOOR")),
        mask_transition_width=getattr(
            config_module, _config_key(config_prefix, "MASK_TRANSITION_WIDTH")
        ),
        generator_stack_source=_default_adaptive_generator_stack_source(
            config_module, stack_prefix
        ),
    )


def _adaptive_generator_stack_options_from_kwargs(
    kwargs: dict[str, Any], config_module: ModuleType, *, provided: Any
) -> Any:
    options = provided or _default_adaptive_generator_stack_options(config_module)
    updates = _pop_updates(
        kwargs,
        {
            "adaptive_generator_stack_hidden_dim": "hidden_dim",
            "adaptive_generator_stack_layer_norm_position": "layer_norm_position",
            "adaptive_generator_stack_num_layers": "num_layers",
            "adaptive_generator_stack_activation": "activation",
            "adaptive_generator_stack_residual_connection_option": "residual_connection_option",
            "adaptive_generator_stack_dropout_probability": "dropout_probability",
            "adaptive_generator_stack_last_layer_bias_option": "last_layer_bias_option",
            "adaptive_generator_stack_apply_output_pipeline_flag": "apply_output_pipeline_flag",
            "adaptive_generator_stack_bias_flag": "bias_flag",
        },
    )
    return replace(options, **updates) if updates else options


def _adaptive_generator_stack_source_from_kwargs(
    kwargs: dict[str, Any], config_module: ModuleType, prefix: str, *, provided: Any
) -> Any:
    source = provided or _default_adaptive_generator_stack_source(config_module, prefix)
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
    provided: Any,
    flat_prefix: str = "",
    config_prefix: str = "",
    stack_prefix: str = "weight_generator_stack",
) -> Any:
    options = provided or _default_hidden_adaptive_weight_options(
        config_module, config_prefix=config_prefix, stack_prefix=stack_prefix
    )
    updates = _pop_updates(
        kwargs, _prefixed_field_map(_HIDDEN_ADAPTIVE_WEIGHT_FIELD_MAP, flat_prefix)
    )
    updates["generator_stack_source"] = _adaptive_generator_stack_source_from_kwargs(
        kwargs,
        config_module,
        stack_prefix,
        provided=kwargs.pop(
            f"{flat_prefix}weight_generator_stack_source",
            options.generator_stack_source,
        ),
    )
    return replace(options, **updates)


def _hidden_adaptive_bias_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: Any,
    flat_prefix: str = "",
    config_prefix: str = "",
    stack_prefix: str = "bias_generator_stack",
) -> Any:
    options = provided or _default_hidden_adaptive_bias_options(
        config_module, config_prefix=config_prefix, stack_prefix=stack_prefix
    )
    updates = _pop_updates(
        kwargs, _prefixed_field_map(_HIDDEN_ADAPTIVE_BIAS_FIELD_MAP, flat_prefix)
    )
    updates["generator_stack_source"] = _adaptive_generator_stack_source_from_kwargs(
        kwargs,
        config_module,
        stack_prefix,
        provided=kwargs.pop(
            f"{flat_prefix}bias_generator_stack_source", options.generator_stack_source
        ),
    )
    return replace(options, **updates)


def _hidden_adaptive_diagonal_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: Any,
    flat_prefix: str = "",
    config_prefix: str = "",
    stack_prefix: str = "diagonal_generator_stack",
) -> Any:
    options = provided or _default_hidden_adaptive_diagonal_options(
        config_module, config_prefix=config_prefix, stack_prefix=stack_prefix
    )
    updates = _pop_updates(
        kwargs, _prefixed_field_map(_HIDDEN_ADAPTIVE_DIAGONAL_FIELD_MAP, flat_prefix)
    )
    updates["generator_stack_source"] = _adaptive_generator_stack_source_from_kwargs(
        kwargs,
        config_module,
        stack_prefix,
        provided=kwargs.pop(
            f"{flat_prefix}diagonal_generator_stack_source",
            options.generator_stack_source,
        ),
    )
    return replace(options, **updates)


def _hidden_adaptive_mask_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: Any,
    flat_prefix: str = "",
    config_prefix: str = "",
    stack_prefix: str = "mask_generator_stack",
) -> Any:
    options = provided or _default_hidden_adaptive_mask_options(
        config_module, config_prefix=config_prefix, stack_prefix=stack_prefix
    )
    updates = _pop_updates(
        kwargs, _prefixed_field_map(_HIDDEN_ADAPTIVE_MASK_FIELD_MAP, flat_prefix)
    )
    updates["generator_stack_source"] = _adaptive_generator_stack_source_from_kwargs(
        kwargs,
        config_module,
        stack_prefix,
        provided=kwargs.pop(
            f"{flat_prefix}mask_generator_stack_source", options.generator_stack_source
        ),
    )
    return replace(options, **updates)


def _pop_updates(kwargs: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for flat_key, option_field in mapping.items():
        if flat_key in kwargs:
            updates[option_field] = kwargs.pop(flat_key)
    return updates


def _prefixed_field_map(mapping: dict[str, str], prefix: str) -> dict[str, str]:
    if not prefix:
        return mapping
    return {f"{prefix}{flat_key}": field for flat_key, field in mapping.items()}


def _adaptive_options():
    return import_module("models.vit.linear_adaptive.runtime_options")
