from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import replace
from typing import Final, Union, get_args, get_origin, get_type_hints

import models.neuron.linear_adaptive.config as config
from emperor.layers import GateConfig
from models.neuron.linear_adaptive._hidden.runtime_options import (
    AdaptiveBiasOptions,
    AdaptiveDiagonalOptions,
    AdaptiveMaskOptions,
    AdaptiveProjectionOptions,
    AdaptiveWeightOptions,
    GateOptions,
    GeneratorStackOptions,
    HaltingOptions,
    MemoryOptions,
    RecurrenceOptions,
    RuntimeOptions,
    StackOptions,
)

_PACKAGE = "models.neuron.linear_adaptive._hidden"
_TOP_LEVEL_CONSTANTS = {
    "BATCH_SIZE",
    "LEARNING_RATE",
    "INPUT_DIM",
    "HIDDEN_DIM",
    "OUTPUT_DIM",
}
_RUNTIME_PREFIXES = (
    "STACK_",
    "SUBMODULE_STACK_",
    "ADAPTIVE_GENERATOR_STACK_",
    "GATE_",
    "HALTING_",
    "MEMORY_",
    "RECURRENT_",
    "WEIGHT_",
    "BIAS_",
    "DIAGONAL_",
    "MASK_",
    "INPUT_LAYER_",
    "OUTPUT_LAYER_",
)
_EXTRA_RUNTIME_CONSTANTS = {"ROW_MASK_OPTION"}
_ALIASES = {
    "gate_flag": "stack_gate_flag",
    "halting_flag": "stack_halting_flag",
    "stack_layer_norm_position": "layer_norm_position",
    "weight_generator_depth": "generator_depth",
}
_CANONICAL_CONSTANT_KEYS = {
    "GATE_FLAG": "stack_gate_flag",
    "HALTING_FLAG": "stack_halting_flag",
    "STACK_LAYER_NORM_POSITION": "layer_norm_position",
    "WEIGHT_GENERATOR_DEPTH": "generator_depth",
}


def _is_runtime_constant(name: str) -> bool:
    return (
        name in _TOP_LEVEL_CONSTANTS
        or name in _EXTRA_RUNTIME_CONSTANTS
        or any(name.startswith(prefix) for prefix in _RUNTIME_PREFIXES)
    )


def _flat_defaults() -> tuple[dict[str, object], dict[str, str]]:
    values: dict[str, object] = {}
    constants: dict[str, str] = {}
    for name, value in vars(config).items():
        if not name.isupper() or not _is_runtime_constant(name):
            continue
        key = _CANONICAL_CONSTANT_KEYS.get(name, name.lower())
        values[key] = value
        constants[key] = name
    values["shared_gate_config"] = None
    return values, constants


_FLAT_DEFAULTS, _CONSTANT_BY_KEY = _flat_defaults()
_CONFIG_TYPES = get_type_hints(config)
_ACCEPTED_KEYS = frozenset(_FLAT_DEFAULTS) | frozenset(_ALIASES)


def _expected_type(key: str):
    if key == "shared_gate_config":
        return GateConfig | None
    return _CONFIG_TYPES[_CONSTANT_BY_KEY[key]]


def _type_name(expected: object) -> str:
    return str(expected).replace("<class '", "").replace("'>", "")


def _matches_type(value: object, expected: object) -> bool:
    origin = get_origin(expected)
    if origin in (Union, types.UnionType):
        return any(_matches_type(value, option) for option in get_args(expected))
    if origin is type:
        if not isinstance(value, type):
            return False
        expected_base = get_args(expected)[0]
        return expected_base is object or issubclass(value, expected_base)
    if expected is type(None):
        return value is None
    if expected is bool:
        return type(value) is bool
    if expected is int:
        return type(value) is int
    if expected is float:
        return type(value) in (int, float)
    try:
        return isinstance(value, expected)
    except TypeError:
        return True


def _normalize_overrides(overrides: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    sources: dict[str, str] = {}
    for supplied_key, value in overrides.items():
        if not isinstance(supplied_key, str):
            raise TypeError(
                f"{_PACKAGE}: runtime override keys must be str, got "
                f"{type(supplied_key).__name__}"
            )
        raw_key = supplied_key.strip().replace("-", "_").lower()
        if raw_key not in _ACCEPTED_KEYS:
            accepted = ", ".join(sorted(_ACCEPTED_KEYS))
            raise ValueError(
                f"{_PACKAGE}: unknown runtime key {supplied_key!r}; "
                f"accepted keys: {accepted}"
            )
        key = _ALIASES.get(raw_key, raw_key)
        if key in normalized and sources[key] != raw_key:
            raise ValueError(
                f"{_PACKAGE}: conflicting aliases {sources[key]!r} and "
                f"{raw_key!r} target {key!r}"
            )
        expected = _expected_type(key)
        if not _matches_type(value, expected):
            raise TypeError(
                f"{_PACKAGE}: runtime key {supplied_key!r} has type "
                f"{type(value).__name__}; expected {_type_name(expected)}"
            )
        normalized[key] = value
        sources[key] = raw_key
    return normalized


def _positive(values: Mapping[str, object], *keys: str) -> None:
    for key in keys:
        value = values[key]
        if value <= 0:  # type: ignore[operator]
            raise ValueError(f"{_PACKAGE}: runtime key {key!r} must be positive")


def _probability(values: Mapping[str, object], *keys: str) -> None:
    for key in keys:
        value = values[key]
        if not 0.0 <= value <= 1.0:  # type: ignore[operator]
            raise ValueError(f"{_PACKAGE}: runtime key {key!r} must be between 0 and 1")


def _validate_values(values: Mapping[str, object]) -> None:
    _positive(
        values,
        "batch_size",
        "learning_rate",
        "input_dim",
        "hidden_dim",
        "output_dim",
        "stack_num_layers",
        "submodule_stack_hidden_dim",
        "submodule_stack_num_layers",
        "adaptive_generator_stack_hidden_dim",
        "adaptive_generator_stack_num_layers",
        "recurrent_max_steps",
    )
    for key, value in values.items():
        if value is None:
            continue
        if key.endswith("_num_layers") and value <= 0:  # type: ignore[operator]
            raise ValueError(f"{_PACKAGE}: runtime key {key!r} must be positive")
        if key.endswith("_hidden_dim") and value <= 0:  # type: ignore[operator]
            raise ValueError(f"{_PACKAGE}: runtime key {key!r} must be positive")
        if key.endswith("_dropout_probability"):
            _probability(values, key)
    _probability(
        values,
        "halting_dropout",
        "halting_threshold",
        "recurrent_halting_dropout",
        "recurrent_halting_threshold",
        "mask_threshold",
        "mask_floor",
        "input_layer_mask_threshold",
        "input_layer_mask_floor",
        "output_layer_mask_threshold",
        "output_layer_mask_floor",
    )
    for key in (
        "mask_surrogate_scale",
        "mask_transition_width",
        "input_layer_mask_surrogate_scale",
        "input_layer_mask_transition_width",
        "output_layer_mask_surrogate_scale",
        "output_layer_mask_transition_width",
    ):
        if values[key] <= 0:  # type: ignore[operator]
            raise ValueError(f"{_PACKAGE}: runtime key {key!r} must be positive")
    for key in (
        "weight_decay_rate",
        "bias_decay_rate",
        "input_layer_weight_decay_rate",
        "input_layer_bias_decay_rate",
        "output_layer_weight_decay_rate",
        "output_layer_bias_decay_rate",
    ):
        if values[key] < 0:  # type: ignore[operator]
            raise ValueError(f"{_PACKAGE}: runtime key {key!r} must be non-negative")
    for key in (
        "weight_decay_warmup_batches",
        "bias_decay_warmup_batches",
        "input_layer_weight_decay_warmup_batches",
        "input_layer_bias_decay_warmup_batches",
        "output_layer_weight_decay_warmup_batches",
        "output_layer_bias_decay_warmup_batches",
    ):
        if values[key] < 0:  # type: ignore[operator]
            raise ValueError(f"{_PACKAGE}: runtime key {key!r} must be non-negative")
    memory_learning_rate = values["memory_test_time_training_learning_rate"]
    if memory_learning_rate is not None and memory_learning_rate <= 0:  # type: ignore[operator]
        raise ValueError(
            f"{_PACKAGE}: runtime key "
            "'memory_test_time_training_learning_rate' must be positive"
        )
    memory_steps = values["memory_test_time_training_num_inner_steps"]
    if memory_steps is not None and memory_steps <= 0:  # type: ignore[operator]
        raise ValueError(
            f"{_PACKAGE}: runtime key "
            "'memory_test_time_training_num_inner_steps' must be positive"
        )
    for flag_key, option_key in (
        ("weight_option_flag", "weight_option"),
        ("bias_option_flag", "bias_option"),
        ("diagonal_option_flag", "diagonal_option"),
        ("mask_option_flag", "row_mask_option"),
    ):
        if values[flag_key] and values[option_key] is None:
            raise ValueError(
                f"{_PACKAGE}: runtime key {option_key!r} must be set when "
                f"{flag_key!r} is True"
            )
    if values["stack_gate_flag"] and values["shared_gate_config"] is not None:
        raise ValueError(
            f"{_PACKAGE}: 'stack_gate_flag' and 'shared_gate_config' are "
            "mutually exclusive"
        )


def _stack(values: Mapping[str, object], prefix: str) -> StackOptions:
    return StackOptions(
        hidden_dim=values[f"{prefix}_hidden_dim"],  # type: ignore[arg-type]
        num_layers=values[f"{prefix}_num_layers"],  # type: ignore[arg-type]
        last_layer_bias_option=values[f"{prefix}_last_layer_bias_option"],  # type: ignore[arg-type]
        apply_output_pipeline_flag=values[f"{prefix}_apply_output_pipeline_flag"],  # type: ignore[arg-type]
        activation=values[f"{prefix}_activation"],  # type: ignore[arg-type]
        layer_norm_position=values[f"{prefix}_layer_norm_position"],  # type: ignore[arg-type]
        residual_connection_option=values[f"{prefix}_residual_connection_option"],  # type: ignore[arg-type]
        dropout_probability=values[f"{prefix}_dropout_probability"],  # type: ignore[arg-type]
        bias_flag=values[f"{prefix}_bias_flag"],  # type: ignore[arg-type]
    )


def _resolved_stack(
    values: Mapping[str, object],
    prefix: str,
    defaults: StackOptions,
) -> StackOptions:
    if not values[f"{prefix}_independent_flag"]:
        return defaults
    updates = {
        field: values[f"{prefix}_{field}"]
        for field in (
            "hidden_dim",
            "num_layers",
            "last_layer_bias_option",
            "apply_output_pipeline_flag",
            "activation",
            "layer_norm_position",
            "residual_connection_option",
            "dropout_probability",
            "bias_flag",
        )
        if values[f"{prefix}_{field}"] is not None
    }
    return replace(defaults, **updates)


def _generator_stack(
    values: Mapping[str, object],
    prefix: str,
    defaults: StackOptions,
) -> GeneratorStackOptions:
    independent = values[f"{prefix}_independent_flag"]
    return GeneratorStackOptions(
        independent=independent,  # type: ignore[arg-type]
        stack=_resolved_stack(values, prefix, defaults),
    )


def _projection(
    values: Mapping[str, object],
    prefix: str,
) -> AdaptiveProjectionOptions:
    def value(name: str) -> object:
        return values[f"{prefix}_{name}"]

    return AdaptiveProjectionOptions(
        weight_option=value("weight_option"),  # type: ignore[arg-type]
        weight_generator_depth=value("weight_generator_depth"),  # type: ignore[arg-type]
        weight_decay_schedule=value("weight_decay_schedule"),  # type: ignore[arg-type]
        weight_decay_rate=value("weight_decay_rate"),  # type: ignore[arg-type]
        weight_decay_warmup_batches=value("weight_decay_warmup_batches"),  # type: ignore[arg-type]
        weight_normalization_option=value("weight_normalization_option"),  # type: ignore[arg-type]
        weight_normalization_position_option=value(
            "weight_normalization_position_option"
        ),  # type: ignore[arg-type]
        weight_bank_expansion_factor=value("weight_bank_expansion_factor"),  # type: ignore[arg-type]
        bias_option=value("bias_option"),  # type: ignore[arg-type]
        bias_decay_schedule=value("bias_decay_schedule"),  # type: ignore[arg-type]
        bias_decay_rate=value("bias_decay_rate"),  # type: ignore[arg-type]
        bias_decay_warmup_batches=value("bias_decay_warmup_batches"),  # type: ignore[arg-type]
        bias_bank_expansion_factor=value("bias_bank_expansion_factor"),  # type: ignore[arg-type]
        diagonal_option=value("diagonal_option"),  # type: ignore[arg-type]
        row_mask_option=value("row_mask_option"),  # type: ignore[arg-type]
        mask_dimension_option=value("mask_dimension_option"),  # type: ignore[arg-type]
        mask_threshold=value("mask_threshold"),  # type: ignore[arg-type]
        mask_surrogate_scale=value("mask_surrogate_scale"),  # type: ignore[arg-type]
        mask_floor=value("mask_floor"),  # type: ignore[arg-type]
        mask_transition_width=value("mask_transition_width"),  # type: ignore[arg-type]
    )


def _runtime(values: Mapping[str, object]) -> RuntimeOptions:
    stack = StackOptions(
        hidden_dim=values["hidden_dim"],  # type: ignore[arg-type]
        num_layers=values["stack_num_layers"],  # type: ignore[arg-type]
        last_layer_bias_option=values["stack_last_layer_bias_option"],  # type: ignore[arg-type]
        apply_output_pipeline_flag=values["stack_apply_output_pipeline_flag"],  # type: ignore[arg-type]
        activation=values["stack_activation"],  # type: ignore[arg-type]
        layer_norm_position=values["layer_norm_position"],  # type: ignore[arg-type]
        residual_connection_option=values["stack_residual_connection_option"],  # type: ignore[arg-type]
        dropout_probability=values["stack_dropout_probability"],  # type: ignore[arg-type]
        bias_flag=values["stack_bias_flag"],  # type: ignore[arg-type]
    )
    submodule_stack = _stack(values, "submodule_stack")
    gate_stack = _resolved_stack(values, "gate_stack", submodule_stack)
    halting_defaults = replace(
        submodule_stack,
        last_layer_bias_option=config.LastLayerBiasOptions.DISABLED,
    )
    halting_stack = _resolved_stack(values, "halting_stack", halting_defaults)
    memory_stack = _resolved_stack(values, "memory_stack", submodule_stack)
    recurrent_gate_stack = _resolved_stack(
        values,
        "recurrent_gate_stack",
        gate_stack,
    )
    recurrent_halting_stack = _resolved_stack(
        values,
        "recurrent_halting_stack",
        halting_stack,
    )
    adaptive_generator_stack = _stack(values, "adaptive_generator_stack")
    return RuntimeOptions(
        batch_size=values["batch_size"],  # type: ignore[arg-type]
        learning_rate=values["learning_rate"],  # type: ignore[arg-type]
        input_dim=values["input_dim"],  # type: ignore[arg-type]
        hidden_dim=values["hidden_dim"],  # type: ignore[arg-type]
        output_dim=values["output_dim"],  # type: ignore[arg-type]
        stack=stack,
        submodule_stack=submodule_stack,
        gate=GateOptions(
            enabled=values["stack_gate_flag"],  # type: ignore[arg-type]
            option=values["gate_option"],  # type: ignore[arg-type]
            activation=values["gate_activation"],  # type: ignore[arg-type]
            stack=gate_stack,
            shared_config=values["shared_gate_config"],  # type: ignore[arg-type]
        ),
        halting=HaltingOptions(
            enabled=values["stack_halting_flag"],  # type: ignore[arg-type]
            threshold=values["halting_threshold"],  # type: ignore[arg-type]
            dropout_probability=values["halting_dropout"],  # type: ignore[arg-type]
            hidden_state_mode=values["halting_hidden_state_mode"],  # type: ignore[arg-type]
            stack=halting_stack,
        ),
        memory=MemoryOptions(
            enabled=values["memory_flag"],  # type: ignore[arg-type]
            option=values["memory_option"],  # type: ignore[arg-type]
            position=values["memory_position_option"],  # type: ignore[arg-type]
            test_time_training_learning_rate=values[
                "memory_test_time_training_learning_rate"
            ],  # type: ignore[arg-type]
            test_time_training_num_inner_steps=values[
                "memory_test_time_training_num_inner_steps"
            ],  # type: ignore[arg-type]
            stack=memory_stack,
        ),
        recurrence=RecurrenceOptions(
            enabled=values["recurrent_flag"],  # type: ignore[arg-type]
            max_steps=values["recurrent_max_steps"],  # type: ignore[arg-type]
            layer_norm_position=values["recurrent_layer_norm_position"],  # type: ignore[arg-type]
            gate=GateOptions(
                enabled=values["recurrent_gate_flag"],  # type: ignore[arg-type]
                option=values["recurrent_gate_option"],  # type: ignore[arg-type]
                activation=values["recurrent_gate_activation"],  # type: ignore[arg-type]
                stack=recurrent_gate_stack,
            ),
            halting=HaltingOptions(
                enabled=values["recurrent_halting_flag"],  # type: ignore[arg-type]
                threshold=values["recurrent_halting_threshold"],  # type: ignore[arg-type]
                dropout_probability=values["recurrent_halting_dropout"],  # type: ignore[arg-type]
                hidden_state_mode=values["recurrent_halting_hidden_state_mode"],  # type: ignore[arg-type]
                stack=recurrent_halting_stack,
            ),
        ),
        adaptive_generator_stack=adaptive_generator_stack,
        weight=AdaptiveWeightOptions(
            enabled=values["weight_option_flag"],  # type: ignore[arg-type]
            option=values["weight_option"],  # type: ignore[arg-type]
            generator_depth=values["generator_depth"],  # type: ignore[arg-type]
            normalization_option=values["weight_normalization_option"],  # type: ignore[arg-type]
            normalization_position_option=values[
                "weight_normalization_position_option"
            ],  # type: ignore[arg-type]
            decay_schedule=values["weight_decay_schedule"],  # type: ignore[arg-type]
            decay_rate=values["weight_decay_rate"],  # type: ignore[arg-type]
            decay_warmup_batches=values["weight_decay_warmup_batches"],  # type: ignore[arg-type]
            bank_expansion_factor=values["weight_bank_expansion_factor"],  # type: ignore[arg-type]
            generator_stack=_generator_stack(
                values,
                "weight_generator_stack",
                adaptive_generator_stack,
            ),
        ),
        bias=AdaptiveBiasOptions(
            enabled=values["bias_option_flag"],  # type: ignore[arg-type]
            option=values["bias_option"],  # type: ignore[arg-type]
            decay_schedule=values["bias_decay_schedule"],  # type: ignore[arg-type]
            decay_rate=values["bias_decay_rate"],  # type: ignore[arg-type]
            decay_warmup_batches=values["bias_decay_warmup_batches"],  # type: ignore[arg-type]
            bank_expansion_factor=values["bias_bank_expansion_factor"],  # type: ignore[arg-type]
            generator_stack=_generator_stack(
                values,
                "bias_generator_stack",
                adaptive_generator_stack,
            ),
        ),
        diagonal=AdaptiveDiagonalOptions(
            enabled=values["diagonal_option_flag"],  # type: ignore[arg-type]
            option=values["diagonal_option"],  # type: ignore[arg-type]
            generator_stack=_generator_stack(
                values,
                "diagonal_generator_stack",
                adaptive_generator_stack,
            ),
        ),
        mask=AdaptiveMaskOptions(
            enabled=values["mask_option_flag"],  # type: ignore[arg-type]
            row_mask_option=values["row_mask_option"],  # type: ignore[arg-type]
            dimension_option=values["mask_dimension_option"],  # type: ignore[arg-type]
            threshold=values["mask_threshold"],  # type: ignore[arg-type]
            surrogate_scale=values["mask_surrogate_scale"],  # type: ignore[arg-type]
            floor=values["mask_floor"],  # type: ignore[arg-type]
            transition_width=values["mask_transition_width"],  # type: ignore[arg-type]
            generator_stack=_generator_stack(
                values,
                "mask_generator_stack",
                adaptive_generator_stack,
            ),
        ),
        input_projection=_projection(values, "input_layer"),
        output_projection=_projection(values, "output_layer"),
    )


def runtime_from_flat(
    overrides: Mapping[str, object] | None = None,
) -> RuntimeOptions:
    normalized = _normalize_overrides(overrides or {})
    values = {**_FLAT_DEFAULTS, **normalized}
    _validate_values(values)
    return _runtime(values)


DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_flat()
