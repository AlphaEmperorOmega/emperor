from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from enum import Enum
from typing import Final, TypeVar

from emperor.base.layer.gate import LayerGateOptions
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.memory.config import DynamicMemoryConfig
from emperor.memory.options import MemoryPositionOptions

import models.linears.linear.config as config
from models.linears.linear.runtime_options import (
    ControllerStackOptions,
    GateOptions,
    HaltingOptions,
    MainStackOptions,
    MemoryOptions,
    RecurrenceOptions,
    RuntimeOptions,
)


_PACKAGE_NAME = "models.linears.linear"
_ALIASES = {
    "gate_flag": "stack_gate_flag",
    "halting_flag": "stack_halting_flag",
    "stack_layer_norm_position": "layer_norm_position",
}
_CONTROLLER_STACK_FIELDS = (
    "independent_flag",
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
_CONTROLLER_STACK_PREFIXES = (
    "gate_stack",
    "halting_stack",
    "memory_stack",
    "recurrent_gate_stack",
    "recurrent_halting_stack",
)

_EnumT = TypeVar("_EnumT", bound=Enum)


def _controller_stack_flat_defaults(prefix: str) -> dict[str, object]:
    config_prefix = prefix.upper()
    return {
        f"{prefix}_{field}": getattr(config, f"{config_prefix}_{field.upper()}")
        for field in _CONTROLLER_STACK_FIELDS
    }


def _flat_defaults() -> dict[str, object]:
    defaults: dict[str, object] = {
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "input_dim": config.INPUT_DIM,
        "hidden_dim": config.HIDDEN_DIM,
        "output_dim": config.OUTPUT_DIM,
        "stack_bias_flag": config.STACK_BIAS_FLAG,
        "layer_norm_position": config.STACK_LAYER_NORM_POSITION,
        "stack_num_layers": config.STACK_NUM_LAYERS,
        "stack_activation": config.STACK_ACTIVATION,
        "stack_residual_connection_option": (
            config.STACK_RESIDUAL_CONNECTION_OPTION
        ),
        "stack_dropout_probability": config.STACK_DROPOUT_PROBABILITY,
        "stack_last_layer_bias_option": config.STACK_LAST_LAYER_BIAS_OPTION,
        "stack_apply_output_pipeline_flag": (
            config.STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        "submodule_stack_hidden_dim": config.SUBMODULE_STACK_HIDDEN_DIM,
        "submodule_stack_num_layers": config.SUBMODULE_STACK_NUM_LAYERS,
        "submodule_stack_last_layer_bias_option": (
            config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION
        ),
        "submodule_stack_apply_output_pipeline_flag": (
            config.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        "submodule_stack_activation": config.SUBMODULE_STACK_ACTIVATION,
        "submodule_stack_layer_norm_position": (
            config.SUBMODULE_STACK_LAYER_NORM_POSITION
        ),
        "submodule_stack_residual_connection_option": (
            config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        "submodule_stack_dropout_probability": (
            config.SUBMODULE_STACK_DROPOUT_PROBABILITY
        ),
        "submodule_stack_bias_flag": config.SUBMODULE_STACK_BIAS_FLAG,
        "stack_gate_flag": config.GATE_FLAG,
        "gate_option": config.GATE_OPTION,
        "gate_activation": config.GATE_ACTIVATION,
        "stack_halting_flag": config.HALTING_FLAG,
        "halting_threshold": config.HALTING_THRESHOLD,
        "halting_dropout": config.HALTING_DROPOUT,
        "halting_hidden_state_mode": config.HALTING_HIDDEN_STATE_MODE,
        "memory_flag": config.MEMORY_FLAG,
        "memory_option": config.MEMORY_OPTION,
        "memory_position_option": config.MEMORY_POSITION_OPTION,
        "memory_test_time_training_learning_rate": (
            config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        ),
        "memory_test_time_training_num_inner_steps": (
            config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
        ),
        "recurrent_flag": config.RECURRENT_FLAG,
        "recurrent_max_steps": config.RECURRENT_MAX_STEPS,
        "recurrent_layer_norm_position": config.RECURRENT_LAYER_NORM_POSITION,
        "recurrent_gate_flag": config.RECURRENT_GATE_FLAG,
        "recurrent_gate_option": config.RECURRENT_GATE_OPTION,
        "recurrent_gate_activation": config.RECURRENT_GATE_ACTIVATION,
        "recurrent_halting_flag": config.RECURRENT_HALTING_FLAG,
        "recurrent_halting_threshold": config.RECURRENT_HALTING_THRESHOLD,
        "recurrent_halting_dropout": config.RECURRENT_HALTING_DROPOUT,
        "recurrent_halting_hidden_state_mode": (
            config.RECURRENT_HALTING_HIDDEN_STATE_MODE
        ),
    }
    for prefix in _CONTROLLER_STACK_PREFIXES:
        defaults.update(_controller_stack_flat_defaults(prefix))
    return defaults


_DEFAULT_FLAT: Final = _flat_defaults()
_ACCEPTED_KEYS: Final = frozenset((*_DEFAULT_FLAT, *_ALIASES))


def _canonical_flat_values(
    overrides: Mapping[str, object] | None,
) -> tuple[dict[str, object], dict[str, str]]:
    values = dict(_DEFAULT_FLAT)
    source_keys = {key: key for key in values}
    if overrides is None:
        return values, source_keys

    normalized_overrides: list[tuple[str, str, object]] = []
    unknown_keys: list[object] = []
    for supplied_key, value in overrides.items():
        if not isinstance(supplied_key, str):
            raise TypeError(
                f"{_PACKAGE_NAME}: runtime override keys must be str; got "
                f"{type(supplied_key).__name__}"
            )
        normalized_key = supplied_key.strip().replace("-", "_").lower()
        if normalized_key not in _ACCEPTED_KEYS:
            unknown_keys.append(supplied_key)
            continue
        normalized_overrides.append((supplied_key, normalized_key, value))
    if unknown_keys:
        accepted = ", ".join(sorted(_ACCEPTED_KEYS))
        raise ValueError(
            f"{_PACKAGE_NAME}: unknown runtime override {unknown_keys[0]!r}; "
            f"accepted keys: {accepted}"
        )

    provided: dict[str, tuple[str, object]] = {}
    for supplied_key, source_key, value in normalized_overrides:
        canonical_key = _ALIASES.get(source_key, source_key)
        previous = provided.get(canonical_key)
        if previous is not None and previous[1] != value:
            previous_key, previous_value = previous
            raise ValueError(
                f"{_PACKAGE_NAME}: conflicting aliases {previous_key!r}="
                f"{previous_value!r} and {supplied_key!r}={value!r} target "
                f"{canonical_key!r}"
            )
        provided[canonical_key] = (supplied_key, value)
        values[canonical_key] = value
        source_keys[canonical_key] = supplied_key
    return values, source_keys


def _raise_type_error(key: str, value: object, expected: str) -> None:
    raise TypeError(
        f"{_PACKAGE_NAME}: {key!r} has type {type(value).__name__}; "
        f"expected {expected}"
    )


def _bool(values: Mapping[str, object], sources: Mapping[str, str], key: str) -> bool:
    value = values[key]
    if type(value) is not bool:
        _raise_type_error(sources[key], value, "bool")
    return value


def _int(values: Mapping[str, object], sources: Mapping[str, str], key: str) -> int:
    value = values[key]
    if type(value) is not int:
        _raise_type_error(sources[key], value, "int")
    return value


def _optional_int(
    values: Mapping[str, object], sources: Mapping[str, str], key: str
) -> int | None:
    value = values[key]
    if value is None:
        return None
    return _int(values, sources, key)


def _float(
    values: Mapping[str, object], sources: Mapping[str, str], key: str
) -> float:
    value = values[key]
    if type(value) is not float:
        _raise_type_error(sources[key], value, "float")
    return value


def _optional_float(
    values: Mapping[str, object], sources: Mapping[str, str], key: str
) -> float | None:
    value = values[key]
    if value is None:
        return None
    return _float(values, sources, key)


def _enum(
    values: Mapping[str, object],
    sources: Mapping[str, str],
    key: str,
    expected_type: type[_EnumT],
) -> _EnumT:
    value = values[key]
    if not isinstance(value, expected_type):
        _raise_type_error(sources[key], value, expected_type.__name__)
    return value


def _optional_enum(
    values: Mapping[str, object],
    sources: Mapping[str, str],
    key: str,
    expected_type: type[_EnumT],
) -> _EnumT | None:
    value = values[key]
    if value is None:
        return None
    return _enum(values, sources, key, expected_type)


def _positive(key: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{_PACKAGE_NAME}: {key!r} must be positive; got {value!r}")


def _probability(key: str, value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(
            f"{_PACKAGE_NAME}: {key!r} must be in [0.0, 1.0); got {value!r}"
        )


def _threshold(key: str, value: float) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(
            f"{_PACKAGE_NAME}: {key!r} must be in (0.0, 1.0]; got {value!r}"
        )


def _main_stack(
    values: Mapping[str, object], sources: Mapping[str, str]
) -> MainStackOptions:
    num_layers = _int(values, sources, "stack_num_layers")
    dropout = _float(values, sources, "stack_dropout_probability")
    _positive(sources["stack_num_layers"], num_layers)
    _probability(sources["stack_dropout_probability"], dropout)
    return MainStackOptions(
        bias_flag=_bool(values, sources, "stack_bias_flag"),
        layer_norm_position=_enum(
            values, sources, "layer_norm_position", LayerNormPositionOptions
        ),
        num_layers=num_layers,
        activation=_enum(values, sources, "stack_activation", ActivationOptions),
        residual_connection_option=_enum(
            values,
            sources,
            "stack_residual_connection_option",
            ResidualConnectionOptions,
        ),
        dropout_probability=dropout,
        last_layer_bias_option=_enum(
            values,
            sources,
            "stack_last_layer_bias_option",
            LastLayerBiasOptions,
        ),
        apply_output_pipeline_flag=_bool(
            values, sources, "stack_apply_output_pipeline_flag"
        ),
    )


def _submodule_stack(
    values: Mapping[str, object], sources: Mapping[str, str]
) -> ControllerStackOptions:
    hidden_dim = _int(values, sources, "submodule_stack_hidden_dim")
    num_layers = _int(values, sources, "submodule_stack_num_layers")
    dropout = _float(values, sources, "submodule_stack_dropout_probability")
    _positive(sources["submodule_stack_hidden_dim"], hidden_dim)
    _positive(sources["submodule_stack_num_layers"], num_layers)
    _probability(sources["submodule_stack_dropout_probability"], dropout)
    return ControllerStackOptions(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=_enum(
            values,
            sources,
            "submodule_stack_last_layer_bias_option",
            LastLayerBiasOptions,
        ),
        apply_output_pipeline_flag=_bool(
            values, sources, "submodule_stack_apply_output_pipeline_flag"
        ),
        activation=_enum(
            values, sources, "submodule_stack_activation", ActivationOptions
        ),
        layer_norm_position=_enum(
            values,
            sources,
            "submodule_stack_layer_norm_position",
            LayerNormPositionOptions,
        ),
        residual_connection_option=_enum(
            values,
            sources,
            "submodule_stack_residual_connection_option",
            ResidualConnectionOptions,
        ),
        dropout_probability=dropout,
        bias_flag=_bool(values, sources, "submodule_stack_bias_flag"),
    )


def _resolved_controller_stack(
    values: Mapping[str, object],
    sources: Mapping[str, str],
    prefix: str,
    defaults: ControllerStackOptions,
) -> ControllerStackOptions:
    independent = _bool(values, sources, f"{prefix}_independent_flag")
    hidden_dim = _optional_int(values, sources, f"{prefix}_hidden_dim")
    num_layers = _optional_int(values, sources, f"{prefix}_num_layers")
    last_layer_bias_option = _optional_enum(
        values,
        sources,
        f"{prefix}_last_layer_bias_option",
        LastLayerBiasOptions,
    )
    apply_output_pipeline_flag = values[f"{prefix}_apply_output_pipeline_flag"]
    if apply_output_pipeline_flag is not None:
        apply_output_pipeline_flag = _bool(
            values, sources, f"{prefix}_apply_output_pipeline_flag"
        )
    activation = _optional_enum(
        values, sources, f"{prefix}_activation", ActivationOptions
    )
    layer_norm_position = _optional_enum(
        values,
        sources,
        f"{prefix}_layer_norm_position",
        LayerNormPositionOptions,
    )
    residual_connection_option = _optional_enum(
        values,
        sources,
        f"{prefix}_residual_connection_option",
        ResidualConnectionOptions,
    )
    dropout = _optional_float(values, sources, f"{prefix}_dropout_probability")
    bias_flag = values[f"{prefix}_bias_flag"]
    if bias_flag is not None:
        bias_flag = _bool(values, sources, f"{prefix}_bias_flag")

    if hidden_dim is not None:
        _positive(sources[f"{prefix}_hidden_dim"], hidden_dim)
    if num_layers is not None:
        _positive(sources[f"{prefix}_num_layers"], num_layers)
    if dropout is not None:
        _probability(sources[f"{prefix}_dropout_probability"], dropout)
    if not independent:
        return defaults
    return ControllerStackOptions(
        hidden_dim=defaults.hidden_dim if hidden_dim is None else hidden_dim,
        num_layers=defaults.num_layers if num_layers is None else num_layers,
        last_layer_bias_option=(
            defaults.last_layer_bias_option
            if last_layer_bias_option is None
            else last_layer_bias_option
        ),
        apply_output_pipeline_flag=(
            defaults.apply_output_pipeline_flag
            if apply_output_pipeline_flag is None
            else apply_output_pipeline_flag
        ),
        activation=defaults.activation if activation is None else activation,
        layer_norm_position=(
            defaults.layer_norm_position
            if layer_norm_position is None
            else layer_norm_position
        ),
        residual_connection_option=(
            defaults.residual_connection_option
            if residual_connection_option is None
            else residual_connection_option
        ),
        dropout_probability=(
            defaults.dropout_probability if dropout is None else dropout
        ),
        bias_flag=defaults.bias_flag if bias_flag is None else bias_flag,
    )


def _memory_implementation(
    values: Mapping[str, object], sources: Mapping[str, str]
) -> type[DynamicMemoryConfig]:
    value = values["memory_option"]
    if not isinstance(value, type) or not issubclass(value, DynamicMemoryConfig):
        _raise_type_error(
            sources["memory_option"], value, "type[DynamicMemoryConfig]"
        )
    return value


def runtime_from_flat(
    overrides: Mapping[str, object] | None = None,
) -> RuntimeOptions:
    values, sources = _canonical_flat_values(overrides)

    batch_size = _int(values, sources, "batch_size")
    learning_rate = _float(values, sources, "learning_rate")
    input_dim = _int(values, sources, "input_dim")
    hidden_dim = _int(values, sources, "hidden_dim")
    output_dim = _int(values, sources, "output_dim")
    for key, value in (
        ("batch_size", batch_size),
        ("learning_rate", learning_rate),
        ("input_dim", input_dim),
        ("hidden_dim", hidden_dim),
        ("output_dim", output_dim),
    ):
        _positive(sources[key], value)

    main_stack = _main_stack(values, sources)
    submodule_stack = _submodule_stack(values, sources)
    gate_stack = _resolved_controller_stack(
        values, sources, "gate_stack", submodule_stack
    )
    halting_stack = _resolved_controller_stack(
        values,
        sources,
        "halting_stack",
        replace(
            submodule_stack,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
        ),
    )
    memory_stack = _resolved_controller_stack(
        values, sources, "memory_stack", submodule_stack
    )
    recurrent_gate_stack = _resolved_controller_stack(
        values, sources, "recurrent_gate_stack", gate_stack
    )
    recurrent_halting_stack = _resolved_controller_stack(
        values, sources, "recurrent_halting_stack", halting_stack
    )

    gate_enabled = _bool(values, sources, "stack_gate_flag")
    gate_option = _optional_enum(values, sources, "gate_option", LayerGateOptions)
    if gate_enabled and gate_option is None:
        raise ValueError(
            f"{_PACKAGE_NAME}: 'gate_option' must be set when "
            "'stack_gate_flag' is True"
        )
    gate = GateOptions(
        enabled=gate_enabled,
        option=gate_option,
        activation=_optional_enum(
            values, sources, "gate_activation", ActivationOptions
        ),
        stack=gate_stack,
    )

    halting_threshold = _float(values, sources, "halting_threshold")
    halting_dropout = _float(values, sources, "halting_dropout")
    _threshold(sources["halting_threshold"], halting_threshold)
    _probability(sources["halting_dropout"], halting_dropout)
    halting = HaltingOptions(
        enabled=_bool(values, sources, "stack_halting_flag"),
        threshold=halting_threshold,
        dropout_probability=halting_dropout,
        hidden_state_mode=_enum(
            values,
            sources,
            "halting_hidden_state_mode",
            HaltingHiddenStateModeOptions,
        ),
        stack=halting_stack,
    )

    memory_learning_rate = _optional_float(
        values, sources, "memory_test_time_training_learning_rate"
    )
    memory_inner_steps = _optional_int(
        values, sources, "memory_test_time_training_num_inner_steps"
    )
    if memory_learning_rate is not None:
        _positive(
            sources["memory_test_time_training_learning_rate"],
            memory_learning_rate,
        )
    if memory_inner_steps is not None:
        _positive(
            sources["memory_test_time_training_num_inner_steps"],
            memory_inner_steps,
        )
    memory = MemoryOptions(
        enabled=_bool(values, sources, "memory_flag"),
        implementation=_memory_implementation(values, sources),
        position=_enum(
            values, sources, "memory_position_option", MemoryPositionOptions
        ),
        test_time_training_learning_rate=memory_learning_rate,
        test_time_training_num_inner_steps=memory_inner_steps,
        stack=memory_stack,
    )

    recurrent_max_steps = _int(values, sources, "recurrent_max_steps")
    _positive(sources["recurrent_max_steps"], recurrent_max_steps)
    recurrent_gate_enabled = _bool(values, sources, "recurrent_gate_flag")
    recurrent_gate_option = _optional_enum(
        values, sources, "recurrent_gate_option", LayerGateOptions
    )
    if recurrent_gate_enabled and recurrent_gate_option is None:
        raise ValueError(
            f"{_PACKAGE_NAME}: 'recurrent_gate_option' must be set when "
            "'recurrent_gate_flag' is True"
        )
    recurrent_halting_threshold = _float(
        values, sources, "recurrent_halting_threshold"
    )
    recurrent_halting_dropout = _float(
        values, sources, "recurrent_halting_dropout"
    )
    _threshold(
        sources["recurrent_halting_threshold"], recurrent_halting_threshold
    )
    _probability(
        sources["recurrent_halting_dropout"], recurrent_halting_dropout
    )
    recurrence = RecurrenceOptions(
        enabled=_bool(values, sources, "recurrent_flag"),
        max_steps=recurrent_max_steps,
        layer_norm_position=_enum(
            values,
            sources,
            "recurrent_layer_norm_position",
            LayerNormPositionOptions,
        ),
        gate=GateOptions(
            enabled=recurrent_gate_enabled,
            option=recurrent_gate_option,
            activation=_optional_enum(
                values, sources, "recurrent_gate_activation", ActivationOptions
            ),
            stack=recurrent_gate_stack,
        ),
        halting=HaltingOptions(
            enabled=_bool(values, sources, "recurrent_halting_flag"),
            threshold=recurrent_halting_threshold,
            dropout_probability=recurrent_halting_dropout,
            hidden_state_mode=_enum(
                values,
                sources,
                "recurrent_halting_hidden_state_mode",
                HaltingHiddenStateModeOptions,
            ),
            stack=recurrent_halting_stack,
        ),
    )
    return RuntimeOptions(
        batch_size=batch_size,
        learning_rate=learning_rate,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        stack=main_stack,
        submodule_stack=submodule_stack,
        gate=gate,
        halting=halting,
        memory=memory,
        recurrence=recurrence,
    )


DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_flat()
