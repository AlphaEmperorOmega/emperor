from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeGuard, TypeVar

import models.neuron.linear.config as config
from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions

_PACKAGE_NAME = "models.neuron.linear._hidden"
_EnumT = TypeVar("_EnumT", bound=Enum)
_ImplementationT = TypeVar("_ImplementationT")


def _is_bool(value: object) -> TypeGuard[bool]:
    return type(value) is bool


def _is_int(value: object) -> TypeGuard[int]:
    return type(value) is int


def _is_float(value: object) -> TypeGuard[float]:
    return type(value) is float


def _is_implementation(
    value: object,
    base_type: type[_ImplementationT],
) -> TypeGuard[type[_ImplementationT]]:
    return isinstance(value, type) and issubclass(value, base_type)


def _is_concrete_residual(value: object) -> TypeGuard[type[ResidualConfig]]:
    return (
        _is_implementation(value, ResidualConfig)
        and "_registry_owner" in value.__dict__
    )


@dataclass(slots=True)
class RuntimeOverrideReader:
    supplied: Mapping[str, object]
    remaining: dict[str, object] = field(init=False)
    accepted: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.remaining = dict(self.supplied)

    def reject_unknown(self, accepted: frozenset[str]) -> None:
        unknown_keys = sorted(
            (key for key in self.remaining if key not in accepted),
            key=repr,
        )
        if not unknown_keys:
            return
        accepted_text = ", ".join(sorted(accepted))
        raise ValueError(
            f"{_PACKAGE_NAME}: unknown runtime override {unknown_keys[0]!r}; "
            f"accepted keys: {accepted_text}"
        )

    def _take(self, key: str, default: object) -> object:
        self.accepted.add(key)
        return self.remaining.pop(key, default)

    def boolean(self, key: str, default: bool) -> bool:
        value = self._take(key, default)
        if not _is_bool(value):
            self._raise_type_error(key, value, "bool")
        return value

    def optional_boolean(self, key: str, default: bool | None) -> bool | None:
        value = self._take(key, default)
        if value is None:
            return None
        if not _is_bool(value):
            self._raise_type_error(key, value, "bool")
        return value

    def integer(self, key: str, default: int) -> int:
        value = self._take(key, default)
        if not _is_int(value):
            self._raise_type_error(key, value, "int")
        return value

    def optional_integer(self, key: str, default: int | None) -> int | None:
        value = self._take(key, default)
        if value is None:
            return None
        if not _is_int(value):
            self._raise_type_error(key, value, "int")
        return value

    def floating(self, key: str, default: float) -> float:
        value = self._take(key, default)
        if not _is_float(value):
            self._raise_type_error(key, value, "float")
        return value

    def optional_floating(self, key: str, default: float | None) -> float | None:
        value = self._take(key, default)
        if value is None:
            return None
        if not _is_float(value):
            self._raise_type_error(key, value, "float")
        return value

    def enum(self, key: str, default: _EnumT, enum_type: type[_EnumT]) -> _EnumT:
        value = self._take(key, default)
        if not isinstance(value, enum_type):
            self._raise_type_error(key, value, enum_type.__name__)
        return value

    def optional_enum(
        self,
        key: str,
        default: _EnumT | None,
        enum_type: type[_EnumT],
    ) -> _EnumT | None:
        value = self._take(key, default)
        if value is None:
            return None
        if not isinstance(value, enum_type):
            self._raise_type_error(key, value, enum_type.__name__)
        return value

    def implementation(
        self,
        key: str,
        default: type[_ImplementationT],
        base_type: type[_ImplementationT],
    ) -> type[_ImplementationT]:
        value = self._take(key, default)
        if not _is_implementation(value, base_type):
            self._raise_type_error(key, value, f"type[{base_type.__name__}]")
        return value

    def optional_residual(
        self,
        key: str,
        default: type[ResidualConfig] | None,
    ) -> type[ResidualConfig] | None:
        value = self._take(key, default)
        if value is None:
            return None
        if not _is_concrete_residual(value):
            self._raise_type_error(key, value, "concrete type[ResidualConfig]")
        return value

    @staticmethod
    def _raise_type_error(key: str, value: object, expected: str) -> None:
        raise TypeError(
            f"{_PACKAGE_NAME}: {key!r} has type {type(value).__name__}; "
            f"expected {expected}"
        )


@dataclass(frozen=True, slots=True)
class RuntimeDimensionsValues:
    batch_size: int
    learning_rate: float
    input_dim: int
    hidden_dim: int
    output_dim: int


@dataclass(frozen=True, slots=True)
class MainStackMeasureValues:
    num_layers: int
    dropout_probability: float


@dataclass(frozen=True, slots=True)
class SubmoduleStackMeasureValues:
    hidden_dim: int
    num_layers: int
    dropout_probability: float


@dataclass(frozen=True, slots=True)
class OptionalStackMeasures:
    hidden_dim: int | None
    num_layers: int | None
    dropout_probability: float | None


@dataclass(frozen=True, slots=True)
class MainStackValues:
    bias_flag: bool
    layer_norm_position: LayerNormPositionOptions
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool


@dataclass(frozen=True, slots=True)
class SubmoduleStackValues:
    hidden_dim: int
    num_layers: int
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    dropout_probability: float
    bias_flag: bool


@dataclass(frozen=True, slots=True)
class OptionalStackValues:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_pipeline_flag: bool | None
    activation: ActivationOptions | None
    layer_norm_position: LayerNormPositionOptions | None
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    dropout_probability: float | None
    bias_flag: bool | None


@dataclass(frozen=True, slots=True)
class OptionalStackFields:
    independent_flag: str
    hidden_dim: str
    num_layers: str
    last_layer_bias_option: str
    apply_output_pipeline_flag: str
    activation: str
    layer_norm_position: str
    residual_connection_option: str
    residual_model_flag: str
    dropout_probability: str
    bias_flag: str


@dataclass(frozen=True, slots=True)
class OptionalStackInput:
    fields: OptionalStackFields
    values: OptionalStackValues


@dataclass(frozen=True, slots=True)
class GateConditionValues:
    enabled: bool
    option: LayerGateOptions | None


@dataclass(frozen=True, slots=True)
class HaltingMeasureValues:
    threshold: float
    dropout_probability: float


@dataclass(frozen=True, slots=True)
class HaltingValues:
    enabled: bool
    hidden_state_mode: HaltingHiddenStateModeOptions


@dataclass(frozen=True, slots=True)
class MemoryMeasureValues:
    learning_rate: float | None
    num_inner_steps: int | None


@dataclass(frozen=True, slots=True)
class MemoryValues:
    enabled: bool
    implementation: type[DynamicMemoryConfig]
    position: MemoryPositionOptions


@dataclass(frozen=True, slots=True)
class RecurrenceValues:
    enabled: bool
    initial_iterations: int | None
    gradient_transition_count: int | None
    iteration_increment: int
    forward_calls_before_iteration_increment: int
    smooth_iteration_growth_flag: bool
    layer_norm_position: LayerNormPositionOptions
    gate_activation: ActivationOptions | None
    halting_enabled: bool
    halting_hidden_state_mode: HaltingHiddenStateModeOptions


@dataclass(frozen=True, slots=True)
class HaltingImplementationValues:
    main: type[HaltingConfig]
    recurrent: type[HaltingConfig]


@dataclass(frozen=True, slots=True)
class ControlDefaultValues:
    stack_gate_flag: bool
    gate_option: LayerGateOptions | None
    gate_activation: ActivationOptions | None
    stack_halting_flag: bool
    halting_option: type[HaltingConfig]
    halting_threshold: float
    halting_dropout: float
    halting_hidden_state_mode: HaltingHiddenStateModeOptions
    memory_flag: bool
    memory_option: type[DynamicMemoryConfig]
    memory_position_option: MemoryPositionOptions
    memory_test_time_training_learning_rate: float | None
    memory_test_time_training_num_inner_steps: int | None
    recurrent_flag: bool
    recurrent_max_steps: int
    recurrent_initial_iterations: int | None
    recurrent_gradient_transition_count: int | None
    recurrent_iteration_increment: int
    recurrent_forward_calls_before_iteration_increment: int
    recurrent_smooth_iteration_growth_flag: bool
    recurrent_layer_norm_position: LayerNormPositionOptions
    recurrent_stack_gate_flag: bool
    recurrent_gate_option: LayerGateOptions | None
    recurrent_gate_activation: ActivationOptions | None
    recurrent_stack_halting_flag: bool
    recurrent_halting_option: type[HaltingConfig]
    recurrent_halting_threshold: float
    recurrent_halting_dropout: float
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions


@dataclass(frozen=True, slots=True)
class RuntimeDefaultValues:
    dimensions: RuntimeDimensionsValues
    main_stack: MainStackValues
    submodule_stack: SubmoduleStackValues
    residual_stack: OptionalStackValues
    gate_stack: OptionalStackValues
    halting_stack: OptionalStackValues
    memory_stack: OptionalStackValues
    recurrent_gate_stack: OptionalStackValues
    recurrent_halting_stack: OptionalStackValues
    control: ControlDefaultValues


_RESIDUAL_STACK_FIELDS = OptionalStackFields(
    independent_flag="residual_stack_independent_flag",
    hidden_dim="residual_stack_hidden_dim",
    num_layers="residual_stack_num_layers",
    last_layer_bias_option="residual_stack_last_layer_bias_option",
    apply_output_pipeline_flag="residual_stack_apply_output_pipeline_flag",
    activation="residual_stack_activation",
    layer_norm_position="residual_stack_layer_norm_position",
    residual_connection_option="residual_stack_residual_connection_option",
    residual_model_flag="residual_stack_residual_model_flag",
    dropout_probability="residual_stack_dropout_probability",
    bias_flag="residual_stack_bias_flag",
)
_GATE_STACK_FIELDS = OptionalStackFields(
    independent_flag="gate_stack_independent_flag",
    hidden_dim="gate_stack_hidden_dim",
    num_layers="gate_stack_num_layers",
    last_layer_bias_option="gate_stack_last_layer_bias_option",
    apply_output_pipeline_flag="gate_stack_apply_output_pipeline_flag",
    activation="gate_stack_activation",
    layer_norm_position="gate_stack_layer_norm_position",
    residual_connection_option="gate_stack_residual_connection_option",
    residual_model_flag="gate_stack_residual_model_flag",
    dropout_probability="gate_stack_dropout_probability",
    bias_flag="gate_stack_bias_flag",
)
_HALTING_STACK_FIELDS = OptionalStackFields(
    independent_flag="halting_stack_independent_flag",
    hidden_dim="halting_stack_hidden_dim",
    num_layers="halting_stack_num_layers",
    last_layer_bias_option="halting_stack_last_layer_bias_option",
    apply_output_pipeline_flag="halting_stack_apply_output_pipeline_flag",
    activation="halting_stack_activation",
    layer_norm_position="halting_stack_layer_norm_position",
    residual_connection_option="halting_stack_residual_connection_option",
    residual_model_flag="halting_stack_residual_model_flag",
    dropout_probability="halting_stack_dropout_probability",
    bias_flag="halting_stack_bias_flag",
)
_MEMORY_STACK_FIELDS = OptionalStackFields(
    independent_flag="memory_stack_independent_flag",
    hidden_dim="memory_stack_hidden_dim",
    num_layers="memory_stack_num_layers",
    last_layer_bias_option="memory_stack_last_layer_bias_option",
    apply_output_pipeline_flag="memory_stack_apply_output_pipeline_flag",
    activation="memory_stack_activation",
    layer_norm_position="memory_stack_layer_norm_position",
    residual_connection_option="memory_stack_residual_connection_option",
    residual_model_flag="memory_stack_residual_model_flag",
    dropout_probability="memory_stack_dropout_probability",
    bias_flag="memory_stack_bias_flag",
)
_RECURRENT_GATE_STACK_FIELDS = OptionalStackFields(
    independent_flag="recurrent_gate_stack_independent_flag",
    hidden_dim="recurrent_gate_stack_hidden_dim",
    num_layers="recurrent_gate_stack_num_layers",
    last_layer_bias_option="recurrent_gate_stack_last_layer_bias_option",
    apply_output_pipeline_flag="recurrent_gate_stack_apply_output_pipeline_flag",
    activation="recurrent_gate_stack_activation",
    layer_norm_position="recurrent_gate_stack_layer_norm_position",
    residual_connection_option="recurrent_gate_stack_residual_connection_option",
    residual_model_flag="recurrent_gate_stack_residual_model_flag",
    dropout_probability="recurrent_gate_stack_dropout_probability",
    bias_flag="recurrent_gate_stack_bias_flag",
)
_RECURRENT_HALTING_STACK_FIELDS = OptionalStackFields(
    independent_flag="recurrent_halting_stack_independent_flag",
    hidden_dim="recurrent_halting_stack_hidden_dim",
    num_layers="recurrent_halting_stack_num_layers",
    last_layer_bias_option="recurrent_halting_stack_last_layer_bias_option",
    apply_output_pipeline_flag="recurrent_halting_stack_apply_output_pipeline_flag",
    activation="recurrent_halting_stack_activation",
    layer_norm_position="recurrent_halting_stack_layer_norm_position",
    residual_connection_option="recurrent_halting_stack_residual_connection_option",
    residual_model_flag="recurrent_halting_stack_residual_model_flag",
    dropout_probability="recurrent_halting_stack_dropout_probability",
    bias_flag="recurrent_halting_stack_bias_flag",
)

_DEFAULT_VALUES = RuntimeDefaultValues(
    dimensions=RuntimeDimensionsValues(
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
    ),
    main_stack=MainStackValues(
        bias_flag=config.STACK_BIAS_FLAG,
        layer_norm_position=config.LAYER_NORM_POSITION,
        num_layers=config.STACK_NUM_LAYERS,
        activation=config.STACK_ACTIVATION,
        residual_connection_option=config.STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.STACK_DROPOUT_PROBABILITY,
        last_layer_bias_option=config.STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
    ),
    submodule_stack=SubmoduleStackValues(
        hidden_dim=config.SUBMODULE_STACK_HIDDEN_DIM,
        num_layers=config.SUBMODULE_STACK_NUM_LAYERS,
        last_layer_bias_option=config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.SUBMODULE_STACK_ACTIVATION,
        layer_norm_position=config.SUBMODULE_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.SUBMODULE_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.SUBMODULE_STACK_BIAS_FLAG,
    ),
    residual_stack=OptionalStackValues(
        independent_flag=config.RESIDUAL_STACK_INDEPENDENT_FLAG,
        hidden_dim=config.RESIDUAL_STACK_HIDDEN_DIM,
        num_layers=config.RESIDUAL_STACK_NUM_LAYERS,
        last_layer_bias_option=config.RESIDUAL_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.RESIDUAL_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.RESIDUAL_STACK_ACTIVATION,
        layer_norm_position=config.RESIDUAL_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.RESIDUAL_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.RESIDUAL_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.RESIDUAL_STACK_BIAS_FLAG,
    ),
    gate_stack=OptionalStackValues(
        independent_flag=config.GATE_STACK_INDEPENDENT_FLAG,
        hidden_dim=config.GATE_STACK_HIDDEN_DIM,
        num_layers=config.GATE_STACK_NUM_LAYERS,
        last_layer_bias_option=config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.GATE_STACK_ACTIVATION,
        layer_norm_position=config.GATE_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.GATE_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.GATE_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.GATE_STACK_BIAS_FLAG,
    ),
    halting_stack=OptionalStackValues(
        independent_flag=config.HALTING_STACK_INDEPENDENT_FLAG,
        hidden_dim=config.HALTING_STACK_HIDDEN_DIM,
        num_layers=config.HALTING_STACK_NUM_LAYERS,
        last_layer_bias_option=config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.HALTING_STACK_ACTIVATION,
        layer_norm_position=config.HALTING_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.HALTING_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.HALTING_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.HALTING_STACK_BIAS_FLAG,
    ),
    memory_stack=OptionalStackValues(
        independent_flag=config.MEMORY_STACK_INDEPENDENT_FLAG,
        hidden_dim=config.MEMORY_STACK_HIDDEN_DIM,
        num_layers=config.MEMORY_STACK_NUM_LAYERS,
        last_layer_bias_option=config.MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.MEMORY_STACK_ACTIVATION,
        layer_norm_position=config.MEMORY_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.MEMORY_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.MEMORY_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.MEMORY_STACK_BIAS_FLAG,
    ),
    recurrent_gate_stack=OptionalStackValues(
        independent_flag=config.RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
        hidden_dim=config.RECURRENT_GATE_STACK_HIDDEN_DIM,
        num_layers=config.RECURRENT_GATE_STACK_NUM_LAYERS,
        last_layer_bias_option=config.RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=(
            config.RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        activation=config.RECURRENT_GATE_STACK_ACTIVATION,
        layer_norm_position=config.RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
        residual_connection_option=(
            config.RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_model_flag=config.RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.RECURRENT_GATE_STACK_BIAS_FLAG,
    ),
    recurrent_halting_stack=OptionalStackValues(
        independent_flag=config.RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        hidden_dim=config.RECURRENT_HALTING_STACK_HIDDEN_DIM,
        num_layers=config.RECURRENT_HALTING_STACK_NUM_LAYERS,
        last_layer_bias_option=config.RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=(
            config.RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        activation=config.RECURRENT_HALTING_STACK_ACTIVATION,
        layer_norm_position=config.RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        residual_connection_option=(
            config.RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_model_flag=config.RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.RECURRENT_HALTING_STACK_BIAS_FLAG,
    ),
    control=ControlDefaultValues(
        stack_gate_flag=config.STACK_GATE_FLAG,
        gate_option=config.GATE_OPTION,
        gate_activation=config.GATE_ACTIVATION,
        stack_halting_flag=config.STACK_HALTING_FLAG,
        halting_option=config.HALTING_OPTION,
        halting_threshold=config.HALTING_THRESHOLD,
        halting_dropout=config.HALTING_DROPOUT,
        halting_hidden_state_mode=config.HALTING_HIDDEN_STATE_MODE,
        memory_flag=config.MEMORY_FLAG,
        memory_option=config.MEMORY_OPTION,
        memory_position_option=config.MEMORY_POSITION_OPTION,
        memory_test_time_training_learning_rate=(
            config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        ),
        memory_test_time_training_num_inner_steps=(
            config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
        ),
        recurrent_flag=config.RECURRENT_FLAG,
        recurrent_max_steps=config.RECURRENT_MAX_STEPS,
        recurrent_initial_iterations=config.RECURRENT_INITIAL_ITERATIONS,
        recurrent_gradient_transition_count=(
            config.RECURRENT_GRADIENT_TRANSITION_COUNT
        ),
        recurrent_iteration_increment=config.RECURRENT_ITERATION_INCREMENT,
        recurrent_forward_calls_before_iteration_increment=(
            config.RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT
        ),
        recurrent_smooth_iteration_growth_flag=(
            config.RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG
        ),
        recurrent_layer_norm_position=config.RECURRENT_LAYER_NORM_POSITION,
        recurrent_stack_gate_flag=config.RECURRENT_STACK_GATE_FLAG,
        recurrent_gate_option=config.RECURRENT_GATE_OPTION,
        recurrent_gate_activation=config.RECURRENT_GATE_ACTIVATION,
        recurrent_stack_halting_flag=config.RECURRENT_STACK_HALTING_FLAG,
        recurrent_halting_option=config.RECURRENT_HALTING_OPTION,
        recurrent_halting_threshold=config.RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout=config.RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode=(
            config.RECURRENT_HALTING_HIDDEN_STATE_MODE
        ),
    ),
)


def read_runtime_dimensions(
    reader: RuntimeOverrideReader,
) -> RuntimeDimensionsValues:
    defaults = _DEFAULT_VALUES.dimensions
    return RuntimeDimensionsValues(
        batch_size=reader.integer("batch_size", defaults.batch_size),
        learning_rate=reader.floating("learning_rate", defaults.learning_rate),
        input_dim=reader.integer("input_dim", defaults.input_dim),
        hidden_dim=reader.integer("hidden_dim", defaults.hidden_dim),
        output_dim=reader.integer("output_dim", defaults.output_dim),
    )


def read_main_stack_measures(
    reader: RuntimeOverrideReader,
) -> MainStackMeasureValues:
    defaults = _DEFAULT_VALUES.main_stack
    return MainStackMeasureValues(
        num_layers=reader.integer("stack_num_layers", defaults.num_layers),
        dropout_probability=reader.floating(
            "stack_dropout_probability", defaults.dropout_probability
        ),
    )


def read_main_stack_values(
    reader: RuntimeOverrideReader,
    measures: MainStackMeasureValues,
) -> MainStackValues:
    defaults = _DEFAULT_VALUES.main_stack
    return MainStackValues(
        bias_flag=reader.boolean("stack_bias_flag", defaults.bias_flag),
        layer_norm_position=reader.enum(
            "layer_norm_position",
            defaults.layer_norm_position,
            LayerNormPositionOptions,
        ),
        num_layers=measures.num_layers,
        activation=reader.enum(
            "stack_activation", defaults.activation, ActivationOptions
        ),
        residual_connection_option=reader.optional_residual(
            "stack_residual_connection_option",
            defaults.residual_connection_option,
        ),
        residual_model_flag=reader.boolean(
            "stack_residual_model_flag", defaults.residual_model_flag
        ),
        dropout_probability=measures.dropout_probability,
        last_layer_bias_option=reader.enum(
            "stack_last_layer_bias_option",
            defaults.last_layer_bias_option,
            LastLayerBiasOptions,
        ),
        apply_output_pipeline_flag=reader.boolean(
            "stack_apply_output_pipeline_flag",
            defaults.apply_output_pipeline_flag,
        ),
    )


def read_submodule_stack_measures(
    reader: RuntimeOverrideReader,
) -> SubmoduleStackMeasureValues:
    defaults = _DEFAULT_VALUES.submodule_stack
    return SubmoduleStackMeasureValues(
        hidden_dim=reader.integer("submodule_stack_hidden_dim", defaults.hidden_dim),
        num_layers=reader.integer("submodule_stack_num_layers", defaults.num_layers),
        dropout_probability=reader.floating(
            "submodule_stack_dropout_probability", defaults.dropout_probability
        ),
    )


def read_submodule_stack_values(
    reader: RuntimeOverrideReader,
    measures: SubmoduleStackMeasureValues,
) -> SubmoduleStackValues:
    defaults = _DEFAULT_VALUES.submodule_stack
    return SubmoduleStackValues(
        hidden_dim=measures.hidden_dim,
        num_layers=measures.num_layers,
        last_layer_bias_option=reader.enum(
            "submodule_stack_last_layer_bias_option",
            defaults.last_layer_bias_option,
            LastLayerBiasOptions,
        ),
        apply_output_pipeline_flag=reader.boolean(
            "submodule_stack_apply_output_pipeline_flag",
            defaults.apply_output_pipeline_flag,
        ),
        activation=reader.enum(
            "submodule_stack_activation", defaults.activation, ActivationOptions
        ),
        layer_norm_position=reader.enum(
            "submodule_stack_layer_norm_position",
            defaults.layer_norm_position,
            LayerNormPositionOptions,
        ),
        residual_connection_option=reader.optional_residual(
            "submodule_stack_residual_connection_option",
            defaults.residual_connection_option,
        ),
        residual_model_flag=reader.boolean(
            "submodule_stack_residual_model_flag", defaults.residual_model_flag
        ),
        dropout_probability=measures.dropout_probability,
        bias_flag=reader.boolean("submodule_stack_bias_flag", defaults.bias_flag),
    )


def read_residual_stack_measures(
    reader: RuntimeOverrideReader,
) -> OptionalStackMeasures:
    defaults = _DEFAULT_VALUES.residual_stack
    fields = _RESIDUAL_STACK_FIELDS
    return OptionalStackMeasures(
        hidden_dim=reader.optional_integer(fields.hidden_dim, defaults.hidden_dim),
        num_layers=reader.optional_integer(fields.num_layers, defaults.num_layers),
        dropout_probability=reader.optional_floating(
            fields.dropout_probability, defaults.dropout_probability
        ),
    )


def read_residual_stack_values(
    reader: RuntimeOverrideReader,
    measures: OptionalStackMeasures,
) -> OptionalStackInput:
    defaults = _DEFAULT_VALUES.residual_stack
    fields = _RESIDUAL_STACK_FIELDS
    return OptionalStackInput(
        fields=fields,
        values=OptionalStackValues(
            independent_flag=reader.boolean(
                fields.independent_flag, defaults.independent_flag
            ),
            hidden_dim=measures.hidden_dim,
            num_layers=measures.num_layers,
            activation=reader.optional_enum(
                fields.activation, defaults.activation, ActivationOptions
            ),
            layer_norm_position=reader.optional_enum(
                fields.layer_norm_position,
                defaults.layer_norm_position,
                LayerNormPositionOptions,
            ),
            residual_connection_option=reader.optional_residual(
                fields.residual_connection_option,
                defaults.residual_connection_option,
            ),
            residual_model_flag=reader.boolean(
                fields.residual_model_flag, defaults.residual_model_flag
            ),
            dropout_probability=measures.dropout_probability,
            last_layer_bias_option=reader.optional_enum(
                fields.last_layer_bias_option,
                defaults.last_layer_bias_option,
                LastLayerBiasOptions,
            ),
            apply_output_pipeline_flag=reader.optional_boolean(
                fields.apply_output_pipeline_flag,
                defaults.apply_output_pipeline_flag,
            ),
            bias_flag=reader.optional_boolean(fields.bias_flag, defaults.bias_flag),
        ),
    )


def _read_optional_stack(
    reader: RuntimeOverrideReader,
    fields: OptionalStackFields,
    defaults: OptionalStackValues,
) -> OptionalStackInput:
    return OptionalStackInput(
        fields=fields,
        values=OptionalStackValues(
            independent_flag=reader.boolean(
                fields.independent_flag, defaults.independent_flag
            ),
            hidden_dim=reader.optional_integer(fields.hidden_dim, defaults.hidden_dim),
            num_layers=reader.optional_integer(fields.num_layers, defaults.num_layers),
            last_layer_bias_option=reader.optional_enum(
                fields.last_layer_bias_option,
                defaults.last_layer_bias_option,
                LastLayerBiasOptions,
            ),
            apply_output_pipeline_flag=reader.optional_boolean(
                fields.apply_output_pipeline_flag,
                defaults.apply_output_pipeline_flag,
            ),
            activation=reader.optional_enum(
                fields.activation, defaults.activation, ActivationOptions
            ),
            layer_norm_position=reader.optional_enum(
                fields.layer_norm_position,
                defaults.layer_norm_position,
                LayerNormPositionOptions,
            ),
            residual_connection_option=reader.optional_residual(
                fields.residual_connection_option,
                defaults.residual_connection_option,
            ),
            residual_model_flag=reader.boolean(
                fields.residual_model_flag, defaults.residual_model_flag
            ),
            dropout_probability=reader.optional_floating(
                fields.dropout_probability, defaults.dropout_probability
            ),
            bias_flag=reader.optional_boolean(fields.bias_flag, defaults.bias_flag),
        ),
    )


def read_gate_stack_values(reader: RuntimeOverrideReader) -> OptionalStackInput:
    return _read_optional_stack(reader, _GATE_STACK_FIELDS, _DEFAULT_VALUES.gate_stack)


def read_halting_stack_values(reader: RuntimeOverrideReader) -> OptionalStackInput:
    return _read_optional_stack(
        reader, _HALTING_STACK_FIELDS, _DEFAULT_VALUES.halting_stack
    )


def read_memory_stack_values(reader: RuntimeOverrideReader) -> OptionalStackInput:
    return _read_optional_stack(
        reader, _MEMORY_STACK_FIELDS, _DEFAULT_VALUES.memory_stack
    )


def read_recurrent_gate_stack_values(
    reader: RuntimeOverrideReader,
) -> OptionalStackInput:
    return _read_optional_stack(
        reader,
        _RECURRENT_GATE_STACK_FIELDS,
        _DEFAULT_VALUES.recurrent_gate_stack,
    )


def read_recurrent_halting_stack_values(
    reader: RuntimeOverrideReader,
) -> OptionalStackInput:
    return _read_optional_stack(
        reader,
        _RECURRENT_HALTING_STACK_FIELDS,
        _DEFAULT_VALUES.recurrent_halting_stack,
    )


def read_gate_condition(reader: RuntimeOverrideReader) -> GateConditionValues:
    defaults = _DEFAULT_VALUES.control
    return GateConditionValues(
        enabled=reader.boolean("stack_gate_flag", defaults.stack_gate_flag),
        option=reader.optional_enum(
            "gate_option", defaults.gate_option, LayerGateOptions
        ),
    )


def read_gate_activation(reader: RuntimeOverrideReader) -> ActivationOptions | None:
    defaults = _DEFAULT_VALUES.control
    return reader.optional_enum(
        "gate_activation", defaults.gate_activation, ActivationOptions
    )


def read_halting_measures(reader: RuntimeOverrideReader) -> HaltingMeasureValues:
    defaults = _DEFAULT_VALUES.control
    return HaltingMeasureValues(
        threshold=reader.floating("halting_threshold", defaults.halting_threshold),
        dropout_probability=reader.floating(
            "halting_dropout", defaults.halting_dropout
        ),
    )


def read_halting_values(reader: RuntimeOverrideReader) -> HaltingValues:
    defaults = _DEFAULT_VALUES.control
    return HaltingValues(
        enabled=reader.boolean("stack_halting_flag", defaults.stack_halting_flag),
        hidden_state_mode=reader.enum(
            "halting_hidden_state_mode",
            defaults.halting_hidden_state_mode,
            HaltingHiddenStateModeOptions,
        ),
    )


def read_memory_measures(reader: RuntimeOverrideReader) -> MemoryMeasureValues:
    defaults = _DEFAULT_VALUES.control
    return MemoryMeasureValues(
        learning_rate=reader.optional_floating(
            "memory_test_time_training_learning_rate",
            defaults.memory_test_time_training_learning_rate,
        ),
        num_inner_steps=reader.optional_integer(
            "memory_test_time_training_num_inner_steps",
            defaults.memory_test_time_training_num_inner_steps,
        ),
    )


def read_memory_values(reader: RuntimeOverrideReader) -> MemoryValues:
    defaults = _DEFAULT_VALUES.control
    return MemoryValues(
        enabled=reader.boolean("memory_flag", defaults.memory_flag),
        implementation=reader.implementation(
            "memory_option", defaults.memory_option, DynamicMemoryConfig
        ),
        position=reader.enum(
            "memory_position_option",
            defaults.memory_position_option,
            MemoryPositionOptions,
        ),
    )


def read_recurrent_max_steps(reader: RuntimeOverrideReader) -> int:
    return reader.integer(
        "recurrent_max_steps", _DEFAULT_VALUES.control.recurrent_max_steps
    )


def read_recurrent_gate_condition(
    reader: RuntimeOverrideReader,
) -> GateConditionValues:
    defaults = _DEFAULT_VALUES.control
    return GateConditionValues(
        enabled=reader.boolean(
            "recurrent_stack_gate_flag", defaults.recurrent_stack_gate_flag
        ),
        option=reader.optional_enum(
            "recurrent_gate_option",
            defaults.recurrent_gate_option,
            LayerGateOptions,
        ),
    )


def read_recurrent_halting_measures(
    reader: RuntimeOverrideReader,
) -> HaltingMeasureValues:
    defaults = _DEFAULT_VALUES.control
    return HaltingMeasureValues(
        threshold=reader.floating(
            "recurrent_halting_threshold", defaults.recurrent_halting_threshold
        ),
        dropout_probability=reader.floating(
            "recurrent_halting_dropout", defaults.recurrent_halting_dropout
        ),
    )


def read_recurrence_values(reader: RuntimeOverrideReader) -> RecurrenceValues:
    defaults = _DEFAULT_VALUES.control
    return RecurrenceValues(
        enabled=reader.boolean("recurrent_flag", defaults.recurrent_flag),
        initial_iterations=reader.optional_integer(
            "recurrent_initial_iterations", defaults.recurrent_initial_iterations
        ),
        gradient_transition_count=reader.optional_integer(
            "recurrent_gradient_transition_count",
            defaults.recurrent_gradient_transition_count,
        ),
        iteration_increment=reader.integer(
            "recurrent_iteration_increment", defaults.recurrent_iteration_increment
        ),
        forward_calls_before_iteration_increment=reader.integer(
            "recurrent_forward_calls_before_iteration_increment",
            defaults.recurrent_forward_calls_before_iteration_increment,
        ),
        smooth_iteration_growth_flag=reader.boolean(
            "recurrent_smooth_iteration_growth_flag",
            defaults.recurrent_smooth_iteration_growth_flag,
        ),
        layer_norm_position=reader.enum(
            "recurrent_layer_norm_position",
            defaults.recurrent_layer_norm_position,
            LayerNormPositionOptions,
        ),
        gate_activation=reader.optional_enum(
            "recurrent_gate_activation",
            defaults.recurrent_gate_activation,
            ActivationOptions,
        ),
        halting_enabled=reader.boolean(
            "recurrent_stack_halting_flag",
            defaults.recurrent_stack_halting_flag,
        ),
        halting_hidden_state_mode=reader.enum(
            "recurrent_halting_hidden_state_mode",
            defaults.recurrent_halting_hidden_state_mode,
            HaltingHiddenStateModeOptions,
        ),
    )


def read_halting_implementations(
    reader: RuntimeOverrideReader,
) -> HaltingImplementationValues:
    defaults = _DEFAULT_VALUES.control
    return HaltingImplementationValues(
        main=reader.implementation(
            "halting_option", defaults.halting_option, HaltingConfig
        ),
        recurrent=reader.implementation(
            "recurrent_halting_option",
            defaults.recurrent_halting_option,
            HaltingConfig,
        ),
    )


def _accepted_keys() -> frozenset[str]:
    reader = RuntimeOverrideReader({})
    read_runtime_dimensions(reader)
    main_measures = read_main_stack_measures(reader)
    read_main_stack_values(reader, main_measures)
    submodule_measures = read_submodule_stack_measures(reader)
    read_submodule_stack_values(reader, submodule_measures)
    residual_measures = read_residual_stack_measures(reader)
    read_residual_stack_values(reader, residual_measures)
    read_gate_stack_values(reader)
    read_halting_stack_values(reader)
    read_memory_stack_values(reader)
    read_recurrent_gate_stack_values(reader)
    read_recurrent_halting_stack_values(reader)
    read_gate_condition(reader)
    read_gate_activation(reader)
    read_halting_measures(reader)
    read_halting_values(reader)
    read_memory_measures(reader)
    read_memory_values(reader)
    read_recurrent_max_steps(reader)
    read_recurrent_gate_condition(reader)
    read_recurrent_halting_measures(reader)
    read_recurrence_values(reader)
    read_halting_implementations(reader)
    return frozenset(reader.accepted)


_ACCEPTED_KEYS = _accepted_keys()


def runtime_override_reader(
    overrides: Mapping[str, object] | None,
) -> RuntimeOverrideReader:
    reader = RuntimeOverrideReader({} if overrides is None else overrides)
    reader.reject_unknown(_ACCEPTED_KEYS)
    return reader
