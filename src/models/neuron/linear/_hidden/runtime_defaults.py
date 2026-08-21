from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final, cast

from emperor.layers import LastLayerBiasOptions
from models.neuron.linear._hidden._runtime_default_values import (
    GateConditionValues,
    HaltingMeasureValues,
    HaltingValues,
    MainStackValues,
    MemoryMeasureValues,
    MemoryValues,
    OptionalStackInput,
    OptionalStackMeasures,
    RecurrenceValues,
    RuntimeDimensionsValues,
    RuntimeOverrideReader,
    SubmoduleStackMeasureValues,
    SubmoduleStackValues,
    read_gate_activation,
    read_gate_condition,
    read_gate_stack_values,
    read_halting_implementations,
    read_halting_measures,
    read_halting_stack_values,
    read_halting_values,
    read_main_stack_measures,
    read_main_stack_values,
    read_memory_measures,
    read_memory_stack_values,
    read_memory_values,
    read_recurrence_values,
    read_recurrent_gate_condition,
    read_recurrent_gate_stack_values,
    read_recurrent_halting_measures,
    read_recurrent_halting_stack_values,
    read_recurrent_max_steps,
    read_residual_stack_measures,
    read_residual_stack_values,
    read_runtime_dimensions,
    read_submodule_stack_measures,
    read_submodule_stack_values,
    runtime_override_reader,
)
from models.neuron.linear._hidden.runtime_options import (
    ControllerStackOptions,
    GateOptions,
    HaltingOptions,
    MainStackOptions,
    MemoryOptions,
    RecurrenceOptions,
    RuntimeOptions,
)
from models.neuron.linear._residual import (
    ResidualStackOptions,
    ResidualStackSource,
    resolve_residual_stack_options,
)

_PACKAGE_NAME = "models.neuron.linear._hidden"


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


def _runtime_dimensions(
    reader: RuntimeOverrideReader,
) -> RuntimeDimensionsValues:
    dimensions = read_runtime_dimensions(reader)
    for key, value in (
        ("batch_size", dimensions.batch_size),
        ("learning_rate", dimensions.learning_rate),
        ("input_dim", dimensions.input_dim),
        ("hidden_dim", dimensions.hidden_dim),
        ("output_dim", dimensions.output_dim),
    ):
        _positive(key, value)
    return dimensions


def _main_stack(reader: RuntimeOverrideReader) -> MainStackOptions:
    measures = read_main_stack_measures(reader)
    _positive("stack_num_layers", measures.num_layers)
    _probability("stack_dropout_probability", measures.dropout_probability)
    values: MainStackValues = read_main_stack_values(reader, measures)
    return MainStackOptions(
        bias_flag=values.bias_flag,
        layer_norm_position=values.layer_norm_position,
        num_layers=values.num_layers,
        activation=values.activation,
        residual_connection_option=values.residual_connection_option,
        residual_model_flag=values.residual_model_flag,
        dropout_probability=values.dropout_probability,
        last_layer_bias_option=values.last_layer_bias_option,
        apply_output_postprocessing_flag=values.apply_output_postprocessing_flag,
    )


def _submodule_stack(reader: RuntimeOverrideReader) -> ControllerStackOptions:
    measures: SubmoduleStackMeasureValues = read_submodule_stack_measures(reader)
    _positive("submodule_stack_hidden_dim", measures.hidden_dim)
    _positive("submodule_stack_num_layers", measures.num_layers)
    _probability(
        "submodule_stack_dropout_probability",
        measures.dropout_probability,
    )
    values: SubmoduleStackValues = read_submodule_stack_values(reader, measures)
    return ControllerStackOptions(
        hidden_dim=values.hidden_dim,
        num_layers=values.num_layers,
        last_layer_bias_option=values.last_layer_bias_option,
        apply_output_postprocessing_flag=values.apply_output_postprocessing_flag,
        activation=values.activation,
        layer_norm_position=values.layer_norm_position,
        residual_connection_option=values.residual_connection_option,
        residual_model_flag=values.residual_model_flag,
        dropout_probability=values.dropout_probability,
        bias_flag=values.bias_flag,
    )


def _validate_optional_stack_measures(
    stack: OptionalStackInput,
) -> None:
    fields = stack.fields
    values = stack.values
    if values.hidden_dim is not None:
        _positive(fields.hidden_dim, values.hidden_dim)
    if values.num_layers is not None:
        _positive(fields.num_layers, values.num_layers)
    if values.dropout_probability is not None:
        _probability(fields.dropout_probability, values.dropout_probability)


def _resolved_controller_stack(
    stack: OptionalStackInput,
    defaults: ControllerStackOptions,
) -> ControllerStackOptions:
    _validate_optional_stack_measures(stack)
    values = stack.values
    if not values.independent_flag:
        return defaults
    return ControllerStackOptions(
        hidden_dim=(
            defaults.hidden_dim if values.hidden_dim is None else values.hidden_dim
        ),
        num_layers=(
            defaults.num_layers if values.num_layers is None else values.num_layers
        ),
        last_layer_bias_option=(
            defaults.last_layer_bias_option
            if values.last_layer_bias_option is None
            else values.last_layer_bias_option
        ),
        apply_output_postprocessing_flag=(
            defaults.apply_output_postprocessing_flag
            if values.apply_output_postprocessing_flag is None
            else values.apply_output_postprocessing_flag
        ),
        activation=(
            defaults.activation if values.activation is None else values.activation
        ),
        layer_norm_position=(
            defaults.layer_norm_position
            if values.layer_norm_position is None
            else values.layer_norm_position
        ),
        residual_connection_option=(
            defaults.residual_connection_option
            if values.residual_connection_option is None
            else values.residual_connection_option
        ),
        residual_model_flag=values.residual_model_flag,
        dropout_probability=(
            defaults.dropout_probability
            if values.dropout_probability is None
            else values.dropout_probability
        ),
        bias_flag=(
            defaults.bias_flag if values.bias_flag is None else values.bias_flag
        ),
        residual_stack_options=defaults.residual_stack_options,
    )


def _residual_stack(
    reader: RuntimeOverrideReader,
    submodule_stack: ControllerStackOptions,
) -> ResidualStackOptions:
    measures: OptionalStackMeasures = read_residual_stack_measures(reader)
    if measures.hidden_dim is not None:
        _positive("residual_stack_hidden_dim", measures.hidden_dim)
    if measures.num_layers is not None:
        _positive("residual_stack_num_layers", measures.num_layers)
    if measures.dropout_probability is not None:
        _probability(
            "residual_stack_dropout_probability",
            measures.dropout_probability,
        )
    stack = read_residual_stack_values(reader, measures)
    values = stack.values
    return resolve_residual_stack_options(
        ResidualStackSource(
            independent_flag=values.independent_flag,
            hidden_dim=values.hidden_dim,
            num_layers=values.num_layers,
            activation=values.activation,
            layer_norm_position=values.layer_norm_position,
            residual_connection_option=values.residual_connection_option,
            residual_model_flag=values.residual_model_flag,
            dropout_probability=values.dropout_probability,
            last_layer_bias_option=values.last_layer_bias_option,
            apply_output_postprocessing_flag=values.apply_output_postprocessing_flag,
            bias_flag=values.bias_flag,
        ),
        submodule_stack,
    )


@dataclass(frozen=True, slots=True)
class _RuntimeStacks:
    main: MainStackOptions
    submodule: ControllerStackOptions
    residual: ResidualStackOptions
    gate: ControllerStackOptions
    halting: ControllerStackOptions
    memory: ControllerStackOptions
    recurrent_gate: ControllerStackOptions
    recurrent_halting: ControllerStackOptions


def _runtime_stacks(reader: RuntimeOverrideReader) -> _RuntimeStacks:
    main_stack = _main_stack(reader)
    submodule_stack = _submodule_stack(reader)
    residual_stack = _residual_stack(reader, submodule_stack)
    main_stack = replace(main_stack, residual_stack_options=residual_stack)
    submodule_stack = replace(
        submodule_stack,
        residual_stack_options=residual_stack,
    )
    gate_stack = _resolved_controller_stack(
        read_gate_stack_values(reader),
        submodule_stack,
    )
    halting_stack = _resolved_controller_stack(
        read_halting_stack_values(reader),
        replace(
            submodule_stack,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
        ),
    )
    memory_stack = _resolved_controller_stack(
        read_memory_stack_values(reader),
        submodule_stack,
    )
    recurrent_gate_stack = _resolved_controller_stack(
        read_recurrent_gate_stack_values(reader),
        gate_stack,
    )
    recurrent_halting_stack = _resolved_controller_stack(
        read_recurrent_halting_stack_values(reader),
        halting_stack,
    )
    return _RuntimeStacks(
        main=main_stack,
        submodule=submodule_stack,
        residual=residual_stack,
        gate=gate_stack,
        halting=halting_stack,
        memory=memory_stack,
        recurrent_gate=recurrent_gate_stack,
        recurrent_halting=recurrent_halting_stack,
    )


def _gate_options(
    reader: RuntimeOverrideReader,
    stack: ControllerStackOptions,
) -> GateOptions:
    condition: GateConditionValues = read_gate_condition(reader)
    if condition.enabled and condition.option is None:
        raise ValueError(
            f"{_PACKAGE_NAME}: 'gate_option' must be set when 'stack_gate_flag' is True"
        )
    return GateOptions(
        enabled=condition.enabled,
        option=condition.option,
        activation=read_gate_activation(reader),
        stack=stack,
    )


def _halting_options(
    reader: RuntimeOverrideReader,
    stack: ControllerStackOptions,
) -> HaltingOptions:
    measures: HaltingMeasureValues = read_halting_measures(reader)
    _threshold("halting_threshold", measures.threshold)
    _probability("halting_dropout", measures.dropout_probability)
    values: HaltingValues = read_halting_values(reader)
    return HaltingOptions(
        enabled=values.enabled,
        threshold=measures.threshold,
        dropout_probability=measures.dropout_probability,
        hidden_state_mode=values.hidden_state_mode,
        stack=stack,
    )


def _memory_options(
    reader: RuntimeOverrideReader,
    stack: ControllerStackOptions,
) -> MemoryOptions:
    measures: MemoryMeasureValues = read_memory_measures(reader)
    if measures.learning_rate is not None:
        _positive(
            "memory_test_time_training_learning_rate",
            measures.learning_rate,
        )
    if measures.num_inner_steps is not None:
        _positive(
            "memory_test_time_training_num_inner_steps",
            measures.num_inner_steps,
        )
    values: MemoryValues = read_memory_values(reader)
    return MemoryOptions(
        enabled=values.enabled,
        implementation=values.implementation,
        position=values.position,
        test_time_training_learning_rate=measures.learning_rate,
        test_time_training_num_inner_steps=measures.num_inner_steps,
        stack=stack,
    )


def _recurrence_options(
    reader: RuntimeOverrideReader,
    gate_stack: ControllerStackOptions,
    halting_stack: ControllerStackOptions,
) -> RecurrenceOptions:
    max_steps = read_recurrent_max_steps(reader)
    _positive("recurrent_max_steps", max_steps)
    gate_condition = read_recurrent_gate_condition(reader)
    if gate_condition.enabled and gate_condition.option is None:
        raise ValueError(
            f"{_PACKAGE_NAME}: 'recurrent_gate_option' must be set when "
            "'recurrent_stack_gate_flag' is True"
        )
    halting_measures = read_recurrent_halting_measures(reader)
    _threshold("recurrent_halting_threshold", halting_measures.threshold)
    _probability(
        "recurrent_halting_dropout",
        halting_measures.dropout_probability,
    )
    values: RecurrenceValues = read_recurrence_values(reader)
    return RecurrenceOptions(
        enabled=values.enabled,
        max_steps=max_steps,
        initial_iterations=cast(int, values.initial_iterations),
        gradient_transition_count=values.gradient_transition_count,
        no_gradient_transition_count=values.no_gradient_transition_count,
        iteration_increment=values.iteration_increment,
        forward_calls_before_iteration_increment=(
            values.forward_calls_before_iteration_increment
        ),
        smooth_iteration_growth_flag=values.smooth_iteration_growth_flag,
        layer_norm_position=values.layer_norm_position,
        gate=GateOptions(
            enabled=gate_condition.enabled,
            option=gate_condition.option,
            activation=values.gate_activation,
            stack=gate_stack,
        ),
        halting=HaltingOptions(
            enabled=values.halting_enabled,
            threshold=halting_measures.threshold,
            dropout_probability=halting_measures.dropout_probability,
            hidden_state_mode=values.halting_hidden_state_mode,
            stack=halting_stack,
        ),
    )


def runtime_from_flat(
    overrides: Mapping[str, object] | None = None,
) -> RuntimeOptions:
    reader = runtime_override_reader(overrides)
    dimensions = _runtime_dimensions(reader)
    stacks = _runtime_stacks(reader)
    gate = _gate_options(reader, stacks.gate)
    halting = _halting_options(reader, stacks.halting)
    memory = _memory_options(reader, stacks.memory)
    recurrence = _recurrence_options(
        reader,
        stacks.recurrent_gate,
        stacks.recurrent_halting,
    )
    implementations = read_halting_implementations(reader)
    return RuntimeOptions(
        batch_size=dimensions.batch_size,
        learning_rate=dimensions.learning_rate,
        input_dim=dimensions.input_dim,
        hidden_dim=dimensions.hidden_dim,
        output_dim=dimensions.output_dim,
        stack=stacks.main,
        gate=gate,
        halting=halting,
        memory=memory,
        recurrence=recurrence,
        halting_option=implementations.main,
        recurrent_halting_option=implementations.recurrent,
    )


DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_flat()
