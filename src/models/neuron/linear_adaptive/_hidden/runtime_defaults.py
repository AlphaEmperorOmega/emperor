from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final

from emperor.layers import LastLayerBiasOptions
from models.neuron.linear_adaptive._hidden._runtime_default_values import (
    _ADAPTIVE_GENERATOR_STACK_FIELDS,
    _BIAS_FIELDS,
    _BIAS_GENERATOR_STACK_FIELDS,
    _CONTROL_FIELDS,
    _DIAGONAL_FIELDS,
    _DIAGONAL_GENERATOR_STACK_FIELDS,
    _DIMENSION_FIELDS,
    _GATE_STACK_FIELDS,
    _HALTING_STACK_FIELDS,
    _INPUT_PROJECTION_FIELDS,
    _MAIN_STACK_FIELDS,
    _MASK_FIELDS,
    _MASK_GENERATOR_STACK_FIELDS,
    _MEMORY_STACK_FIELDS,
    _OUTPUT_PROJECTION_FIELDS,
    _RECURRENT_GATE_STACK_FIELDS,
    _RECURRENT_HALTING_STACK_FIELDS,
    _RESIDUAL_STACK_FIELDS,
    _SUBMODULE_STACK_FIELDS,
    _WEIGHT_FIELDS,
    _WEIGHT_GENERATOR_STACK_FIELDS,
    BiasValues,
    DiagonalValues,
    MaskValues,
    OptionalStackFields,
    OptionalStackValues,
    ProjectionValues,
    RuntimeDefaultValues,
    StackFields,
    StackValues,
    WeightValues,
    runtime_default_values,
)
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
from models.neuron.linear_adaptive._residual import (
    ResidualStackOptions,
    ResidualStackSource,
    resolve_residual_stack_options,
)

_PACKAGE = "models.neuron.linear_adaptive._hidden"


def _positive(key: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{_PACKAGE}: runtime key {key!r} must be positive")


def _nonnegative(key: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{_PACKAGE}: runtime key {key!r} must be non-negative")


def _probability(key: str, value: int | float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{_PACKAGE}: runtime key {key!r} must be between 0 and 1")


def _validate_required_positive(values: RuntimeDefaultValues) -> None:
    dimensions = values.dimensions
    for field, value in (
        (_DIMENSION_FIELDS.batch_size, dimensions.batch_size),
        (_DIMENSION_FIELDS.learning_rate, dimensions.learning_rate),
        (_DIMENSION_FIELDS.input_dim, dimensions.input_dim),
        (_DIMENSION_FIELDS.hidden_dim, dimensions.hidden_dim),
        (_DIMENSION_FIELDS.output_dim, dimensions.output_dim),
        (_MAIN_STACK_FIELDS.num_layers, values.main_stack.num_layers),
        (_SUBMODULE_STACK_FIELDS.hidden_dim, values.submodule_stack.hidden_dim),
        (_SUBMODULE_STACK_FIELDS.num_layers, values.submodule_stack.num_layers),
        (
            _ADAPTIVE_GENERATOR_STACK_FIELDS.hidden_dim,
            values.adaptive_generator_stack.hidden_dim,
        ),
        (
            _ADAPTIVE_GENERATOR_STACK_FIELDS.num_layers,
            values.adaptive_generator_stack.num_layers,
        ),
        (_CONTROL_FIELDS.recurrent_max_steps, values.control.recurrent_max_steps),
    ):
        _positive(field.key, value)


def _validate_stack_probability(fields: StackFields, values: StackValues) -> None:
    _probability(fields.dropout_probability.key, values.dropout_probability)


def _validate_optional_stack_pattern(
    fields: OptionalStackFields,
    values: OptionalStackValues,
) -> None:
    if values.hidden_dim is not None:
        _positive(fields.hidden_dim.key, values.hidden_dim)
    if values.num_layers is not None:
        _positive(fields.num_layers.key, values.num_layers)
    if values.dropout_probability is not None:
        _probability(fields.dropout_probability.key, values.dropout_probability)


def _validate_stack_patterns(values: RuntimeDefaultValues) -> None:
    for fields, stack_values in (
        (_MAIN_STACK_FIELDS, values.main_stack),
        (_SUBMODULE_STACK_FIELDS, values.submodule_stack),
        (_ADAPTIVE_GENERATOR_STACK_FIELDS, values.adaptive_generator_stack),
    ):
        _validate_stack_probability(fields, stack_values)
    for fields, stack_values in (
        (_RESIDUAL_STACK_FIELDS, values.residual_stack),
        (_GATE_STACK_FIELDS, values.gate_stack),
        (_HALTING_STACK_FIELDS, values.halting_stack),
        (_MEMORY_STACK_FIELDS, values.memory_stack),
        (_RECURRENT_GATE_STACK_FIELDS, values.recurrent_gate_stack),
        (_RECURRENT_HALTING_STACK_FIELDS, values.recurrent_halting_stack),
        (_WEIGHT_GENERATOR_STACK_FIELDS, values.weight_generator_stack),
        (_BIAS_GENERATOR_STACK_FIELDS, values.bias_generator_stack),
        (_DIAGONAL_GENERATOR_STACK_FIELDS, values.diagonal_generator_stack),
        (_MASK_GENERATOR_STACK_FIELDS, values.mask_generator_stack),
    ):
        _validate_optional_stack_pattern(fields, stack_values)


def _validate_probabilities(values: RuntimeDefaultValues) -> None:
    control = values.control
    mask = values.mask
    input_projection = values.input_projection
    output_projection = values.output_projection
    for field, value in (
        (_CONTROL_FIELDS.halting_dropout, control.halting_dropout),
        (_CONTROL_FIELDS.halting_threshold, control.halting_threshold),
        (_CONTROL_FIELDS.recurrent_halting_dropout, control.recurrent_halting_dropout),
        (
            _CONTROL_FIELDS.recurrent_halting_threshold,
            control.recurrent_halting_threshold,
        ),
        (_MASK_FIELDS.threshold, mask.threshold),
        (_MASK_FIELDS.floor, mask.floor),
        (_INPUT_PROJECTION_FIELDS.mask_threshold, input_projection.mask_threshold),
        (_INPUT_PROJECTION_FIELDS.mask_floor, input_projection.mask_floor),
        (
            _OUTPUT_PROJECTION_FIELDS.mask_threshold,
            output_projection.mask_threshold,
        ),
        (_OUTPUT_PROJECTION_FIELDS.mask_floor, output_projection.mask_floor),
    ):
        _probability(field.key, value)


def _validate_positive_controls(values: RuntimeDefaultValues) -> None:
    mask = values.mask
    input_projection = values.input_projection
    output_projection = values.output_projection
    for field, value in (
        (_MASK_FIELDS.surrogate_scale, mask.surrogate_scale),
        (_MASK_FIELDS.transition_width, mask.transition_width),
        (
            _INPUT_PROJECTION_FIELDS.mask_surrogate_scale,
            input_projection.mask_surrogate_scale,
        ),
        (
            _INPUT_PROJECTION_FIELDS.mask_transition_width,
            input_projection.mask_transition_width,
        ),
        (
            _OUTPUT_PROJECTION_FIELDS.mask_surrogate_scale,
            output_projection.mask_surrogate_scale,
        ),
        (
            _OUTPUT_PROJECTION_FIELDS.mask_transition_width,
            output_projection.mask_transition_width,
        ),
    ):
        _positive(field.key, value)


def _validate_decay_values(values: RuntimeDefaultValues) -> None:
    input_projection = values.input_projection
    output_projection = values.output_projection
    for field, value in (
        (_WEIGHT_FIELDS.decay_rate, values.weight.decay_rate),
        (_BIAS_FIELDS.decay_rate, values.bias.decay_rate),
        (
            _INPUT_PROJECTION_FIELDS.weight_decay_rate,
            input_projection.weight_decay_rate,
        ),
        (
            _INPUT_PROJECTION_FIELDS.bias_decay_rate,
            input_projection.bias_decay_rate,
        ),
        (
            _OUTPUT_PROJECTION_FIELDS.weight_decay_rate,
            output_projection.weight_decay_rate,
        ),
        (
            _OUTPUT_PROJECTION_FIELDS.bias_decay_rate,
            output_projection.bias_decay_rate,
        ),
    ):
        _nonnegative(field.key, value)
    for field, value in (
        (_WEIGHT_FIELDS.decay_warmup_batches, values.weight.decay_warmup_batches),
        (_BIAS_FIELDS.decay_warmup_batches, values.bias.decay_warmup_batches),
        (
            _INPUT_PROJECTION_FIELDS.weight_decay_warmup_batches,
            input_projection.weight_decay_warmup_batches,
        ),
        (
            _INPUT_PROJECTION_FIELDS.bias_decay_warmup_batches,
            input_projection.bias_decay_warmup_batches,
        ),
        (
            _OUTPUT_PROJECTION_FIELDS.weight_decay_warmup_batches,
            output_projection.weight_decay_warmup_batches,
        ),
        (
            _OUTPUT_PROJECTION_FIELDS.bias_decay_warmup_batches,
            output_projection.bias_decay_warmup_batches,
        ),
    ):
        _nonnegative(field.key, value)


def _validate_control_invariants(values: RuntimeDefaultValues) -> None:
    control = values.control
    if control.memory_learning_rate is not None:
        _positive(
            _CONTROL_FIELDS.memory_learning_rate.key, control.memory_learning_rate
        )
    if control.memory_num_inner_steps is not None:
        _positive(
            _CONTROL_FIELDS.memory_num_inner_steps.key,
            control.memory_num_inner_steps,
        )
    for enabled, option, flag_field, option_field in (
        (
            values.weight.enabled,
            values.weight.option,
            _WEIGHT_FIELDS.enabled,
            _WEIGHT_FIELDS.option,
        ),
        (
            values.bias.enabled,
            values.bias.option,
            _BIAS_FIELDS.enabled,
            _BIAS_FIELDS.option,
        ),
        (
            values.diagonal.enabled,
            values.diagonal.option,
            _DIAGONAL_FIELDS.enabled,
            _DIAGONAL_FIELDS.option,
        ),
        (
            values.mask.enabled,
            values.mask.row_mask_option,
            _MASK_FIELDS.enabled,
            _MASK_FIELDS.row_mask_option,
        ),
    ):
        if enabled and option is None:
            raise ValueError(
                f"{_PACKAGE}: runtime key {option_field.key!r} must be set when "
                f"{flag_field.key!r} is True"
            )
    if control.stack_gate_flag and control.shared_gate_config is not None:
        raise ValueError(
            f"{_PACKAGE}: 'stack_gate_flag' and 'shared_gate_config' are "
            "mutually exclusive"
        )


def _validate_values(values: RuntimeDefaultValues) -> None:
    _validate_required_positive(values)
    _validate_stack_patterns(values)
    _validate_probabilities(values)
    _validate_positive_controls(values)
    _validate_decay_values(values)
    _validate_control_invariants(values)


def _stack(values: StackValues) -> StackOptions:
    return StackOptions(
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


def _resolved_stack(
    values: OptionalStackValues,
    defaults: StackOptions,
) -> StackOptions:
    if not values.independent_flag:
        return defaults
    return replace(
        defaults,
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
    )


def _generator_stack(
    values: OptionalStackValues,
    defaults: StackOptions,
) -> GeneratorStackOptions:
    return GeneratorStackOptions(
        independent=values.independent_flag,
        stack=_resolved_stack(values, defaults),
    )


@dataclass(frozen=True, slots=True)
class _ResolvedStacks:
    main: StackOptions
    submodule: StackOptions
    residual: ResidualStackOptions
    gate: StackOptions
    halting: StackOptions
    memory: StackOptions
    recurrent_gate: StackOptions
    recurrent_halting: StackOptions
    adaptive_generator: StackOptions


def _resolve_stacks(values: RuntimeDefaultValues) -> _ResolvedStacks:
    main = _stack(values.main_stack)
    submodule = _stack(values.submodule_stack)
    residual_values = values.residual_stack
    residual = resolve_residual_stack_options(
        ResidualStackSource(
            independent_flag=residual_values.independent_flag,
            hidden_dim=residual_values.hidden_dim,
            num_layers=residual_values.num_layers,
            activation=residual_values.activation,
            layer_norm_position=residual_values.layer_norm_position,
            residual_connection_option=residual_values.residual_connection_option,
            residual_model_flag=residual_values.residual_model_flag,
            dropout_probability=residual_values.dropout_probability,
            last_layer_bias_option=residual_values.last_layer_bias_option,
            apply_output_postprocessing_flag=(
                residual_values.apply_output_postprocessing_flag
            ),
            bias_flag=residual_values.bias_flag,
        ),
        submodule,
    )
    gate = _resolved_stack(values.gate_stack, submodule)
    halting = _resolved_stack(
        values.halting_stack,
        replace(
            submodule,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
        ),
    )
    memory = _resolved_stack(values.memory_stack, submodule)
    return _ResolvedStacks(
        main=main,
        submodule=submodule,
        residual=residual,
        gate=gate,
        halting=halting,
        memory=memory,
        recurrent_gate=_resolved_stack(values.recurrent_gate_stack, gate),
        recurrent_halting=_resolved_stack(values.recurrent_halting_stack, halting),
        adaptive_generator=_stack(values.adaptive_generator_stack),
    )


@dataclass(frozen=True, slots=True)
class _ControlDefaults:
    gate: GateOptions
    halting: HaltingOptions
    memory: MemoryOptions
    recurrence: RecurrenceOptions


def _resolve_control_defaults(
    values: RuntimeDefaultValues,
    stacks: _ResolvedStacks,
) -> _ControlDefaults:
    control = values.control
    return _ControlDefaults(
        gate=GateOptions(
            enabled=control.stack_gate_flag,
            option=control.gate_option,
            activation=control.gate_activation,
            stack=stacks.gate,
            shared_config=control.shared_gate_config,
        ),
        halting=HaltingOptions(
            enabled=control.stack_halting_flag,
            threshold=control.halting_threshold,
            dropout_probability=control.halting_dropout,
            hidden_state_mode=control.halting_hidden_state_mode,
            stack=stacks.halting,
        ),
        memory=MemoryOptions(
            enabled=control.memory_flag,
            option=control.memory_option,
            position=control.memory_position_option,
            test_time_training_learning_rate=control.memory_learning_rate,
            test_time_training_num_inner_steps=control.memory_num_inner_steps,
            stack=stacks.memory,
        ),
        recurrence=RecurrenceOptions(
            enabled=control.recurrent_flag,
            max_steps=control.recurrent_max_steps,
            initial_iterations=control.recurrent_initial_iterations,
            gradient_transition_count=control.recurrent_gradient_transition_count,
            no_gradient_transition_count=control.recurrent_no_gradient_transition_count,
            iteration_increment=control.recurrent_iteration_increment,
            forward_calls_before_iteration_increment=(
                control.recurrent_forward_calls_before_iteration_increment
            ),
            smooth_iteration_growth_flag=(
                control.recurrent_smooth_iteration_growth_flag
            ),
            layer_norm_position=control.recurrent_layer_norm_position,
            gate=GateOptions(
                enabled=control.recurrent_stack_gate_flag,
                option=control.recurrent_gate_option,
                activation=control.recurrent_gate_activation,
                stack=stacks.recurrent_gate,
            ),
            halting=HaltingOptions(
                enabled=control.recurrent_stack_halting_flag,
                threshold=control.recurrent_halting_threshold,
                dropout_probability=control.recurrent_halting_dropout,
                hidden_state_mode=control.recurrent_halting_hidden_state_mode,
                stack=stacks.recurrent_halting,
            ),
        ),
    )


def _projection(values: ProjectionValues) -> AdaptiveProjectionOptions:
    return AdaptiveProjectionOptions(
        weight_option=values.weight_option,
        generator_depth=values.generator_depth,
        weight_decay_schedule=values.weight_decay_schedule,
        weight_decay_rate=values.weight_decay_rate,
        weight_decay_warmup_batches=values.weight_decay_warmup_batches,
        weight_normalization_option=values.weight_normalization_option,
        weight_normalization_position_option=(
            values.weight_normalization_position_option
        ),
        weight_bank_expansion_factor=values.weight_bank_expansion_factor,
        bias_option=values.bias_option,
        bias_decay_schedule=values.bias_decay_schedule,
        bias_decay_rate=values.bias_decay_rate,
        bias_decay_warmup_batches=values.bias_decay_warmup_batches,
        bias_bank_expansion_factor=values.bias_bank_expansion_factor,
        diagonal_option=values.diagonal_option,
        row_mask_option=values.row_mask_option,
        mask_dimension_option=values.mask_dimension_option,
        mask_threshold=values.mask_threshold,
        mask_surrogate_scale=values.mask_surrogate_scale,
        mask_floor=values.mask_floor,
        mask_transition_width=values.mask_transition_width,
    )


@dataclass(frozen=True, slots=True)
class _AdaptiveDefaults:
    weight: AdaptiveWeightOptions
    bias: AdaptiveBiasOptions
    diagonal: AdaptiveDiagonalOptions
    mask: AdaptiveMaskOptions
    input_projection: AdaptiveProjectionOptions
    output_projection: AdaptiveProjectionOptions


def _resolve_adaptive_defaults(
    values: RuntimeDefaultValues,
    adaptive_generator_stack: StackOptions,
) -> _AdaptiveDefaults:
    weight: WeightValues = values.weight
    bias: BiasValues = values.bias
    diagonal: DiagonalValues = values.diagonal
    mask: MaskValues = values.mask
    return _AdaptiveDefaults(
        weight=AdaptiveWeightOptions(
            enabled=weight.enabled,
            option=weight.option,
            generator_depth=weight.generator_depth,
            normalization_option=weight.normalization_option,
            normalization_position_option=weight.normalization_position_option,
            decay_schedule=weight.decay_schedule,
            decay_rate=weight.decay_rate,
            decay_warmup_batches=weight.decay_warmup_batches,
            bank_expansion_factor=weight.bank_expansion_factor,
            generator_stack=_generator_stack(
                values.weight_generator_stack,
                adaptive_generator_stack,
            ),
        ),
        bias=AdaptiveBiasOptions(
            enabled=bias.enabled,
            option=bias.option,
            decay_schedule=bias.decay_schedule,
            decay_rate=bias.decay_rate,
            decay_warmup_batches=bias.decay_warmup_batches,
            bank_expansion_factor=bias.bank_expansion_factor,
            generator_stack=_generator_stack(
                values.bias_generator_stack,
                adaptive_generator_stack,
            ),
        ),
        diagonal=AdaptiveDiagonalOptions(
            enabled=diagonal.enabled,
            option=diagonal.option,
            generator_stack=_generator_stack(
                values.diagonal_generator_stack,
                adaptive_generator_stack,
            ),
        ),
        mask=AdaptiveMaskOptions(
            enabled=mask.enabled,
            row_mask_option=mask.row_mask_option,
            dimension_option=mask.dimension_option,
            threshold=mask.threshold,
            surrogate_scale=mask.surrogate_scale,
            floor=mask.floor,
            transition_width=mask.transition_width,
            generator_stack=_generator_stack(
                values.mask_generator_stack,
                adaptive_generator_stack,
            ),
        ),
        input_projection=_projection(values.input_projection),
        output_projection=_projection(values.output_projection),
    )


def _runtime(values: RuntimeDefaultValues) -> RuntimeOptions:
    stacks = _resolve_stacks(values)
    control = _resolve_control_defaults(values, stacks)
    adaptive = _resolve_adaptive_defaults(values, stacks.adaptive_generator)
    dimensions = values.dimensions
    return RuntimeOptions(
        batch_size=dimensions.batch_size,
        learning_rate=dimensions.learning_rate,
        input_dim=dimensions.input_dim,
        hidden_dim=dimensions.hidden_dim,
        output_dim=dimensions.output_dim,
        stack=stacks.main,
        submodule_stack=stacks.submodule,
        residual_stack=stacks.residual,
        gate=control.gate,
        halting=control.halting,
        memory=control.memory,
        recurrence=control.recurrence,
        adaptive_generator_stack=stacks.adaptive_generator,
        weight=adaptive.weight,
        bias=adaptive.bias,
        diagonal=adaptive.diagonal,
        mask=adaptive.mask,
        input_projection=adaptive.input_projection,
        output_projection=adaptive.output_projection,
        halting_option=values.control.halting_option,
        recurrent_halting_option=values.control.recurrent_halting_option,
    )


def runtime_from_flat(
    overrides: Mapping[str, object] | None = None,
) -> RuntimeOptions:
    values = runtime_default_values(overrides)
    _validate_values(values)
    return _runtime(values)


DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_flat()
