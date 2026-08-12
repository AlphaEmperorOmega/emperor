from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from numbers import Integral, Real
from typing import Final

from emperor.layers import RecurrentCompositionConfig

from .runtime_options import RuntimeOptions


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _nonnegative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _positive_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number.")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")


def _nonnegative_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number.")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be non-negative and finite.")


def _probability(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number.")
    if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0].")


def _optional_positive_integer(name: str, value: object | None) -> None:
    if value is not None:
        _positive_integer(name, value)


def _optional_positive_number(name: str, value: object | None) -> None:
    if value is not None:
        _positive_number(name, value)


def _boolean(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def _validate_controller_stack(name: str, options) -> None:
    _optional_positive_integer(f"{name}_hidden_dim", options.hidden_dim)
    _optional_positive_integer(f"{name}_num_layers", options.num_layers)
    if options.dropout_probability is not None:
        _probability(
            f"{name}_dropout_probability",
            options.dropout_probability,
        )


@dataclass(frozen=True)
class _RecurrentFieldRule:
    option_field: str
    runtime_field: str
    validator: Callable[[str, object], None]


_RECURRENT_FIELD_RULES: Final[tuple[_RecurrentFieldRule, ...]] = (
    _RecurrentFieldRule("max_steps", "recurrent_max_steps", _positive_integer),
    _RecurrentFieldRule(
        "latent_updates_per_answer_update",
        "recurrent_latent_updates_per_answer_update",
        _positive_integer,
    ),
    _RecurrentFieldRule(
        "answer_update_count",
        "recurrent_answer_update_count",
        _positive_integer,
    ),
    _RecurrentFieldRule("high_cycles", "recurrent_high_cycles", _positive_integer),
    _RecurrentFieldRule("low_cycles", "recurrent_low_cycles", _positive_integer),
    _RecurrentFieldRule(
        "initialization_standard_deviation",
        "recurrent_initialization_standard_deviation",
        _nonnegative_number,
    ),
)


def _selected_transition_count(options, option_fields: set[str]) -> int | None:
    if "max_steps" in option_fields:
        return options.recurrent_max_steps
    if {"answer_update_count", "latent_updates_per_answer_update"} <= option_fields:
        return options.recurrent_answer_update_count * (
            options.recurrent_latent_updates_per_answer_update + 1
        )
    if {"high_cycles", "low_cycles"} <= option_fields:
        return options.recurrent_high_cycles * (options.recurrent_low_cycles + 1)
    return None


def _validate_no_gradient_transition_count(
    name: str,
    options,
    option_fields: set[str],
) -> None:
    transition_count = options.recurrent_no_gradient_transition_count
    if transition_count is None:
        return
    field_name = f"{name}_recurrent_no_gradient_transition_count"
    _nonnegative_integer(field_name, transition_count)
    total_transition_count = _selected_transition_count(options, option_fields)
    if (
        total_transition_count is not None
        and transition_count >= total_transition_count
    ):
        raise ValueError(
            f"{field_name} must be less than the selected recurrent "
            f"schedule's {total_transition_count} transitions."
        )


def _validate_recurrent_options(name: str, options) -> None:
    _boolean(
        f"{name}_recurrent_reinject_original_hidden_flag",
        options.recurrent_reinject_original_hidden_flag,
    )
    option = options.recurrent_composition_option
    if not isinstance(option, type) or not issubclass(
        option, RecurrentCompositionConfig
    ):
        raise TypeError(
            f"{name}_recurrent_composition_option must be a concrete "
            "RecurrentCompositionConfig subclass."
        )
    try:
        option().registry_owner()
    except (NotImplementedError, ValueError) as exc:
        raise ValueError(
            f"{name}_recurrent_composition_option must select a concrete "
            "recurrent config."
        ) from exc
    option_fields = {field.name for field in fields(option)}
    for rule in _RECURRENT_FIELD_RULES:
        if rule.option_field in option_fields:
            rule.validator(
                f"{name}_{rule.runtime_field}",
                getattr(options, rule.runtime_field),
            )
    _validate_no_gradient_transition_count(name, options, option_fields)

    if (
        "reinject_original_hidden_flag" not in option_fields
        and options.recurrent_reinject_original_hidden_flag
    ):
        raise ValueError(
            f"{name}_recurrent_composition_option does not support standard "
            "fixed-input reinjection."
        )


def _validate_path(name: str, options) -> None:
    stack = options.stack_options
    _positive_integer(f"{name}_stack_hidden_dim", stack.hidden_dim)
    _positive_integer(f"{name}_num_layers", stack.num_layers)
    _probability(f"{name}_stack_dropout_probability", stack.dropout_probability)
    controllers = options.layer_controller_options
    _probability(f"{name}_halting_threshold", controllers.halting_threshold)
    _probability(f"{name}_halting_dropout", controllers.halting_dropout)
    _validate_controller_stack(f"{name}_gate_stack", controllers.gate_stack_options)
    _validate_controller_stack(
        f"{name}_halting_stack",
        controllers.halting_stack_options,
    )

    memory = options.dynamic_memory_options
    _optional_positive_number(
        f"{name}_memory_test_time_training_learning_rate",
        memory.memory_test_time_training_learning_rate,
    )
    _optional_positive_integer(
        f"{name}_memory_test_time_training_num_inner_steps",
        memory.memory_test_time_training_num_inner_steps,
    )
    _validate_controller_stack(f"{name}_memory_stack", memory.memory_stack_options)

    recurrent = options.recurrent_controller_options
    _validate_recurrent_options(name, recurrent)
    _probability(
        f"{name}_recurrent_halting_threshold",
        recurrent.recurrent_halting_threshold,
    )
    _probability(
        f"{name}_recurrent_halting_dropout",
        recurrent.recurrent_halting_dropout,
    )
    _validate_controller_stack(
        f"{name}_recurrent_gate_stack",
        recurrent.recurrent_gate_stack_options,
    )
    _validate_controller_stack(
        f"{name}_recurrent_halting_stack",
        recurrent.recurrent_halting_stack_options,
    )


def validate_runtime(runtime: RuntimeOptions) -> None:
    _positive_integer("batch_size", runtime.batch_size)
    _positive_number("learning_rate", runtime.learning_rate)
    _positive_integer("vocab_size", runtime.vocab_size)
    if runtime.vocab_size <= 3:
        raise ValueError("vocab_size must be greater than the PAD/BOS/EOS token IDs.")
    _positive_integer("model_dim", runtime.model_dim)
    for name, length in (
        ("source_sequence_length", runtime.source_sequence_length),
        ("target_sequence_length", runtime.target_sequence_length),
    ):
        _positive_integer(name, length)
        if length < 2:
            raise ValueError(f"{name} must be at least 2.")
    _probability("dropout_probability", runtime.dropout_probability)

    for name, options in (
        ("encoder", runtime.encoder_options),
        ("decoder", runtime.decoder_options),
    ):
        _positive_integer(f"{name}_num_layers", options.num_layers)
        _validate_recurrent_options(name, options)

    attention_paths = (
        ("encoder_attn", runtime.encoder_attention_options),
        ("decoder_self_attn", runtime.decoder_self_attention_options),
        ("decoder_cross_attn", runtime.decoder_cross_attention_options),
    )
    for name, options in attention_paths:
        _positive_integer(f"{name}_num_heads", options.num_heads)
        if runtime.model_dim % options.num_heads:
            raise ValueError(f"{name}_num_heads must divide model_dim.")
        _validate_path(name, options)

    for name, options in (
        ("encoder_ff", runtime.encoder_feed_forward_options),
        ("decoder_ff", runtime.decoder_feed_forward_options),
    ):
        _validate_path(name, options)


__all__ = ["validate_runtime"]
