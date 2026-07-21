from __future__ import annotations

import math
from numbers import Integral, Real

from .runtime_options import RuntimeOptions


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number.")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _positive_number(name: str, value: object) -> None:
    if _number(name, value) <= 0:
        raise ValueError(f"{name} must be positive.")


def _non_negative_number(name: str, value: object) -> None:
    if _number(name, value) < 0:
        raise ValueError(f"{name} must be non-negative.")


def _probability(name: str, value: object) -> None:
    resolved = _number(name, value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0].")


def _optional_positive_integer(name: str, value: object | None) -> None:
    if value is not None:
        _positive_integer(name, value)


def _optional_positive_number(name: str, value: object | None) -> None:
    if value is not None:
        _positive_number(name, value)


def _validate_controller_stack(name: str, options) -> None:
    _optional_positive_integer(f"{name}_hidden_dim", options.hidden_dim)
    _optional_positive_integer(f"{name}_num_layers", options.num_layers)
    if options.dropout_probability is not None:
        _probability(
            f"{name}_dropout_probability",
            options.dropout_probability,
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
    _positive_integer(f"{name}_recurrent_max_steps", recurrent.recurrent_max_steps)
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


def _validate_experts(name: str, options) -> None:
    _positive_integer(f"{name}_num_experts", options.num_experts)
    _positive_integer(f"{name}_top_k", options.top_k)
    if options.top_k > options.num_experts:
        raise ValueError(f"{name}_top_k cannot exceed {name}_num_experts.")
    _non_negative_number(f"{name}_switch_loss_weight", options.switch_loss_weight)
    _non_negative_number(f"{name}_capacity_factor", options.capacity_factor)
    if options.capacity_factor > 0 and options.top_k == options.num_experts:
        raise ValueError(
            f"{name}_capacity_factor must be 0 when top_k equals num_experts."
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
        _positive_integer(f"{name}_recurrent_max_steps", options.recurrent_max_steps)

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

    _validate_experts("attention_expert", runtime.attention_expert_options)
    _validate_experts("feed_forward_expert", runtime.feed_forward_expert_options)


__all__ = ["validate_runtime"]
