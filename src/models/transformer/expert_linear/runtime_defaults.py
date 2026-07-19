from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Final

from emperor.transformer import (
    TransformerPathOptions,
    attention_options_from_config,
    feed_forward_options_from_config,
    resolve_transformer_path_options,
)

from . import config
from .runtime_options import ExpertOptions, RuntimeOptions, TransformerStackOptions


def _scoped_feed_forward(options, *, hidden_dim: int, num_layers: int):
    return replace(
        options,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )


def runtime_from_config() -> RuntimeOptions:
    stack = TransformerStackOptions(
        num_layers=config.ENCODER_NUM_LAYERS,
        layer_norm_position=config.ENCODER_LAYER_NORM_POSITION,
        stack_gate_flag=config.STACK_GATE_FLAG,
        stack_halting_flag=config.STACK_HALTING_FLAG,
        memory_flag=config.MEMORY_FLAG,
        recurrent_flag=config.RECURRENT_FLAG,
        recurrent_gate_flag=config.RECURRENT_GATE_FLAG,
        recurrent_halting_flag=config.RECURRENT_HALTING_FLAG,
        recurrent_max_steps=config.RECURRENT_MAX_STEPS,
        stack_residual_connection_option=config.STACK_RESIDUAL_CONNECTION_OPTION,
        recurrent_residual_connection_option=(
            config.RECURRENT_RESIDUAL_CONNECTION_OPTION
        ),
    )
    attention = attention_options_from_config(config)
    feed_forward = feed_forward_options_from_config(config)
    experts = ExpertOptions(
        num_experts=config.EXPERT_NUM_EXPERTS,
        top_k=config.EXPERT_TOP_K,
        normalize_probabilities_flag=config.EXPERT_NORMALIZE_PROBABILITIES_FLAG,
        switch_loss_weight=config.EXPERT_SWITCH_LOSS_WEIGHT,
        capacity_factor=config.EXPERT_CAPACITY_FACTOR,
    )
    return RuntimeOptions(
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        vocab_size=config.VOCAB_SIZE,
        model_dim=config.MODEL_DIM,
        source_sequence_length=config.SOURCE_SEQUENCE_LENGTH,
        target_sequence_length=config.TARGET_SEQUENCE_LENGTH,
        dropout_probability=config.DROPOUT_PROBABILITY,
        positional_embedding_option=config.POSITIONAL_EMBEDDING_OPTION,
        encoder_options=stack,
        decoder_options=replace(
            stack,
            num_layers=config.DECODER_NUM_LAYERS,
            layer_norm_position=config.DECODER_LAYER_NORM_POSITION,
        ),
        encoder_attention_options=replace(
            attention, num_heads=config.ENCODER_ATTN_NUM_HEADS
        ),
        decoder_self_attention_options=replace(
            attention, num_heads=config.DECODER_SELF_ATTN_NUM_HEADS
        ),
        decoder_cross_attention_options=replace(
            attention, num_heads=config.DECODER_CROSS_ATTN_NUM_HEADS
        ),
        encoder_feed_forward_options=_scoped_feed_forward(
            feed_forward,
            hidden_dim=config.ENCODER_FEED_FORWARD_HIDDEN_DIM,
            num_layers=config.ENCODER_FEED_FORWARD_NUM_LAYERS,
        ),
        decoder_feed_forward_options=_scoped_feed_forward(
            feed_forward,
            hidden_dim=config.DECODER_FEED_FORWARD_HIDDEN_DIM,
            num_layers=config.DECODER_FEED_FORWARD_NUM_LAYERS,
        ),
        attention_expert_options=experts,
        feed_forward_expert_options=experts,
    )


_TOP_LEVEL_FIELDS = {item.name for item in fields(RuntimeOptions)}
_STACK_FIELDS = {item.name for item in fields(TransformerStackOptions)}
_PATH_FIELDS = {
    "encoder_attention_options",
    "decoder_self_attention_options",
    "decoder_cross_attention_options",
    "encoder_feed_forward_options",
    "decoder_feed_forward_options",
}
_EXPERT_FIELDS = {item.name for item in fields(ExpertOptions)}
_EXPERT_ALIASES = {
    "num_experts": "num_experts",
    "top_k": "top_k",
    "normalize_probabilities_flag": "normalize_probabilities_flag",
    "switch_loss_weight": "switch_loss_weight",
    "capacity_factor": "capacity_factor",
    "expert_num_experts": "num_experts",
    "expert_top_k": "top_k",
    "expert_normalize_probabilities_flag": "normalize_probabilities_flag",
    "expert_switch_loss_weight": "switch_loss_weight",
    "expert_capacity_factor": "capacity_factor",
}


def runtime_from_flat(
    values: dict[str, Any] | None = None,
    base: RuntimeOptions | None = None,
) -> RuntimeOptions:
    values = dict(values or {})
    runtime = DEFAULT_RUNTIME if base is None else base
    scalar_aliases = {
        "input_dim": "vocab_size",
        "output_dim": "vocab_size",
        "hidden_dim": "model_dim",
    }
    scalar_updates: dict[str, Any] = {}
    model_dim_changed = False
    dropout_changed = False
    for key in list(values):
        target = scalar_aliases.get(key, key)
        if target == "sequence_length":
            length = values.pop(key)
            scalar_updates.update(
                source_sequence_length=length,
                target_sequence_length=length,
            )
        elif target in _TOP_LEVEL_FIELDS - _PATH_FIELDS - {
            "encoder_options",
            "decoder_options",
        }:
            value = values.pop(key)
            scalar_updates[target] = value
            model_dim_changed |= target == "model_dim"
            dropout_changed |= target == "dropout_probability"
    runtime = replace(runtime, **scalar_updates)
    if model_dim_changed:
        values.setdefault("attn_stack_hidden_dim", runtime.model_dim)
    if dropout_changed:
        values.setdefault("ff_stack_dropout_probability", runtime.dropout_probability)

    stack_broadcast = {
        key: values.pop(key) for key in list(values) if key in _STACK_FIELDS
    }
    encoder = replace(
        values.pop("encoder_options", runtime.encoder_options), **stack_broadcast
    )
    decoder = replace(
        values.pop("decoder_options", runtime.decoder_options), **stack_broadcast
    )
    for prefix, current in (("encoder_", encoder), ("decoder_", decoder)):
        updates = {}
        for field_name in _STACK_FIELDS:
            key = f"{prefix}{field_name}"
            if key in values:
                updates[field_name] = values.pop(key)
        if prefix == "encoder_":
            encoder = replace(current, **updates)
        else:
            decoder = replace(current, **updates)

    paths = resolve_transformer_path_options(
        values,
        TransformerPathOptions(
            encoder_attention_options=runtime.encoder_attention_options,
            decoder_self_attention_options=runtime.decoder_self_attention_options,
            decoder_cross_attention_options=runtime.decoder_cross_attention_options,
            encoder_feed_forward_options=runtime.encoder_feed_forward_options,
            decoder_feed_forward_options=runtime.decoder_feed_forward_options,
        ),
    )
    expert_broadcast = {
        field_name: values.pop(key)
        for key, field_name in _EXPERT_ALIASES.items()
        if key in values
    }
    attention_experts = replace(
        values.pop("attention_expert_options", runtime.attention_expert_options),
        **expert_broadcast,
    )
    feed_forward_experts = replace(
        values.pop("feed_forward_expert_options", runtime.feed_forward_expert_options),
        **expert_broadcast,
    )
    for prefix, current in (
        ("attention_expert_", attention_experts),
        ("feed_forward_expert_", feed_forward_experts),
    ):
        updates = {}
        for field_name in _EXPERT_FIELDS:
            key = f"{prefix}{field_name}"
            if key in values:
                updates[field_name] = values.pop(key)
        if prefix == "attention_expert_":
            attention_experts = replace(current, **updates)
        else:
            feed_forward_experts = replace(current, **updates)
    if values:
        unknown = sorted(values)[0]
        raise TypeError(
            "TransformerExpertLinearConfigBuilder.__init__() got an unexpected keyword "
            f"argument {unknown!r}"
        )
    return replace(
        runtime,
        encoder_options=encoder,
        decoder_options=decoder,
        encoder_attention_options=paths.encoder_attention_options,
        decoder_self_attention_options=paths.decoder_self_attention_options,
        decoder_cross_attention_options=paths.decoder_cross_attention_options,
        encoder_feed_forward_options=paths.encoder_feed_forward_options,
        decoder_feed_forward_options=paths.decoder_feed_forward_options,
        attention_expert_options=attention_experts,
        feed_forward_expert_options=feed_forward_experts,
    )


DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_config()

__all__ = ["DEFAULT_RUNTIME", "runtime_from_config", "runtime_from_flat"]
