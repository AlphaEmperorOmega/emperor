from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, fields, replace
from types import ModuleType
from typing import Any, Final

from model_runtime.packages.runtime_values import validate_runtime_default_values

from . import config
from .runtime_options import (
    AdaptiveParameterOptions,
    ControllerStackOptions,
    DynamicMemoryOptions,
    ExpertOptions,
    LayerControllerOptions,
    RecurrentControllerOptions,
    RuntimeOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerFeedForwardOptions,
    TransformerStackOptions,
)


@dataclass(frozen=True)
class TransformerPathOptions:
    encoder_attention_options: TransformerAttentionOptions
    decoder_self_attention_options: TransformerAttentionOptions
    decoder_cross_attention_options: TransformerAttentionOptions
    encoder_feed_forward_options: TransformerFeedForwardOptions
    decoder_feed_forward_options: TransformerFeedForwardOptions


def _controller_stack_from_config(
    config_module: ModuleType,
    prefix: str,
) -> ControllerStackOptions:
    return ControllerStackOptions(
        independent_flag=getattr(config_module, f"{prefix}_INDEPENDENT_FLAG"),
        hidden_dim=getattr(config_module, f"{prefix}_HIDDEN_DIM"),
        num_layers=getattr(config_module, f"{prefix}_NUM_LAYERS"),
        last_layer_bias_option=getattr(
            config_module, f"{prefix}_LAST_LAYER_BIAS_OPTION"
        ),
        apply_output_pipeline_flag=getattr(
            config_module, f"{prefix}_APPLY_OUTPUT_PIPELINE_FLAG"
        ),
        activation=getattr(config_module, f"{prefix}_ACTIVATION"),
        layer_norm_position=getattr(config_module, f"{prefix}_LAYER_NORM_POSITION"),
        residual_connection_option=getattr(
            config_module, f"{prefix}_RESIDUAL_CONNECTION_OPTION"
        ),
        dropout_probability=getattr(config_module, f"{prefix}_DROPOUT_PROBABILITY"),
        bias_flag=getattr(config_module, f"{prefix}_BIAS_FLAG"),
    )


def _layer_controller_from_config(
    config_module: ModuleType,
    prefix: str,
) -> LayerControllerOptions:
    return LayerControllerOptions(
        stack_gate_flag=getattr(config_module, f"{prefix}_STACK_GATE_FLAG"),
        gate_option=getattr(config_module, f"{prefix}_GATE_OPTION"),
        gate_activation=getattr(config_module, f"{prefix}_GATE_ACTIVATION"),
        gate_stack_options=_controller_stack_from_config(
            config_module, f"{prefix}_GATE_STACK"
        ),
        stack_halting_flag=getattr(config_module, f"{prefix}_STACK_HALTING_FLAG"),
        halting_option=getattr(config_module, f"{prefix}_HALTING_OPTION"),
        halting_threshold=getattr(config_module, f"{prefix}_HALTING_THRESHOLD"),
        halting_dropout=getattr(config_module, f"{prefix}_HALTING_DROPOUT"),
        halting_hidden_state_mode=getattr(
            config_module, f"{prefix}_HALTING_HIDDEN_STATE_MODE"
        ),
        halting_stack_options=_controller_stack_from_config(
            config_module, f"{prefix}_HALTING_STACK"
        ),
    )


def _memory_from_config(
    config_module: ModuleType,
    prefix: str,
) -> DynamicMemoryOptions:
    return DynamicMemoryOptions(
        memory_flag=getattr(config_module, f"{prefix}_MEMORY_FLAG"),
        memory_option=getattr(config_module, f"{prefix}_MEMORY_OPTION"),
        memory_position_option=getattr(
            config_module, f"{prefix}_MEMORY_POSITION_OPTION"
        ),
        memory_test_time_training_learning_rate=getattr(
            config_module, f"{prefix}_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE"
        ),
        memory_test_time_training_num_inner_steps=getattr(
            config_module, f"{prefix}_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS"
        ),
        memory_stack_options=_controller_stack_from_config(
            config_module, f"{prefix}_MEMORY_STACK"
        ),
    )


def _recurrent_from_config(
    config_module: ModuleType,
    prefix: str,
) -> RecurrentControllerOptions:
    return RecurrentControllerOptions(
        recurrent_flag=getattr(config_module, f"{prefix}_RECURRENT_FLAG"),
        recurrent_max_steps=getattr(config_module, f"{prefix}_RECURRENT_MAX_STEPS"),
        recurrent_layer_norm_position=getattr(
            config_module, f"{prefix}_RECURRENT_LAYER_NORM_POSITION"
        ),
        recurrent_stack_gate_flag=getattr(
            config_module, f"{prefix}_RECURRENT_STACK_GATE_FLAG"
        ),
        recurrent_gate_option=getattr(config_module, f"{prefix}_RECURRENT_GATE_OPTION"),
        recurrent_gate_activation=getattr(
            config_module, f"{prefix}_RECURRENT_GATE_ACTIVATION"
        ),
        recurrent_gate_stack_options=_controller_stack_from_config(
            config_module, f"{prefix}_RECURRENT_GATE_STACK"
        ),
        recurrent_stack_halting_flag=getattr(
            config_module, f"{prefix}_RECURRENT_STACK_HALTING_FLAG"
        ),
        recurrent_halting_option=getattr(
            config_module, f"{prefix}_RECURRENT_HALTING_OPTION"
        ),
        recurrent_halting_threshold=getattr(
            config_module, f"{prefix}_RECURRENT_HALTING_THRESHOLD"
        ),
        recurrent_halting_dropout=getattr(
            config_module, f"{prefix}_RECURRENT_HALTING_DROPOUT"
        ),
        recurrent_halting_hidden_state_mode=getattr(
            config_module, f"{prefix}_RECURRENT_HALTING_HIDDEN_STATE_MODE"
        ),
        recurrent_halting_stack_options=_controller_stack_from_config(
            config_module, f"{prefix}_RECURRENT_HALTING_STACK"
        ),
    )


def attention_options_from_config(
    config_module: ModuleType,
) -> TransformerAttentionOptions:
    return TransformerAttentionOptions(
        num_heads=config_module.ATTN_NUM_HEADS,
        add_key_value_bias_flag=config_module.ATTN_ADD_KEY_VALUE_BIAS_FLAG,
        zero_attention_flag=config_module.ATTN_ZERO_ATTENTION_FLAG,
        stack_options=SubmoduleStackOptions(
            hidden_dim=config_module.ATTN_STACK_HIDDEN_DIM,
            num_layers=config_module.ATTN_NUM_LAYERS,
            last_layer_bias_option=(config_module.ATTN_STACK_LAST_LAYER_BIAS_OPTION),
            apply_output_pipeline_flag=(
                config_module.ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config_module.ATTN_STACK_ACTIVATION,
            layer_norm_position=config_module.ATTN_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config_module.ATTN_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            dropout_probability=(config_module.ATTN_STACK_DROPOUT_PROBABILITY),
            bias_flag=config_module.ATTN_BIAS_FLAG,
        ),
        layer_controller_options=_layer_controller_from_config(config_module, "ATTN"),
        dynamic_memory_options=_memory_from_config(config_module, "ATTN"),
        recurrent_controller_options=_recurrent_from_config(config_module, "ATTN"),
    )


def feed_forward_options_from_config(
    config_module: ModuleType,
) -> TransformerFeedForwardOptions:
    return TransformerFeedForwardOptions(
        stack_options=SubmoduleStackOptions(
            hidden_dim=config_module.FF_STACK_HIDDEN_DIM,
            num_layers=config_module.FF_NUM_LAYERS,
            last_layer_bias_option=(config_module.FF_STACK_LAST_LAYER_BIAS_OPTION),
            apply_output_pipeline_flag=(
                config_module.FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            activation=config_module.FF_STACK_ACTIVATION,
            layer_norm_position=config_module.FF_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config_module.FF_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            dropout_probability=config_module.FF_STACK_DROPOUT_PROBABILITY,
            bias_flag=config_module.FF_BIAS_FLAG,
        ),
        layer_controller_options=_layer_controller_from_config(config_module, "FF"),
        dynamic_memory_options=_memory_from_config(config_module, "FF"),
        recurrent_controller_options=_recurrent_from_config(config_module, "FF"),
    )


_STACK_OPTION_FIELDS = (
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
_CONTROLLER_STACK_FIELDS = ("independent_flag", *_STACK_OPTION_FIELDS)


def _path_field_map(*, attention: bool) -> dict[str, tuple[str, str]]:
    mapping = {
        "num_layers": ("stack", "num_layers"),
        "bias_flag": ("stack", "bias_flag"),
    }
    if attention:
        mapping.update(
            {
                "num_heads": ("path", "num_heads"),
                "add_key_value_bias_flag": (
                    "path",
                    "add_key_value_bias_flag",
                ),
                "zero_attention_flag": ("path", "zero_attention_flag"),
            }
        )
    for field_name in _STACK_OPTION_FIELDS:
        if field_name not in {"num_layers", "bias_flag"}:
            mapping[f"stack_{field_name}"] = ("stack", field_name)
    mapping.update(
        {
            "stack_gate_flag": ("controller", "stack_gate_flag"),
            "gate_option": ("controller", "gate_option"),
            "gate_activation": ("controller", "gate_activation"),
            "stack_halting_flag": ("controller", "stack_halting_flag"),
            "halting_option": ("controller", "halting_option"),
            "halting_threshold": ("controller", "halting_threshold"),
            "halting_dropout": ("controller", "halting_dropout"),
            "halting_hidden_state_mode": (
                "controller",
                "halting_hidden_state_mode",
            ),
            "memory_flag": ("memory", "memory_flag"),
            "memory_option": ("memory", "memory_option"),
            "memory_position_option": (
                "memory",
                "memory_position_option",
            ),
            "memory_test_time_training_learning_rate": (
                "memory",
                "memory_test_time_training_learning_rate",
            ),
            "memory_test_time_training_num_inner_steps": (
                "memory",
                "memory_test_time_training_num_inner_steps",
            ),
            "recurrent_flag": ("recurrent", "recurrent_flag"),
            "recurrent_max_steps": ("recurrent", "recurrent_max_steps"),
            "recurrent_layer_norm_position": (
                "recurrent",
                "recurrent_layer_norm_position",
            ),
            "recurrent_stack_gate_flag": (
                "recurrent",
                "recurrent_stack_gate_flag",
            ),
            "recurrent_gate_option": (
                "recurrent",
                "recurrent_gate_option",
            ),
            "recurrent_gate_activation": (
                "recurrent",
                "recurrent_gate_activation",
            ),
            "recurrent_stack_halting_flag": (
                "recurrent",
                "recurrent_stack_halting_flag",
            ),
            "recurrent_halting_option": (
                "recurrent",
                "recurrent_halting_option",
            ),
            "recurrent_halting_threshold": (
                "recurrent",
                "recurrent_halting_threshold",
            ),
            "recurrent_halting_dropout": (
                "recurrent",
                "recurrent_halting_dropout",
            ),
            "recurrent_halting_hidden_state_mode": (
                "recurrent",
                "recurrent_halting_hidden_state_mode",
            ),
        }
    )
    for role, component, field_name in (
        ("gate", "gate_stack", "gate_stack_options"),
        ("halting", "halting_stack", "halting_stack_options"),
        ("memory", "memory_stack", "memory_stack_options"),
        (
            "recurrent_gate",
            "recurrent_gate_stack",
            "recurrent_gate_stack_options",
        ),
        (
            "recurrent_halting",
            "recurrent_halting_stack",
            "recurrent_halting_stack_options",
        ),
    ):
        for field in _CONTROLLER_STACK_FIELDS:
            mapping[f"{role}_stack_{field}"] = (
                component,
                f"{field_name}.{field}",
            )
    return mapping


_ATTENTION_FIELD_MAP = _path_field_map(attention=True)
_FEED_FORWARD_FIELD_MAP = _path_field_map(attention=False)


def _replace_nested(source: Any, dotted_field: str, value: Any) -> Any:
    outer_field, inner_field = dotted_field.split(".", 1)
    return replace(
        source,
        **{
            outer_field: replace(
                getattr(source, outer_field),
                **{inner_field: value},
            )
        },
    )


def _apply_path_updates(
    options: TransformerAttentionOptions | TransformerFeedForwardOptions,
    updates: dict[str, Any],
    *,
    attention: bool,
) -> TransformerAttentionOptions | TransformerFeedForwardOptions:
    field_map = _ATTENTION_FIELD_MAP if attention else _FEED_FORWARD_FIELD_MAP
    path = options
    stack = path.stack_options
    controller = path.layer_controller_options
    memory = path.dynamic_memory_options
    recurrent = path.recurrent_controller_options
    for suffix, value in updates.items():
        component, field_name = field_map[suffix]
        if component == "path":
            path = replace(path, **{field_name: value})
        elif component == "stack":
            stack = replace(stack, **{field_name: value})
        elif component == "controller":
            controller = replace(controller, **{field_name: value})
        elif component == "memory":
            memory = replace(memory, **{field_name: value})
        elif component == "recurrent":
            recurrent = replace(recurrent, **{field_name: value})
        elif component == "gate_stack":
            controller = _replace_nested(controller, field_name, value)
        elif component == "halting_stack":
            controller = _replace_nested(controller, field_name, value)
        elif component == "memory_stack":
            memory = _replace_nested(memory, field_name, value)
        elif component == "recurrent_gate_stack":
            recurrent = _replace_nested(recurrent, field_name, value)
        elif component == "recurrent_halting_stack":
            recurrent = _replace_nested(recurrent, field_name, value)
        else:
            raise ValueError(component)
    return replace(
        path,
        stack_options=stack,
        layer_controller_options=controller,
        dynamic_memory_options=memory,
        recurrent_controller_options=recurrent,
    )


def _pop_updates(
    values: MutableMapping[str, Any],
    prefix: str,
    field_map: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    updates = {}
    for suffix in field_map:
        key = f"{prefix}{suffix}"
        if key in values:
            updates[suffix] = values.pop(key)
    return updates


def _pop_scoped_feed_forward_updates(
    values: MutableMapping[str, Any], prefix: str
) -> dict[str, Any]:
    updates = {}
    for field_name, suffix in (
        ("hidden_dim", "stack_hidden_dim"),
        ("num_layers", "num_layers"),
    ):
        key = f"{prefix}{field_name}"
        if key in values:
            updates[suffix] = values.pop(key)
    return updates


def resolve_transformer_path_options(
    values: MutableMapping[str, Any],
    defaults: TransformerPathOptions,
) -> TransformerPathOptions:
    encoder_attention = values.pop(
        "encoder_attention_options", defaults.encoder_attention_options
    )
    decoder_self_attention = values.pop(
        "decoder_self_attention_options",
        defaults.decoder_self_attention_options,
    )
    decoder_cross_attention = values.pop(
        "decoder_cross_attention_options",
        defaults.decoder_cross_attention_options,
    )
    encoder_feed_forward = values.pop(
        "encoder_feed_forward_options",
        defaults.encoder_feed_forward_options,
    )
    decoder_feed_forward = values.pop(
        "decoder_feed_forward_options",
        defaults.decoder_feed_forward_options,
    )

    attention_updates = _pop_updates(values, "attn_", _ATTENTION_FIELD_MAP)
    feed_forward_updates = _pop_updates(values, "ff_", _FEED_FORWARD_FIELD_MAP)
    encoder_attention = _apply_path_updates(
        encoder_attention, attention_updates, attention=True
    )
    decoder_self_attention = _apply_path_updates(
        decoder_self_attention, attention_updates, attention=True
    )
    decoder_cross_attention = _apply_path_updates(
        decoder_cross_attention, attention_updates, attention=True
    )
    encoder_feed_forward = _apply_path_updates(
        encoder_feed_forward, feed_forward_updates, attention=False
    )
    decoder_feed_forward = _apply_path_updates(
        decoder_feed_forward, feed_forward_updates, attention=False
    )

    encoder_attention = _apply_path_updates(
        encoder_attention,
        _pop_updates(values, "encoder_attn_", _ATTENTION_FIELD_MAP),
        attention=True,
    )
    decoder_self_attention = _apply_path_updates(
        decoder_self_attention,
        _pop_updates(values, "decoder_self_attn_", _ATTENTION_FIELD_MAP),
        attention=True,
    )
    decoder_cross_attention = _apply_path_updates(
        decoder_cross_attention,
        _pop_updates(values, "decoder_cross_attn_", _ATTENTION_FIELD_MAP),
        attention=True,
    )
    encoder_feed_forward = _apply_path_updates(
        encoder_feed_forward,
        _pop_scoped_feed_forward_updates(values, "encoder_feed_forward_"),
        attention=False,
    )
    decoder_feed_forward = _apply_path_updates(
        decoder_feed_forward,
        _pop_scoped_feed_forward_updates(values, "decoder_feed_forward_"),
        attention=False,
    )
    return TransformerPathOptions(
        encoder_attention_options=encoder_attention,
        decoder_self_attention_options=decoder_self_attention,
        decoder_cross_attention_options=decoder_cross_attention,
        encoder_feed_forward_options=encoder_feed_forward,
        decoder_feed_forward_options=decoder_feed_forward,
    )


def _scoped_feed_forward(options, *, hidden_dim: int, num_layers: int):
    return replace(
        options,
        stack_options=replace(
            options.stack_options,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ),
    )


def runtime_from_config() -> RuntimeOptions:
    stack = TransformerStackOptions(
        num_layers=config.ENCODER_NUM_LAYERS,
        layer_norm_position=config.ENCODER_LAYER_NORM_POSITION,
        stack_gate_flag=config.STACK_GATE_FLAG,
        stack_halting_flag=config.STACK_HALTING_FLAG,
        memory_flag=config.MEMORY_FLAG,
        recurrent_flag=config.RECURRENT_FLAG,
        recurrent_stack_gate_flag=config.RECURRENT_STACK_GATE_FLAG,
        recurrent_stack_halting_flag=config.RECURRENT_STACK_HALTING_FLAG,
        recurrent_max_steps=config.RECURRENT_MAX_STEPS,
        stack_residual_connection_option=config.STACK_RESIDUAL_CONNECTION_OPTION,
        recurrent_residual_connection_option=(
            config.RECURRENT_RESIDUAL_CONNECTION_OPTION
        ),
    )
    attention = attention_options_from_config(config)
    feed_forward = feed_forward_options_from_config(config)
    experts = ExpertOptions(
        num_experts=config.NUM_EXPERTS,
        top_k=config.TOP_K,
        normalize_probabilities_flag=config.NORMALIZE_PROBABILITIES_FLAG,
        switch_loss_weight=config.SWITCH_LOSS_WEIGHT,
        capacity_factor=config.CAPACITY_FACTOR,
    )

    def adaptive_options(prefix: str) -> AdaptiveParameterOptions:
        return AdaptiveParameterOptions(
            weight_option=getattr(config, f"{prefix}_WEIGHT_OPTION"),
            bias_option=getattr(config, f"{prefix}_BIAS_OPTION"),
            diagonal_option=getattr(config, f"{prefix}_DIAGONAL_OPTION"),
            row_mask_option=getattr(config, f"{prefix}_ROW_MASK_OPTION"),
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
        attention_projection_adaptive_options=adaptive_options(
            "ATTENTION_PROJECTION_ADAPTIVE"
        ),
        attention_expert_adaptive_options=adaptive_options("ATTENTION_EXPERT_ADAPTIVE"),
        router_adaptive_options=adaptive_options("ROUTER_ADAPTIVE"),
        feed_forward_adaptive_options=adaptive_options("FEED_FORWARD_ADAPTIVE"),
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
_ADAPTIVE_FIELDS = {item.name for item in fields(AdaptiveParameterOptions)}


def runtime_from_flat(
    values: dict[str, Any] | None = None,
    base: RuntimeOptions | None = None,
) -> RuntimeOptions:
    values = validate_runtime_default_values(
        values,
        package="models.transformer.expert_linear_adaptive",
        config_module=config,
    )
    runtime = DEFAULT_RUNTIME if base is None else base
    scalar_updates: dict[str, Any] = {}
    model_dim_changed = False
    dropout_changed = False
    for key in list(values):
        target = key
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
        key: values.pop(key) for key in list(values) if key in _EXPERT_FIELDS
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

    adaptive_groups = {
        "attention_projection_adaptive_": (
            runtime.attention_projection_adaptive_options
        ),
        "attention_expert_adaptive_": runtime.attention_expert_adaptive_options,
        "router_adaptive_": runtime.router_adaptive_options,
        "feed_forward_adaptive_": runtime.feed_forward_adaptive_options,
    }
    resolved_adaptive = {}
    for prefix, current in adaptive_groups.items():
        updates = {}
        for field_name in _ADAPTIVE_FIELDS:
            key = f"{prefix}{field_name}"
            if key in values:
                updates[field_name] = values.pop(key)
        resolved_adaptive[prefix] = replace(current, **updates)
    if values:
        unknown = sorted(values)[0]
        raise TypeError(
            "TransformerExpertLinearAdaptiveConfigBuilder.__init__() got an "
            "unexpected keyword "
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
        attention_projection_adaptive_options=resolved_adaptive[
            "attention_projection_adaptive_"
        ],
        attention_expert_adaptive_options=resolved_adaptive[
            "attention_expert_adaptive_"
        ],
        router_adaptive_options=resolved_adaptive["router_adaptive_"],
        feed_forward_adaptive_options=resolved_adaptive["feed_forward_adaptive_"],
    )


DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_config()

__all__ = ["DEFAULT_RUNTIME", "runtime_from_config", "runtime_from_flat"]
