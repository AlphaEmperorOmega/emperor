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
        residual_model_flag=getattr(config_module, f"{prefix}_RESIDUAL_MODEL_FLAG"),
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
    prefix: str = "ATTN",
) -> TransformerAttentionOptions:
    return TransformerAttentionOptions(
        num_heads=getattr(config_module, f"{prefix}_NUM_HEADS"),
        add_key_value_bias_flag=getattr(
            config_module,
            f"{prefix}_ADD_KEY_VALUE_BIAS_FLAG",
        ),
        zero_attention_flag=getattr(config_module, f"{prefix}_ZERO_ATTENTION_FLAG"),
        stack_options=SubmoduleStackOptions(
            hidden_dim=getattr(config_module, f"{prefix}_STACK_HIDDEN_DIM"),
            num_layers=getattr(config_module, f"{prefix}_NUM_LAYERS"),
            last_layer_bias_option=getattr(
                config_module,
                f"{prefix}_STACK_LAST_LAYER_BIAS_OPTION",
            ),
            apply_output_pipeline_flag=(
                getattr(
                    config_module,
                    f"{prefix}_STACK_APPLY_OUTPUT_PIPELINE_FLAG",
                )
            ),
            activation=getattr(config_module, f"{prefix}_STACK_ACTIVATION"),
            layer_norm_position=getattr(
                config_module,
                f"{prefix}_STACK_LAYER_NORM_POSITION",
            ),
            residual_connection_option=(
                getattr(
                    config_module,
                    f"{prefix}_STACK_RESIDUAL_CONNECTION_OPTION",
                )
            ),
            residual_model_flag=getattr(
                config_module,
                f"{prefix}_STACK_RESIDUAL_MODEL_FLAG",
            ),
            dropout_probability=getattr(
                config_module,
                f"{prefix}_STACK_DROPOUT_PROBABILITY",
            ),
            bias_flag=getattr(config_module, f"{prefix}_BIAS_FLAG"),
        ),
        layer_controller_options=_layer_controller_from_config(config_module, prefix),
        dynamic_memory_options=_memory_from_config(config_module, prefix),
        recurrent_controller_options=_recurrent_from_config(config_module, prefix),
    )


def feed_forward_options_from_config(
    config_module: ModuleType,
    prefix: str = "FF",
) -> TransformerFeedForwardOptions:
    return TransformerFeedForwardOptions(
        stack_options=SubmoduleStackOptions(
            hidden_dim=getattr(config_module, f"{prefix}_STACK_HIDDEN_DIM"),
            num_layers=getattr(
                config_module,
                f"{prefix}_STACK_NUM_LAYERS",
                getattr(config_module, f"{prefix}_NUM_LAYERS"),
            ),
            last_layer_bias_option=getattr(
                config_module,
                f"{prefix}_STACK_LAST_LAYER_BIAS_OPTION",
            ),
            apply_output_pipeline_flag=(
                getattr(
                    config_module,
                    f"{prefix}_STACK_APPLY_OUTPUT_PIPELINE_FLAG",
                )
            ),
            activation=getattr(config_module, f"{prefix}_STACK_ACTIVATION"),
            layer_norm_position=getattr(
                config_module,
                f"{prefix}_STACK_LAYER_NORM_POSITION",
            ),
            residual_connection_option=(
                getattr(
                    config_module,
                    f"{prefix}_STACK_RESIDUAL_CONNECTION_OPTION",
                )
            ),
            residual_model_flag=getattr(
                config_module,
                f"{prefix}_STACK_RESIDUAL_MODEL_FLAG",
            ),
            dropout_probability=getattr(
                config_module,
                f"{prefix}_STACK_DROPOUT_PROBABILITY",
            ),
            bias_flag=getattr(
                config_module,
                f"{prefix}_STACK_BIAS_FLAG",
                getattr(config_module, f"{prefix}_BIAS_FLAG"),
            ),
        ),
        layer_controller_options=_layer_controller_from_config(config_module, prefix),
        dynamic_memory_options=_memory_from_config(config_module, prefix),
        recurrent_controller_options=_recurrent_from_config(config_module, prefix),
    )


def _adaptive_controller_stack_from_config(
    config_module: ModuleType,
    prefix: str,
) -> ControllerStackOptions:
    return ControllerStackOptions(
        independent_flag=getattr(config_module, f"{prefix}_INDEPENDENT_FLAG"),
        hidden_dim=getattr(config_module, f"{prefix}_HIDDEN_DIM"),
        num_layers=getattr(config_module, f"{prefix}_NUM_LAYERS"),
        last_layer_bias_option=getattr(
            config_module,
            f"{prefix}_LAST_LAYER_BIAS_OPTION",
        ),
        apply_output_pipeline_flag=getattr(
            config_module,
            f"{prefix}_APPLY_OUTPUT_PIPELINE_FLAG",
        ),
        activation=getattr(config_module, f"{prefix}_ACTIVATION"),
        layer_norm_position=getattr(
            config_module,
            f"{prefix}_LAYER_NORM_POSITION",
        ),
        residual_connection_option=getattr(
            config_module,
            f"{prefix}_RESIDUAL_CONNECTION_OPTION",
        ),
        residual_model_flag=getattr(
            config_module,
            f"{prefix}_RESIDUAL_MODEL_FLAG",
        ),
        dropout_probability=getattr(
            config_module,
            f"{prefix}_DROPOUT_PROBABILITY",
        ),
        bias_flag=getattr(config_module, f"{prefix}_BIAS_FLAG"),
    )


def adaptive_options_from_config(
    config_module: ModuleType,
    prefix: str,
) -> AdaptiveParameterOptions:
    return AdaptiveParameterOptions(
        grouping_scope=getattr(config_module, f"{prefix}_GROUPING_SCOPE"),
        group_count=getattr(config_module, f"{prefix}_GROUP_COUNT"),
        weight_option_flag=getattr(config_module, f"{prefix}_WEIGHT_OPTION_FLAG"),
        weight_option=getattr(config_module, f"{prefix}_WEIGHT_OPTION"),
        generator_depth=getattr(config_module, f"{prefix}_GENERATOR_DEPTH"),
        weight_decay_schedule=getattr(
            config_module,
            f"{prefix}_WEIGHT_DECAY_SCHEDULE",
        ),
        weight_decay_rate=getattr(config_module, f"{prefix}_WEIGHT_DECAY_RATE"),
        weight_decay_warmup_batches=getattr(
            config_module,
            f"{prefix}_WEIGHT_DECAY_WARMUP_BATCHES",
        ),
        weight_normalization_option=getattr(
            config_module,
            f"{prefix}_WEIGHT_NORMALIZATION_OPTION",
        ),
        weight_normalization_position_option=getattr(
            config_module,
            f"{prefix}_WEIGHT_NORMALIZATION_POSITION_OPTION",
        ),
        weight_bank_expansion_factor=getattr(
            config_module,
            f"{prefix}_WEIGHT_BANK_EXPANSION_FACTOR",
        ),
        bias_option_flag=getattr(config_module, f"{prefix}_BIAS_OPTION_FLAG"),
        bias_option=getattr(config_module, f"{prefix}_BIAS_OPTION"),
        bias_decay_schedule=getattr(
            config_module,
            f"{prefix}_BIAS_DECAY_SCHEDULE",
        ),
        bias_decay_rate=getattr(config_module, f"{prefix}_BIAS_DECAY_RATE"),
        bias_decay_warmup_batches=getattr(
            config_module,
            f"{prefix}_BIAS_DECAY_WARMUP_BATCHES",
        ),
        bias_bank_expansion_factor=getattr(
            config_module,
            f"{prefix}_BIAS_BANK_EXPANSION_FACTOR",
        ),
        diagonal_option_flag=getattr(
            config_module,
            f"{prefix}_DIAGONAL_OPTION_FLAG",
        ),
        diagonal_option=getattr(config_module, f"{prefix}_DIAGONAL_OPTION"),
        mask_option_flag=getattr(config_module, f"{prefix}_MASK_OPTION_FLAG"),
        row_mask_option=getattr(config_module, f"{prefix}_ROW_MASK_OPTION"),
        mask_threshold=getattr(config_module, f"{prefix}_MASK_THRESHOLD"),
        mask_surrogate_scale=getattr(
            config_module,
            f"{prefix}_MASK_SURROGATE_SCALE",
        ),
        mask_floor=getattr(config_module, f"{prefix}_MASK_FLOOR"),
        mask_dimension_option=getattr(
            config_module,
            f"{prefix}_MASK_DIMENSION_OPTION",
        ),
        mask_transition_width=getattr(
            config_module,
            f"{prefix}_MASK_TRANSITION_WIDTH",
        ),
        generator_stack_options=SubmoduleStackOptions(
            hidden_dim=getattr(
                config_module,
                f"{prefix}_GENERATOR_STACK_HIDDEN_DIM",
            ),
            num_layers=getattr(
                config_module,
                f"{prefix}_GENERATOR_STACK_NUM_LAYERS",
            ),
            last_layer_bias_option=getattr(
                config_module,
                f"{prefix}_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION",
            ),
            apply_output_pipeline_flag=getattr(
                config_module,
                f"{prefix}_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG",
            ),
            activation=getattr(
                config_module,
                f"{prefix}_GENERATOR_STACK_ACTIVATION",
            ),
            layer_norm_position=getattr(
                config_module,
                f"{prefix}_GENERATOR_STACK_LAYER_NORM_POSITION",
            ),
            residual_connection_option=getattr(
                config_module,
                f"{prefix}_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION",
            ),
            residual_model_flag=getattr(
                config_module,
                f"{prefix}_GENERATOR_STACK_RESIDUAL_MODEL_FLAG",
            ),
            dropout_probability=getattr(
                config_module,
                f"{prefix}_GENERATOR_STACK_DROPOUT_PROBABILITY",
            ),
            bias_flag=getattr(
                config_module,
                f"{prefix}_GENERATOR_STACK_BIAS_FLAG",
            ),
        ),
        weight_generator_stack_options=_adaptive_controller_stack_from_config(
            config_module,
            f"{prefix}_WEIGHT_GENERATOR_STACK",
        ),
        bias_generator_stack_options=_adaptive_controller_stack_from_config(
            config_module,
            f"{prefix}_BIAS_GENERATOR_STACK",
        ),
        diagonal_generator_stack_options=_adaptive_controller_stack_from_config(
            config_module,
            f"{prefix}_DIAGONAL_GENERATOR_STACK",
        ),
        mask_generator_stack_options=_adaptive_controller_stack_from_config(
            config_module,
            f"{prefix}_MASK_GENERATOR_STACK",
        ),
    )


_STACK_OPTION_FIELDS = (
    "hidden_dim",
    "num_layers",
    "last_layer_bias_option",
    "apply_output_pipeline_flag",
    "activation",
    "layer_norm_position",
    "residual_connection_option",
    "residual_model_flag",
    "dropout_probability",
    "bias_flag",
)
_CONTROLLER_STACK_FIELDS = ("independent_flag", *_STACK_OPTION_FIELDS)


def _path_field_map(*, attention: bool) -> dict[str, tuple[str, str]]:
    mapping = {
        "num_layers": ("stack", "num_layers"),
        "bias_flag": ("stack", "bias_flag"),
        "stack_num_layers": ("stack", "num_layers"),
        "stack_bias_flag": ("stack", "bias_flag"),
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


_ADAPTIVE_NESTED_FIELDS = {
    "generator_stack_options",
    "weight_generator_stack_options",
    "bias_generator_stack_options",
    "diagonal_generator_stack_options",
    "mask_generator_stack_options",
}
_ADAPTIVE_VALUE_FIELDS = {
    item.name for item in fields(AdaptiveParameterOptions)
} - _ADAPTIVE_NESTED_FIELDS


def _pop_adaptive_options(
    values: MutableMapping[str, Any],
    prefix: str,
    current: AdaptiveParameterOptions,
) -> AdaptiveParameterOptions:
    updates = {}
    for field_name in _ADAPTIVE_VALUE_FIELDS:
        key = f"{prefix}{field_name}"
        if key in values:
            updates[field_name] = values.pop(key)

    generator = current.generator_stack_options
    generator_updates = {}
    generator_prefix = (
        f"{prefix}generator_stack_" if prefix else "adaptive_generator_stack_"
    )
    for field_name in _STACK_OPTION_FIELDS:
        key = f"{generator_prefix}{field_name}"
        if key in values:
            generator_updates[field_name] = values.pop(key)
    generator = replace(generator, **generator_updates)

    component_stacks = {}
    for component in ("weight", "bias", "diagonal", "mask"):
        field_name = f"{component}_generator_stack_options"
        stack = getattr(current, field_name)
        stack_updates = {}
        component_prefix = f"{prefix}{component}_generator_stack_"
        for stack_field in _CONTROLLER_STACK_FIELDS:
            key = f"{component_prefix}{stack_field}"
            if key in values:
                stack_updates[stack_field] = values.pop(key)
        component_stacks[field_name] = replace(stack, **stack_updates)

    return replace(
        current,
        **updates,
        generator_stack_options=generator,
        **component_stacks,
    )


def _pop_adaptive_broadcast(
    values: MutableMapping[str, Any],
    prefix: str,
    groups: dict[str, AdaptiveParameterOptions],
    names: tuple[str, ...],
) -> None:
    before = dict(values)
    _pop_adaptive_options(values, prefix, AdaptiveParameterOptions())
    consumed = {key: value for key, value in before.items() if key not in values}
    for name in names:
        groups[name] = _pop_adaptive_options(
            dict(consumed),
            prefix,
            groups[name],
        )


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
        _pop_updates(values, "encoder_ff_", _FEED_FORWARD_FIELD_MAP),
        attention=False,
    )
    decoder_feed_forward = _apply_path_updates(
        decoder_feed_forward,
        _pop_updates(values, "decoder_ff_", _FEED_FORWARD_FIELD_MAP),
        attention=False,
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


def runtime_from_config() -> RuntimeOptions:
    stack = TransformerStackOptions(
        num_layers=config.ENCODER_NUM_LAYERS,
        layer_norm_position=config.ENCODER_LAYER_NORM_POSITION,
        stack_gate_flag=config.STACK_GATE_FLAG,
        stack_halting_flag=config.STACK_HALTING_FLAG,
        halting_threshold=config.HALTING_THRESHOLD,
        memory_flag=config.MEMORY_FLAG,
        recurrent_flag=config.RECURRENT_FLAG,
        recurrent_stack_gate_flag=config.RECURRENT_STACK_GATE_FLAG,
        recurrent_stack_halting_flag=config.RECURRENT_STACK_HALTING_FLAG,
        recurrent_halting_threshold=config.RECURRENT_HALTING_THRESHOLD,
        recurrent_max_steps=config.RECURRENT_MAX_STEPS,
        stack_residual_connection_option=config.STACK_RESIDUAL_CONNECTION_OPTION,
        stack_residual_model_flag=config.STACK_RESIDUAL_MODEL_FLAG,
        recurrent_residual_connection_option=(
            config.RECURRENT_RESIDUAL_CONNECTION_OPTION
        ),
        recurrent_residual_model_flag=config.RECURRENT_RESIDUAL_MODEL_FLAG,
    )
    experts = ExpertOptions(
        use_kv_expert_models_flag=(config.EXPERT_ATTENTION_USE_KV_EXPERT_MODELS_FLAG),
        num_experts=config.NUM_EXPERTS,
        top_k=config.TOP_K,
        dropped_token_behavior=config.DROPPED_TOKEN_BEHAVIOR,
        compute_expert_mixture_flag=config.COMPUTE_EXPERT_MIXTURE_FLAG,
        weighted_parameters_flag=config.WEIGHTED_PARAMETERS_FLAG,
        weighting_position_option=config.WEIGHTING_POSITION_OPTION,
        routing_initialization_mode=config.ROUTING_INITIALIZATION_MODE,
        sampler_threshold=config.SAMPLER_THRESHOLD,
        sampler_filter_above_threshold=config.SAMPLER_FILTER_ABOVE_THRESHOLD,
        sampler_num_topk_samples=config.SAMPLER_NUM_TOPK_SAMPLES,
        normalize_probabilities_flag=config.NORMALIZE_PROBABILITIES_FLAG,
        sampler_noisy_topk_flag=config.SAMPLER_NOISY_TOPK_FLAG,
        coefficient_of_variation_loss_weight=(
            config.COEFFICIENT_OF_VARIATION_LOSS_WEIGHT
        ),
        switch_loss_weight=config.SWITCH_LOSS_WEIGHT,
        zero_centred_loss_weight=config.ZERO_CENTRED_LOSS_WEIGHT,
        mutual_information_loss_weight=config.MUTUAL_INFORMATION_LOSS_WEIGHT,
        capacity_factor=config.CAPACITY_FACTOR,
        router_noisy_topk_flag=config.ROUTER_NOISY_TOPK_FLAG,
        router_path_options=feed_forward_options_from_config(config, "ROUTER"),
        expert_path_options=feed_forward_options_from_config(config, "EXPERT"),
    )

    return RuntimeOptions(
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        vocab_size=config.VOCAB_SIZE,
        model_dim=config.MODEL_DIM,
        source_sequence_length=config.SOURCE_SEQUENCE_LENGTH,
        target_sequence_length=config.TARGET_SEQUENCE_LENGTH,
        dropout_probability=config.DROPOUT_PROBABILITY,
        residual_stack_independent_flag=(
            config.RESIDUAL_STACK_INDEPENDENT_FLAG
        ),
        residual_stack_hidden_dim=config.RESIDUAL_STACK_HIDDEN_DIM,
        residual_stack_layer_norm_position=(
            config.RESIDUAL_STACK_LAYER_NORM_POSITION
        ),
        residual_stack_num_layers=config.RESIDUAL_STACK_NUM_LAYERS,
        residual_stack_activation=config.RESIDUAL_STACK_ACTIVATION,
        residual_stack_residual_connection_option=(
            config.RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_stack_residual_model_flag=(
            config.RESIDUAL_STACK_RESIDUAL_MODEL_FLAG
        ),
        residual_stack_dropout_probability=(
            config.RESIDUAL_STACK_DROPOUT_PROBABILITY
        ),
        residual_stack_last_layer_bias_option=(
            config.RESIDUAL_STACK_LAST_LAYER_BIAS_OPTION
        ),
        residual_stack_apply_output_pipeline_flag=(
            config.RESIDUAL_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        residual_stack_bias_flag=config.RESIDUAL_STACK_BIAS_FLAG,
        positional_embedding_option=config.POSITIONAL_EMBEDDING_OPTION,
        encoder_options=stack,
        decoder_options=replace(
            stack,
            num_layers=config.DECODER_NUM_LAYERS,
            layer_norm_position=config.DECODER_LAYER_NORM_POSITION,
        ),
        encoder_attention_options=attention_options_from_config(config, "ENCODER_ATTN"),
        decoder_self_attention_options=attention_options_from_config(
            config,
            "DECODER_SELF_ATTN",
        ),
        decoder_cross_attention_options=attention_options_from_config(
            config,
            "DECODER_CROSS_ATTN",
        ),
        encoder_feed_forward_options=feed_forward_options_from_config(
            config,
            "ENCODER_FF",
        ),
        decoder_feed_forward_options=feed_forward_options_from_config(
            config,
            "DECODER_FF",
        ),
        attention_expert_options=experts,
        feed_forward_expert_options=experts,
        attention_projection_adaptive_options=adaptive_options_from_config(
            config, "ATTENTION_PROJECTION_ADAPTIVE"
        ),
        attention_expert_adaptive_options=adaptive_options_from_config(
            config,
            "ATTENTION_EXPERT_ADAPTIVE",
        ),
        router_adaptive_options=adaptive_options_from_config(
            config,
            "ROUTER_ADAPTIVE",
        ),
        feed_forward_adaptive_options=adaptive_options_from_config(
            config,
            "FEED_FORWARD_ADAPTIVE",
        ),
        encoder_attention_adaptive_options=adaptive_options_from_config(
            config,
            "ENCODER_ATTN_ADAPTIVE",
        ),
        decoder_self_attention_adaptive_options=adaptive_options_from_config(
            config,
            "DECODER_SELF_ATTN_ADAPTIVE",
        ),
        decoder_cross_attention_adaptive_options=adaptive_options_from_config(
            config,
            "DECODER_CROSS_ATTN_ADAPTIVE",
        ),
        encoder_feed_forward_adaptive_options=adaptive_options_from_config(
            config,
            "ENCODER_FF_ADAPTIVE",
        ),
        decoder_feed_forward_adaptive_options=adaptive_options_from_config(
            config,
            "DECODER_FF_ADAPTIVE",
        ),
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
    if "expert_attention_use_kv_expert_models_flag" in values:
        values["use_kv_expert_models_flag"] = values.pop(
            "expert_attention_use_kv_expert_models_flag"
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

    router_updates = _pop_updates(values, "router_", _FEED_FORWARD_FIELD_MAP)
    expert_path_updates = _pop_updates(values, "expert_", _FEED_FORWARD_FIELD_MAP)
    attention_experts = replace(
        attention_experts,
        router_path_options=_apply_path_updates(
            attention_experts.router_path_options,
            router_updates,
            attention=False,
        ),
        expert_path_options=_apply_path_updates(
            attention_experts.expert_path_options,
            expert_path_updates,
            attention=False,
        ),
    )
    feed_forward_experts = replace(
        feed_forward_experts,
        router_path_options=_apply_path_updates(
            feed_forward_experts.router_path_options,
            router_updates,
            attention=False,
        ),
        expert_path_options=_apply_path_updates(
            feed_forward_experts.expert_path_options,
            expert_path_updates,
            attention=False,
        ),
    )

    adaptive_groups = {
        "attention_projection": runtime.attention_projection_adaptive_options,
        "attention_expert": runtime.attention_expert_adaptive_options,
        "router": runtime.router_adaptive_options,
        "feed_forward": runtime.feed_forward_adaptive_options,
        "encoder_attention": runtime.encoder_attention_adaptive_options,
        "decoder_self_attention": runtime.decoder_self_attention_adaptive_options,
        "decoder_cross_attention": runtime.decoder_cross_attention_adaptive_options,
        "encoder_feed_forward": runtime.encoder_feed_forward_adaptive_options,
        "decoder_feed_forward": runtime.decoder_feed_forward_adaptive_options,
    }
    _pop_adaptive_broadcast(
        values,
        "",
        adaptive_groups,
        tuple(adaptive_groups),
    )
    for prefix, names in (
        (
            "attention_projection_adaptive_",
            (
                "attention_projection",
                "encoder_attention",
                "decoder_self_attention",
                "decoder_cross_attention",
            ),
        ),
        (
            "attn_adaptive_",
            (
                "attention_projection",
                "encoder_attention",
                "decoder_self_attention",
                "decoder_cross_attention",
            ),
        ),
        ("attention_expert_adaptive_", ("attention_expert",)),
        (
            "expert_adaptive_",
            (
                "attention_expert",
                "feed_forward",
                "encoder_feed_forward",
                "decoder_feed_forward",
            ),
        ),
        ("router_adaptive_", ("router",)),
        (
            "feed_forward_adaptive_",
            (
                "feed_forward",
                "encoder_feed_forward",
                "decoder_feed_forward",
            ),
        ),
        (
            "ff_adaptive_",
            (
                "feed_forward",
                "encoder_feed_forward",
                "decoder_feed_forward",
            ),
        ),
    ):
        _pop_adaptive_broadcast(
            values,
            prefix,
            adaptive_groups,
            names,
        )
    for name, prefixes in (
        ("encoder_attention", ("encoder_attn_", "encoder_attn_adaptive_")),
        (
            "decoder_self_attention",
            ("decoder_self_attn_", "decoder_self_attn_adaptive_"),
        ),
        (
            "decoder_cross_attention",
            ("decoder_cross_attn_", "decoder_cross_attn_adaptive_"),
        ),
        ("encoder_feed_forward", ("encoder_ff_", "encoder_ff_adaptive_")),
        ("decoder_feed_forward", ("decoder_ff_", "decoder_ff_adaptive_")),
    ):
        for prefix in prefixes:
            adaptive_groups[name] = _pop_adaptive_options(
                values,
                prefix,
                adaptive_groups[name],
            )
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
        attention_projection_adaptive_options=adaptive_groups["attention_projection"],
        attention_expert_adaptive_options=adaptive_groups["attention_expert"],
        router_adaptive_options=adaptive_groups["router"],
        feed_forward_adaptive_options=adaptive_groups["feed_forward"],
        encoder_attention_adaptive_options=adaptive_groups["encoder_attention"],
        decoder_self_attention_adaptive_options=adaptive_groups[
            "decoder_self_attention"
        ],
        decoder_cross_attention_adaptive_options=adaptive_groups[
            "decoder_cross_attention"
        ],
        encoder_feed_forward_adaptive_options=adaptive_groups["encoder_feed_forward"],
        decoder_feed_forward_adaptive_options=adaptive_groups["decoder_feed_forward"],
    )


DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_config()

__all__ = ["DEFAULT_RUNTIME", "runtime_from_config", "runtime_from_flat"]
