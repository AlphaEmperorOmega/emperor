from collections.abc import MutableMapping
from dataclasses import replace
from typing import Any

from emperor.transformer._options.records import (
    TransformerAttentionOptions,
    TransformerFeedForwardOptions,
    TransformerPathOptions,
)

_STACK_FIELDS = (
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
_CONTROLLER_STACK_FIELDS = ("independent_flag", *_STACK_FIELDS)


def _path_field_map(*, attention: bool) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    if attention:
        mapping.update(
            {
                "num_heads": ("path", "num_heads"),
                "add_key_value_bias_flag": (
                    "path",
                    "add_key_value_bias_flag",
                ),
                "zero_attention_flag": ("path", "zero_attention_flag"),
                "num_layers": ("stack", "num_layers"),
                "bias_flag": ("stack", "bias_flag"),
            }
        )
    else:
        mapping.update(
            {
                "num_layers": ("stack", "num_layers"),
                "bias_flag": ("stack", "bias_flag"),
            }
        )
    for field in _STACK_FIELDS:
        if field not in {"num_layers", "bias_flag"}:
            mapping[f"stack_{field}"] = ("stack", field)
    mapping.update(
        {
            "gate_flag": ("controller", "stack_gate_flag"),
            "gate_option": ("controller", "gate_option"),
            "gate_activation": ("controller", "gate_activation"),
            "halting_flag": ("controller", "stack_halting_flag"),
            "halting_option": ("controller", "halting_option"),
            "halting_threshold": ("controller", "halting_threshold"),
            "halting_dropout": ("controller", "halting_dropout"),
            "halting_hidden_state_mode": (
                "controller",
                "halting_hidden_state_mode",
            ),
            "memory_flag": ("memory", "memory_flag"),
            "memory_option": ("memory", "memory_option"),
            "memory_position_option": ("memory", "memory_position_option"),
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
            "recurrent_gate_flag": ("recurrent", "recurrent_gate_flag"),
            "recurrent_gate_option": ("recurrent", "recurrent_gate_option"),
            "recurrent_gate_activation": (
                "recurrent",
                "recurrent_gate_activation",
            ),
            "recurrent_halting_flag": (
                "recurrent",
                "recurrent_halting_flag",
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
        **{outer_field: replace(getattr(source, outer_field), **{inner_field: value})},
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
        component, field = field_map[suffix]
        if component == "path":
            path = replace(path, **{field: value})
        elif component == "stack":
            if attention and field == "num_layers":
                path = replace(path, num_layers=value)
                stack = path.stack_options
            elif attention and field == "bias_flag":
                path = replace(path, projection_bias_flag=value)
                stack = path.stack_options
            elif not attention and field in {
                "hidden_dim",
                "num_layers",
                "bias_flag",
            }:
                path = replace(path, **{field: value})
                stack = path.stack_options
            else:
                stack = replace(stack, **{field: value})
        elif component == "controller":
            controller = replace(controller, **{field: value})
        elif component == "memory":
            memory = replace(memory, **{field: value})
        elif component == "recurrent":
            recurrent = replace(recurrent, **{field: value})
        elif component == "gate_stack":
            controller = _replace_nested(controller, field, value)
        elif component == "halting_stack":
            controller = _replace_nested(controller, field, value)
        elif component == "memory_stack":
            memory = _replace_nested(memory, field, value)
        elif component == "recurrent_gate_stack":
            recurrent = _replace_nested(recurrent, field, value)
        elif component == "recurrent_halting_stack":
            recurrent = _replace_nested(recurrent, field, value)
        else:
            raise ValueError(f"Unsupported transformer override component: {component}")
    return replace(
        path,
        stack_options=stack,
        layer_controller_options=controller,
        dynamic_memory_options=memory,
        recurrent_controller_options=recurrent,
    )


def _values_match(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        result = left == right
        return result if isinstance(result, bool) else bool(result)
    except (TypeError, ValueError):
        return False


def _normalize_legacy_aliases(values: MutableMapping[str, Any]) -> None:
    aliases = {
        "attn_projection_bias_flag": "attn_bias_flag",
        "feed_forward_hidden_dim": "ff_stack_hidden_dim",
        "feed_forward_num_layers": "ff_num_layers",
        "encoder_attn_projection_bias_flag": "encoder_attn_bias_flag",
        "decoder_self_attn_projection_bias_flag": ("decoder_self_attn_bias_flag"),
        "decoder_cross_attn_projection_bias_flag": ("decoder_cross_attn_bias_flag"),
        "encoder_feed_forward_hidden_dim": "encoder_ff_stack_hidden_dim",
        "encoder_feed_forward_num_layers": "encoder_ff_num_layers",
        "decoder_feed_forward_hidden_dim": "decoder_ff_stack_hidden_dim",
        "decoder_feed_forward_num_layers": "decoder_ff_num_layers",
    }
    for legacy, canonical in aliases.items():
        if legacy not in values:
            continue
        legacy_value = values.pop(legacy)
        if canonical in values and not _values_match(values[canonical], legacy_value):
            raise ValueError(
                f"Conflicting values for canonical option {canonical!r} and "
                f"legacy alias {legacy!r}."
            )
        values.setdefault(canonical, legacy_value)


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


def resolve_transformer_path_options(
    values: MutableMapping[str, Any],
    defaults: TransformerPathOptions,
) -> TransformerPathOptions:
    """Consume and resolve flat attention/feed-forward path options.

    Resolution is defaults, then unscoped broadcast values, then a scoped path
    override. Legacy feed-forward and projection-bias names are normalized first.
    Unrecognized values are intentionally left in ``values`` for the package-local
    runtime resolver to validate.
    """

    _normalize_legacy_aliases(values)
    encoder_attention = values.pop(
        "encoder_attention_options", defaults.encoder_attention_options
    )
    decoder_self_attention = values.pop(
        "decoder_self_attention_options", defaults.decoder_self_attention_options
    )
    decoder_cross_attention = values.pop(
        "decoder_cross_attention_options", defaults.decoder_cross_attention_options
    )
    encoder_feed_forward = values.pop(
        "encoder_feed_forward_options", defaults.encoder_feed_forward_options
    )
    decoder_feed_forward = values.pop(
        "decoder_feed_forward_options", defaults.decoder_feed_forward_options
    )

    attention_broadcast = _pop_updates(values, "attn_", _ATTENTION_FIELD_MAP)
    feed_forward_broadcast = _pop_updates(values, "ff_", _FEED_FORWARD_FIELD_MAP)
    encoder_attention = _apply_path_updates(
        encoder_attention, attention_broadcast, attention=True
    )
    decoder_self_attention = _apply_path_updates(
        decoder_self_attention, attention_broadcast, attention=True
    )
    decoder_cross_attention = _apply_path_updates(
        decoder_cross_attention, attention_broadcast, attention=True
    )
    encoder_feed_forward = _apply_path_updates(
        encoder_feed_forward, feed_forward_broadcast, attention=False
    )
    decoder_feed_forward = _apply_path_updates(
        decoder_feed_forward, feed_forward_broadcast, attention=False
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
    return TransformerPathOptions(
        encoder_attention_options=encoder_attention,
        decoder_self_attention_options=decoder_self_attention,
        decoder_cross_attention_options=decoder_cross_attention,
        encoder_feed_forward_options=encoder_feed_forward,
        decoder_feed_forward_options=decoder_feed_forward,
    )


def expand_transformer_path_locks(locks: dict[str, Any]) -> dict[str, Any]:
    """Expand an unscoped preset lock to every affected Transformer path."""

    expanded = dict(locks)
    for key, value in tuple(locks.items()):
        if key.startswith("attn_"):
            suffix = key[len("attn_") :]
            for prefix in (
                "encoder_attn_",
                "decoder_self_attn_",
                "decoder_cross_attn_",
            ):
                expanded[f"{prefix}{suffix}"] = value
            if suffix == "bias_flag":
                for prefix in (
                    "encoder_attn_",
                    "decoder_self_attn_",
                    "decoder_cross_attn_",
                ):
                    expanded[f"{prefix}projection_bias_flag"] = value
        elif key.startswith("ff_"):
            suffix = key[len("ff_") :]
            expanded[f"encoder_ff_{suffix}"] = value
            expanded[f"decoder_ff_{suffix}"] = value
        elif key == "feed_forward_hidden_dim":
            expanded["encoder_feed_forward_hidden_dim"] = value
            expanded["decoder_feed_forward_hidden_dim"] = value
            expanded["encoder_ff_stack_hidden_dim"] = value
            expanded["decoder_ff_stack_hidden_dim"] = value
        elif key == "feed_forward_num_layers":
            expanded["encoder_feed_forward_num_layers"] = value
            expanded["decoder_feed_forward_num_layers"] = value
            expanded["encoder_ff_num_layers"] = value
            expanded["decoder_ff_num_layers"] = value
    return expanded
