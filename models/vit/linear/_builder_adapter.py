# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from types import ModuleType
from typing import Any

from models.vit.linear import _config_defaults as config_defaults
from models.vit.linear.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    SubmoduleStackSource,
)

_TOP_LEVEL_KEYS = {"batch_size", "learning_rate", "input_dim", "output_dim"}
_VIT_GROUPED_KEYS = {
    *_TOP_LEVEL_KEYS,
    "patch_options",
    "encoder_options",
    "positional_embedding_options",
    "attention_options",
    "feed_forward_options",
    "output_options",
    "attention_projection_stack_options",
    "attention_projection_layer_controller_options",
    "attention_projection_dynamic_memory_options",
    "attention_projection_recurrent_controller_options",
    "feed_forward_stack_options",
    "feed_forward_layer_controller_options",
    "feed_forward_dynamic_memory_options",
    "feed_forward_recurrent_controller_options",
    "stack_options",
    "submodule_stack_options",
    "layer_controller_options",
    "dynamic_memory_options",
    "recurrent_controller_options",
}
_CONTROLLER_STACK_FIELD_MAP = {
    "independent_flag": "independent_flag",
    "hidden_dim": "hidden_dim",
    "num_layers": "num_layers",
    "last_layer_bias_option": "last_layer_bias_option",
    "apply_output_pipeline_flag": "apply_output_pipeline_flag",
    "activation": "activation",
    "layer_norm_position": "layer_norm_position",
    "residual_connection_option": "residual_connection_option",
    "dropout_probability": "dropout_probability",
    "bias_flag": "bias_flag",
}
_SUBMODULE_STACK_FIELD_MAP = {
    "hidden_dim": "hidden_dim",
    "num_layers": "num_layers",
    "last_layer_bias_option": "last_layer_bias_option",
    "apply_output_pipeline_flag": "apply_output_pipeline_flag",
    "activation": "activation",
    "layer_norm_position": "layer_norm_position",
    "residual_connection_option": "residual_connection_option",
    "dropout_probability": "dropout_probability",
    "bias_flag": "bias_flag",
}


def linear_builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any], config_module: ModuleType
) -> dict[str, Any]:
    kwargs = dict(flat_kwargs)
    consumed: set[str] = set()
    builder_kwargs = _top_level_kwargs(kwargs, consumed)
    builder_kwargs.update(_vit_builder_kwargs(kwargs, config_module, consumed))
    builder_kwargs.update(_leftover_kwargs(kwargs, consumed))
    return builder_kwargs


def _top_level_kwargs(kwargs: dict[str, Any], consumed: set[str]) -> dict[str, Any]:
    consumed.update(key for key in _TOP_LEVEL_KEYS if key in kwargs)
    return {key: kwargs[key] for key in _TOP_LEVEL_KEYS if key in kwargs}


def _vit_builder_kwargs(
    kwargs: dict[str, Any], config_module: ModuleType, consumed: set[str]
) -> dict[str, Any]:
    builder_kwargs: dict[str, Any] = {}
    _maybe_set(
        builder_kwargs,
        "patch_options",
        _patch_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "PATCH_OPTIONS",
    )
    _maybe_set(
        builder_kwargs,
        "encoder_options",
        _encoder_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "ENCODER_OPTIONS",
    )
    _maybe_set(
        builder_kwargs,
        "positional_embedding_options",
        _positional_embedding_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "POSITIONAL_EMBEDDING_OPTIONS",
    )
    _maybe_set(
        builder_kwargs,
        "attention_options",
        _attention_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "ATTENTION_OPTIONS",
    )
    _maybe_set(
        builder_kwargs,
        "feed_forward_options",
        _feed_forward_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "FEED_FORWARD_OPTIONS",
    )
    _maybe_set(
        builder_kwargs,
        "output_options",
        _output_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "OUTPUT_OPTIONS",
    )
    _maybe_set(
        builder_kwargs,
        "stack_options",
        _main_stack_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "STACK_OPTIONS",
    )
    _maybe_set(
        builder_kwargs,
        "submodule_stack_options",
        _submodule_stack_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "SUBMODULE_STACK_OPTIONS",
        flat_prefix="submodule_stack",
    )
    _maybe_set(
        builder_kwargs,
        "layer_controller_options",
        _layer_controller_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "LAYER_CONTROLLER_OPTIONS",
        flat_prefix="",
        gate_stack_prefix="gate_stack",
        halting_stack_prefix="halting_stack",
    )
    _maybe_set(
        builder_kwargs,
        "dynamic_memory_options",
        _dynamic_memory_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "DYNAMIC_MEMORY_OPTIONS",
        flat_prefix="",
        memory_stack_prefix="memory_stack",
    )
    _maybe_set(
        builder_kwargs,
        "recurrent_controller_options",
        _recurrent_controller_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "RECURRENT_CONTROLLER_OPTIONS",
        flat_prefix="recurrent",
        gate_stack_prefix="recurrent_gate_stack",
        halting_stack_prefix="recurrent_halting_stack",
    )
    _maybe_set(
        builder_kwargs,
        "attention_projection_stack_options",
        _submodule_stack_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "ATTENTION_PROJECTION_STACK_OPTIONS",
        flat_prefix="attn_stack",
    )
    _maybe_set(
        builder_kwargs,
        "attention_projection_layer_controller_options",
        _layer_controller_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "ATTENTION_PROJECTION_LAYER_CONTROLLER_OPTIONS",
        flat_prefix="attn",
        gate_stack_prefix="attn_gate_stack",
        halting_stack_prefix="attn_halting_stack",
    )
    _maybe_set(
        builder_kwargs,
        "attention_projection_dynamic_memory_options",
        _dynamic_memory_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "ATTENTION_PROJECTION_DYNAMIC_MEMORY_OPTIONS",
        flat_prefix="attn",
        memory_stack_prefix="attn_memory_stack",
    )
    _maybe_set(
        builder_kwargs,
        "attention_projection_recurrent_controller_options",
        _recurrent_controller_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "ATTENTION_PROJECTION_RECURRENT_CONTROLLER_OPTIONS",
        flat_prefix="attn_recurrent",
        gate_stack_prefix="attn_recurrent_gate_stack",
        halting_stack_prefix="attn_recurrent_halting_stack",
    )
    _maybe_set(
        builder_kwargs,
        "feed_forward_stack_options",
        _submodule_stack_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "FEED_FORWARD_STACK_OPTIONS",
        flat_prefix="ff_stack",
    )
    _maybe_set(
        builder_kwargs,
        "feed_forward_layer_controller_options",
        _layer_controller_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "FEED_FORWARD_LAYER_CONTROLLER_OPTIONS",
        flat_prefix="ff",
        gate_stack_prefix="ff_gate_stack",
        halting_stack_prefix="ff_halting_stack",
    )
    _maybe_set(
        builder_kwargs,
        "feed_forward_dynamic_memory_options",
        _dynamic_memory_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "FEED_FORWARD_DYNAMIC_MEMORY_OPTIONS",
        flat_prefix="ff",
        memory_stack_prefix="ff_memory_stack",
    )
    _maybe_set(
        builder_kwargs,
        "feed_forward_recurrent_controller_options",
        _recurrent_controller_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        "FEED_FORWARD_RECURRENT_CONTROLLER_OPTIONS",
        flat_prefix="ff_recurrent",
        gate_stack_prefix="ff_recurrent_gate_stack",
        halting_stack_prefix="ff_recurrent_halting_stack",
    )
    _ensure_control_dependencies(builder_kwargs, config_module)
    return builder_kwargs


def _patch_options_from_kwargs(options, kwargs: dict[str, Any]):
    return replace(
        options,
        **_updates(
            kwargs,
            {
                "image_patch_size": "patch_size",
                "input_channels": "input_channels",
                "image_height": "image_height",
                "patch_dropout_probability": "dropout_probability",
                "patch_bias_flag": "bias_flag",
            },
        ),
    )


def _encoder_options_from_kwargs(options, kwargs: dict[str, Any]):
    return replace(
        options,
        **_updates(
            kwargs,
            {
                "hidden_dim": "hidden_dim",
                "stack_num_layers": "num_layers",
                "stack_activation": "activation",
                "stack_dropout_probability": "dropout_probability",
                "layer_norm_position": "layer_norm_position",
                "stack_layer_norm_position": "layer_norm_position",
            },
        ),
    )


def _positional_embedding_options_from_kwargs(options, kwargs: dict[str, Any]):
    return replace(
        options,
        **_updates(
            kwargs,
            {
                "positional_embedding_option": "option",
                "positional_embedding_padding_idx": "padding_idx",
                "positional_embedding_auto_expand_flag": "auto_expand_flag",
            },
        ),
    )


def _attention_options_from_kwargs(options, kwargs: dict[str, Any]):
    return replace(
        options,
        **_updates(
            kwargs,
            {
                "attn_num_heads": "num_heads",
                "attn_num_layers": "num_layers",
                "attn_bias_flag": "bias_flag",
                "attn_add_key_value_bias_flag": "add_key_value_bias_flag",
            },
        ),
    )


def _feed_forward_options_from_kwargs(options, kwargs: dict[str, Any]):
    return replace(
        options,
        **_updates(
            kwargs, {"ff_num_layers": "num_layers", "ff_bias_flag": "bias_flag"}
        ),
    )


def _output_options_from_kwargs(options, kwargs: dict[str, Any]):
    return replace(options, **_updates(kwargs, {"output_bias_flag": "bias_flag"}))


def _main_stack_options_from_kwargs(
    options: MainLayerStackOptions, kwargs: dict[str, Any]
):
    return replace(
        options,
        **_updates(
            kwargs,
            {
                "stack_bias_flag": "bias_flag",
                "stack_layer_norm_position": "layer_norm_position",
                "layer_norm_position": "layer_norm_position",
                "stack_num_layers": "num_layers",
                "stack_activation": "activation",
                "stack_residual_connection_option": "residual_connection_option",
                "stack_dropout_probability": "dropout_probability",
                "stack_last_layer_bias_option": "last_layer_bias_option",
                "stack_apply_output_pipeline_flag": "apply_output_pipeline_flag",
            },
        ),
    )


def _submodule_stack_options_from_kwargs(
    options: SubmoduleStackOptions, kwargs: dict[str, Any], *, flat_prefix: str
):
    return replace(
        options,
        **_updates(
            kwargs,
            {
                f"{flat_prefix}_{flat_field}": dataclass_field
                for flat_field, dataclass_field in _SUBMODULE_STACK_FIELD_MAP.items()
            },
        ),
    )


def _layer_controller_options_from_kwargs(
    options: LayerControllerOptions,
    kwargs: dict[str, Any],
    *,
    flat_prefix: str,
    gate_stack_prefix: str,
    halting_stack_prefix: str,
):
    prefix = f"{flat_prefix}_" if flat_prefix else ""
    flag_map = (
        {
            "stack_gate_flag": "stack_gate_flag",
            "gate_flag": "stack_gate_flag",
            "gate_option": "gate_option",
            "gate_activation": "gate_activation",
            "stack_halting_flag": "stack_halting_flag",
            "halting_flag": "stack_halting_flag",
            "halting_threshold": "halting_threshold",
            "halting_dropout": "halting_dropout",
            "halting_hidden_state_mode": "halting_hidden_state_mode",
        }
        if not flat_prefix
        else {
            f"{prefix}gate_flag": "stack_gate_flag",
            f"{prefix}gate_option": "gate_option",
            f"{prefix}gate_activation": "gate_activation",
            f"{prefix}halting_flag": "stack_halting_flag",
            f"{prefix}halting_threshold": "halting_threshold",
            f"{prefix}halting_dropout": "halting_dropout",
            f"{prefix}halting_hidden_state_mode": "halting_hidden_state_mode",
        }
    )
    updates = _updates(kwargs, flag_map)
    updates["gate_stack_source"] = _controller_stack_source_from_kwargs(
        options.gate_stack_source, kwargs, gate_stack_prefix
    )
    updates["halting_stack_source"] = _controller_stack_source_from_kwargs(
        options.halting_stack_source, kwargs, halting_stack_prefix
    )
    return replace(options, **updates)


def _dynamic_memory_options_from_kwargs(
    options: DynamicMemoryOptions,
    kwargs: dict[str, Any],
    *,
    flat_prefix: str,
    memory_stack_prefix: str,
):
    prefix = f"{flat_prefix}_" if flat_prefix else ""
    updates = _updates(
        kwargs,
        {
            f"{prefix}memory_flag": "memory_flag",
            f"{prefix}memory_option": "memory_option",
            f"{prefix}memory_position_option": "memory_position_option",
            f"{prefix}memory_test_time_training_learning_rate": "memory_test_time_training_learning_rate",
            f"{prefix}memory_test_time_training_num_inner_steps": "memory_test_time_training_num_inner_steps",
        },
    )
    updates["memory_stack_source"] = _controller_stack_source_from_kwargs(
        options.memory_stack_source, kwargs, memory_stack_prefix
    )
    return replace(options, **updates)


def _recurrent_controller_options_from_kwargs(
    options: RecurrentControllerOptions,
    kwargs: dict[str, Any],
    *,
    flat_prefix: str,
    gate_stack_prefix: str,
    halting_stack_prefix: str,
):
    prefix = f"{flat_prefix}_"
    updates = _updates(
        kwargs,
        {
            f"{prefix}flag": "recurrent_flag",
            f"{prefix}max_steps": "recurrent_max_steps",
            f"{prefix}layer_norm_position": "recurrent_layer_norm_position",
            f"{prefix}gate_flag": "recurrent_gate_flag",
            f"{prefix}gate_option": "recurrent_gate_option",
            f"{prefix}gate_activation": "recurrent_gate_activation",
            f"{prefix}halting_flag": "recurrent_halting_flag",
            f"{prefix}halting_threshold": "recurrent_halting_threshold",
            f"{prefix}halting_dropout": "recurrent_halting_dropout",
            f"{prefix}halting_hidden_state_mode": "recurrent_halting_hidden_state_mode",
        },
    )
    updates["recurrent_gate_stack_source"] = _controller_stack_source_from_kwargs(
        options.recurrent_gate_stack_source, kwargs, gate_stack_prefix
    )
    updates["recurrent_halting_stack_source"] = _controller_stack_source_from_kwargs(
        options.recurrent_halting_stack_source, kwargs, halting_stack_prefix
    )
    return replace(options, **updates)


def _controller_stack_source_from_kwargs(
    source: SubmoduleStackSource, kwargs: dict[str, Any], flat_prefix: str
):
    return replace(
        source,
        **_updates(
            kwargs,
            {
                f"{flat_prefix}_{flat_field}": dataclass_field
                for flat_field, dataclass_field in _CONTROLLER_STACK_FIELD_MAP.items()
            },
        ),
    )


def _maybe_set(
    builder_kwargs: dict[str, Any],
    builder_key: str,
    factory: Callable,
    kwargs: dict[str, Any],
    consumed: set[str],
    config_module: ModuleType,
    config_attr: str,
    **factory_kwargs,
) -> None:
    provided = kwargs.get(builder_key)
    relevant = _factory_relevant_keys(factory, factory_kwargs)
    if provided is None and (not any(key in kwargs for key in relevant)):
        return
    base = (
        provided
        if provided is not None
        else _default_config_options(config_module, config_attr)
    )
    consumed.add(builder_key)
    consumed.update(key for key in relevant if key in kwargs)
    builder_kwargs[builder_key] = factory(base, kwargs, **factory_kwargs)


def _default_config_options(config_module: ModuleType, config_attr: str) -> Any:
    option_factories: dict[str, Callable[[ModuleType], Any]] = {
        "PATCH_OPTIONS": config_defaults.vit_patch_options,
        "POSITIONAL_EMBEDDING_OPTIONS": config_defaults.vit_positional_embedding_options,
        "ENCODER_OPTIONS": config_defaults.vit_encoder_options,
        "ATTENTION_OPTIONS": config_defaults.vit_attention_options,
        "FEED_FORWARD_OPTIONS": config_defaults.vit_feed_forward_options,
        "OUTPUT_OPTIONS": config_defaults.vit_output_options,
        "STACK_OPTIONS": config_defaults.main_layer_stack_options,
        "SUBMODULE_STACK_OPTIONS": lambda config_object: (
            config_defaults.linears_submodule_stack_options(
                config_object, "SUBMODULE_STACK"
            )
        ),
        "ATTENTION_PROJECTION_STACK_OPTIONS": lambda config_object: (
            config_defaults.linears_submodule_stack_options(
                config_object,
                "ATTN_STACK",
                num_layers_key="ATTN_NUM_LAYERS",
                bias_key="ATTN_BIAS_FLAG",
            )
        ),
        "FEED_FORWARD_STACK_OPTIONS": lambda config_object: (
            config_defaults.linears_submodule_stack_options(
                config_object,
                "FF_STACK",
                num_layers_key="FF_NUM_LAYERS",
                bias_key="FF_BIAS_FLAG",
            )
        ),
        "LAYER_CONTROLLER_OPTIONS": lambda config_object: (
            config_defaults.linears_layer_controller_options(
                config_object,
                gate_prefix="GATE",
                gate_stack_prefix="GATE_STACK",
                halting_prefix="HALTING",
                halting_stack_prefix="HALTING_STACK",
            )
        ),
        "ATTENTION_PROJECTION_LAYER_CONTROLLER_OPTIONS": lambda config_object: (
            config_defaults.linears_layer_controller_options(
                config_object,
                gate_prefix="ATTN_GATE",
                gate_stack_prefix="ATTN_GATE_STACK",
                halting_prefix="ATTN_HALTING",
                halting_stack_prefix="ATTN_HALTING_STACK",
            )
        ),
        "FEED_FORWARD_LAYER_CONTROLLER_OPTIONS": lambda config_object: (
            config_defaults.linears_layer_controller_options(
                config_object,
                gate_prefix="FF_GATE",
                gate_stack_prefix="FF_GATE_STACK",
                halting_prefix="FF_HALTING",
                halting_stack_prefix="FF_HALTING_STACK",
            )
        ),
        "DYNAMIC_MEMORY_OPTIONS": lambda config_object: (
            config_defaults.linears_dynamic_memory_options(
                config_object,
                memory_prefix="MEMORY",
                memory_stack_prefix="MEMORY_STACK",
            )
        ),
        "ATTENTION_PROJECTION_DYNAMIC_MEMORY_OPTIONS": lambda config_object: (
            config_defaults.linears_dynamic_memory_options(
                config_object,
                memory_prefix="ATTN_MEMORY",
                memory_stack_prefix="ATTN_MEMORY_STACK",
            )
        ),
        "FEED_FORWARD_DYNAMIC_MEMORY_OPTIONS": lambda config_object: (
            config_defaults.linears_dynamic_memory_options(
                config_object,
                memory_prefix="FF_MEMORY",
                memory_stack_prefix="FF_MEMORY_STACK",
            )
        ),
        "RECURRENT_CONTROLLER_OPTIONS": lambda config_object: (
            config_defaults.linears_recurrent_controller_options(
                config_object,
                recurrent_prefix="RECURRENT",
                gate_stack_prefix="RECURRENT_GATE_STACK",
                halting_stack_prefix="RECURRENT_HALTING_STACK",
            )
        ),
        "ATTENTION_PROJECTION_RECURRENT_CONTROLLER_OPTIONS": lambda config_object: (
            config_defaults.linears_recurrent_controller_options(
                config_object,
                recurrent_prefix="ATTN_RECURRENT",
                gate_stack_prefix="ATTN_RECURRENT_GATE_STACK",
                halting_stack_prefix="ATTN_RECURRENT_HALTING_STACK",
            )
        ),
        "FEED_FORWARD_RECURRENT_CONTROLLER_OPTIONS": lambda config_object: (
            config_defaults.linears_recurrent_controller_options(
                config_object,
                recurrent_prefix="FF_RECURRENT",
                gate_stack_prefix="FF_RECURRENT_GATE_STACK",
                halting_stack_prefix="FF_RECURRENT_HALTING_STACK",
            )
        ),
    }
    try:
        return option_factories[config_attr](config_module)
    except KeyError as error:
        raise AttributeError(
            f"{config_module.__name__} has no default option factory for {config_attr}"
        ) from error


def _ensure_control_dependencies(
    builder_kwargs: dict[str, Any], config_module: ModuleType
) -> None:
    encoder_control_keys = {
        "submodule_stack_options",
        "layer_controller_options",
        "dynamic_memory_options",
        "recurrent_controller_options",
    }
    if encoder_control_keys & set(builder_kwargs):
        _ensure_default(
            builder_kwargs,
            "submodule_stack_options",
            config_module,
            "SUBMODULE_STACK_OPTIONS",
        )
        _ensure_default(
            builder_kwargs,
            "layer_controller_options",
            config_module,
            "LAYER_CONTROLLER_OPTIONS",
        )
        _ensure_default(
            builder_kwargs,
            "dynamic_memory_options",
            config_module,
            "DYNAMIC_MEMORY_OPTIONS",
        )
        _ensure_default(
            builder_kwargs,
            "recurrent_controller_options",
            config_module,
            "RECURRENT_CONTROLLER_OPTIONS",
        )
    attention_control_keys = {
        "attention_projection_layer_controller_options",
        "attention_projection_dynamic_memory_options",
        "attention_projection_recurrent_controller_options",
    }
    if attention_control_keys & set(builder_kwargs):
        _ensure_default(
            builder_kwargs,
            "attention_projection_layer_controller_options",
            config_module,
            "ATTENTION_PROJECTION_LAYER_CONTROLLER_OPTIONS",
        )
        _ensure_default(
            builder_kwargs,
            "attention_projection_dynamic_memory_options",
            config_module,
            "ATTENTION_PROJECTION_DYNAMIC_MEMORY_OPTIONS",
        )
        _ensure_default(
            builder_kwargs,
            "attention_projection_recurrent_controller_options",
            config_module,
            "ATTENTION_PROJECTION_RECURRENT_CONTROLLER_OPTIONS",
        )
    feed_forward_control_keys = {
        "feed_forward_layer_controller_options",
        "feed_forward_dynamic_memory_options",
        "feed_forward_recurrent_controller_options",
    }
    if feed_forward_control_keys & set(builder_kwargs):
        _ensure_default(
            builder_kwargs,
            "feed_forward_layer_controller_options",
            config_module,
            "FEED_FORWARD_LAYER_CONTROLLER_OPTIONS",
        )
        _ensure_default(
            builder_kwargs,
            "feed_forward_dynamic_memory_options",
            config_module,
            "FEED_FORWARD_DYNAMIC_MEMORY_OPTIONS",
        )
        _ensure_default(
            builder_kwargs,
            "feed_forward_recurrent_controller_options",
            config_module,
            "FEED_FORWARD_RECURRENT_CONTROLLER_OPTIONS",
        )


def _ensure_default(
    builder_kwargs: dict[str, Any],
    builder_key: str,
    config_module: ModuleType,
    config_attr: str,
) -> None:
    if builder_key not in builder_kwargs:
        builder_kwargs[builder_key] = _default_config_options(
            config_module, config_attr
        )


def _factory_relevant_keys(
    factory: Callable, factory_kwargs: dict[str, Any]
) -> set[str]:
    if factory is _patch_options_from_kwargs:
        return {
            "image_patch_size",
            "input_channels",
            "image_height",
            "patch_dropout_probability",
            "patch_bias_flag",
        }
    if factory is _encoder_options_from_kwargs:
        return {
            "hidden_dim",
            "stack_num_layers",
            "stack_activation",
            "stack_dropout_probability",
            "layer_norm_position",
            "stack_layer_norm_position",
        }
    if factory is _positional_embedding_options_from_kwargs:
        return {
            "positional_embedding_option",
            "positional_embedding_padding_idx",
            "positional_embedding_auto_expand_flag",
        }
    if factory is _attention_options_from_kwargs:
        return {
            "attn_num_heads",
            "attn_num_layers",
            "attn_bias_flag",
            "attn_add_key_value_bias_flag",
        }
    if factory is _feed_forward_options_from_kwargs:
        return {"ff_num_layers", "ff_bias_flag"}
    if factory is _output_options_from_kwargs:
        return {"output_bias_flag"}
    if factory is _main_stack_options_from_kwargs:
        return {
            "stack_bias_flag",
            "stack_layer_norm_position",
            "layer_norm_position",
            "stack_num_layers",
            "stack_activation",
            "stack_residual_connection_option",
            "stack_dropout_probability",
            "stack_last_layer_bias_option",
            "stack_apply_output_pipeline_flag",
        }
    if factory is _submodule_stack_options_from_kwargs:
        prefix = factory_kwargs["flat_prefix"]
        return {f"{prefix}_{flat_field}" for flat_field in _SUBMODULE_STACK_FIELD_MAP}
    if factory is _layer_controller_options_from_kwargs:
        flat_prefix = factory_kwargs["flat_prefix"]
        if not flat_prefix:
            base = {
                "stack_gate_flag",
                "gate_flag",
                "gate_option",
                "gate_activation",
                "stack_halting_flag",
                "halting_flag",
                "halting_threshold",
                "halting_dropout",
                "halting_hidden_state_mode",
            }
        else:
            base = {
                f"{flat_prefix}_gate_flag",
                f"{flat_prefix}_gate_option",
                f"{flat_prefix}_gate_activation",
                f"{flat_prefix}_halting_flag",
                f"{flat_prefix}_halting_threshold",
                f"{flat_prefix}_halting_dropout",
                f"{flat_prefix}_halting_hidden_state_mode",
            }
        return (
            base
            | _controller_stack_keys(factory_kwargs["gate_stack_prefix"])
            | _controller_stack_keys(factory_kwargs["halting_stack_prefix"])
        )
    if factory is _dynamic_memory_options_from_kwargs:
        flat_prefix = factory_kwargs["flat_prefix"]
        prefix = f"{flat_prefix}_" if flat_prefix else ""
        return {
            f"{prefix}memory_flag",
            f"{prefix}memory_option",
            f"{prefix}memory_position_option",
            f"{prefix}memory_test_time_training_learning_rate",
            f"{prefix}memory_test_time_training_num_inner_steps",
        } | _controller_stack_keys(factory_kwargs["memory_stack_prefix"])
    if factory is _recurrent_controller_options_from_kwargs:
        flat_prefix = factory_kwargs["flat_prefix"]
        return (
            {
                f"{flat_prefix}_flag",
                f"{flat_prefix}_max_steps",
                f"{flat_prefix}_layer_norm_position",
                f"{flat_prefix}_gate_flag",
                f"{flat_prefix}_gate_option",
                f"{flat_prefix}_gate_activation",
                f"{flat_prefix}_halting_flag",
                f"{flat_prefix}_halting_threshold",
                f"{flat_prefix}_halting_dropout",
                f"{flat_prefix}_halting_hidden_state_mode",
            }
            | _controller_stack_keys(factory_kwargs["gate_stack_prefix"])
            | _controller_stack_keys(factory_kwargs["halting_stack_prefix"])
        )
    return set()


def _controller_stack_keys(prefix: str) -> set[str]:
    return {f"{prefix}_{flat_field}" for flat_field in _CONTROLLER_STACK_FIELD_MAP}


def _updates(kwargs: dict[str, Any], field_map: dict[str, str]) -> dict[str, Any]:
    return {
        dataclass_field: kwargs[flat_field]
        for flat_field, dataclass_field in field_map.items()
        if flat_field in kwargs
    }


def _leftover_kwargs(kwargs: dict[str, Any], consumed: set[str]) -> dict[str, Any]:
    consumed.update(_VIT_GROUPED_KEYS & set(kwargs))
    return {key: value for key, value in kwargs.items() if key not in consumed}
