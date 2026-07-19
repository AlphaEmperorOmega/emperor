# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass, replace
from types import ModuleType
from typing import Any

from models.vit.expert_linear_adaptive import _config_defaults as config_defaults
from models.vit.expert_linear_adaptive import (
    _experts_builder_adapter as experts_builder_adapter,
)
from models.vit.expert_linear_adaptive._experts_builder_adapter import (
    _dynamic_memory_options_from_kwargs as _experts_dynamic_memory_options_from_kwargs,
)
from models.vit.expert_linear_adaptive._experts_builder_adapter import (
    _expert_dynamic_memory_options_from_kwargs,
    _expert_layer_controller_options_from_kwargs,
    _expert_recurrent_controller_options_from_kwargs,
    _mixture_options_from_kwargs,
    _role_stack_options_from_kwargs,
    _router_dynamic_memory_options_from_kwargs,
    _router_layer_controller_options_from_kwargs,
    _router_options_from_kwargs,
    _router_recurrent_controller_options_from_kwargs,
    _sampler_options_from_kwargs,
)
from models.vit.expert_linear_adaptive._linears_builder_adapter import (
    _adaptive_generator_stack_options_from_kwargs,
    _hidden_adaptive_bias_options_from_kwargs,
    _hidden_adaptive_diagonal_options_from_kwargs,
    _hidden_adaptive_mask_options_from_kwargs,
    _hidden_adaptive_weight_options_from_kwargs,
)
from models.vit.expert_linear_adaptive.runtime_options import (
    DynamicMemoryOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    SubmoduleStackSource,
)

_experts_layer_controller_options_from_kwargs = (
    experts_builder_adapter._layer_controller_options_from_kwargs
)
_experts_recurrent_controller_options_from_kwargs = (
    experts_builder_adapter._recurrent_controller_options_from_kwargs
)
_experts_submodule_stack_options_from_kwargs = (
    experts_builder_adapter._submodule_stack_options_from_kwargs
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
_ADAPTIVE_GENERATOR_STACK_FIELD_MAP = {
    "hidden_dim": "hidden_dim",
    "layer_norm_position": "layer_norm_position",
    "num_layers": "num_layers",
    "activation": "activation",
    "residual_connection_option": "residual_connection_option",
    "dropout_probability": "dropout_probability",
    "last_layer_bias_option": "last_layer_bias_option",
    "apply_output_pipeline_flag": "apply_output_pipeline_flag",
    "bias_flag": "bias_flag",
}


def expert_linear_adaptive_builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any], config_module: ModuleType
) -> dict[str, Any]:
    kwargs = dict(flat_kwargs)
    consumed: set[str] = set()
    builder_kwargs = _top_level_kwargs(kwargs, consumed)
    builder_kwargs.update(_vit_builder_kwargs(kwargs, config_module, consumed))
    builder_kwargs.update(
        _expert_adaptive_builder_kwargs(kwargs, config_module, consumed)
    )
    builder_kwargs.update(_adaptive_builder_kwargs(kwargs, config_module, consumed))
    builder_kwargs.update(
        _router_adaptive_builder_kwargs(kwargs, config_module, consumed)
    )
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


def _adaptive_builder_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    consumed: set[str],
    *,
    include_role_overrides: bool = False,
) -> dict[str, Any]:
    adaptive_generator_stack_options = _private_helper_value(
        _adaptive_generator_stack_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        provided=kwargs.get("adaptive_generator_stack_options"),
    )
    hidden_adaptive_weight_options = _private_helper_value(
        _hidden_adaptive_weight_options_with_aliases_from_kwargs,
        kwargs,
        consumed,
        config_module,
        provided=kwargs.get("hidden_adaptive_weight_options"),
    )
    hidden_adaptive_bias_options = _private_helper_value(
        _hidden_adaptive_bias_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        provided=kwargs.get("hidden_adaptive_bias_options"),
    )
    hidden_adaptive_diagonal_options = _private_helper_value(
        _hidden_adaptive_diagonal_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        provided=kwargs.get("hidden_adaptive_diagonal_options"),
    )
    hidden_adaptive_mask_options = _private_helper_value(
        _hidden_adaptive_mask_options_from_kwargs,
        kwargs,
        consumed,
        config_module,
        provided=kwargs.get("hidden_adaptive_mask_options"),
    )
    builder_kwargs = {
        "adaptive_generator_stack_options": adaptive_generator_stack_options,
        "hidden_adaptive_weight_options": hidden_adaptive_weight_options,
        "hidden_adaptive_bias_options": hidden_adaptive_bias_options,
        "hidden_adaptive_diagonal_options": hidden_adaptive_diagonal_options,
        "hidden_adaptive_mask_options": hidden_adaptive_mask_options,
    }
    if not include_role_overrides:
        return builder_kwargs
    builder_kwargs.update(
        _role_adaptive_builder_kwargs(
            kwargs,
            config_module,
            consumed,
            builder_prefix="attention",
            flat_prefix="attn_",
            config_prefix="ATTN_",
            adaptive_generator_stack_options=adaptive_generator_stack_options,
            hidden_adaptive_weight_options=hidden_adaptive_weight_options,
            hidden_adaptive_bias_options=hidden_adaptive_bias_options,
            hidden_adaptive_diagonal_options=hidden_adaptive_diagonal_options,
            hidden_adaptive_mask_options=hidden_adaptive_mask_options,
        )
    )
    builder_kwargs.update(
        _role_adaptive_builder_kwargs(
            kwargs,
            config_module,
            consumed,
            builder_prefix="feed_forward",
            flat_prefix="ff_",
            config_prefix="FF_",
            adaptive_generator_stack_options=adaptive_generator_stack_options,
            hidden_adaptive_weight_options=hidden_adaptive_weight_options,
            hidden_adaptive_bias_options=hidden_adaptive_bias_options,
            hidden_adaptive_diagonal_options=hidden_adaptive_diagonal_options,
            hidden_adaptive_mask_options=hidden_adaptive_mask_options,
        )
    )
    return builder_kwargs


def _role_adaptive_builder_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    consumed: set[str],
    *,
    builder_prefix: str,
    flat_prefix: str,
    config_prefix: str,
    adaptive_generator_stack_options: Any,
    hidden_adaptive_weight_options: Any,
    hidden_adaptive_bias_options: Any,
    hidden_adaptive_diagonal_options: Any,
    hidden_adaptive_mask_options: Any,
) -> dict[str, Any]:
    return {
        f"{builder_prefix}_adaptive_generator_stack_options": _private_helper_value(
            _adaptive_generator_stack_options_with_prefix_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get(f"{builder_prefix}_adaptive_generator_stack_options")
            or adaptive_generator_stack_options,
            flat_prefix=f"{flat_prefix}adaptive_generator_stack",
        ),
        f"{builder_prefix}_hidden_adaptive_weight_options": _private_helper_value(
            _hidden_adaptive_weight_options_with_aliases_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get(f"{builder_prefix}_hidden_adaptive_weight_options")
            or _role_config_default(
                hidden_adaptive_weight_options,
                config_module,
                global_default_factory=config_defaults.hidden_adaptive_weight_options,
                role_default_factory=lambda config_object: (
                    config_defaults.hidden_adaptive_weight_options(
                        config_object,
                        prefix=config_prefix,
                        stack_prefix=f"{flat_prefix}weight_generator_stack".upper(),
                    )
                ),
            ),
            flat_prefix=flat_prefix,
            config_prefix=config_prefix,
            stack_prefix=f"{flat_prefix}weight_generator_stack",
        ),
        f"{builder_prefix}_hidden_adaptive_bias_options": _private_helper_value(
            _hidden_adaptive_bias_options_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get(f"{builder_prefix}_hidden_adaptive_bias_options")
            or _role_config_default(
                hidden_adaptive_bias_options,
                config_module,
                global_default_factory=config_defaults.hidden_adaptive_bias_options,
                role_default_factory=lambda config_object: (
                    config_defaults.hidden_adaptive_bias_options(
                        config_object,
                        prefix=config_prefix,
                        stack_prefix=f"{flat_prefix}bias_generator_stack".upper(),
                    )
                ),
            ),
            flat_prefix=flat_prefix,
            config_prefix=config_prefix,
            stack_prefix=f"{flat_prefix}bias_generator_stack",
        ),
        f"{builder_prefix}_hidden_adaptive_diagonal_options": _private_helper_value(
            _hidden_adaptive_diagonal_options_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get(f"{builder_prefix}_hidden_adaptive_diagonal_options")
            or _role_config_default(
                hidden_adaptive_diagonal_options,
                config_module,
                global_default_factory=config_defaults.hidden_adaptive_diagonal_options,
                role_default_factory=lambda config_object: (
                    config_defaults.hidden_adaptive_diagonal_options(
                        config_object,
                        prefix=config_prefix,
                        stack_prefix=f"{flat_prefix}diagonal_generator_stack".upper(),
                    )
                ),
            ),
            flat_prefix=flat_prefix,
            config_prefix=config_prefix,
            stack_prefix=f"{flat_prefix}diagonal_generator_stack",
        ),
        f"{builder_prefix}_hidden_adaptive_mask_options": _private_helper_value(
            _hidden_adaptive_mask_options_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get(f"{builder_prefix}_hidden_adaptive_mask_options")
            or _role_config_default(
                hidden_adaptive_mask_options,
                config_module,
                global_default_factory=config_defaults.hidden_adaptive_mask_options,
                role_default_factory=lambda config_object: (
                    config_defaults.hidden_adaptive_mask_options(
                        config_object,
                        prefix=config_prefix,
                        stack_prefix=f"{flat_prefix}mask_generator_stack".upper(),
                    )
                ),
            ),
            flat_prefix=flat_prefix,
            config_prefix=config_prefix,
            stack_prefix=f"{flat_prefix}mask_generator_stack",
        ),
    }


def _adaptive_generator_stack_options_with_prefix_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: Any,
    flat_prefix: str,
) -> Any:
    options = provided or config_defaults.adaptive_generator_stack_options(
        config_module
    )
    updates = _pop_updates(
        kwargs,
        {
            f"{flat_prefix}_{flat_field}": dataclass_field
            for flat_field, dataclass_field in _ADAPTIVE_GENERATOR_STACK_FIELD_MAP.items()
        },
    )
    return replace(options, **updates) if updates else options


def _role_config_default(
    global_options: Any,
    config_module: ModuleType,
    *,
    global_default_factory: Callable[[ModuleType], Any],
    role_default_factory: Callable[[ModuleType], Any],
) -> Any:
    return _merge_role_config_overrides(
        global_options,
        global_default_factory(config_module),
        role_default_factory(config_module),
    )


def _merge_role_config_overrides(
    base: Any, global_default: Any, role_default: Any
) -> Any:
    if (
        not is_dataclass(base)
        or not is_dataclass(global_default)
        or (not is_dataclass(role_default))
    ):
        return role_default if role_default != global_default else base
    updates: dict[str, Any] = {}
    for field in fields(role_default):
        base_value = getattr(base, field.name)
        global_value = getattr(global_default, field.name)
        role_value = getattr(role_default, field.name)
        if (
            is_dataclass(base_value)
            and is_dataclass(global_value)
            and is_dataclass(role_value)
        ):
            merged_value = _merge_role_config_overrides(
                base_value, global_value, role_value
            )
            if merged_value != base_value:
                updates[field.name] = merged_value
        elif role_value != global_value:
            updates[field.name] = role_value
    return replace(base, **updates) if updates else base


def _router_adaptive_builder_kwargs(
    kwargs: dict[str, Any], config_module: ModuleType, consumed: set[str]
) -> dict[str, Any]:
    return {
        "router_adaptive_weight_options": _private_helper_value(
            _hidden_adaptive_weight_options_with_aliases_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get("router_adaptive_weight_options"),
            flat_prefix="router_",
            config_prefix="ROUTER_",
            stack_prefix="router_weight_generator_stack",
        ),
        "router_adaptive_bias_options": _private_helper_value(
            _hidden_adaptive_bias_options_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get("router_adaptive_bias_options"),
            flat_prefix="router_",
            config_prefix="ROUTER_",
            stack_prefix="router_bias_generator_stack",
        ),
        "router_adaptive_diagonal_options": _private_helper_value(
            _hidden_adaptive_diagonal_options_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get("router_adaptive_diagonal_options"),
            flat_prefix="router_",
            config_prefix="ROUTER_",
            stack_prefix="router_diagonal_generator_stack",
        ),
        "router_adaptive_mask_options": _private_helper_value(
            _hidden_adaptive_mask_options_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get("router_adaptive_mask_options"),
            flat_prefix="router_",
            config_prefix="ROUTER_",
            stack_prefix="router_mask_generator_stack",
        ),
    }


def _expert_builder_kwargs(
    kwargs: dict[str, Any], config_module: ModuleType, consumed: set[str]
) -> dict[str, Any]:
    builder_kwargs = {
        "mixture_options": _private_helper_value(
            _mixture_options_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get("mixture_options"),
        ),
        "expert_stack_options": _private_helper_value(
            _role_stack_options_from_kwargs,
            kwargs,
            consumed,
            config_module,
            "expert_stack",
            defaults=_default_config_options(config_module, "EXPERT_STACK_OPTIONS"),
            provided=kwargs.get("expert_stack_options"),
            extra_mapping={"expert_bias_flag": "bias_flag"},
        ),
        "sampler_options": _private_helper_value(
            _sampler_options_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get("sampler_options"),
        ),
        "router_options": _private_helper_value(
            _router_options_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get("router_options"),
        ),
        "router_stack_options": _private_helper_value(
            _role_stack_options_from_kwargs,
            kwargs,
            consumed,
            config_module,
            "router_stack",
            defaults=_default_config_options(config_module, "ROUTER_STACK_OPTIONS"),
            provided=kwargs.get("router_stack_options"),
            extra_mapping={"router_bias_flag": "bias_flag"},
        ),
        "expert_layer_controller_options": _private_helper_value(
            _expert_layer_controller_options_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get("expert_layer_controller_options"),
        ),
        "expert_dynamic_memory_options": _private_helper_value(
            _expert_dynamic_memory_options_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get("expert_dynamic_memory_options"),
        ),
        "expert_recurrent_controller_options": _private_helper_value(
            _expert_recurrent_controller_options_from_kwargs,
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get("expert_recurrent_controller_options"),
        ),
    }
    _copy_direct(
        builder_kwargs,
        kwargs,
        consumed,
        {"expert_attention_use_kv_expert_models_flag"},
    )
    return builder_kwargs


def _hidden_adaptive_weight_options_with_aliases_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: Any,
    flat_prefix: str = "",
    config_prefix: str = "",
    stack_prefix: str = "weight_generator_stack",
) -> Any:
    expected_depth_key = f"{flat_prefix}generator_depth"
    public_depth_key = f"{flat_prefix}weight_generator_depth"
    if public_depth_key in kwargs and expected_depth_key not in kwargs:
        kwargs[expected_depth_key] = kwargs.pop(public_depth_key)
    return _hidden_adaptive_weight_options_from_kwargs(
        kwargs,
        config_module,
        provided=provided,
        flat_prefix=flat_prefix,
        config_prefix=config_prefix,
        stack_prefix=stack_prefix,
    )


def _expert_adaptive_builder_kwargs(
    kwargs: dict[str, Any], config_module: ModuleType, consumed: set[str]
) -> dict[str, Any]:
    builder_kwargs = _expert_builder_kwargs(kwargs, config_module, consumed)
    builder_kwargs.update(
        {
            "mixture_submodule_stack_options": _private_helper_value(
                _experts_submodule_stack_options_from_kwargs,
                kwargs,
                consumed,
                config_module,
                provided=kwargs.get("mixture_submodule_stack_options"),
            ),
            "mixture_layer_controller_options": _private_helper_value(
                _experts_layer_controller_options_from_kwargs,
                kwargs,
                consumed,
                config_module,
                provided=kwargs.get("mixture_layer_controller_options"),
            ),
            "mixture_dynamic_memory_options": _private_helper_value(
                _experts_dynamic_memory_options_from_kwargs,
                kwargs,
                consumed,
                config_module,
                provided=kwargs.get("mixture_dynamic_memory_options"),
            ),
            "mixture_recurrent_controller_options": _private_helper_value(
                _experts_recurrent_controller_options_from_kwargs,
                kwargs,
                consumed,
                config_module,
                provided=kwargs.get("mixture_recurrent_controller_options"),
            ),
            "router_layer_controller_options": _private_helper_value(
                _router_layer_controller_options_from_kwargs,
                kwargs,
                consumed,
                config_module,
                provided=kwargs.get("router_layer_controller_options"),
            ),
            "router_dynamic_memory_options": _private_helper_value(
                _router_dynamic_memory_options_from_kwargs,
                kwargs,
                consumed,
                config_module,
                provided=kwargs.get("router_dynamic_memory_options"),
            ),
            "router_recurrent_controller_options": _private_helper_value(
                _router_recurrent_controller_options_from_kwargs,
                kwargs,
                consumed,
                config_module,
                provided=kwargs.get("router_recurrent_controller_options"),
            ),
        }
    )
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
            "halting_option": "halting_option",
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
            f"{prefix}halting_option": "halting_option",
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
            f"{prefix}halting_option": "recurrent_halting_option",
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
        "EXPERT_STACK_OPTIONS": lambda config_object: (
            config_defaults.experts_submodule_stack_options(
                config_object, "EXPERT_STACK", bias_key="EXPERT_BIAS_FLAG"
            )
        ),
        "ROUTER_STACK_OPTIONS": lambda config_object: (
            config_defaults.experts_submodule_stack_options(
                config_object, "ROUTER_STACK", bias_key="ROUTER_BIAS_FLAG"
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
                "halting_option",
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
                f"{flat_prefix}_halting_option",
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
                f"{flat_prefix}_halting_option",
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


def _private_helper_value(
    helper: Callable,
    kwargs: dict[str, Any],
    consumed: set[str],
    config_module: ModuleType,
    *args,
    **helper_kwargs,
) -> Any:
    temp = dict(kwargs)
    value = helper(temp, config_module, *args, **helper_kwargs)
    consumed.update(set(kwargs) - set(temp))
    for key, value_arg in helper_kwargs.items():
        if key == "provided" and value_arg is not None:
            continue
    return value


def _copy_direct(
    builder_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
    consumed: set[str],
    keys: set[str],
) -> None:
    for key in keys:
        if key in kwargs:
            builder_kwargs[key] = kwargs[key]
            consumed.add(key)


def _updates(kwargs: dict[str, Any], field_map: dict[str, str]) -> dict[str, Any]:
    return {
        dataclass_field: kwargs[flat_field]
        for flat_field, dataclass_field in field_map.items()
        if flat_field in kwargs
    }


def _pop_updates(kwargs: dict[str, Any], field_map: dict[str, str]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for flat_field, dataclass_field in field_map.items():
        if flat_field in kwargs:
            updates[dataclass_field] = kwargs.pop(flat_field)
    return updates


def _leftover_kwargs(kwargs: dict[str, Any], consumed: set[str]) -> dict[str, Any]:
    consumed.update(_VIT_GROUPED_KEYS & set(kwargs))
    consumed.update(
        {
            "adaptive_generator_stack_options",
            "hidden_adaptive_weight_options",
            "hidden_adaptive_bias_options",
            "hidden_adaptive_diagonal_options",
            "hidden_adaptive_mask_options",
            "attention_adaptive_generator_stack_options",
            "attention_hidden_adaptive_weight_options",
            "attention_hidden_adaptive_bias_options",
            "attention_hidden_adaptive_diagonal_options",
            "attention_hidden_adaptive_mask_options",
            "feed_forward_adaptive_generator_stack_options",
            "feed_forward_hidden_adaptive_weight_options",
            "feed_forward_hidden_adaptive_bias_options",
            "feed_forward_hidden_adaptive_diagonal_options",
            "feed_forward_hidden_adaptive_mask_options",
            "mixture_options",
            "mixture_submodule_stack_options",
            "mixture_layer_controller_options",
            "mixture_dynamic_memory_options",
            "mixture_recurrent_controller_options",
            "expert_stack_options",
            "sampler_options",
            "router_options",
            "router_stack_options",
            "router_layer_controller_options",
            "router_dynamic_memory_options",
            "router_recurrent_controller_options",
            "expert_layer_controller_options",
            "expert_dynamic_memory_options",
            "expert_recurrent_controller_options",
            "router_adaptive_weight_options",
            "router_adaptive_bias_options",
            "router_adaptive_diagonal_options",
            "router_adaptive_mask_options",
        }
        & set(kwargs)
    )
    return {key: value for key, value in kwargs.items() if key not in consumed}
