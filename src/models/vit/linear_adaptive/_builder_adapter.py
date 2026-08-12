# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from types import ModuleType
from typing import Any, TypeVar

from models.vit.linear_adaptive import _config_defaults as config_defaults
from models.vit.linear_adaptive._flat_updates import pop_updates as _pop_updates
from models.vit.linear_adaptive._linears_builder_adapter import (
    _adaptive_generator_stack_options_from_kwargs,
    _hidden_adaptive_bias_options_from_kwargs,
    _hidden_adaptive_diagonal_options_from_kwargs,
    _hidden_adaptive_mask_options_from_kwargs,
    _hidden_adaptive_weight_options_from_kwargs,
)
from models.vit.linear_adaptive._residual import (
    ResidualStackSource,
    resolve_residual_stack_options,
)
from models.vit.linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
    DynamicMemoryOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
    LayerControllerOptions,
    MainLayerStackOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    SubmoduleStackSource,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
    VitOutputOptions,
    VitPatchOptions,
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
    "residual_model_flag": "residual_model_flag",
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
    "residual_model_flag": "residual_model_flag",
    "dropout_probability": "dropout_probability",
    "bias_flag": "bias_flag",
}
_ADAPTIVE_GENERATOR_STACK_FIELD_MAP = {
    "hidden_dim": "hidden_dim",
    "layer_norm_position": "layer_norm_position",
    "num_layers": "num_layers",
    "activation": "activation",
    "residual_connection_option": "residual_connection_option",
    "residual_model_flag": "residual_model_flag",
    "dropout_probability": "dropout_probability",
    "last_layer_bias_option": "last_layer_bias_option",
    "apply_output_pipeline_flag": "apply_output_pipeline_flag",
    "bias_flag": "bias_flag",
}


def linear_adaptive_builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any], config_module: ModuleType
) -> dict[str, Any]:
    kwargs = dict(flat_kwargs)
    consumed: set[str] = set()
    builder_kwargs = _top_level_kwargs(kwargs, consumed)
    builder_kwargs.update(_vit_builder_kwargs(kwargs, config_module, consumed))
    builder_kwargs.update(
        _adaptive_builder_kwargs(
            kwargs, config_module, consumed, include_role_overrides=True
        )
    )
    _attach_residual_stack_options(
        builder_kwargs,
        kwargs,
        config_module,
        consumed,
    )
    builder_kwargs.update(_leftover_kwargs(kwargs, consumed))
    return builder_kwargs


_RESIDUAL_STACK_FLAT_FIELDS = {
    "independent_flag",
    "hidden_dim",
    "layer_norm_position",
    "num_layers",
    "activation",
    "residual_connection_option",
    "residual_model_flag",
    "dropout_probability",
    "last_layer_bias_option",
    "apply_output_pipeline_flag",
    "bias_flag",
}


def _attach_residual_stack_options(
    builder_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
    config_module: ModuleType,
    consumed: set[str],
) -> None:
    residual_stack_keys = {
        f"residual_stack_{field}" for field in _RESIDUAL_STACK_FLAT_FIELDS
    }
    submodule_stack_options = builder_kwargs.get("submodule_stack_options")
    if submodule_stack_options is None:
        submodule_stack_options = config_defaults.linears_submodule_stack_options(
            config_module, config_defaults.LinearRole.MAIN
        )
    residual_stack_options = resolve_residual_stack_options(
        ResidualStackSource(
            independent_flag=kwargs.get(
                "residual_stack_independent_flag",
                config_module.RESIDUAL_STACK_INDEPENDENT_FLAG,
            ),
            hidden_dim=kwargs.get(
                "residual_stack_hidden_dim", config_module.RESIDUAL_STACK_HIDDEN_DIM
            ),
            layer_norm_position=kwargs.get(
                "residual_stack_layer_norm_position",
                config_module.RESIDUAL_STACK_LAYER_NORM_POSITION,
            ),
            num_layers=kwargs.get(
                "residual_stack_num_layers", config_module.RESIDUAL_STACK_NUM_LAYERS
            ),
            activation=kwargs.get(
                "residual_stack_activation", config_module.RESIDUAL_STACK_ACTIVATION
            ),
            residual_connection_option=kwargs.get(
                "residual_stack_residual_connection_option",
                config_module.RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION,
            ),
            residual_model_flag=kwargs.get(
                "residual_stack_residual_model_flag",
                config_module.RESIDUAL_STACK_RESIDUAL_MODEL_FLAG,
            ),
            dropout_probability=kwargs.get(
                "residual_stack_dropout_probability",
                config_module.RESIDUAL_STACK_DROPOUT_PROBABILITY,
            ),
            last_layer_bias_option=kwargs.get(
                "residual_stack_last_layer_bias_option",
                config_module.RESIDUAL_STACK_LAST_LAYER_BIAS_OPTION,
            ),
            apply_output_pipeline_flag=kwargs.get(
                "residual_stack_apply_output_pipeline_flag",
                config_module.RESIDUAL_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            ),
            bias_flag=kwargs.get(
                "residual_stack_bias_flag", config_module.RESIDUAL_STACK_BIAS_FLAG
            ),
        ),
        submodule_stack_options,
    )
    builder_kwargs["submodule_stack_options"] = replace(
        submodule_stack_options,
        residual_stack_options=residual_stack_options,
    )
    for key, value in tuple(builder_kwargs.items()):
        if isinstance(
            value,
            (
                MainLayerStackOptions,
                SubmoduleStackOptions,
                AdaptiveGeneratorStackOptions,
            ),
        ):
            builder_kwargs[key] = replace(
                value,
                residual_stack_options=residual_stack_options,
            )
    consumed.update(residual_stack_keys.intersection(kwargs))


def _top_level_kwargs(kwargs: dict[str, Any], consumed: set[str]) -> dict[str, Any]:
    consumed.update(key for key in _TOP_LEVEL_KEYS if key in kwargs)
    return {key: kwargs[key] for key in _TOP_LEVEL_KEYS if key in kwargs}


def _vit_builder_kwargs(
    kwargs: dict[str, Any], config_module: ModuleType, consumed: set[str]
) -> dict[str, Any]:
    builder_kwargs: dict[str, Any] = {}
    _set_vit_structure_options(builder_kwargs, kwargs, config_module, consumed)
    _set_main_control_options(builder_kwargs, kwargs, config_module, consumed)
    _set_attention_control_options(builder_kwargs, kwargs, config_module, consumed)
    _set_feed_forward_control_options(builder_kwargs, kwargs, config_module, consumed)
    _ensure_control_dependencies(builder_kwargs, config_module)
    return builder_kwargs


def _set_vit_structure_options(
    builder_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
    config_module: ModuleType,
    consumed: set[str],
) -> None:
    patch_options = _resolve_patch_options(kwargs, config_module, consumed)
    if patch_options is not None:
        builder_kwargs["patch_options"] = patch_options
    encoder_options = _resolve_encoder_options(kwargs, config_module, consumed)
    if encoder_options is not None:
        builder_kwargs["encoder_options"] = encoder_options
    positional_embedding_options = _resolve_positional_embedding_options(
        kwargs, config_module, consumed
    )
    if positional_embedding_options is not None:
        builder_kwargs["positional_embedding_options"] = positional_embedding_options
    attention_options = _resolve_attention_options(kwargs, config_module, consumed)
    if attention_options is not None:
        builder_kwargs["attention_options"] = attention_options
    feed_forward_options = _resolve_feed_forward_options(
        kwargs, config_module, consumed
    )
    if feed_forward_options is not None:
        builder_kwargs["feed_forward_options"] = feed_forward_options
    output_options = _resolve_output_options(kwargs, config_module, consumed)
    if output_options is not None:
        builder_kwargs["output_options"] = output_options


def _set_main_control_options(
    builder_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
    config_module: ModuleType,
    consumed: set[str],
) -> None:
    stack_options = _resolve_main_stack_options(kwargs, config_module, consumed)
    if stack_options is not None:
        builder_kwargs["stack_options"] = stack_options
    submodule_stack_options = _resolve_submodule_stack_options(
        kwargs,
        consumed,
        option_key="submodule_stack_options",
        provided=kwargs.get("submodule_stack_options"),
        defaults_factory=lambda: config_defaults.linears_submodule_stack_options(
            config_module, config_defaults.LinearRole.MAIN
        ),
        flat_prefix="submodule_stack",
    )
    if submodule_stack_options is not None:
        builder_kwargs["submodule_stack_options"] = submodule_stack_options
    layer_controller_options = _resolve_layer_controller_options(
        kwargs,
        consumed,
        option_key="layer_controller_options",
        provided=kwargs.get("layer_controller_options"),
        defaults_factory=lambda: config_defaults.linears_layer_controller_options(
            config_module, config_defaults.LinearRole.MAIN
        ),
        flat_prefix="",
        gate_stack_prefix="gate_stack",
        halting_stack_prefix="halting_stack",
    )
    if layer_controller_options is not None:
        builder_kwargs["layer_controller_options"] = layer_controller_options
    dynamic_memory_options = _resolve_dynamic_memory_options(
        kwargs,
        consumed,
        option_key="dynamic_memory_options",
        provided=kwargs.get("dynamic_memory_options"),
        defaults_factory=lambda: config_defaults.linears_dynamic_memory_options(
            config_module, config_defaults.LinearRole.MAIN
        ),
        flat_prefix="",
        memory_stack_prefix="memory_stack",
    )
    if dynamic_memory_options is not None:
        builder_kwargs["dynamic_memory_options"] = dynamic_memory_options
    recurrent_controller_options = _resolve_recurrent_controller_options(
        kwargs,
        consumed,
        option_key="recurrent_controller_options",
        provided=kwargs.get("recurrent_controller_options"),
        defaults_factory=lambda: config_defaults.linears_recurrent_controller_options(
            config_module, config_defaults.LinearRole.MAIN
        ),
        flat_prefix="recurrent",
        gate_stack_prefix="recurrent_gate_stack",
        halting_stack_prefix="recurrent_halting_stack",
    )
    if recurrent_controller_options is not None:
        builder_kwargs["recurrent_controller_options"] = recurrent_controller_options


def _set_attention_control_options(
    builder_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
    config_module: ModuleType,
    consumed: set[str],
) -> None:
    stack_options = _resolve_submodule_stack_options(
        kwargs,
        consumed,
        option_key="attention_projection_stack_options",
        provided=kwargs.get("attention_projection_stack_options"),
        defaults_factory=lambda: config_defaults.linears_submodule_stack_options(
            config_module, config_defaults.LinearRole.ATTENTION
        ),
        flat_prefix="attn_stack",
    )
    if stack_options is not None:
        builder_kwargs["attention_projection_stack_options"] = stack_options
    layer_controller_options = _resolve_layer_controller_options(
        kwargs,
        consumed,
        option_key="attention_projection_layer_controller_options",
        provided=kwargs.get("attention_projection_layer_controller_options"),
        defaults_factory=lambda: config_defaults.linears_layer_controller_options(
            config_module, config_defaults.LinearRole.ATTENTION
        ),
        flat_prefix="attn",
        gate_stack_prefix="attn_gate_stack",
        halting_stack_prefix="attn_halting_stack",
    )
    if layer_controller_options is not None:
        builder_kwargs["attention_projection_layer_controller_options"] = (
            layer_controller_options
        )
    dynamic_memory_options = _resolve_dynamic_memory_options(
        kwargs,
        consumed,
        option_key="attention_projection_dynamic_memory_options",
        provided=kwargs.get("attention_projection_dynamic_memory_options"),
        defaults_factory=lambda: config_defaults.linears_dynamic_memory_options(
            config_module, config_defaults.LinearRole.ATTENTION
        ),
        flat_prefix="attn",
        memory_stack_prefix="attn_memory_stack",
    )
    if dynamic_memory_options is not None:
        builder_kwargs["attention_projection_dynamic_memory_options"] = (
            dynamic_memory_options
        )
    recurrent_controller_options = _resolve_recurrent_controller_options(
        kwargs,
        consumed,
        option_key="attention_projection_recurrent_controller_options",
        provided=kwargs.get("attention_projection_recurrent_controller_options"),
        defaults_factory=lambda: config_defaults.linears_recurrent_controller_options(
            config_module, config_defaults.LinearRole.ATTENTION
        ),
        flat_prefix="attn_recurrent",
        gate_stack_prefix="attn_recurrent_gate_stack",
        halting_stack_prefix="attn_recurrent_halting_stack",
    )
    if recurrent_controller_options is not None:
        builder_kwargs["attention_projection_recurrent_controller_options"] = (
            recurrent_controller_options
        )


def _set_feed_forward_control_options(
    builder_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
    config_module: ModuleType,
    consumed: set[str],
) -> None:
    stack_options = _resolve_submodule_stack_options(
        kwargs,
        consumed,
        option_key="feed_forward_stack_options",
        provided=kwargs.get("feed_forward_stack_options"),
        defaults_factory=lambda: config_defaults.linears_submodule_stack_options(
            config_module, config_defaults.LinearRole.FEED_FORWARD
        ),
        flat_prefix="ff_stack",
    )
    if stack_options is not None:
        builder_kwargs["feed_forward_stack_options"] = stack_options
    layer_controller_options = _resolve_layer_controller_options(
        kwargs,
        consumed,
        option_key="feed_forward_layer_controller_options",
        provided=kwargs.get("feed_forward_layer_controller_options"),
        defaults_factory=lambda: config_defaults.linears_layer_controller_options(
            config_module, config_defaults.LinearRole.FEED_FORWARD
        ),
        flat_prefix="ff",
        gate_stack_prefix="ff_gate_stack",
        halting_stack_prefix="ff_halting_stack",
    )
    if layer_controller_options is not None:
        builder_kwargs["feed_forward_layer_controller_options"] = (
            layer_controller_options
        )
    dynamic_memory_options = _resolve_dynamic_memory_options(
        kwargs,
        consumed,
        option_key="feed_forward_dynamic_memory_options",
        provided=kwargs.get("feed_forward_dynamic_memory_options"),
        defaults_factory=lambda: config_defaults.linears_dynamic_memory_options(
            config_module, config_defaults.LinearRole.FEED_FORWARD
        ),
        flat_prefix="ff",
        memory_stack_prefix="ff_memory_stack",
    )
    if dynamic_memory_options is not None:
        builder_kwargs["feed_forward_dynamic_memory_options"] = dynamic_memory_options
    recurrent_controller_options = _resolve_recurrent_controller_options(
        kwargs,
        consumed,
        option_key="feed_forward_recurrent_controller_options",
        provided=kwargs.get("feed_forward_recurrent_controller_options"),
        defaults_factory=lambda: config_defaults.linears_recurrent_controller_options(
            config_module, config_defaults.LinearRole.FEED_FORWARD
        ),
        flat_prefix="ff_recurrent",
        gate_stack_prefix="ff_recurrent_gate_stack",
        halting_stack_prefix="ff_recurrent_halting_stack",
    )
    if recurrent_controller_options is not None:
        builder_kwargs["feed_forward_recurrent_controller_options"] = (
            recurrent_controller_options
        )


def _record_consumed_helper_keys(
    kwargs: dict[str, Any], remaining: dict[str, Any], consumed: set[str]
) -> None:
    consumed.update(kwargs.keys() - remaining.keys())


def _resolved_adaptive_generator_stack_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    config_module: ModuleType,
    *,
    provided: AdaptiveGeneratorStackOptions | None,
    flat_prefix: str | None = None,
) -> AdaptiveGeneratorStackOptions:
    remaining = dict(kwargs)
    if flat_prefix is None:
        options = _adaptive_generator_stack_options_from_kwargs(
            remaining, config_module, provided=provided
        )
    else:
        options = _adaptive_generator_stack_options_with_prefix_from_kwargs(
            remaining,
            config_module,
            provided=provided,
            flat_prefix=flat_prefix,
        )
    _record_consumed_helper_keys(kwargs, remaining, consumed)
    return options


def _resolved_hidden_adaptive_weight_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveWeightOptions | None,
    flat_prefix: str = "",
    stack_prefix: str = "weight_generator_stack",
    role: config_defaults.AdaptiveRole = config_defaults.AdaptiveRole.MAIN,
) -> HiddenAdaptiveWeightOptions:
    remaining = dict(kwargs)
    options = _hidden_adaptive_weight_options_from_kwargs(
        remaining,
        config_module,
        provided=provided,
        flat_prefix=flat_prefix,
        stack_prefix=stack_prefix,
        role=role,
    )
    _record_consumed_helper_keys(kwargs, remaining, consumed)
    return options


def _resolved_hidden_adaptive_bias_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveBiasOptions | None,
    flat_prefix: str = "",
    stack_prefix: str = "bias_generator_stack",
    role: config_defaults.AdaptiveRole = config_defaults.AdaptiveRole.MAIN,
) -> HiddenAdaptiveBiasOptions:
    remaining = dict(kwargs)
    options = _hidden_adaptive_bias_options_from_kwargs(
        remaining,
        config_module,
        provided=provided,
        flat_prefix=flat_prefix,
        stack_prefix=stack_prefix,
        role=role,
    )
    _record_consumed_helper_keys(kwargs, remaining, consumed)
    return options


def _resolved_hidden_adaptive_diagonal_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveDiagonalOptions | None,
    flat_prefix: str = "",
    stack_prefix: str = "diagonal_generator_stack",
    role: config_defaults.AdaptiveRole = config_defaults.AdaptiveRole.MAIN,
) -> HiddenAdaptiveDiagonalOptions:
    remaining = dict(kwargs)
    options = _hidden_adaptive_diagonal_options_from_kwargs(
        remaining,
        config_module,
        provided=provided,
        flat_prefix=flat_prefix,
        stack_prefix=stack_prefix,
        role=role,
    )
    _record_consumed_helper_keys(kwargs, remaining, consumed)
    return options


def _resolved_hidden_adaptive_mask_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveMaskOptions | None,
    flat_prefix: str = "",
    stack_prefix: str = "mask_generator_stack",
    role: config_defaults.AdaptiveRole = config_defaults.AdaptiveRole.MAIN,
) -> HiddenAdaptiveMaskOptions:
    remaining = dict(kwargs)
    options = _hidden_adaptive_mask_options_from_kwargs(
        remaining,
        config_module,
        provided=provided,
        flat_prefix=flat_prefix,
        stack_prefix=stack_prefix,
        role=role,
    )
    _record_consumed_helper_keys(kwargs, remaining, consumed)
    return options


def _adaptive_builder_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    consumed: set[str],
    *,
    include_role_overrides: bool = False,
) -> dict[str, Any]:
    adaptive_generator_stack_options = _resolved_adaptive_generator_stack_options(
        kwargs,
        consumed,
        config_module,
        provided=kwargs.get("adaptive_generator_stack_options"),
    )
    hidden_adaptive_weight_options = _resolved_hidden_adaptive_weight_options(
        kwargs,
        consumed,
        config_module,
        provided=kwargs.get("hidden_adaptive_weight_options"),
    )
    hidden_adaptive_bias_options = _resolved_hidden_adaptive_bias_options(
        kwargs,
        consumed,
        config_module,
        provided=kwargs.get("hidden_adaptive_bias_options"),
    )
    hidden_adaptive_diagonal_options = _resolved_hidden_adaptive_diagonal_options(
        kwargs,
        consumed,
        config_module,
        provided=kwargs.get("hidden_adaptive_diagonal_options"),
    )
    hidden_adaptive_mask_options = _resolved_hidden_adaptive_mask_options(
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
            role=config_defaults.AdaptiveRole.ATTENTION,
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
            role=config_defaults.AdaptiveRole.FEED_FORWARD,
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
    role: config_defaults.AdaptiveRole,
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions,
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions,
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions,
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions,
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions,
) -> dict[str, Any]:
    return {
        f"{builder_prefix}_adaptive_generator_stack_options": _resolved_adaptive_generator_stack_options(
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get(f"{builder_prefix}_adaptive_generator_stack_options")
            or adaptive_generator_stack_options,
            flat_prefix=f"{flat_prefix}adaptive_generator_stack",
        ),
        f"{builder_prefix}_hidden_adaptive_weight_options": _resolved_hidden_adaptive_weight_options(
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get(f"{builder_prefix}_hidden_adaptive_weight_options")
            or _role_hidden_adaptive_weight_default(
                hidden_adaptive_weight_options,
                config_module,
                role,
            ),
            flat_prefix=flat_prefix,
            stack_prefix=f"{flat_prefix}weight_generator_stack",
            role=role,
        ),
        f"{builder_prefix}_hidden_adaptive_bias_options": _resolved_hidden_adaptive_bias_options(
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get(f"{builder_prefix}_hidden_adaptive_bias_options")
            or _role_hidden_adaptive_bias_default(
                hidden_adaptive_bias_options,
                config_module,
                role,
            ),
            flat_prefix=flat_prefix,
            stack_prefix=f"{flat_prefix}bias_generator_stack",
            role=role,
        ),
        f"{builder_prefix}_hidden_adaptive_diagonal_options": _resolved_hidden_adaptive_diagonal_options(
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get(f"{builder_prefix}_hidden_adaptive_diagonal_options")
            or _role_hidden_adaptive_diagonal_default(
                hidden_adaptive_diagonal_options,
                config_module,
                role,
            ),
            flat_prefix=flat_prefix,
            stack_prefix=f"{flat_prefix}diagonal_generator_stack",
            role=role,
        ),
        f"{builder_prefix}_hidden_adaptive_mask_options": _resolved_hidden_adaptive_mask_options(
            kwargs,
            consumed,
            config_module,
            provided=kwargs.get(f"{builder_prefix}_hidden_adaptive_mask_options")
            or _role_hidden_adaptive_mask_default(
                hidden_adaptive_mask_options,
                config_module,
                role,
            ),
            flat_prefix=flat_prefix,
            stack_prefix=f"{flat_prefix}mask_generator_stack",
            role=role,
        ),
    }


def _adaptive_generator_stack_options_with_prefix_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: AdaptiveGeneratorStackOptions | None,
    flat_prefix: str,
) -> AdaptiveGeneratorStackOptions:
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


_RoleValue = TypeVar("_RoleValue")


def _role_value(
    base: _RoleValue,
    global_default: _RoleValue,
    role_default: _RoleValue,
) -> _RoleValue:
    return role_default if role_default != global_default else base


def _merge_generator_stack_source(
    base: AdaptiveGeneratorStackSource,
    global_default: AdaptiveGeneratorStackSource,
    role_default: AdaptiveGeneratorStackSource,
) -> AdaptiveGeneratorStackSource:
    merged = replace(
        base,
        independent_flag=_role_value(
            base.independent_flag,
            global_default.independent_flag,
            role_default.independent_flag,
        ),
        hidden_dim=_role_value(
            base.hidden_dim, global_default.hidden_dim, role_default.hidden_dim
        ),
        layer_norm_position=_role_value(
            base.layer_norm_position,
            global_default.layer_norm_position,
            role_default.layer_norm_position,
        ),
        num_layers=_role_value(
            base.num_layers, global_default.num_layers, role_default.num_layers
        ),
        activation=_role_value(
            base.activation, global_default.activation, role_default.activation
        ),
        residual_connection_option=_role_value(
            base.residual_connection_option,
            global_default.residual_connection_option,
            role_default.residual_connection_option,
        ),
        residual_model_flag=_role_value(
            base.residual_model_flag,
            global_default.residual_model_flag,
            role_default.residual_model_flag,
        ),
        dropout_probability=_role_value(
            base.dropout_probability,
            global_default.dropout_probability,
            role_default.dropout_probability,
        ),
        last_layer_bias_option=_role_value(
            base.last_layer_bias_option,
            global_default.last_layer_bias_option,
            role_default.last_layer_bias_option,
        ),
        apply_output_pipeline_flag=_role_value(
            base.apply_output_pipeline_flag,
            global_default.apply_output_pipeline_flag,
            role_default.apply_output_pipeline_flag,
        ),
        bias_flag=_role_value(
            base.bias_flag, global_default.bias_flag, role_default.bias_flag
        ),
    )
    return base if merged == base else merged


def _merge_hidden_adaptive_weight_options(
    base: HiddenAdaptiveWeightOptions,
    global_default: HiddenAdaptiveWeightOptions,
    role_default: HiddenAdaptiveWeightOptions,
) -> HiddenAdaptiveWeightOptions:
    merged = replace(
        base,
        generator_depth=_role_value(
            base.generator_depth,
            global_default.generator_depth,
            role_default.generator_depth,
        ),
        option_flag=_role_value(
            base.option_flag, global_default.option_flag, role_default.option_flag
        ),
        option=_role_value(base.option, global_default.option, role_default.option),
        normalization_option=_role_value(
            base.normalization_option,
            global_default.normalization_option,
            role_default.normalization_option,
        ),
        normalization_position_option=_role_value(
            base.normalization_position_option,
            global_default.normalization_position_option,
            role_default.normalization_position_option,
        ),
        decay_schedule=_role_value(
            base.decay_schedule,
            global_default.decay_schedule,
            role_default.decay_schedule,
        ),
        decay_rate=_role_value(
            base.decay_rate, global_default.decay_rate, role_default.decay_rate
        ),
        decay_warmup_batches=_role_value(
            base.decay_warmup_batches,
            global_default.decay_warmup_batches,
            role_default.decay_warmup_batches,
        ),
        bank_expansion_factor=_role_value(
            base.bank_expansion_factor,
            global_default.bank_expansion_factor,
            role_default.bank_expansion_factor,
        ),
        generator_stack_source=_merge_generator_stack_source(
            base.generator_stack_source,
            global_default.generator_stack_source,
            role_default.generator_stack_source,
        ),
    )
    return base if merged == base else merged


def _merge_hidden_adaptive_bias_options(
    base: HiddenAdaptiveBiasOptions,
    global_default: HiddenAdaptiveBiasOptions,
    role_default: HiddenAdaptiveBiasOptions,
) -> HiddenAdaptiveBiasOptions:
    merged = replace(
        base,
        option_flag=_role_value(
            base.option_flag, global_default.option_flag, role_default.option_flag
        ),
        option=_role_value(base.option, global_default.option, role_default.option),
        decay_schedule=_role_value(
            base.decay_schedule,
            global_default.decay_schedule,
            role_default.decay_schedule,
        ),
        decay_rate=_role_value(
            base.decay_rate, global_default.decay_rate, role_default.decay_rate
        ),
        decay_warmup_batches=_role_value(
            base.decay_warmup_batches,
            global_default.decay_warmup_batches,
            role_default.decay_warmup_batches,
        ),
        bank_expansion_factor=_role_value(
            base.bank_expansion_factor,
            global_default.bank_expansion_factor,
            role_default.bank_expansion_factor,
        ),
        generator_stack_source=_merge_generator_stack_source(
            base.generator_stack_source,
            global_default.generator_stack_source,
            role_default.generator_stack_source,
        ),
    )
    return base if merged == base else merged


def _merge_hidden_adaptive_diagonal_options(
    base: HiddenAdaptiveDiagonalOptions,
    global_default: HiddenAdaptiveDiagonalOptions,
    role_default: HiddenAdaptiveDiagonalOptions,
) -> HiddenAdaptiveDiagonalOptions:
    merged = replace(
        base,
        option_flag=_role_value(
            base.option_flag, global_default.option_flag, role_default.option_flag
        ),
        option=_role_value(base.option, global_default.option, role_default.option),
        generator_stack_source=_merge_generator_stack_source(
            base.generator_stack_source,
            global_default.generator_stack_source,
            role_default.generator_stack_source,
        ),
    )
    return base if merged == base else merged


def _merge_hidden_adaptive_mask_options(
    base: HiddenAdaptiveMaskOptions,
    global_default: HiddenAdaptiveMaskOptions,
    role_default: HiddenAdaptiveMaskOptions,
) -> HiddenAdaptiveMaskOptions:
    merged = replace(
        base,
        option_flag=_role_value(
            base.option_flag, global_default.option_flag, role_default.option_flag
        ),
        row_mask_option=_role_value(
            base.row_mask_option,
            global_default.row_mask_option,
            role_default.row_mask_option,
        ),
        mask_dimension_option=_role_value(
            base.mask_dimension_option,
            global_default.mask_dimension_option,
            role_default.mask_dimension_option,
        ),
        mask_threshold=_role_value(
            base.mask_threshold,
            global_default.mask_threshold,
            role_default.mask_threshold,
        ),
        mask_surrogate_scale=_role_value(
            base.mask_surrogate_scale,
            global_default.mask_surrogate_scale,
            role_default.mask_surrogate_scale,
        ),
        mask_floor=_role_value(
            base.mask_floor, global_default.mask_floor, role_default.mask_floor
        ),
        mask_transition_width=_role_value(
            base.mask_transition_width,
            global_default.mask_transition_width,
            role_default.mask_transition_width,
        ),
        generator_stack_source=_merge_generator_stack_source(
            base.generator_stack_source,
            global_default.generator_stack_source,
            role_default.generator_stack_source,
        ),
    )
    return base if merged == base else merged


def _role_hidden_adaptive_weight_default(
    global_options: HiddenAdaptiveWeightOptions,
    config_module: ModuleType,
    role: config_defaults.AdaptiveRole,
) -> HiddenAdaptiveWeightOptions:
    return _merge_hidden_adaptive_weight_options(
        global_options,
        config_defaults.hidden_adaptive_weight_options(config_module),
        config_defaults.hidden_adaptive_weight_options(config_module, role),
    )


def _role_hidden_adaptive_bias_default(
    global_options: HiddenAdaptiveBiasOptions,
    config_module: ModuleType,
    role: config_defaults.AdaptiveRole,
) -> HiddenAdaptiveBiasOptions:
    return _merge_hidden_adaptive_bias_options(
        global_options,
        config_defaults.hidden_adaptive_bias_options(config_module),
        config_defaults.hidden_adaptive_bias_options(config_module, role),
    )


def _role_hidden_adaptive_diagonal_default(
    global_options: HiddenAdaptiveDiagonalOptions,
    config_module: ModuleType,
    role: config_defaults.AdaptiveRole,
) -> HiddenAdaptiveDiagonalOptions:
    return _merge_hidden_adaptive_diagonal_options(
        global_options,
        config_defaults.hidden_adaptive_diagonal_options(config_module),
        config_defaults.hidden_adaptive_diagonal_options(config_module, role),
    )


def _role_hidden_adaptive_mask_default(
    global_options: HiddenAdaptiveMaskOptions,
    config_module: ModuleType,
    role: config_defaults.AdaptiveRole,
) -> HiddenAdaptiveMaskOptions:
    return _merge_hidden_adaptive_mask_options(
        global_options,
        config_defaults.hidden_adaptive_mask_options(config_module),
        config_defaults.hidden_adaptive_mask_options(config_module, role),
    )


def _patch_options_from_kwargs(
    options: VitPatchOptions, kwargs: dict[str, Any]
) -> VitPatchOptions:
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


def _encoder_options_from_kwargs(
    options: TransformerEncoderOptions, kwargs: dict[str, Any]
) -> TransformerEncoderOptions:
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
            },
        ),
    )


def _positional_embedding_options_from_kwargs(
    options: TransformerPositionalEmbeddingOptions,
    kwargs: dict[str, Any],
) -> TransformerPositionalEmbeddingOptions:
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


def _attention_options_from_kwargs(
    options: TransformerAttentionOptions, kwargs: dict[str, Any]
) -> TransformerAttentionOptions:
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


def _feed_forward_options_from_kwargs(
    options: TransformerFeedForwardOptions, kwargs: dict[str, Any]
) -> TransformerFeedForwardOptions:
    return replace(
        options,
        **_updates(
            kwargs, {"ff_num_layers": "num_layers", "ff_bias_flag": "bias_flag"}
        ),
    )


def _output_options_from_kwargs(
    options: VitOutputOptions, kwargs: dict[str, Any]
) -> VitOutputOptions:
    return replace(options, **_updates(kwargs, {"output_bias_flag": "bias_flag"}))


def _main_stack_options_from_kwargs(
    options: MainLayerStackOptions, kwargs: dict[str, Any]
) -> MainLayerStackOptions:
    return replace(
        options,
        **_updates(
            kwargs,
            {
                "stack_bias_flag": "bias_flag",
                "layer_norm_position": "layer_norm_position",
                "stack_num_layers": "num_layers",
                "stack_activation": "activation",
                "stack_residual_connection_option": "residual_connection_option",
                "stack_residual_model_flag": "residual_model_flag",
                "stack_dropout_probability": "dropout_probability",
                "stack_last_layer_bias_option": "last_layer_bias_option",
                "stack_apply_output_pipeline_flag": "apply_output_pipeline_flag",
            },
        ),
    )


def _submodule_stack_options_from_kwargs(
    options: SubmoduleStackOptions, kwargs: dict[str, Any], *, flat_prefix: str
) -> SubmoduleStackOptions:
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
) -> LayerControllerOptions:
    prefix = f"{flat_prefix}_" if flat_prefix else ""
    flag_map = (
        {
            "stack_gate_flag": "stack_gate_flag",
            "gate_option": "gate_option",
            "gate_activation": "gate_activation",
            "stack_halting_flag": "stack_halting_flag",
            "halting_option": "halting_option",
            "halting_threshold": "halting_threshold",
            "halting_dropout": "halting_dropout",
            "halting_hidden_state_mode": "halting_hidden_state_mode",
        }
        if not flat_prefix
        else {
            f"{prefix}stack_gate_flag": "stack_gate_flag",
            f"{prefix}gate_option": "gate_option",
            f"{prefix}gate_activation": "gate_activation",
            f"{prefix}stack_halting_flag": "stack_halting_flag",
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
) -> DynamicMemoryOptions:
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
) -> RecurrentControllerOptions:
    prefix = f"{flat_prefix}_"
    updates = _updates(
        kwargs,
        {
            f"{prefix}flag": "recurrent_flag",
            f"{prefix}max_steps": "recurrent_max_steps",
            f"{prefix}initial_iterations": "recurrent_initial_iterations",
            f"{prefix}gradient_transition_count": "recurrent_gradient_transition_count",
            f"{prefix}iteration_increment": "recurrent_iteration_increment",
            f"{prefix}forward_calls_before_iteration_increment": (
                "recurrent_forward_calls_before_iteration_increment"
            ),
            f"{prefix}layer_norm_position": "recurrent_layer_norm_position",
            f"{prefix}stack_gate_flag": "recurrent_stack_gate_flag",
            f"{prefix}gate_option": "recurrent_gate_option",
            f"{prefix}gate_activation": "recurrent_gate_activation",
            f"{prefix}stack_halting_flag": "recurrent_stack_halting_flag",
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
) -> SubmoduleStackSource:
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


_PATCH_FLAT_KEYS = frozenset(
    {
        "image_patch_size",
        "input_channels",
        "image_height",
        "patch_dropout_probability",
        "patch_bias_flag",
    }
)
_ENCODER_FLAT_KEYS = frozenset(
    {
        "hidden_dim",
        "stack_num_layers",
        "stack_activation",
        "stack_dropout_probability",
        "layer_norm_position",
    }
)
_POSITIONAL_EMBEDDING_FLAT_KEYS = frozenset(
    {
        "positional_embedding_option",
        "positional_embedding_padding_idx",
        "positional_embedding_auto_expand_flag",
    }
)
_ATTENTION_FLAT_KEYS = frozenset(
    {
        "attn_num_heads",
        "attn_num_layers",
        "attn_bias_flag",
        "attn_add_key_value_bias_flag",
    }
)
_FEED_FORWARD_FLAT_KEYS = frozenset({"ff_num_layers", "ff_bias_flag"})
_OUTPUT_FLAT_KEYS = frozenset({"output_bias_flag"})
_MAIN_STACK_FLAT_KEYS = frozenset(
    {
        "stack_bias_flag",
        "layer_norm_position",
        "stack_num_layers",
        "stack_activation",
        "stack_residual_connection_option",
        "stack_residual_model_flag",
        "stack_dropout_probability",
        "stack_last_layer_bias_option",
        "stack_apply_output_pipeline_flag",
    }
)


def _resolve_patch_options(
    kwargs: dict[str, Any], config_module: ModuleType, consumed: set[str]
) -> VitPatchOptions | None:
    provided: VitPatchOptions | None = kwargs.get("patch_options")
    if provided is None and not _PATCH_FLAT_KEYS.intersection(kwargs):
        return None
    options = provided or config_defaults.vit_patch_options(config_module)
    consumed.add("patch_options")
    consumed.update(_PATCH_FLAT_KEYS.intersection(kwargs))
    return _patch_options_from_kwargs(options, kwargs)


def _resolve_encoder_options(
    kwargs: dict[str, Any], config_module: ModuleType, consumed: set[str]
) -> TransformerEncoderOptions | None:
    provided: TransformerEncoderOptions | None = kwargs.get("encoder_options")
    if provided is None and not _ENCODER_FLAT_KEYS.intersection(kwargs):
        return None
    options = provided or config_defaults.vit_encoder_options(config_module)
    consumed.add("encoder_options")
    consumed.update(_ENCODER_FLAT_KEYS.intersection(kwargs))
    return _encoder_options_from_kwargs(options, kwargs)


def _resolve_positional_embedding_options(
    kwargs: dict[str, Any], config_module: ModuleType, consumed: set[str]
) -> TransformerPositionalEmbeddingOptions | None:
    provided: TransformerPositionalEmbeddingOptions | None = kwargs.get(
        "positional_embedding_options"
    )
    if provided is None and not _POSITIONAL_EMBEDDING_FLAT_KEYS.intersection(kwargs):
        return None
    options = provided or config_defaults.vit_positional_embedding_options(
        config_module
    )
    consumed.add("positional_embedding_options")
    consumed.update(_POSITIONAL_EMBEDDING_FLAT_KEYS.intersection(kwargs))
    return _positional_embedding_options_from_kwargs(options, kwargs)


def _resolve_attention_options(
    kwargs: dict[str, Any], config_module: ModuleType, consumed: set[str]
) -> TransformerAttentionOptions | None:
    provided: TransformerAttentionOptions | None = kwargs.get("attention_options")
    if provided is None and not _ATTENTION_FLAT_KEYS.intersection(kwargs):
        return None
    options = provided or config_defaults.vit_attention_options(config_module)
    consumed.add("attention_options")
    consumed.update(_ATTENTION_FLAT_KEYS.intersection(kwargs))
    return _attention_options_from_kwargs(options, kwargs)


def _resolve_feed_forward_options(
    kwargs: dict[str, Any], config_module: ModuleType, consumed: set[str]
) -> TransformerFeedForwardOptions | None:
    provided: TransformerFeedForwardOptions | None = kwargs.get("feed_forward_options")
    if provided is None and not _FEED_FORWARD_FLAT_KEYS.intersection(kwargs):
        return None
    options = provided or config_defaults.vit_feed_forward_options(config_module)
    consumed.add("feed_forward_options")
    consumed.update(_FEED_FORWARD_FLAT_KEYS.intersection(kwargs))
    return _feed_forward_options_from_kwargs(options, kwargs)


def _resolve_output_options(
    kwargs: dict[str, Any], config_module: ModuleType, consumed: set[str]
) -> VitOutputOptions | None:
    provided: VitOutputOptions | None = kwargs.get("output_options")
    if provided is None and not _OUTPUT_FLAT_KEYS.intersection(kwargs):
        return None
    options = provided or config_defaults.vit_output_options(config_module)
    consumed.add("output_options")
    consumed.update(_OUTPUT_FLAT_KEYS.intersection(kwargs))
    return _output_options_from_kwargs(options, kwargs)


def _resolve_main_stack_options(
    kwargs: dict[str, Any], config_module: ModuleType, consumed: set[str]
) -> MainLayerStackOptions | None:
    provided: MainLayerStackOptions | None = kwargs.get("stack_options")
    if provided is None and not _MAIN_STACK_FLAT_KEYS.intersection(kwargs):
        return None
    options = provided or config_defaults.main_layer_stack_options(config_module)
    consumed.add("stack_options")
    consumed.update(_MAIN_STACK_FLAT_KEYS.intersection(kwargs))
    return _main_stack_options_from_kwargs(options, kwargs)


def _resolve_submodule_stack_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    *,
    option_key: str,
    provided: SubmoduleStackOptions | None,
    defaults_factory: Callable[[], SubmoduleStackOptions],
    flat_prefix: str,
) -> SubmoduleStackOptions | None:
    relevant = _submodule_stack_relevant_keys(flat_prefix)
    if provided is None and not relevant.intersection(kwargs):
        return None
    consumed.add(option_key)
    consumed.update(relevant.intersection(kwargs))
    return _submodule_stack_options_from_kwargs(
        provided or defaults_factory(), kwargs, flat_prefix=flat_prefix
    )


def _resolve_layer_controller_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    *,
    option_key: str,
    provided: LayerControllerOptions | None,
    defaults_factory: Callable[[], LayerControllerOptions],
    flat_prefix: str,
    gate_stack_prefix: str,
    halting_stack_prefix: str,
) -> LayerControllerOptions | None:
    relevant = _layer_controller_relevant_keys(
        flat_prefix, gate_stack_prefix, halting_stack_prefix
    )
    if provided is None and not relevant.intersection(kwargs):
        return None
    consumed.add(option_key)
    consumed.update(relevant.intersection(kwargs))
    return _layer_controller_options_from_kwargs(
        provided or defaults_factory(),
        kwargs,
        flat_prefix=flat_prefix,
        gate_stack_prefix=gate_stack_prefix,
        halting_stack_prefix=halting_stack_prefix,
    )


def _resolve_dynamic_memory_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    *,
    option_key: str,
    provided: DynamicMemoryOptions | None,
    defaults_factory: Callable[[], DynamicMemoryOptions],
    flat_prefix: str,
    memory_stack_prefix: str,
) -> DynamicMemoryOptions | None:
    relevant = _dynamic_memory_relevant_keys(flat_prefix, memory_stack_prefix)
    if provided is None and not relevant.intersection(kwargs):
        return None
    consumed.add(option_key)
    consumed.update(relevant.intersection(kwargs))
    return _dynamic_memory_options_from_kwargs(
        provided or defaults_factory(),
        kwargs,
        flat_prefix=flat_prefix,
        memory_stack_prefix=memory_stack_prefix,
    )


def _resolve_recurrent_controller_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    *,
    option_key: str,
    provided: RecurrentControllerOptions | None,
    defaults_factory: Callable[[], RecurrentControllerOptions],
    flat_prefix: str,
    gate_stack_prefix: str,
    halting_stack_prefix: str,
) -> RecurrentControllerOptions | None:
    relevant = _recurrent_controller_relevant_keys(
        flat_prefix, gate_stack_prefix, halting_stack_prefix
    )
    if provided is None and not relevant.intersection(kwargs):
        return None
    consumed.add(option_key)
    consumed.update(relevant.intersection(kwargs))
    return _recurrent_controller_options_from_kwargs(
        provided or defaults_factory(),
        kwargs,
        flat_prefix=flat_prefix,
        gate_stack_prefix=gate_stack_prefix,
        halting_stack_prefix=halting_stack_prefix,
    )


def _ensure_control_dependencies(
    builder_kwargs: dict[str, Any], config_module: ModuleType
) -> None:
    _ensure_main_control_dependencies(builder_kwargs, config_module)
    _ensure_attention_control_dependencies(builder_kwargs, config_module)
    _ensure_feed_forward_control_dependencies(builder_kwargs, config_module)


def _ensure_main_control_dependencies(
    builder_kwargs: dict[str, Any], config_module: ModuleType
) -> None:
    control_keys = {
        "submodule_stack_options",
        "layer_controller_options",
        "dynamic_memory_options",
        "recurrent_controller_options",
    }
    if not control_keys.intersection(builder_kwargs):
        return
    if "submodule_stack_options" not in builder_kwargs:
        builder_kwargs["submodule_stack_options"] = (
            config_defaults.linears_submodule_stack_options(
                config_module, config_defaults.LinearRole.MAIN
            )
        )
    if "layer_controller_options" not in builder_kwargs:
        builder_kwargs["layer_controller_options"] = (
            config_defaults.linears_layer_controller_options(
                config_module, config_defaults.LinearRole.MAIN
            )
        )
    if "dynamic_memory_options" not in builder_kwargs:
        builder_kwargs["dynamic_memory_options"] = (
            config_defaults.linears_dynamic_memory_options(
                config_module, config_defaults.LinearRole.MAIN
            )
        )
    if "recurrent_controller_options" not in builder_kwargs:
        builder_kwargs["recurrent_controller_options"] = (
            config_defaults.linears_recurrent_controller_options(
                config_module, config_defaults.LinearRole.MAIN
            )
        )


def _ensure_attention_control_dependencies(
    builder_kwargs: dict[str, Any], config_module: ModuleType
) -> None:
    control_keys = {
        "attention_projection_layer_controller_options",
        "attention_projection_dynamic_memory_options",
        "attention_projection_recurrent_controller_options",
    }
    if not control_keys.intersection(builder_kwargs):
        return
    if "attention_projection_layer_controller_options" not in builder_kwargs:
        builder_kwargs["attention_projection_layer_controller_options"] = (
            config_defaults.linears_layer_controller_options(
                config_module, config_defaults.LinearRole.ATTENTION
            )
        )
    if "attention_projection_dynamic_memory_options" not in builder_kwargs:
        builder_kwargs["attention_projection_dynamic_memory_options"] = (
            config_defaults.linears_dynamic_memory_options(
                config_module, config_defaults.LinearRole.ATTENTION
            )
        )
    if "attention_projection_recurrent_controller_options" not in builder_kwargs:
        builder_kwargs["attention_projection_recurrent_controller_options"] = (
            config_defaults.linears_recurrent_controller_options(
                config_module, config_defaults.LinearRole.ATTENTION
            )
        )


def _ensure_feed_forward_control_dependencies(
    builder_kwargs: dict[str, Any], config_module: ModuleType
) -> None:
    control_keys = {
        "feed_forward_layer_controller_options",
        "feed_forward_dynamic_memory_options",
        "feed_forward_recurrent_controller_options",
    }
    if not control_keys.intersection(builder_kwargs):
        return
    if "feed_forward_layer_controller_options" not in builder_kwargs:
        builder_kwargs["feed_forward_layer_controller_options"] = (
            config_defaults.linears_layer_controller_options(
                config_module, config_defaults.LinearRole.FEED_FORWARD
            )
        )
    if "feed_forward_dynamic_memory_options" not in builder_kwargs:
        builder_kwargs["feed_forward_dynamic_memory_options"] = (
            config_defaults.linears_dynamic_memory_options(
                config_module, config_defaults.LinearRole.FEED_FORWARD
            )
        )
    if "feed_forward_recurrent_controller_options" not in builder_kwargs:
        builder_kwargs["feed_forward_recurrent_controller_options"] = (
            config_defaults.linears_recurrent_controller_options(
                config_module, config_defaults.LinearRole.FEED_FORWARD
            )
        )


def _submodule_stack_relevant_keys(flat_prefix: str) -> set[str]:
    return {f"{flat_prefix}_{flat_field}" for flat_field in _SUBMODULE_STACK_FIELD_MAP}


def _layer_controller_relevant_keys(
    flat_prefix: str,
    gate_stack_prefix: str,
    halting_stack_prefix: str,
) -> set[str]:
    if flat_prefix:
        base = {
            f"{flat_prefix}_stack_gate_flag",
            f"{flat_prefix}_gate_option",
            f"{flat_prefix}_gate_activation",
            f"{flat_prefix}_stack_halting_flag",
            f"{flat_prefix}_halting_option",
            f"{flat_prefix}_halting_threshold",
            f"{flat_prefix}_halting_dropout",
            f"{flat_prefix}_halting_hidden_state_mode",
        }
    else:
        base = {
            "stack_gate_flag",
            "gate_option",
            "gate_activation",
            "stack_halting_flag",
            "halting_option",
            "halting_threshold",
            "halting_dropout",
            "halting_hidden_state_mode",
        }
    return (
        base
        | _controller_stack_keys(gate_stack_prefix)
        | _controller_stack_keys(halting_stack_prefix)
    )


def _dynamic_memory_relevant_keys(
    flat_prefix: str,
    memory_stack_prefix: str,
) -> set[str]:
    prefix = f"{flat_prefix}_" if flat_prefix else ""
    return {
        f"{prefix}memory_flag",
        f"{prefix}memory_option",
        f"{prefix}memory_position_option",
        f"{prefix}memory_test_time_training_learning_rate",
        f"{prefix}memory_test_time_training_num_inner_steps",
    } | _controller_stack_keys(memory_stack_prefix)


def _recurrent_controller_relevant_keys(
    flat_prefix: str,
    gate_stack_prefix: str,
    halting_stack_prefix: str,
) -> set[str]:
    return (
        {
            f"{flat_prefix}_flag",
            f"{flat_prefix}_max_steps",
            f"{flat_prefix}_initial_iterations",
            f"{flat_prefix}_gradient_transition_count",
            f"{flat_prefix}_iteration_increment",
            f"{flat_prefix}_forward_calls_before_iteration_increment",
            f"{flat_prefix}_layer_norm_position",
            f"{flat_prefix}_stack_gate_flag",
            f"{flat_prefix}_gate_option",
            f"{flat_prefix}_gate_activation",
            f"{flat_prefix}_stack_halting_flag",
            f"{flat_prefix}_halting_option",
            f"{flat_prefix}_halting_threshold",
            f"{flat_prefix}_halting_dropout",
            f"{flat_prefix}_halting_hidden_state_mode",
        }
        | _controller_stack_keys(gate_stack_prefix)
        | _controller_stack_keys(halting_stack_prefix)
    )


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
