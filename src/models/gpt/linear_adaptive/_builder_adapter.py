from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from typing import Any, TypeVar, cast

from models.gpt.linear_adaptive import _config_defaults as config_defaults
from models.gpt.linear_adaptive._flat_updates import pop_updates as _pop_updates
from models.gpt.linear_adaptive._linears_builder_adapter import (
    _adaptive_generator_stack_options_from_kwargs,
    _auto_enable_adaptive_option_flags,
    _hidden_adaptive_bias_options_from_kwargs,
    _hidden_adaptive_diagonal_options_from_kwargs,
    _hidden_adaptive_mask_options_from_kwargs,
    _hidden_adaptive_weight_options_from_kwargs,
)
from models.gpt.linear_adaptive._residual import (
    ResidualStackSource,
    resolve_residual_stack_options,
)
from models.gpt.linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
    DynamicMemoryOptions,
    GptEmbeddingOptions,
    GptLmHeadOptions,
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
    TransformerDecoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)

_TOP_LEVEL_KEYS = {
    "batch_size",
    "learning_rate",
    "input_dim",
    "output_dim",
    "sequence_length",
}
_GPT_GROUPED_KEYS = {
    *_TOP_LEVEL_KEYS,
    "embedding_options",
    "decoder_options",
    "positional_embedding_options",
    "attention_options",
    "feed_forward_options",
    "lm_head_options",
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
_ADAPTIVE_GROUPED_KEYS = {
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
}
_CONTROLLER_STACK_FIELD_MAP = {
    "independent_flag": "independent_flag",
    "hidden_dim": "hidden_dim",
    "num_layers": "num_layers",
    "last_layer_bias_option": "last_layer_bias_option",
    "apply_output_postprocessing_flag": "apply_output_postprocessing_flag",
    "activation": "activation",
    "layer_norm_position": "layer_norm_position",
    "residual_connection_option": "residual_connection_option",
    "residual_model_flag": "residual_model_flag",
    "dropout_probability": "dropout_probability",
    "bias_flag": "bias_flag",
}
_SUBMODULE_STACK_FIELD_MAP = {
    key: value
    for key, value in _CONTROLLER_STACK_FIELD_MAP.items()
    if key != "independent_flag"
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
    "apply_output_postprocessing_flag": "apply_output_postprocessing_flag",
    "bias_flag": "bias_flag",
}


def linear_adaptive_builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any],
    config_module: ModuleType,
) -> dict[str, Any]:
    kwargs = dict(flat_kwargs)
    _auto_enable_adaptive_option_flags(kwargs)
    consumed: set[str] = set()
    builder_kwargs = _top_level_kwargs(kwargs, consumed)
    builder_kwargs.update(_gpt_builder_kwargs(kwargs, config_module, consumed))
    builder_kwargs.update(
        _adaptive_builder_kwargs(
            kwargs,
            config_module,
            consumed,
            include_role_overrides=True,
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
    "apply_output_postprocessing_flag",
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
            config_module,
            config_defaults.LinearRole.MAIN,
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
            apply_output_postprocessing_flag=kwargs.get(
                "residual_stack_apply_output_postprocessing_flag",
                config_module.RESIDUAL_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
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


def _gpt_builder_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    consumed: set[str],
) -> dict[str, Any]:
    builder_kwargs = _gpt_boundary_builder_kwargs(kwargs, config_module, consumed)
    _set_submodule_stack_option(
        builder_kwargs,
        kwargs,
        consumed,
        builder_key="submodule_stack_options",
        flat_prefix="submodule_stack",
        default=config_defaults.linears_submodule_stack_options(
            config_module, config_defaults.LinearRole.MAIN
        ),
    )
    _set_layer_controller_option(
        builder_kwargs,
        kwargs,
        consumed,
        builder_key="layer_controller_options",
        flat_prefix="",
        gate_stack_prefix="gate_stack",
        halting_stack_prefix="halting_stack",
        default=config_defaults.linears_layer_controller_options(
            config_module,
            config_defaults.LinearRole.MAIN,
        ),
    )
    _set_dynamic_memory_option(
        builder_kwargs,
        kwargs,
        consumed,
        builder_key="dynamic_memory_options",
        flat_prefix="",
        memory_stack_prefix="memory_stack",
        default=config_defaults.linears_dynamic_memory_options(
            config_module,
            config_defaults.LinearRole.MAIN,
        ),
    )
    _set_recurrent_controller_option(
        builder_kwargs,
        kwargs,
        consumed,
        builder_key="recurrent_controller_options",
        flat_prefix="recurrent",
        gate_stack_prefix="recurrent_gate_stack",
        halting_stack_prefix="recurrent_halting_stack",
        default=config_defaults.linears_recurrent_controller_options(
            config_module,
            config_defaults.LinearRole.MAIN,
        ),
    )
    _set_submodule_stack_option(
        builder_kwargs,
        kwargs,
        consumed,
        builder_key="attention_projection_stack_options",
        flat_prefix="attn_stack",
        default=config_defaults.linears_submodule_stack_options(
            config_module,
            config_defaults.LinearRole.ATTENTION,
        ),
    )
    _set_layer_controller_option(
        builder_kwargs,
        kwargs,
        consumed,
        builder_key="attention_projection_layer_controller_options",
        flat_prefix="attn",
        gate_stack_prefix="attn_gate_stack",
        halting_stack_prefix="attn_halting_stack",
        default=config_defaults.linears_layer_controller_options(
            config_module,
            config_defaults.LinearRole.ATTENTION,
        ),
    )
    _set_dynamic_memory_option(
        builder_kwargs,
        kwargs,
        consumed,
        builder_key="attention_projection_dynamic_memory_options",
        flat_prefix="attn",
        memory_stack_prefix="attn_memory_stack",
        default=config_defaults.linears_dynamic_memory_options(
            config_module,
            config_defaults.LinearRole.ATTENTION,
        ),
    )
    _set_recurrent_controller_option(
        builder_kwargs,
        kwargs,
        consumed,
        builder_key="attention_projection_recurrent_controller_options",
        flat_prefix="attn_recurrent",
        gate_stack_prefix="attn_recurrent_gate_stack",
        halting_stack_prefix="attn_recurrent_halting_stack",
        default=config_defaults.linears_recurrent_controller_options(
            config_module,
            config_defaults.LinearRole.ATTENTION,
        ),
    )
    _set_submodule_stack_option(
        builder_kwargs,
        kwargs,
        consumed,
        builder_key="feed_forward_stack_options",
        flat_prefix="ff_stack",
        default=config_defaults.linears_submodule_stack_options(
            config_module,
            config_defaults.LinearRole.FEED_FORWARD,
        ),
    )
    _set_layer_controller_option(
        builder_kwargs,
        kwargs,
        consumed,
        builder_key="feed_forward_layer_controller_options",
        flat_prefix="ff",
        gate_stack_prefix="ff_gate_stack",
        halting_stack_prefix="ff_halting_stack",
        default=config_defaults.linears_layer_controller_options(
            config_module,
            config_defaults.LinearRole.FEED_FORWARD,
        ),
    )
    _set_dynamic_memory_option(
        builder_kwargs,
        kwargs,
        consumed,
        builder_key="feed_forward_dynamic_memory_options",
        flat_prefix="ff",
        memory_stack_prefix="ff_memory_stack",
        default=config_defaults.linears_dynamic_memory_options(
            config_module,
            config_defaults.LinearRole.FEED_FORWARD,
        ),
    )
    _set_recurrent_controller_option(
        builder_kwargs,
        kwargs,
        consumed,
        builder_key="feed_forward_recurrent_controller_options",
        flat_prefix="ff_recurrent",
        gate_stack_prefix="ff_recurrent_gate_stack",
        halting_stack_prefix="ff_recurrent_halting_stack",
        default=config_defaults.linears_recurrent_controller_options(
            config_module,
            config_defaults.LinearRole.FEED_FORWARD,
        ),
    )
    _ensure_control_dependencies(builder_kwargs, config_module)
    return builder_kwargs


def _option_requested(
    kwargs: dict[str, Any],
    builder_key: str,
    relevant_keys: set[str],
) -> bool:
    return kwargs.get(builder_key) is not None or bool(relevant_keys & set(kwargs))


def _record_option_consumption(
    kwargs: dict[str, Any],
    consumed: set[str],
    builder_key: str,
    relevant_keys: set[str],
) -> None:
    consumed.add(builder_key)
    consumed.update(relevant_keys & set(kwargs))


def _gpt_boundary_builder_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    consumed: set[str],
) -> dict[str, Any]:
    builder_kwargs: dict[str, Any] = {}
    embedding_keys = {
        "embedding_layer_norm_flag",
        "embedding_dropout_probability",
    }
    if _option_requested(kwargs, "embedding_options", embedding_keys):
        provided = cast(GptEmbeddingOptions | None, kwargs.get("embedding_options"))
        options = provided or config_defaults.gpt_embedding_options(config_module)
        builder_kwargs["embedding_options"] = _embedding_options_from_kwargs(
            options, kwargs
        )
        _record_option_consumption(
            kwargs, consumed, "embedding_options", embedding_keys
        )

    decoder_keys = {
        "hidden_dim",
        "stack_num_layers",
        "stack_activation",
        "stack_dropout_probability",
        "layer_norm_position",
    }
    if _option_requested(kwargs, "decoder_options", decoder_keys):
        provided = cast(TransformerDecoderOptions | None, kwargs.get("decoder_options"))
        options = provided or config_defaults.gpt_decoder_options(config_module)
        builder_kwargs["decoder_options"] = _decoder_options_from_kwargs(
            options, kwargs
        )
        _record_option_consumption(kwargs, consumed, "decoder_options", decoder_keys)

    positional_keys = {
        "positional_embedding_option",
        "positional_embedding_padding_idx",
        "positional_embedding_auto_expand_flag",
    }
    if _option_requested(kwargs, "positional_embedding_options", positional_keys):
        provided = cast(
            TransformerPositionalEmbeddingOptions | None,
            kwargs.get("positional_embedding_options"),
        )
        options = provided or config_defaults.gpt_positional_embedding_options(
            config_module
        )
        builder_kwargs["positional_embedding_options"] = (
            _positional_embedding_options_from_kwargs(options, kwargs)
        )
        _record_option_consumption(
            kwargs, consumed, "positional_embedding_options", positional_keys
        )

    attention_keys = {
        "attn_num_heads",
        "attn_num_layers",
        "attn_bias_flag",
        "attn_add_key_value_bias_flag",
    }
    if _option_requested(kwargs, "attention_options", attention_keys):
        provided = cast(
            TransformerAttentionOptions | None, kwargs.get("attention_options")
        )
        options = provided or config_defaults.gpt_attention_options(config_module)
        builder_kwargs["attention_options"] = _attention_options_from_kwargs(
            options, kwargs
        )
        _record_option_consumption(
            kwargs, consumed, "attention_options", attention_keys
        )

    feed_forward_keys = {"ff_num_layers", "ff_bias_flag"}
    if _option_requested(kwargs, "feed_forward_options", feed_forward_keys):
        provided = cast(
            TransformerFeedForwardOptions | None,
            kwargs.get("feed_forward_options"),
        )
        options = provided or config_defaults.gpt_feed_forward_options(config_module)
        builder_kwargs["feed_forward_options"] = _feed_forward_options_from_kwargs(
            options, kwargs
        )
        _record_option_consumption(
            kwargs, consumed, "feed_forward_options", feed_forward_keys
        )

    lm_head_keys = {"lm_head_bias_flag", "lm_head_weight_tying_flag"}
    if _option_requested(kwargs, "lm_head_options", lm_head_keys):
        provided = cast(GptLmHeadOptions | None, kwargs.get("lm_head_options"))
        options = provided or config_defaults.gpt_lm_head_options(config_module)
        builder_kwargs["lm_head_options"] = _lm_head_options_from_kwargs(
            options, kwargs
        )
        _record_option_consumption(kwargs, consumed, "lm_head_options", lm_head_keys)

    stack_keys = {
        "stack_bias_flag",
        "layer_norm_position",
        "stack_num_layers",
        "stack_activation",
        "stack_residual_connection_option",
        "stack_residual_model_flag",
        "stack_dropout_probability",
        "stack_last_layer_bias_option",
        "stack_apply_output_postprocessing_flag",
    }
    if _option_requested(kwargs, "stack_options", stack_keys):
        provided = cast(MainLayerStackOptions | None, kwargs.get("stack_options"))
        options = provided or config_defaults.main_layer_stack_options(config_module)
        builder_kwargs["stack_options"] = _main_stack_options_from_kwargs(
            options, kwargs
        )
        _record_option_consumption(kwargs, consumed, "stack_options", stack_keys)
    return builder_kwargs


def _submodule_stack_keys(flat_prefix: str) -> set[str]:
    return {f"{flat_prefix}_{flat_field}" for flat_field in _SUBMODULE_STACK_FIELD_MAP}


def _layer_controller_keys(
    flat_prefix: str,
    gate_stack_prefix: str,
    halting_stack_prefix: str,
) -> set[str]:
    prefix = f"{flat_prefix}_" if flat_prefix else ""
    return (
        {
            f"{prefix}stack_gate_flag",
            f"{prefix}gate_option",
            f"{prefix}gate_activation",
            f"{prefix}stack_halting_flag",
            f"{prefix}halting_option",
            f"{prefix}halting_threshold",
            f"{prefix}halting_dropout",
            f"{prefix}halting_hidden_state_mode",
        }
        | _controller_stack_keys(gate_stack_prefix)
        | _controller_stack_keys(halting_stack_prefix)
    )


def _dynamic_memory_keys(
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


def _recurrent_controller_keys(
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
            f"{flat_prefix}_no_gradient_transition_count",
            f"{flat_prefix}_iteration_increment",
            f"{flat_prefix}_forward_calls_before_iteration_increment",
            f"{flat_prefix}_smooth_iteration_growth_flag",
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


def _set_submodule_stack_option(
    builder_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
    consumed: set[str],
    *,
    builder_key: str,
    flat_prefix: str,
    default: SubmoduleStackOptions,
) -> None:
    relevant_keys = _submodule_stack_keys(flat_prefix)
    if not _option_requested(kwargs, builder_key, relevant_keys):
        return
    provided = cast(SubmoduleStackOptions | None, kwargs.get(builder_key))
    builder_kwargs[builder_key] = _submodule_stack_options_from_kwargs(
        provided or default,
        kwargs,
        flat_prefix=flat_prefix,
    )
    _record_option_consumption(kwargs, consumed, builder_key, relevant_keys)


def _set_layer_controller_option(
    builder_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
    consumed: set[str],
    *,
    builder_key: str,
    flat_prefix: str,
    gate_stack_prefix: str,
    halting_stack_prefix: str,
    default: LayerControllerOptions,
) -> None:
    relevant_keys = _layer_controller_keys(
        flat_prefix, gate_stack_prefix, halting_stack_prefix
    )
    if not _option_requested(kwargs, builder_key, relevant_keys):
        return
    provided = cast(LayerControllerOptions | None, kwargs.get(builder_key))
    builder_kwargs[builder_key] = _layer_controller_options_from_kwargs(
        provided or default,
        kwargs,
        flat_prefix=flat_prefix,
        gate_stack_prefix=gate_stack_prefix,
        halting_stack_prefix=halting_stack_prefix,
    )
    _record_option_consumption(kwargs, consumed, builder_key, relevant_keys)


def _set_dynamic_memory_option(
    builder_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
    consumed: set[str],
    *,
    builder_key: str,
    flat_prefix: str,
    memory_stack_prefix: str,
    default: DynamicMemoryOptions,
) -> None:
    relevant_keys = _dynamic_memory_keys(flat_prefix, memory_stack_prefix)
    if not _option_requested(kwargs, builder_key, relevant_keys):
        return
    provided = cast(DynamicMemoryOptions | None, kwargs.get(builder_key))
    builder_kwargs[builder_key] = _dynamic_memory_options_from_kwargs(
        provided or default,
        kwargs,
        flat_prefix=flat_prefix,
        memory_stack_prefix=memory_stack_prefix,
    )
    _record_option_consumption(kwargs, consumed, builder_key, relevant_keys)


def _set_recurrent_controller_option(
    builder_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
    consumed: set[str],
    *,
    builder_key: str,
    flat_prefix: str,
    gate_stack_prefix: str,
    halting_stack_prefix: str,
    default: RecurrentControllerOptions,
) -> None:
    relevant_keys = _recurrent_controller_keys(
        flat_prefix, gate_stack_prefix, halting_stack_prefix
    )
    if not _option_requested(kwargs, builder_key, relevant_keys):
        return
    provided = cast(RecurrentControllerOptions | None, kwargs.get(builder_key))
    builder_kwargs[builder_key] = _recurrent_controller_options_from_kwargs(
        provided or default,
        kwargs,
        flat_prefix=flat_prefix,
        gate_stack_prefix=gate_stack_prefix,
        halting_stack_prefix=halting_stack_prefix,
    )
    _record_option_consumption(kwargs, consumed, builder_key, relevant_keys)


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
    for builder_prefix, flat_prefix, role in (
        ("attention", "attn_", config_defaults.LinearRole.ATTENTION),
        ("feed_forward", "ff_", config_defaults.LinearRole.FEED_FORWARD),
    ):
        builder_kwargs.update(
            _role_adaptive_builder_kwargs(
                kwargs,
                config_module,
                consumed,
                builder_prefix=builder_prefix,
                flat_prefix=flat_prefix,
                role=role,
                adaptive_generator_stack_options=adaptive_generator_stack_options,
                hidden_adaptive_weight_options=hidden_adaptive_weight_options,
                hidden_adaptive_bias_options=hidden_adaptive_bias_options,
                hidden_adaptive_diagonal_options=hidden_adaptive_diagonal_options,
                hidden_adaptive_mask_options=hidden_adaptive_mask_options,
            )
        )
    return builder_kwargs


def _record_consumed_delta(
    kwargs: dict[str, Any],
    remaining: dict[str, Any],
    consumed: set[str],
) -> None:
    consumed.update(set(kwargs) - set(remaining))


def _resolved_adaptive_generator_stack_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    config_module: ModuleType,
    *,
    provided: AdaptiveGeneratorStackOptions | None,
) -> AdaptiveGeneratorStackOptions:
    remaining = dict(kwargs)
    options = _adaptive_generator_stack_options_from_kwargs(
        remaining,
        config_module,
        provided=provided,
    )
    _record_consumed_delta(kwargs, remaining, consumed)
    return options


def _resolved_role_adaptive_generator_stack_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    config_module: ModuleType,
    *,
    provided: AdaptiveGeneratorStackOptions | None,
    flat_prefix: str,
) -> AdaptiveGeneratorStackOptions:
    remaining = dict(kwargs)
    options = _adaptive_generator_stack_options_with_prefix_from_kwargs(
        remaining,
        config_module,
        provided=provided,
        flat_prefix=flat_prefix,
    )
    _record_consumed_delta(kwargs, remaining, consumed)
    return options


def _resolved_hidden_adaptive_weight_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveWeightOptions | None,
    flat_prefix: str = "",
    role: config_defaults.LinearRole = config_defaults.LinearRole.MAIN,
    stack_prefix: str = "weight_generator_stack",
) -> HiddenAdaptiveWeightOptions:
    remaining = dict(kwargs)
    options = _hidden_adaptive_weight_options_from_kwargs(
        remaining,
        config_module,
        provided=provided,
        flat_prefix=flat_prefix,
        role=role,
        stack_prefix=stack_prefix,
    )
    _record_consumed_delta(kwargs, remaining, consumed)
    return options


def _resolved_hidden_adaptive_bias_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveBiasOptions | None,
    flat_prefix: str = "",
    role: config_defaults.LinearRole = config_defaults.LinearRole.MAIN,
    stack_prefix: str = "bias_generator_stack",
) -> HiddenAdaptiveBiasOptions:
    remaining = dict(kwargs)
    options = _hidden_adaptive_bias_options_from_kwargs(
        remaining,
        config_module,
        provided=provided,
        flat_prefix=flat_prefix,
        role=role,
        stack_prefix=stack_prefix,
    )
    _record_consumed_delta(kwargs, remaining, consumed)
    return options


def _resolved_hidden_adaptive_diagonal_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveDiagonalOptions | None,
    flat_prefix: str = "",
    role: config_defaults.LinearRole = config_defaults.LinearRole.MAIN,
    stack_prefix: str = "diagonal_generator_stack",
) -> HiddenAdaptiveDiagonalOptions:
    remaining = dict(kwargs)
    options = _hidden_adaptive_diagonal_options_from_kwargs(
        remaining,
        config_module,
        provided=provided,
        flat_prefix=flat_prefix,
        role=role,
        stack_prefix=stack_prefix,
    )
    _record_consumed_delta(kwargs, remaining, consumed)
    return options


def _resolved_hidden_adaptive_mask_options(
    kwargs: dict[str, Any],
    consumed: set[str],
    config_module: ModuleType,
    *,
    provided: HiddenAdaptiveMaskOptions | None,
    flat_prefix: str = "",
    role: config_defaults.LinearRole = config_defaults.LinearRole.MAIN,
    stack_prefix: str = "mask_generator_stack",
) -> HiddenAdaptiveMaskOptions:
    remaining = dict(kwargs)
    options = _hidden_adaptive_mask_options_from_kwargs(
        remaining,
        config_module,
        provided=provided,
        flat_prefix=flat_prefix,
        role=role,
        stack_prefix=stack_prefix,
    )
    _record_consumed_delta(kwargs, remaining, consumed)
    return options


def _role_adaptive_builder_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    consumed: set[str],
    *,
    builder_prefix: str,
    flat_prefix: str,
    role: config_defaults.LinearRole,
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions,
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions,
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions,
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions,
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions,
) -> dict[str, Any]:
    return {
        f"{builder_prefix}_adaptive_generator_stack_options": _resolved_role_adaptive_generator_stack_options(
            kwargs,
            consumed,
            config_module,
            provided=(
                kwargs.get(f"{builder_prefix}_adaptive_generator_stack_options")
                or adaptive_generator_stack_options
            ),
            flat_prefix=f"{flat_prefix}adaptive_generator_stack",
        ),
        f"{builder_prefix}_hidden_adaptive_weight_options": _resolved_hidden_adaptive_weight_options(
            kwargs,
            consumed,
            config_module,
            provided=(
                kwargs.get(f"{builder_prefix}_hidden_adaptive_weight_options")
                or _role_hidden_adaptive_weight_default(
                    hidden_adaptive_weight_options,
                    config_module,
                    role,
                )
            ),
            flat_prefix=flat_prefix,
            role=role,
            stack_prefix=f"{flat_prefix}weight_generator_stack",
        ),
        f"{builder_prefix}_hidden_adaptive_bias_options": _resolved_hidden_adaptive_bias_options(
            kwargs,
            consumed,
            config_module,
            provided=(
                kwargs.get(f"{builder_prefix}_hidden_adaptive_bias_options")
                or _role_hidden_adaptive_bias_default(
                    hidden_adaptive_bias_options,
                    config_module,
                    role,
                )
            ),
            flat_prefix=flat_prefix,
            role=role,
            stack_prefix=f"{flat_prefix}bias_generator_stack",
        ),
        f"{builder_prefix}_hidden_adaptive_diagonal_options": _resolved_hidden_adaptive_diagonal_options(
            kwargs,
            consumed,
            config_module,
            provided=(
                kwargs.get(f"{builder_prefix}_hidden_adaptive_diagonal_options")
                or _role_hidden_adaptive_diagonal_default(
                    hidden_adaptive_diagonal_options,
                    config_module,
                    role,
                )
            ),
            flat_prefix=flat_prefix,
            role=role,
            stack_prefix=f"{flat_prefix}diagonal_generator_stack",
        ),
        f"{builder_prefix}_hidden_adaptive_mask_options": _resolved_hidden_adaptive_mask_options(
            kwargs,
            consumed,
            config_module,
            provided=(
                kwargs.get(f"{builder_prefix}_hidden_adaptive_mask_options")
                or _role_hidden_adaptive_mask_default(
                    hidden_adaptive_mask_options,
                    config_module,
                    role,
                )
            ),
            flat_prefix=flat_prefix,
            role=role,
            stack_prefix=f"{flat_prefix}mask_generator_stack",
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
            for flat_field, dataclass_field in (
                _ADAPTIVE_GENERATOR_STACK_FIELD_MAP.items()
            )
        },
    )
    return replace(options, **updates) if updates else options


_OptionValue = TypeVar("_OptionValue")


def _role_value(
    base: _OptionValue,
    global_default: _OptionValue,
    role_default: _OptionValue,
) -> _OptionValue:
    return role_default if role_default != global_default else base


def _merge_adaptive_generator_stack_source(
    base: AdaptiveGeneratorStackSource,
    global_default: AdaptiveGeneratorStackSource,
    role_default: AdaptiveGeneratorStackSource,
) -> AdaptiveGeneratorStackSource:
    return replace(
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
        apply_output_postprocessing_flag=_role_value(
            base.apply_output_postprocessing_flag,
            global_default.apply_output_postprocessing_flag,
            role_default.apply_output_postprocessing_flag,
        ),
        bias_flag=_role_value(
            base.bias_flag, global_default.bias_flag, role_default.bias_flag
        ),
    )


def _merge_hidden_adaptive_weight_options(
    base: HiddenAdaptiveWeightOptions,
    global_default: HiddenAdaptiveWeightOptions,
    role_default: HiddenAdaptiveWeightOptions,
) -> HiddenAdaptiveWeightOptions:
    return replace(
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
        generator_stack_source=_merge_adaptive_generator_stack_source(
            base.generator_stack_source,
            global_default.generator_stack_source,
            role_default.generator_stack_source,
        ),
    )


def _merge_hidden_adaptive_bias_options(
    base: HiddenAdaptiveBiasOptions,
    global_default: HiddenAdaptiveBiasOptions,
    role_default: HiddenAdaptiveBiasOptions,
) -> HiddenAdaptiveBiasOptions:
    return replace(
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
        generator_stack_source=_merge_adaptive_generator_stack_source(
            base.generator_stack_source,
            global_default.generator_stack_source,
            role_default.generator_stack_source,
        ),
    )


def _merge_hidden_adaptive_diagonal_options(
    base: HiddenAdaptiveDiagonalOptions,
    global_default: HiddenAdaptiveDiagonalOptions,
    role_default: HiddenAdaptiveDiagonalOptions,
) -> HiddenAdaptiveDiagonalOptions:
    return replace(
        base,
        option_flag=_role_value(
            base.option_flag, global_default.option_flag, role_default.option_flag
        ),
        option=_role_value(base.option, global_default.option, role_default.option),
        generator_stack_source=_merge_adaptive_generator_stack_source(
            base.generator_stack_source,
            global_default.generator_stack_source,
            role_default.generator_stack_source,
        ),
    )


def _merge_hidden_adaptive_mask_options(
    base: HiddenAdaptiveMaskOptions,
    global_default: HiddenAdaptiveMaskOptions,
    role_default: HiddenAdaptiveMaskOptions,
) -> HiddenAdaptiveMaskOptions:
    return replace(
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
        generator_stack_source=_merge_adaptive_generator_stack_source(
            base.generator_stack_source,
            global_default.generator_stack_source,
            role_default.generator_stack_source,
        ),
    )


def _role_hidden_adaptive_weight_default(
    global_options: HiddenAdaptiveWeightOptions,
    config_module: ModuleType,
    role: config_defaults.LinearRole,
) -> HiddenAdaptiveWeightOptions:
    return _merge_hidden_adaptive_weight_options(
        global_options,
        config_defaults.hidden_adaptive_weight_options(config_module),
        config_defaults.hidden_adaptive_weight_options(config_module, role),
    )


def _role_hidden_adaptive_bias_default(
    global_options: HiddenAdaptiveBiasOptions,
    config_module: ModuleType,
    role: config_defaults.LinearRole,
) -> HiddenAdaptiveBiasOptions:
    return _merge_hidden_adaptive_bias_options(
        global_options,
        config_defaults.hidden_adaptive_bias_options(config_module),
        config_defaults.hidden_adaptive_bias_options(config_module, role),
    )


def _role_hidden_adaptive_diagonal_default(
    global_options: HiddenAdaptiveDiagonalOptions,
    config_module: ModuleType,
    role: config_defaults.LinearRole,
) -> HiddenAdaptiveDiagonalOptions:
    return _merge_hidden_adaptive_diagonal_options(
        global_options,
        config_defaults.hidden_adaptive_diagonal_options(config_module),
        config_defaults.hidden_adaptive_diagonal_options(config_module, role),
    )


def _role_hidden_adaptive_mask_default(
    global_options: HiddenAdaptiveMaskOptions,
    config_module: ModuleType,
    role: config_defaults.LinearRole,
) -> HiddenAdaptiveMaskOptions:
    return _merge_hidden_adaptive_mask_options(
        global_options,
        config_defaults.hidden_adaptive_mask_options(config_module),
        config_defaults.hidden_adaptive_mask_options(config_module, role),
    )


def _embedding_options_from_kwargs(
    options: GptEmbeddingOptions,
    kwargs: dict[str, Any],
) -> GptEmbeddingOptions:
    return replace(
        options,
        **_updates(
            kwargs,
            {
                "embedding_layer_norm_flag": "layer_norm_flag",
                "embedding_dropout_probability": "dropout_probability",
            },
        ),
    )


def _decoder_options_from_kwargs(
    options: TransformerDecoderOptions,
    kwargs: dict[str, Any],
) -> TransformerDecoderOptions:
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
    options: TransformerAttentionOptions,
    kwargs: dict[str, Any],
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
    options: TransformerFeedForwardOptions,
    kwargs: dict[str, Any],
) -> TransformerFeedForwardOptions:
    return replace(
        options,
        **_updates(
            kwargs,
            {"ff_num_layers": "num_layers", "ff_bias_flag": "bias_flag"},
        ),
    )


def _lm_head_options_from_kwargs(
    options: GptLmHeadOptions,
    kwargs: dict[str, Any],
) -> GptLmHeadOptions:
    return replace(
        options,
        **_updates(
            kwargs,
            {
                "lm_head_weight_tying_flag": "weight_tying_flag",
                "lm_head_bias_flag": "bias_flag",
            },
        ),
    )


def _main_stack_options_from_kwargs(
    options: MainLayerStackOptions,
    kwargs: dict[str, Any],
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
                "stack_apply_output_postprocessing_flag": "apply_output_postprocessing_flag",
            },
        ),
    )


def _submodule_stack_options_from_kwargs(
    options: SubmoduleStackOptions,
    kwargs: dict[str, Any],
    *,
    flat_prefix: str,
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
    if flat_prefix:
        flag_map = {
            f"{prefix}stack_gate_flag": "stack_gate_flag",
            f"{prefix}gate_option": "gate_option",
            f"{prefix}gate_activation": "gate_activation",
            f"{prefix}stack_halting_flag": "stack_halting_flag",
            f"{prefix}halting_option": "halting_option",
            f"{prefix}halting_threshold": "halting_threshold",
            f"{prefix}halting_dropout": "halting_dropout",
            f"{prefix}halting_hidden_state_mode": "halting_hidden_state_mode",
        }
    else:
        flag_map = {
            "stack_gate_flag": "stack_gate_flag",
            "gate_option": "gate_option",
            "gate_activation": "gate_activation",
            "stack_halting_flag": "stack_halting_flag",
            "halting_option": "halting_option",
            "halting_threshold": "halting_threshold",
            "halting_dropout": "halting_dropout",
            "halting_hidden_state_mode": "halting_hidden_state_mode",
        }
    updates = _updates(kwargs, flag_map)
    updates["gate_stack_source"] = _controller_stack_source_from_kwargs(
        options.gate_stack_source,
        kwargs,
        gate_stack_prefix,
    )
    updates["halting_stack_source"] = _controller_stack_source_from_kwargs(
        options.halting_stack_source,
        kwargs,
        halting_stack_prefix,
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
            f"{prefix}memory_test_time_training_learning_rate": (
                "memory_test_time_training_learning_rate"
            ),
            f"{prefix}memory_test_time_training_num_inner_steps": (
                "memory_test_time_training_num_inner_steps"
            ),
        },
    )
    updates["memory_stack_source"] = _controller_stack_source_from_kwargs(
        options.memory_stack_source,
        kwargs,
        memory_stack_prefix,
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
            f"{prefix}no_gradient_transition_count": "recurrent_no_gradient_transition_count",
            f"{prefix}iteration_increment": "recurrent_iteration_increment",
            f"{prefix}forward_calls_before_iteration_increment": (
                "recurrent_forward_calls_before_iteration_increment"
            ),
            f"{prefix}smooth_iteration_growth_flag": (
                "recurrent_smooth_iteration_growth_flag"
            ),
            f"{prefix}layer_norm_position": "recurrent_layer_norm_position",
            f"{prefix}stack_gate_flag": "recurrent_stack_gate_flag",
            f"{prefix}gate_option": "recurrent_gate_option",
            f"{prefix}gate_activation": "recurrent_gate_activation",
            f"{prefix}stack_halting_flag": "recurrent_stack_halting_flag",
            f"{prefix}halting_option": "recurrent_halting_option",
            f"{prefix}halting_threshold": "recurrent_halting_threshold",
            f"{prefix}halting_dropout": "recurrent_halting_dropout",
            f"{prefix}halting_hidden_state_mode": (
                "recurrent_halting_hidden_state_mode"
            ),
        },
    )
    updates["recurrent_gate_stack_source"] = _controller_stack_source_from_kwargs(
        options.recurrent_gate_stack_source,
        kwargs,
        gate_stack_prefix,
    )
    updates["recurrent_halting_stack_source"] = _controller_stack_source_from_kwargs(
        options.recurrent_halting_stack_source,
        kwargs,
        halting_stack_prefix,
    )
    return replace(options, **updates)


def _controller_stack_source_from_kwargs(
    source: SubmoduleStackSource,
    kwargs: dict[str, Any],
    flat_prefix: str,
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


def _ensure_control_dependencies(
    builder_kwargs: dict[str, Any],
    config_module: ModuleType,
) -> None:
    _ensure_main_control_dependencies(builder_kwargs, config_module)
    _ensure_attention_control_dependencies(builder_kwargs, config_module)
    _ensure_feed_forward_control_dependencies(builder_kwargs, config_module)


def _ensure_main_control_dependencies(
    builder_kwargs: dict[str, Any],
    config_module: ModuleType,
) -> None:
    decoder_control_keys = {
        "submodule_stack_options",
        "layer_controller_options",
        "dynamic_memory_options",
        "recurrent_controller_options",
    }
    if decoder_control_keys & set(builder_kwargs):
        if "submodule_stack_options" not in builder_kwargs:
            builder_kwargs["submodule_stack_options"] = (
                config_defaults.linears_submodule_stack_options(
                    config_module, config_defaults.LinearRole.MAIN
                )
            )
        if "layer_controller_options" not in builder_kwargs:
            builder_kwargs["layer_controller_options"] = (
                config_defaults.linears_layer_controller_options(
                    config_module,
                    config_defaults.LinearRole.MAIN,
                )
            )
        if "dynamic_memory_options" not in builder_kwargs:
            builder_kwargs["dynamic_memory_options"] = (
                config_defaults.linears_dynamic_memory_options(
                    config_module,
                    config_defaults.LinearRole.MAIN,
                )
            )
        if "recurrent_controller_options" not in builder_kwargs:
            builder_kwargs["recurrent_controller_options"] = (
                config_defaults.linears_recurrent_controller_options(
                    config_module,
                    config_defaults.LinearRole.MAIN,
                )
            )


def _ensure_attention_control_dependencies(
    builder_kwargs: dict[str, Any],
    config_module: ModuleType,
) -> None:
    attention_control_keys = {
        "attention_projection_layer_controller_options",
        "attention_projection_dynamic_memory_options",
        "attention_projection_recurrent_controller_options",
    }
    if attention_control_keys & set(builder_kwargs):
        if "attention_projection_layer_controller_options" not in builder_kwargs:
            builder_kwargs["attention_projection_layer_controller_options"] = (
                config_defaults.linears_layer_controller_options(
                    config_module,
                    config_defaults.LinearRole.ATTENTION,
                )
            )
        if "attention_projection_dynamic_memory_options" not in builder_kwargs:
            builder_kwargs["attention_projection_dynamic_memory_options"] = (
                config_defaults.linears_dynamic_memory_options(
                    config_module,
                    config_defaults.LinearRole.ATTENTION,
                )
            )
        if "attention_projection_recurrent_controller_options" not in builder_kwargs:
            builder_kwargs["attention_projection_recurrent_controller_options"] = (
                config_defaults.linears_recurrent_controller_options(
                    config_module,
                    config_defaults.LinearRole.ATTENTION,
                )
            )


def _ensure_feed_forward_control_dependencies(
    builder_kwargs: dict[str, Any],
    config_module: ModuleType,
) -> None:
    feed_forward_control_keys = {
        "feed_forward_layer_controller_options",
        "feed_forward_dynamic_memory_options",
        "feed_forward_recurrent_controller_options",
    }
    if feed_forward_control_keys & set(builder_kwargs):
        if "feed_forward_layer_controller_options" not in builder_kwargs:
            builder_kwargs["feed_forward_layer_controller_options"] = (
                config_defaults.linears_layer_controller_options(
                    config_module,
                    config_defaults.LinearRole.FEED_FORWARD,
                )
            )
        if "feed_forward_dynamic_memory_options" not in builder_kwargs:
            builder_kwargs["feed_forward_dynamic_memory_options"] = (
                config_defaults.linears_dynamic_memory_options(
                    config_module,
                    config_defaults.LinearRole.FEED_FORWARD,
                )
            )
        if "feed_forward_recurrent_controller_options" not in builder_kwargs:
            builder_kwargs["feed_forward_recurrent_controller_options"] = (
                config_defaults.linears_recurrent_controller_options(
                    config_module,
                    config_defaults.LinearRole.FEED_FORWARD,
                )
            )


def _controller_stack_keys(prefix: str) -> set[str]:
    return {f"{prefix}_{flat_field}" for flat_field in _CONTROLLER_STACK_FIELD_MAP}


def _updates(kwargs: dict[str, Any], field_map: dict[str, str]) -> dict[str, Any]:
    return {
        dataclass_field: kwargs[flat_field]
        for flat_field, dataclass_field in field_map.items()
        if flat_field in kwargs
    }


def _leftover_kwargs(
    kwargs: dict[str, Any],
    consumed: set[str],
) -> dict[str, Any]:
    consumed.update(_GPT_GROUPED_KEYS & set(kwargs))
    consumed.update(_ADAPTIVE_GROUPED_KEYS & set(kwargs))
    return {key: value for key, value in kwargs.items() if key not in consumed}
