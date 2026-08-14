from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from typing import Any, Literal

from models.gpt.linear import _config_defaults as config_defaults
from models.gpt.linear._residual import (
    ResidualStackSource,
    resolve_residual_stack_options,
)
from models.gpt.linear.runtime_options import (
    DynamicMemoryOptions,
    GptEmbeddingOptions,
    GptLmHeadOptions,
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
    field: field for field in _CONTROLLER_STACK_FIELD_MAP if field != "independent_flag"
}

_MODERN_TOP_LEVEL_KEYS = {
    "batch_size",
    "learning_rate",
    "input_dim",
    "output_dim",
    "sequence_length",
}

_MODERN_GROUPED_KEYS = {
    *_MODERN_TOP_LEVEL_KEYS,
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


def linear_builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any],
    config_module: ModuleType,
) -> dict[str, Any]:
    kwargs = dict(flat_kwargs)
    decoder_options = _modern_decoder_options(
        kwargs,
        config_module,
        provided=kwargs.get("decoder_options"),
    )
    attention_options = _modern_attention_options(
        kwargs,
        config_module,
        provided=kwargs.get("attention_options"),
    )
    feed_forward_options = _modern_feed_forward_options(
        kwargs,
        config_module,
        provided=kwargs.get("feed_forward_options"),
    )
    stack_options = _modern_main_stack_options(
        kwargs,
        config_module,
        provided=kwargs.get("stack_options"),
    )
    submodule_stack_options = _modern_submodule_stack_options(
        kwargs,
        config_module,
        stack_options=stack_options,
        provided=kwargs.get("submodule_stack_options"),
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
            num_layers=kwargs.get(
                "residual_stack_num_layers", config_module.RESIDUAL_STACK_NUM_LAYERS
            ),
            activation=kwargs.get(
                "residual_stack_activation", config_module.RESIDUAL_STACK_ACTIVATION
            ),
            layer_norm_position=kwargs.get(
                "residual_stack_layer_norm_position",
                config_module.RESIDUAL_STACK_LAYER_NORM_POSITION,
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
    stack_options = replace(
        stack_options, residual_stack_options=residual_stack_options
    )
    submodule_stack_options = replace(
        submodule_stack_options, residual_stack_options=residual_stack_options
    )
    attention_projection_stack_options = replace(
        _modern_attention_projection_stack_options(
            kwargs,
            config_module,
            decoder_options=decoder_options,
            attention_options=attention_options,
            provided=kwargs.get("attention_projection_stack_options"),
        ),
        residual_stack_options=residual_stack_options,
    )
    feed_forward_stack_options = replace(
        _modern_feed_forward_stack_options(
            kwargs,
            config_module,
            decoder_options=decoder_options,
            feed_forward_options=feed_forward_options,
            provided=kwargs.get("feed_forward_stack_options"),
        ),
        residual_stack_options=residual_stack_options,
    )
    builder_kwargs: dict[str, Any] = {
        key: kwargs[key] for key in _MODERN_TOP_LEVEL_KEYS if key in kwargs
    }
    builder_kwargs.update(
        {
            "embedding_options": _modern_embedding_options(
                kwargs,
                config_module,
                provided=kwargs.get("embedding_options"),
            ),
            "decoder_options": decoder_options,
            "positional_embedding_options": _modern_positional_embedding_options(
                kwargs,
                config_module,
                provided=kwargs.get("positional_embedding_options"),
            ),
            "attention_options": attention_options,
            "feed_forward_options": feed_forward_options,
            "lm_head_options": _modern_lm_head_options(
                kwargs,
                config_module,
                provided=kwargs.get("lm_head_options"),
            ),
            "attention_projection_stack_options": attention_projection_stack_options,
            "attention_projection_layer_controller_options": (
                _modern_layer_controller_options(
                    kwargs,
                    config_module,
                    flat_prefix="attn",
                    role="attention",
                    provided=kwargs.get(
                        "attention_projection_layer_controller_options"
                    ),
                )
            ),
            "attention_projection_dynamic_memory_options": (
                _modern_dynamic_memory_options(
                    kwargs,
                    config_module,
                    flat_prefix="attn",
                    role="attention",
                    provided=kwargs.get("attention_projection_dynamic_memory_options"),
                )
            ),
            "attention_projection_recurrent_controller_options": (
                _modern_recurrent_controller_options(
                    kwargs,
                    config_module,
                    flat_prefix="attn_recurrent",
                    role="attention",
                    provided=kwargs.get(
                        "attention_projection_recurrent_controller_options"
                    ),
                )
            ),
            "feed_forward_stack_options": feed_forward_stack_options,
            "feed_forward_layer_controller_options": (
                _modern_layer_controller_options(
                    kwargs,
                    config_module,
                    flat_prefix="ff",
                    role="feed_forward",
                    provided=kwargs.get("feed_forward_layer_controller_options"),
                )
            ),
            "feed_forward_dynamic_memory_options": (
                _modern_dynamic_memory_options(
                    kwargs,
                    config_module,
                    flat_prefix="ff",
                    role="feed_forward",
                    provided=kwargs.get("feed_forward_dynamic_memory_options"),
                )
            ),
            "feed_forward_recurrent_controller_options": (
                _modern_recurrent_controller_options(
                    kwargs,
                    config_module,
                    flat_prefix="ff_recurrent",
                    role="feed_forward",
                    provided=kwargs.get("feed_forward_recurrent_controller_options"),
                )
            ),
            "stack_options": stack_options,
            "submodule_stack_options": submodule_stack_options,
            "layer_controller_options": _modern_layer_controller_options(
                kwargs,
                config_module,
                flat_prefix="",
                role="main",
                provided=kwargs.get("layer_controller_options"),
            ),
            "dynamic_memory_options": _modern_dynamic_memory_options(
                kwargs,
                config_module,
                flat_prefix="",
                role="main",
                provided=kwargs.get("dynamic_memory_options"),
            ),
            "recurrent_controller_options": _modern_recurrent_controller_options(
                kwargs,
                config_module,
                flat_prefix="recurrent",
                role="main",
                provided=kwargs.get("recurrent_controller_options"),
            ),
        }
    )
    supported_flat_keys = _modern_supported_flat_keys()
    builder_kwargs.update(
        {
            key: value
            for key, value in kwargs.items()
            if key not in _MODERN_GROUPED_KEYS and key not in supported_flat_keys
        }
    )
    return builder_kwargs


def _modern_embedding_options(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: GptEmbeddingOptions | None,
) -> GptEmbeddingOptions:
    options = provided or config_defaults.gpt_embedding_options(config_module)
    return replace(
        options,
        **_modern_option_updates(
            kwargs,
            {
                "embedding_layer_norm_flag": "layer_norm_flag",
                "embedding_dropout_probability": "dropout_probability",
            },
        ),
    )


def _modern_decoder_options(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: TransformerDecoderOptions | None,
) -> TransformerDecoderOptions:
    options = provided or config_defaults.gpt_decoder_options(config_module)
    return replace(
        options,
        **_modern_option_updates(
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


def _modern_positional_embedding_options(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: TransformerPositionalEmbeddingOptions | None,
) -> TransformerPositionalEmbeddingOptions:
    options = provided or config_defaults.gpt_positional_embedding_options(
        config_module
    )
    return replace(
        options,
        **_modern_option_updates(
            kwargs,
            {
                "positional_embedding_option": "option",
                "positional_embedding_padding_idx": "padding_idx",
                "positional_embedding_auto_expand_flag": "auto_expand_flag",
            },
        ),
    )


def _modern_attention_options(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: TransformerAttentionOptions | None,
) -> TransformerAttentionOptions:
    options = provided or config_defaults.gpt_attention_options(config_module)
    return replace(
        options,
        **_modern_option_updates(
            kwargs,
            {
                "attn_num_heads": "num_heads",
                "attn_num_layers": "num_layers",
                "attn_bias_flag": "bias_flag",
                "attn_add_key_value_bias_flag": "add_key_value_bias_flag",
            },
        ),
    )


def _modern_feed_forward_options(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: TransformerFeedForwardOptions | None,
) -> TransformerFeedForwardOptions:
    options = provided or config_defaults.gpt_feed_forward_options(config_module)
    return replace(
        options,
        **_modern_option_updates(
            kwargs,
            {
                "ff_num_layers": "num_layers",
                "ff_bias_flag": "bias_flag",
            },
        ),
    )


def _modern_lm_head_options(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: GptLmHeadOptions | None,
) -> GptLmHeadOptions:
    options = provided or config_defaults.gpt_lm_head_options(config_module)
    return replace(
        options,
        **_modern_option_updates(
            kwargs,
            {
                "lm_head_weight_tying_flag": "weight_tying_flag",
                "lm_head_bias_flag": "bias_flag",
            },
        ),
    )


def _modern_main_stack_options(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: MainLayerStackOptions | None,
) -> MainLayerStackOptions:
    options = provided or config_defaults.main_layer_stack_options(config_module)
    return replace(
        options,
        **_modern_option_updates(
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


def _modern_submodule_stack_options(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    stack_options: MainLayerStackOptions,
    provided: SubmoduleStackOptions | None,
) -> SubmoduleStackOptions:
    options = provided or config_defaults.submodule_stack_options(
        config_module,
        stack_options,
    )
    return _modern_submodule_stack_options_with_prefix(
        options,
        kwargs,
        flat_prefix="submodule_stack",
    )


def _modern_attention_projection_stack_options(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    decoder_options: TransformerDecoderOptions,
    attention_options: TransformerAttentionOptions,
    provided: SubmoduleStackOptions | None,
) -> SubmoduleStackOptions:
    options = provided or config_defaults.attention_projection_stack_options(
        config_module,
        decoder_options,
        attention_options,
    )
    updates = _modern_option_updates(
        kwargs,
        {f"attn_stack_{field}": field for field in _SUBMODULE_STACK_FIELD_MAP},
    )
    if "attn_num_layers" in kwargs:
        updates["num_layers"] = attention_options.num_layers
    if "attn_bias_flag" in kwargs:
        updates["bias_flag"] = attention_options.bias_flag
    return replace(options, **updates)


def _modern_feed_forward_stack_options(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    decoder_options: TransformerDecoderOptions,
    feed_forward_options: TransformerFeedForwardOptions,
    provided: SubmoduleStackOptions | None,
) -> SubmoduleStackOptions:
    options = provided or config_defaults.feed_forward_stack_options(
        config_module,
        decoder_options,
        feed_forward_options,
    )
    updates = _modern_option_updates(
        kwargs,
        {f"ff_stack_{field}": field for field in _SUBMODULE_STACK_FIELD_MAP},
    )
    if "ff_num_layers" in kwargs:
        updates["num_layers"] = feed_forward_options.num_layers
    if "ff_bias_flag" in kwargs:
        updates["bias_flag"] = feed_forward_options.bias_flag
    return replace(options, **updates)


def _modern_submodule_stack_options_with_prefix(
    options: SubmoduleStackOptions,
    kwargs: dict[str, Any],
    *,
    flat_prefix: str,
) -> SubmoduleStackOptions:
    return replace(
        options,
        **_modern_option_updates(
            kwargs,
            {f"{flat_prefix}_{field}": field for field in _SUBMODULE_STACK_FIELD_MAP},
        ),
    )


def _modern_layer_controller_options(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    flat_prefix: str,
    role: Literal["main", "attention", "feed_forward"],
    provided: LayerControllerOptions | None,
) -> LayerControllerOptions:
    flat_lead = f"{flat_prefix}_" if flat_prefix else ""
    gate_stack_flat_prefix = f"{flat_lead}gate_stack"
    halting_stack_flat_prefix = f"{flat_lead}halting_stack"
    options = provided or config_defaults.linears_layer_controller_options(
        config_module,
        role,
    )
    if flat_prefix:
        field_map = {
            f"{flat_lead}stack_gate_flag": "stack_gate_flag",
            f"{flat_lead}gate_option": "gate_option",
            f"{flat_lead}gate_activation": "gate_activation",
            f"{flat_lead}stack_halting_flag": "stack_halting_flag",
            f"{flat_lead}halting_option": "halting_option",
            f"{flat_lead}halting_threshold": "halting_threshold",
            f"{flat_lead}halting_dropout": "halting_dropout",
            f"{flat_lead}halting_hidden_state_mode": "halting_hidden_state_mode",
        }
    else:
        field_map = {
            "stack_gate_flag": "stack_gate_flag",
            "gate_option": "gate_option",
            "gate_activation": "gate_activation",
            "stack_halting_flag": "stack_halting_flag",
            "halting_option": "halting_option",
            "halting_threshold": "halting_threshold",
            "halting_dropout": "halting_dropout",
            "halting_hidden_state_mode": "halting_hidden_state_mode",
        }
    updates = _modern_option_updates(kwargs, field_map)
    updates["gate_stack_source"] = _modern_controller_stack_source_from_kwargs(
        options.gate_stack_source,
        kwargs,
        gate_stack_flat_prefix,
    )
    updates["halting_stack_source"] = _modern_controller_stack_source_from_kwargs(
        options.halting_stack_source,
        kwargs,
        halting_stack_flat_prefix,
    )
    return replace(options, **updates)


def _modern_dynamic_memory_options(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    flat_prefix: str,
    role: Literal["main", "attention", "feed_forward"],
    provided: DynamicMemoryOptions | None,
) -> DynamicMemoryOptions:
    flat_lead = f"{flat_prefix}_" if flat_prefix else ""
    options = provided or config_defaults.linears_dynamic_memory_options(
        config_module,
        role,
    )
    updates = _modern_option_updates(
        kwargs,
        {
            f"{flat_lead}memory_flag": "memory_flag",
            f"{flat_lead}memory_option": "memory_option",
            f"{flat_lead}memory_position_option": "memory_position_option",
            f"{flat_lead}memory_test_time_training_learning_rate": (
                "memory_test_time_training_learning_rate"
            ),
            f"{flat_lead}memory_test_time_training_num_inner_steps": (
                "memory_test_time_training_num_inner_steps"
            ),
        },
    )
    updates["memory_stack_source"] = _modern_controller_stack_source_from_kwargs(
        options.memory_stack_source,
        kwargs,
        f"{flat_lead}memory_stack",
    )
    return replace(options, **updates)


def _modern_recurrent_controller_options(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    flat_prefix: str,
    role: Literal["main", "attention", "feed_forward"],
    provided: RecurrentControllerOptions | None,
) -> RecurrentControllerOptions:
    flat_lead = f"{flat_prefix}_"
    options = provided or config_defaults.linears_recurrent_controller_options(
        config_module,
        role,
    )
    updates = _modern_option_updates(
        kwargs,
        {
            f"{flat_lead}flag": "recurrent_flag",
            f"{flat_lead}max_steps": "recurrent_max_steps",
            f"{flat_lead}initial_iterations": "recurrent_initial_iterations",
            f"{flat_lead}gradient_transition_count": (
                "recurrent_gradient_transition_count"
            ),
            f"{flat_lead}iteration_increment": "recurrent_iteration_increment",
            f"{flat_lead}forward_calls_before_iteration_increment": (
                "recurrent_forward_calls_before_iteration_increment"
            ),
            f"{flat_lead}smooth_iteration_growth_flag": (
                "recurrent_smooth_iteration_growth_flag"
            ),
            f"{flat_lead}layer_norm_position": "recurrent_layer_norm_position",
            f"{flat_lead}stack_gate_flag": "recurrent_stack_gate_flag",
            f"{flat_lead}gate_option": "recurrent_gate_option",
            f"{flat_lead}gate_activation": "recurrent_gate_activation",
            f"{flat_lead}stack_halting_flag": "recurrent_stack_halting_flag",
            f"{flat_lead}halting_option": "recurrent_halting_option",
            f"{flat_lead}halting_threshold": "recurrent_halting_threshold",
            f"{flat_lead}halting_dropout": "recurrent_halting_dropout",
            f"{flat_lead}halting_hidden_state_mode": (
                "recurrent_halting_hidden_state_mode"
            ),
        },
    )
    updates["recurrent_gate_stack_source"] = (
        _modern_controller_stack_source_from_kwargs(
            options.recurrent_gate_stack_source,
            kwargs,
            f"{flat_lead}gate_stack",
        )
    )
    updates["recurrent_halting_stack_source"] = (
        _modern_controller_stack_source_from_kwargs(
            options.recurrent_halting_stack_source,
            kwargs,
            f"{flat_lead}halting_stack",
        )
    )
    return replace(options, **updates)


def _modern_controller_stack_source_from_kwargs(
    source: SubmoduleStackSource,
    kwargs: dict[str, Any],
    flat_prefix: str,
) -> SubmoduleStackSource:
    return replace(
        source,
        **_modern_option_updates(
            kwargs,
            {f"{flat_prefix}_{field}": field for field in _CONTROLLER_STACK_FIELD_MAP},
        ),
    )


def _modern_option_updates(
    kwargs: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    return {
        option_field: kwargs[flat_key]
        for flat_key, option_field in mapping.items()
        if flat_key in kwargs
    }


def _modern_supported_flat_keys() -> set[str]:
    keys = {
        "embedding_layer_norm_flag",
        "embedding_dropout_probability",
        "hidden_dim",
        "stack_num_layers",
        "stack_activation",
        "stack_dropout_probability",
        "layer_norm_position",
        "stack_residual_connection_option",
        "stack_residual_model_flag",
        "stack_last_layer_bias_option",
        "stack_apply_output_pipeline_flag",
        "stack_bias_flag",
        "positional_embedding_option",
        "positional_embedding_padding_idx",
        "positional_embedding_auto_expand_flag",
        "attn_num_heads",
        "attn_num_layers",
        "attn_bias_flag",
        "attn_add_key_value_bias_flag",
        "ff_num_layers",
        "ff_bias_flag",
        "lm_head_weight_tying_flag",
        "lm_head_bias_flag",
    }
    keys.update(
        {
            "residual_stack_independent_flag",
            "residual_stack_hidden_dim",
            "residual_stack_layer_norm_position",
            "residual_stack_num_layers",
            "residual_stack_activation",
            "residual_stack_residual_connection_option",
            "residual_stack_residual_model_flag",
            "residual_stack_dropout_probability",
            "residual_stack_last_layer_bias_option",
            "residual_stack_apply_output_pipeline_flag",
            "residual_stack_bias_flag",
        }
    )
    keys.update(f"submodule_stack_{field}" for field in _SUBMODULE_STACK_FIELD_MAP)
    keys.update(f"attn_stack_{field}" for field in _SUBMODULE_STACK_FIELD_MAP)
    keys.update(f"ff_stack_{field}" for field in _SUBMODULE_STACK_FIELD_MAP)
    keys.update(_modern_role_control_flat_keys(""))
    keys.update(_modern_role_control_flat_keys("attn"))
    keys.update(_modern_role_control_flat_keys("ff"))
    keys.update(_modern_recurrent_flat_keys("recurrent"))
    keys.update(_modern_recurrent_flat_keys("attn_recurrent"))
    keys.update(_modern_recurrent_flat_keys("ff_recurrent"))
    return keys


def _modern_role_control_flat_keys(prefix: str) -> set[str]:
    lead = f"{prefix}_" if prefix else ""
    keys = {
        f"{lead}stack_gate_flag",
        f"{lead}gate_option",
        f"{lead}gate_activation",
        f"{lead}stack_halting_flag",
        f"{lead}halting_option",
        f"{lead}halting_threshold",
        f"{lead}halting_dropout",
        f"{lead}halting_hidden_state_mode",
        f"{lead}memory_flag",
        f"{lead}memory_option",
        f"{lead}memory_position_option",
        f"{lead}memory_test_time_training_learning_rate",
        f"{lead}memory_test_time_training_num_inner_steps",
    }
    if not prefix:
        keys.update({"stack_gate_flag", "stack_halting_flag"})
    for stack_name in ("gate_stack", "halting_stack", "memory_stack"):
        keys.update(
            f"{lead}{stack_name}_{field}" for field in _CONTROLLER_STACK_FIELD_MAP
        )
    return keys


def _modern_recurrent_flat_keys(prefix: str) -> set[str]:
    keys = {
        f"{prefix}_flag",
        f"{prefix}_max_steps",
        f"{prefix}_initial_iterations",
        f"{prefix}_gradient_transition_count",
        f"{prefix}_iteration_increment",
        f"{prefix}_forward_calls_before_iteration_increment",
        f"{prefix}_smooth_iteration_growth_flag",
        f"{prefix}_layer_norm_position",
        f"{prefix}_stack_gate_flag",
        f"{prefix}_gate_option",
        f"{prefix}_gate_activation",
        f"{prefix}_stack_halting_flag",
        f"{prefix}_halting_option",
        f"{prefix}_halting_threshold",
        f"{prefix}_halting_dropout",
        f"{prefix}_halting_hidden_state_mode",
    }
    for stack_name in ("gate_stack", "halting_stack"):
        keys.update(
            f"{prefix}_{stack_name}_{field}" for field in _CONTROLLER_STACK_FIELD_MAP
        )
    return keys
