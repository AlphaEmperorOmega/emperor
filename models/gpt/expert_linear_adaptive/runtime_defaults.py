# ruff: noqa: E501

from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from typing import TYPE_CHECKING, Any

from models.gpt.expert_linear_adaptive._linear_adapter_support import (
    _adaptive_generator_stack_options_from_kwargs,
    _hidden_adaptive_bias_options_from_kwargs,
    _hidden_adaptive_diagonal_options_from_kwargs,
    _hidden_adaptive_mask_options_from_kwargs,
    _hidden_adaptive_weight_options_from_kwargs,
)
from models.gpt.expert_linear_adaptive.runtime_options import (
    GptEmbeddingOptions,
    GptLmHeadOptions,
    TransformerAttentionOptions,
    TransformerDecoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)

if TYPE_CHECKING:
    from models.gpt.expert_linear_adaptive.runtime_options import (
        ExpertsDynamicMemoryOptions,
        ExpertsLayerControllerOptions,
        ExpertsRecurrentControllerOptions,
        ExpertsSubmoduleStackOptions,
        ExpertsSubmoduleStackSource,
    )


def _load_legacy_adapter_dependencies() -> None:
    global ExpertsDynamicMemoryOptions
    global ExpertsLayerControllerOptions
    global ExpertsRecurrentControllerOptions
    global ExpertsSubmoduleStackOptions
    global ExpertsSubmoduleStackSource
    global _expert_adapter_dynamic_memory_options_from_kwargs
    global _expert_adapter_expert_dynamic_memory_options_from_kwargs
    global _expert_adapter_expert_layer_controller_options_from_kwargs
    global _expert_adapter_expert_recurrent_controller_options_from_kwargs
    global _expert_adapter_layer_controller_options_from_kwargs
    global _expert_adapter_mixture_options_from_kwargs
    global _expert_adapter_recurrent_controller_options_from_kwargs
    global _expert_adapter_role_stack_options_from_kwargs
    global _expert_adapter_router_dynamic_memory_options_from_kwargs
    global _expert_adapter_router_layer_controller_options_from_kwargs
    global _expert_adapter_router_options_from_kwargs
    global _expert_adapter_router_recurrent_controller_options_from_kwargs
    global _expert_adapter_router_stack_options_from_config
    global _expert_adapter_sampler_options_from_kwargs
    from models.gpt.expert_linear_adaptive import _expert_adapter_support as adapter
    from models.gpt.expert_linear_adaptive import runtime_options as options

    ExpertsDynamicMemoryOptions = options.ExpertsDynamicMemoryOptions
    ExpertsLayerControllerOptions = options.ExpertsLayerControllerOptions
    ExpertsRecurrentControllerOptions = options.ExpertsRecurrentControllerOptions
    ExpertsSubmoduleStackOptions = options.ExpertsSubmoduleStackOptions
    ExpertsSubmoduleStackSource = options.ExpertsSubmoduleStackSource
    _expert_adapter_dynamic_memory_options_from_kwargs = (
        adapter._dynamic_memory_options_from_kwargs
    )
    _expert_adapter_expert_dynamic_memory_options_from_kwargs = (
        adapter._expert_dynamic_memory_options_from_kwargs
    )
    _expert_adapter_expert_layer_controller_options_from_kwargs = (
        adapter._expert_layer_controller_options_from_kwargs
    )
    _expert_adapter_expert_recurrent_controller_options_from_kwargs = (
        adapter._expert_recurrent_controller_options_from_kwargs
    )
    _expert_adapter_layer_controller_options_from_kwargs = (
        adapter._layer_controller_options_from_kwargs
    )
    _expert_adapter_mixture_options_from_kwargs = adapter._mixture_options_from_kwargs
    _expert_adapter_recurrent_controller_options_from_kwargs = (
        adapter._recurrent_controller_options_from_kwargs
    )
    _expert_adapter_role_stack_options_from_kwargs = (
        adapter._role_stack_options_from_kwargs
    )
    _expert_adapter_router_dynamic_memory_options_from_kwargs = (
        adapter._router_dynamic_memory_options_from_kwargs
    )
    _expert_adapter_router_layer_controller_options_from_kwargs = (
        adapter._router_layer_controller_options_from_kwargs
    )
    _expert_adapter_router_options_from_kwargs = adapter._router_options_from_kwargs
    _expert_adapter_router_recurrent_controller_options_from_kwargs = (
        adapter._router_recurrent_controller_options_from_kwargs
    )
    _expert_adapter_router_stack_options_from_config = (
        adapter._router_stack_options_from_config
    )
    _expert_adapter_sampler_options_from_kwargs = adapter._sampler_options_from_kwargs


_TOP_LEVEL_KEYS = (
    "batch_size",
    "learning_rate",
    "input_dim",
    "output_dim",
    "sequence_length",
)
_GPT_LINEAR_ALLOWED_KWARGS = {
    *_TOP_LEVEL_KEYS,
    "decoder_options",
    "embedding_options",
    "lm_head_options",
    "positional_embedding_options",
    "attention_options",
    "feed_forward_options",
    "attention_projection_stack_options",
    "attention_projection_layer_controller_options",
    "attention_projection_dynamic_memory_options",
    "attention_projection_recurrent_controller_options",
    "feed_forward_stack_options",
    "feed_forward_layer_controller_options",
    "feed_forward_dynamic_memory_options",
    "feed_forward_recurrent_controller_options",
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
    field: field for field in _CONTROLLER_STACK_FIELD_MAP if field != "independent_flag"
}


def _legacy_linear_builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any], config_module: ModuleType
) -> dict[str, Any]:
    _load_legacy_adapter_dependencies()
    kwargs = dict(flat_kwargs)
    builder_kwargs = _pop_top_level_kwargs(kwargs)
    builder_kwargs["embedding_options"] = _embedding_options_from_kwargs(
        kwargs,
        config_module,
        provided=kwargs.pop("embedding_options", None),
    )
    builder_kwargs["lm_head_options"] = _lm_head_options_from_kwargs(
        kwargs,
        config_module,
        provided=kwargs.pop("lm_head_options", None),
    )
    decoder_options = _decoder_options_from_kwargs(
        kwargs, config_module, provided=kwargs.pop("decoder_options", None)
    )
    positional_embedding_options = _positional_embedding_options_from_kwargs(
        kwargs, config_module, provided=kwargs.pop("positional_embedding_options", None)
    )
    attention_options = _attention_options_from_kwargs(
        kwargs, config_module, provided=kwargs.pop("attention_options", None)
    )
    feed_forward_options = _feed_forward_options_from_kwargs(
        kwargs, config_module, provided=kwargs.pop("feed_forward_options", None)
    )
    builder_kwargs.update(
        {
            "decoder_options": decoder_options,
            "positional_embedding_options": positional_embedding_options,
            "attention_options": attention_options,
            "feed_forward_options": feed_forward_options,
            "attention_projection_stack_options": _attention_projection_stack_options_from_kwargs(
                kwargs,
                config_module,
                decoder_options=decoder_options,
                attention_options=attention_options,
                provided=kwargs.pop("attention_projection_stack_options", None),
            ),
            "attention_projection_layer_controller_options": _attention_projection_layer_controller_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop(
                    "attention_projection_layer_controller_options", None
                ),
            ),
            "attention_projection_dynamic_memory_options": _attention_projection_dynamic_memory_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop(
                    "attention_projection_dynamic_memory_options", None
                ),
            ),
            "attention_projection_recurrent_controller_options": _attention_projection_recurrent_controller_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop(
                    "attention_projection_recurrent_controller_options", None
                ),
            ),
            "feed_forward_stack_options": _feed_forward_stack_options_from_kwargs(
                kwargs,
                config_module,
                decoder_options=decoder_options,
                feed_forward_options=feed_forward_options,
                provided=kwargs.pop("feed_forward_stack_options", None),
            ),
            "feed_forward_layer_controller_options": _feed_forward_layer_controller_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("feed_forward_layer_controller_options", None),
            ),
            "feed_forward_dynamic_memory_options": _feed_forward_dynamic_memory_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("feed_forward_dynamic_memory_options", None),
            ),
            "feed_forward_recurrent_controller_options": _feed_forward_recurrent_controller_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("feed_forward_recurrent_controller_options", None),
            ),
            "submodule_stack_options": _submodule_stack_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("submodule_stack_options", None),
            ),
            "layer_controller_options": _layer_controller_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("layer_controller_options", None),
            ),
            "dynamic_memory_options": _dynamic_memory_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("dynamic_memory_options", None),
            ),
            "recurrent_controller_options": _recurrent_controller_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("recurrent_controller_options", None),
            ),
        }
    )
    builder_kwargs.update(kwargs)
    return builder_kwargs


def expert_linear_builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any], config_module: ModuleType
) -> dict[str, Any]:
    kwargs = dict(flat_kwargs)
    builder_kwargs = _linear_builder_kwargs_only(flat_kwargs, config_module)
    submodule_stack_options = builder_kwargs["submodule_stack_options"]
    builder_kwargs.update(
        {
            "mixture_options": _expert_adapter_mixture_options_from_kwargs(
                kwargs, config_module, provided=kwargs.pop("mixture_options", None)
            ),
            "expert_stack_options": _expert_adapter_role_stack_options_from_kwargs(
                kwargs,
                config_module,
                "expert_stack",
                defaults=submodule_stack_options,
                provided=kwargs.pop("expert_stack_options", None),
                extra_mapping={"expert_bias_flag": "bias_flag"},
                default_overrides={
                    "layer_norm_position": config_module.EXPERT_STACK_LAYER_NORM_POSITION,
                    "apply_output_pipeline_flag": config_module.EXPERT_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
                },
            ),
            "sampler_options": _expert_adapter_sampler_options_from_kwargs(
                kwargs, config_module, provided=kwargs.pop("sampler_options", None)
            ),
            "router_options": _expert_adapter_router_options_from_kwargs(
                kwargs, config_module, provided=kwargs.pop("router_options", None)
            ),
            "router_stack_options": _expert_adapter_role_stack_options_from_kwargs(
                kwargs,
                config_module,
                "router_stack",
                defaults=_expert_adapter_router_stack_options_from_config(
                    config_module
                ),
                provided=kwargs.pop("router_stack_options", None),
                extra_mapping={"router_bias_flag": "bias_flag"},
            ),
            "expert_layer_controller_options": _expert_adapter_expert_layer_controller_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("expert_layer_controller_options", None),
            ),
            "expert_dynamic_memory_options": _expert_adapter_expert_dynamic_memory_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("expert_dynamic_memory_options", None),
            ),
            "expert_recurrent_controller_options": _expert_adapter_expert_recurrent_controller_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("expert_recurrent_controller_options", None),
            ),
        }
    )
    _copy_direct(
        builder_kwargs,
        flat_kwargs,
        {"expert_attention_flag", "expert_attention_use_kv_expert_models_flag"},
    )
    return builder_kwargs


def expert_linear_adaptive_builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any], config_module: ModuleType
) -> dict[str, Any]:
    kwargs = dict(flat_kwargs)
    builder_kwargs = expert_linear_builder_kwargs_from_flat(flat_kwargs, config_module)
    _auto_enable_adaptive_option_flags(kwargs)
    _auto_enable_adaptive_option_flags(kwargs, "router_")
    builder_kwargs.update(
        {
            "mixture_submodule_stack_options": _expert_adapter_role_stack_options_from_kwargs(
                kwargs,
                config_module,
                "submodule_stack",
                defaults=builder_kwargs["submodule_stack_options"],
                provided=kwargs.pop("mixture_submodule_stack_options", None),
            ),
            "mixture_layer_controller_options": _expert_adapter_layer_controller_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("mixture_layer_controller_options", None),
            ),
            "mixture_dynamic_memory_options": _expert_adapter_dynamic_memory_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("mixture_dynamic_memory_options", None),
            ),
            "mixture_recurrent_controller_options": _expert_adapter_recurrent_controller_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("mixture_recurrent_controller_options", None),
            ),
            "router_layer_controller_options": _expert_adapter_router_layer_controller_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("router_layer_controller_options", None),
            ),
            "router_dynamic_memory_options": _expert_adapter_router_dynamic_memory_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("router_dynamic_memory_options", None),
            ),
            "router_recurrent_controller_options": _expert_adapter_router_recurrent_controller_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("router_recurrent_controller_options", None),
            ),
            "adaptive_generator_stack_options": _adaptive_generator_stack_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("adaptive_generator_stack_options", None),
            ),
            "hidden_adaptive_weight_options": _hidden_adaptive_weight_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("hidden_adaptive_weight_options", None),
            ),
            "hidden_adaptive_bias_options": _hidden_adaptive_bias_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("hidden_adaptive_bias_options", None),
            ),
            "hidden_adaptive_diagonal_options": _hidden_adaptive_diagonal_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("hidden_adaptive_diagonal_options", None),
            ),
            "hidden_adaptive_mask_options": _hidden_adaptive_mask_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("hidden_adaptive_mask_options", None),
            ),
            "router_adaptive_weight_options": _hidden_adaptive_weight_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("router_adaptive_weight_options", None),
                flat_prefix="router_",
                config_prefix="router_",
                stack_prefix="router_weight_generator_stack",
            ),
            "router_adaptive_bias_options": _hidden_adaptive_bias_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("router_adaptive_bias_options", None),
                flat_prefix="router_",
                config_prefix="router_",
                stack_prefix="router_bias_generator_stack",
            ),
            "router_adaptive_diagonal_options": _hidden_adaptive_diagonal_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("router_adaptive_diagonal_options", None),
                flat_prefix="router_",
                config_prefix="router_",
                stack_prefix="router_diagonal_generator_stack",
            ),
            "router_adaptive_mask_options": _hidden_adaptive_mask_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("router_adaptive_mask_options", None),
                flat_prefix="router_",
                config_prefix="router_",
                stack_prefix="router_mask_generator_stack",
            ),
        }
    )
    return builder_kwargs


def _linear_builder_kwargs_only(
    flat_kwargs: dict[str, Any], config_module: ModuleType
) -> dict[str, Any]:
    return {
        key: value
        for key, value in _legacy_linear_builder_kwargs_from_flat(
            flat_kwargs, config_module
        ).items()
        if key in _GPT_LINEAR_ALLOWED_KWARGS
    }


def _auto_enable_adaptive_option_flags(
    kwargs: dict[str, Any], prefix: str = ""
) -> None:
    for option_key, flag_key in (
        (f"{prefix}weight_option", f"{prefix}weight_option_flag"),
        (f"{prefix}bias_option", f"{prefix}bias_option_flag"),
        (f"{prefix}diagonal_option", f"{prefix}diagonal_option_flag"),
        (f"{prefix}row_mask_option", f"{prefix}mask_option_flag"),
    ):
        if flag_key in kwargs:
            continue
        if option_key in kwargs and kwargs[option_key] is not None:
            kwargs[flag_key] = True


def _copy_direct(
    builder_kwargs: dict[str, Any], flat_kwargs: dict[str, Any], keys: set[str]
) -> None:
    for key in keys:
        if key in flat_kwargs:
            builder_kwargs[key] = flat_kwargs[key]


def _pop_top_level_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: kwargs.pop(key) for key in _TOP_LEVEL_KEYS if key in kwargs}


def _decoder_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: TransformerDecoderOptions | None,
) -> TransformerDecoderOptions:
    options = provided or TransformerDecoderOptions(
        hidden_dim=config_module.HIDDEN_DIM,
        num_layers=config_module.STACK_NUM_LAYERS,
        activation=config_module.STACK_ACTIVATION,
        dropout_probability=config_module.STACK_DROPOUT_PROBABILITY,
        layer_norm_position=config_module.LAYER_NORM_POSITION,
    )
    updates = _pop_updates(
        kwargs,
        {
            "hidden_dim": "hidden_dim",
            "stack_num_layers": "num_layers",
            "stack_activation": "activation",
            "stack_dropout_probability": "dropout_probability",
            "layer_norm_position": "layer_norm_position",
            "stack_layer_norm_position": "layer_norm_position",
        },
    )
    return replace(options, **updates) if updates else options


def _embedding_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: GptEmbeddingOptions | None,
) -> GptEmbeddingOptions:
    options = provided or GptEmbeddingOptions(
        layer_norm_flag=config_module.EMBEDDING_LAYER_NORM_FLAG,
        dropout_probability=config_module.EMBEDDING_DROPOUT_PROBABILITY,
    )
    updates = _pop_updates(
        kwargs,
        {
            "embedding_layer_norm_flag": "layer_norm_flag",
            "embedding_dropout_probability": "dropout_probability",
        },
    )
    return replace(options, **updates) if updates else options


def _lm_head_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: GptLmHeadOptions | None,
) -> GptLmHeadOptions:
    options = provided or GptLmHeadOptions(
        weight_tying_flag=config_module.LM_HEAD_WEIGHT_TYING_FLAG,
        bias_flag=config_module.LM_HEAD_BIAS_FLAG,
    )
    updates = _pop_updates(
        kwargs,
        {
            "lm_head_weight_tying_flag": "weight_tying_flag",
            "lm_head_bias_flag": "bias_flag",
        },
    )
    return replace(options, **updates) if updates else options


def _positional_embedding_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: TransformerPositionalEmbeddingOptions | None,
) -> TransformerPositionalEmbeddingOptions:
    options = provided or TransformerPositionalEmbeddingOptions(
        option=config_module.POSITIONAL_EMBEDDING_OPTION,
        padding_idx=config_module.POSITIONAL_EMBEDDING_PADDING_IDX,
        auto_expand_flag=config_module.POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG,
    )
    updates = _pop_updates(
        kwargs,
        {
            "positional_embedding_option": "option",
            "positional_embedding_padding_idx": "padding_idx",
            "positional_embedding_auto_expand_flag": "auto_expand_flag",
        },
    )
    return replace(options, **updates) if updates else options


def _attention_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: TransformerAttentionOptions | None,
) -> TransformerAttentionOptions:
    options = provided or TransformerAttentionOptions(
        num_heads=config_module.ATTN_NUM_HEADS,
        num_layers=config_module.ATTN_NUM_LAYERS,
        bias_flag=config_module.ATTN_BIAS_FLAG,
        add_key_value_bias_flag=config_module.ATTN_ADD_KEY_VALUE_BIAS_FLAG,
    )
    updates = _pop_updates(
        kwargs,
        {
            "attn_num_heads": "num_heads",
            "attn_num_layers": "num_layers",
            "attn_bias_flag": "bias_flag",
            "attn_add_key_value_bias_flag": "add_key_value_bias_flag",
        },
    )
    return replace(options, **updates) if updates else options


def _feed_forward_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: TransformerFeedForwardOptions | None,
) -> TransformerFeedForwardOptions:
    options = provided or TransformerFeedForwardOptions(
        num_layers=config_module.FF_NUM_LAYERS, bias_flag=config_module.FF_BIAS_FLAG
    )
    updates = _pop_updates(
        kwargs, {"ff_num_layers": "num_layers", "ff_bias_flag": "bias_flag"}
    )
    return replace(options, **updates) if updates else options


def _submodule_stack_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: ExpertsSubmoduleStackOptions | None,
) -> ExpertsSubmoduleStackOptions:
    options = provided or ExpertsSubmoduleStackOptions(
        hidden_dim=config_module.SUBMODULE_STACK_HIDDEN_DIM,
        num_layers=config_module.SUBMODULE_STACK_NUM_LAYERS,
        last_layer_bias_option=config_module.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config_module.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config_module.SUBMODULE_STACK_ACTIVATION,
        layer_norm_position=config_module.SUBMODULE_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config_module.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION,
        dropout_probability=config_module.SUBMODULE_STACK_DROPOUT_PROBABILITY,
        bias_flag=config_module.SUBMODULE_STACK_BIAS_FLAG,
    )
    updates = _pop_updates(
        kwargs,
        {
            f"submodule_stack_{flat_field}": dataclass_field
            for flat_field, dataclass_field in _SUBMODULE_STACK_FIELD_MAP.items()
        },
    )
    return replace(options, **updates) if updates else options


def _attention_projection_stack_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    decoder_options: TransformerDecoderOptions,
    attention_options: TransformerAttentionOptions,
    provided: ExpertsSubmoduleStackOptions | None,
) -> ExpertsSubmoduleStackOptions:
    options = provided or _default_attention_projection_stack_options(
        config_module,
        attention_options,
        hidden_dim=decoder_options.hidden_dim,
        activation=decoder_options.activation,
    )
    options = replace(
        options,
        num_layers=attention_options.num_layers,
        bias_flag=attention_options.bias_flag,
    )
    updates = _pop_updates(
        kwargs,
        {
            "attn_stack_hidden_dim": "hidden_dim",
            "attn_stack_last_layer_bias_option": "last_layer_bias_option",
            "attn_stack_apply_output_pipeline_flag": "apply_output_pipeline_flag",
            "attn_stack_activation": "activation",
            "attn_stack_layer_norm_position": "layer_norm_position",
            "attn_stack_residual_connection_option": "residual_connection_option",
            "attn_stack_dropout_probability": "dropout_probability",
        },
    )
    return replace(options, **updates) if updates else options


def _feed_forward_stack_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    decoder_options: TransformerDecoderOptions,
    feed_forward_options: TransformerFeedForwardOptions,
    provided: ExpertsSubmoduleStackOptions | None,
) -> ExpertsSubmoduleStackOptions:
    options = provided or _default_feed_forward_stack_options(
        config_module, feed_forward_options, hidden_dim=decoder_options.hidden_dim
    )
    options = replace(
        options,
        num_layers=feed_forward_options.num_layers,
        bias_flag=feed_forward_options.bias_flag,
    )
    updates = _pop_updates(
        kwargs,
        {
            "ff_stack_hidden_dim": "hidden_dim",
            "ff_stack_last_layer_bias_option": "last_layer_bias_option",
            "ff_stack_apply_output_pipeline_flag": "apply_output_pipeline_flag",
            "ff_stack_activation": "activation",
            "ff_stack_layer_norm_position": "layer_norm_position",
            "ff_stack_residual_connection_option": "residual_connection_option",
            "ff_stack_dropout_probability": "dropout_probability",
        },
    )
    return replace(options, **updates) if updates else options


def _layer_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: ExpertsLayerControllerOptions | None,
) -> ExpertsLayerControllerOptions:
    options = provided or ExpertsLayerControllerOptions(
        stack_gate_flag=config_module.GATE_FLAG,
        gate_option=config_module.GATE_OPTION,
        gate_activation=config_module.GATE_ACTIVATION,
        gate_stack_source=_default_controller_stack_source(config_module, "gate_stack"),
        stack_halting_flag=config_module.HALTING_FLAG,
        halting_threshold=config_module.HALTING_THRESHOLD,
        halting_dropout=config_module.HALTING_DROPOUT,
        halting_hidden_state_mode=config_module.HALTING_HIDDEN_STATE_MODE,
        halting_stack_source=_default_controller_stack_source(
            config_module, "halting_stack"
        ),
        halting_output_dim=config_module.HALTING_OUTPUT_DIM,
    )
    updates = _pop_updates(
        kwargs,
        {
            "stack_gate_flag": "stack_gate_flag",
            "gate_option": "gate_option",
            "gate_activation": "gate_activation",
            "stack_halting_flag": "stack_halting_flag",
            "halting_threshold": "halting_threshold",
            "halting_dropout": "halting_dropout",
            "halting_hidden_state_mode": "halting_hidden_state_mode",
            "halting_output_dim": "halting_output_dim",
            "shared_gate_config": "shared_gate_config",
        },
    )
    updates["gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs, config_module, "gate_stack", provided=options.gate_stack_source
    )
    updates["halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs, config_module, "halting_stack", provided=options.halting_stack_source
    )
    return replace(options, **updates)


def _attention_projection_layer_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: ExpertsLayerControllerOptions | None,
) -> ExpertsLayerControllerOptions:
    options = provided or _default_attention_projection_layer_controller_options(
        config_module
    )
    updates = _pop_updates(
        kwargs,
        {
            "attn_gate_flag": "stack_gate_flag",
            "attn_gate_option": "gate_option",
            "attn_gate_activation": "gate_activation",
            "attn_halting_flag": "stack_halting_flag",
            "attn_halting_threshold": "halting_threshold",
            "attn_halting_dropout": "halting_dropout",
            "attn_halting_hidden_state_mode": "halting_hidden_state_mode",
            "attn_shared_gate_config": "shared_gate_config",
        },
    )
    updates["gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs, config_module, "attn_gate_stack", provided=options.gate_stack_source
    )
    updates["halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "attn_halting_stack",
        provided=options.halting_stack_source,
    )
    return replace(options, **updates)


def _feed_forward_layer_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: ExpertsLayerControllerOptions | None,
) -> ExpertsLayerControllerOptions:
    options = provided or _default_feed_forward_layer_controller_options(config_module)
    updates = _pop_updates(
        kwargs,
        {
            "ff_gate_flag": "stack_gate_flag",
            "ff_gate_option": "gate_option",
            "ff_gate_activation": "gate_activation",
            "ff_halting_flag": "stack_halting_flag",
            "ff_halting_threshold": "halting_threshold",
            "ff_halting_dropout": "halting_dropout",
            "ff_halting_hidden_state_mode": "halting_hidden_state_mode",
            "ff_shared_gate_config": "shared_gate_config",
        },
    )
    updates["gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs, config_module, "ff_gate_stack", provided=options.gate_stack_source
    )
    updates["halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs, config_module, "ff_halting_stack", provided=options.halting_stack_source
    )
    return replace(options, **updates)


def _dynamic_memory_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: ExpertsDynamicMemoryOptions | None,
) -> ExpertsDynamicMemoryOptions:
    options = provided or ExpertsDynamicMemoryOptions(
        memory_flag=config_module.MEMORY_FLAG,
        memory_option=config_module.MEMORY_OPTION,
        memory_position_option=config_module.MEMORY_POSITION_OPTION,
        memory_test_time_training_learning_rate=config_module.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
        memory_test_time_training_num_inner_steps=config_module.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
        memory_stack_source=_default_controller_stack_source(
            config_module, "memory_stack"
        ),
    )
    updates = _pop_updates(
        kwargs,
        {
            "memory_flag": "memory_flag",
            "memory_option": "memory_option",
            "memory_position_option": "memory_position_option",
            "memory_test_time_training_learning_rate": "memory_test_time_training_learning_rate",
            "memory_test_time_training_num_inner_steps": "memory_test_time_training_num_inner_steps",
        },
    )
    updates["memory_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs, config_module, "memory_stack", provided=options.memory_stack_source
    )
    return replace(options, **updates)


def _attention_projection_dynamic_memory_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: ExpertsDynamicMemoryOptions | None,
) -> ExpertsDynamicMemoryOptions:
    options = provided or _default_attention_projection_dynamic_memory_options(
        config_module
    )
    updates = _pop_updates(
        kwargs,
        {
            "attn_memory_flag": "memory_flag",
            "attn_memory_option": "memory_option",
            "attn_memory_position_option": "memory_position_option",
            "attn_memory_test_time_training_learning_rate": "memory_test_time_training_learning_rate",
            "attn_memory_test_time_training_num_inner_steps": "memory_test_time_training_num_inner_steps",
        },
    )
    updates["memory_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs, config_module, "attn_memory_stack", provided=options.memory_stack_source
    )
    return replace(options, **updates)


def _feed_forward_dynamic_memory_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: ExpertsDynamicMemoryOptions | None,
) -> ExpertsDynamicMemoryOptions:
    options = provided or _default_feed_forward_dynamic_memory_options(config_module)
    updates = _pop_updates(
        kwargs,
        {
            "ff_memory_flag": "memory_flag",
            "ff_memory_option": "memory_option",
            "ff_memory_position_option": "memory_position_option",
            "ff_memory_test_time_training_learning_rate": "memory_test_time_training_learning_rate",
            "ff_memory_test_time_training_num_inner_steps": "memory_test_time_training_num_inner_steps",
        },
    )
    updates["memory_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs, config_module, "ff_memory_stack", provided=options.memory_stack_source
    )
    return replace(options, **updates)


def _recurrent_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: ExpertsRecurrentControllerOptions | None,
) -> ExpertsRecurrentControllerOptions:
    options = provided or ExpertsRecurrentControllerOptions(
        recurrent_flag=config_module.RECURRENT_FLAG,
        recurrent_max_steps=config_module.RECURRENT_MAX_STEPS,
        recurrent_layer_norm_position=config_module.RECURRENT_LAYER_NORM_POSITION,
        recurrent_gate_flag=config_module.RECURRENT_GATE_FLAG,
        recurrent_gate_option=config_module.RECURRENT_GATE_OPTION,
        recurrent_gate_activation=config_module.RECURRENT_GATE_ACTIVATION,
        recurrent_gate_stack_source=_default_controller_stack_source(
            config_module, "recurrent_gate_stack"
        ),
        recurrent_halting_flag=config_module.RECURRENT_HALTING_FLAG,
        recurrent_halting_threshold=config_module.RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout=config_module.RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode=config_module.RECURRENT_HALTING_HIDDEN_STATE_MODE,
        recurrent_halting_stack_source=_default_controller_stack_source(
            config_module, "recurrent_halting_stack"
        ),
    )
    updates = _pop_updates(
        kwargs,
        {
            "recurrent_flag": "recurrent_flag",
            "recurrent_max_steps": "recurrent_max_steps",
            "recurrent_layer_norm_position": "recurrent_layer_norm_position",
            "recurrent_gate_flag": "recurrent_gate_flag",
            "recurrent_gate_option": "recurrent_gate_option",
            "recurrent_gate_activation": "recurrent_gate_activation",
            "recurrent_halting_flag": "recurrent_halting_flag",
            "recurrent_halting_threshold": "recurrent_halting_threshold",
            "recurrent_halting_dropout": "recurrent_halting_dropout",
            "recurrent_halting_hidden_state_mode": "recurrent_halting_hidden_state_mode",
        },
    )
    updates["recurrent_gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "recurrent_gate_stack",
        provided=options.recurrent_gate_stack_source,
    )
    updates["recurrent_halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "recurrent_halting_stack",
        provided=options.recurrent_halting_stack_source,
    )
    return replace(options, **updates)


def _attention_projection_recurrent_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: ExpertsRecurrentControllerOptions | None,
) -> ExpertsRecurrentControllerOptions:
    options = provided or _default_attention_projection_recurrent_controller_options(
        config_module
    )
    updates = _pop_updates(
        kwargs,
        {
            "attn_recurrent_flag": "recurrent_flag",
            "attn_recurrent_max_steps": "recurrent_max_steps",
            "attn_recurrent_layer_norm_position": "recurrent_layer_norm_position",
            "attn_recurrent_gate_flag": "recurrent_gate_flag",
            "attn_recurrent_gate_option": "recurrent_gate_option",
            "attn_recurrent_gate_activation": "recurrent_gate_activation",
            "attn_recurrent_halting_flag": "recurrent_halting_flag",
            "attn_recurrent_halting_threshold": "recurrent_halting_threshold",
            "attn_recurrent_halting_dropout": "recurrent_halting_dropout",
            "attn_recurrent_halting_hidden_state_mode": "recurrent_halting_hidden_state_mode",
        },
    )
    updates["recurrent_gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "attn_recurrent_gate_stack",
        provided=options.recurrent_gate_stack_source,
    )
    updates["recurrent_halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "attn_recurrent_halting_stack",
        provided=options.recurrent_halting_stack_source,
    )
    return replace(options, **updates)


def _feed_forward_recurrent_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: ExpertsRecurrentControllerOptions | None,
) -> ExpertsRecurrentControllerOptions:
    options = provided or _default_feed_forward_recurrent_controller_options(
        config_module
    )
    updates = _pop_updates(
        kwargs,
        {
            "ff_recurrent_flag": "recurrent_flag",
            "ff_recurrent_max_steps": "recurrent_max_steps",
            "ff_recurrent_layer_norm_position": "recurrent_layer_norm_position",
            "ff_recurrent_gate_flag": "recurrent_gate_flag",
            "ff_recurrent_gate_option": "recurrent_gate_option",
            "ff_recurrent_gate_activation": "recurrent_gate_activation",
            "ff_recurrent_halting_flag": "recurrent_halting_flag",
            "ff_recurrent_halting_threshold": "recurrent_halting_threshold",
            "ff_recurrent_halting_dropout": "recurrent_halting_dropout",
            "ff_recurrent_halting_hidden_state_mode": "recurrent_halting_hidden_state_mode",
        },
    )
    updates["recurrent_gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "ff_recurrent_gate_stack",
        provided=options.recurrent_gate_stack_source,
    )
    updates["recurrent_halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "ff_recurrent_halting_stack",
        provided=options.recurrent_halting_stack_source,
    )
    return replace(options, **updates)


def _controller_stack_source_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    prefix: str,
    *,
    provided: ExpertsSubmoduleStackSource | None,
) -> ExpertsSubmoduleStackSource:
    source = provided or _default_controller_stack_source(config_module, prefix)
    updates = _pop_updates(
        kwargs,
        {
            f"{prefix}_{flat_field}": dataclass_field
            for flat_field, dataclass_field in _CONTROLLER_STACK_FIELD_MAP.items()
        },
    )
    return replace(source, **updates) if updates else source


def _default_attention_projection_stack_options(
    config_module: ModuleType,
    attention_options: TransformerAttentionOptions,
    *,
    hidden_dim: int | None = None,
    activation: Any | None = None,
) -> ExpertsSubmoduleStackOptions:
    return ExpertsSubmoduleStackOptions(
        hidden_dim=config_module.ATTN_STACK_HIDDEN_DIM
        if hidden_dim is None
        else hidden_dim,
        num_layers=attention_options.num_layers,
        last_layer_bias_option=config_module.ATTN_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config_module.ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config_module.ATTN_STACK_ACTIVATION
        if activation is None
        else activation,
        layer_norm_position=config_module.ATTN_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config_module.ATTN_STACK_RESIDUAL_CONNECTION_OPTION,
        dropout_probability=config_module.ATTN_STACK_DROPOUT_PROBABILITY,
        bias_flag=attention_options.bias_flag,
    )


def _default_feed_forward_stack_options(
    config_module: ModuleType,
    feed_forward_options: TransformerFeedForwardOptions,
    *,
    hidden_dim: int | None = None,
) -> ExpertsSubmoduleStackOptions:
    return ExpertsSubmoduleStackOptions(
        hidden_dim=config_module.FF_STACK_HIDDEN_DIM
        if hidden_dim is None
        else hidden_dim,
        num_layers=feed_forward_options.num_layers,
        last_layer_bias_option=config_module.FF_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config_module.FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config_module.FF_STACK_ACTIVATION,
        layer_norm_position=config_module.FF_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config_module.FF_STACK_RESIDUAL_CONNECTION_OPTION,
        dropout_probability=config_module.FF_STACK_DROPOUT_PROBABILITY,
        bias_flag=feed_forward_options.bias_flag,
    )


def _default_attention_projection_layer_controller_options(
    config_module: ModuleType,
) -> ExpertsLayerControllerOptions:
    return ExpertsLayerControllerOptions(
        stack_gate_flag=config_module.ATTN_GATE_FLAG,
        gate_option=config_module.ATTN_GATE_OPTION,
        gate_activation=config_module.ATTN_GATE_ACTIVATION,
        gate_stack_source=_default_controller_stack_source(
            config_module, "attn_gate_stack"
        ),
        stack_halting_flag=config_module.ATTN_HALTING_FLAG,
        halting_threshold=config_module.ATTN_HALTING_THRESHOLD,
        halting_dropout=config_module.ATTN_HALTING_DROPOUT,
        halting_hidden_state_mode=config_module.ATTN_HALTING_HIDDEN_STATE_MODE,
        halting_stack_source=_default_controller_stack_source(
            config_module, "attn_halting_stack"
        ),
        halting_output_dim=config_module.HALTING_OUTPUT_DIM,
    )


def _default_feed_forward_layer_controller_options(
    config_module: ModuleType,
) -> ExpertsLayerControllerOptions:
    return ExpertsLayerControllerOptions(
        stack_gate_flag=config_module.FF_GATE_FLAG,
        gate_option=config_module.FF_GATE_OPTION,
        gate_activation=config_module.FF_GATE_ACTIVATION,
        gate_stack_source=_default_controller_stack_source(
            config_module, "ff_gate_stack"
        ),
        stack_halting_flag=config_module.FF_HALTING_FLAG,
        halting_threshold=config_module.FF_HALTING_THRESHOLD,
        halting_dropout=config_module.FF_HALTING_DROPOUT,
        halting_hidden_state_mode=config_module.FF_HALTING_HIDDEN_STATE_MODE,
        halting_stack_source=_default_controller_stack_source(
            config_module, "ff_halting_stack"
        ),
        halting_output_dim=config_module.HALTING_OUTPUT_DIM,
    )


def _default_attention_projection_dynamic_memory_options(
    config_module: ModuleType,
) -> ExpertsDynamicMemoryOptions:
    return ExpertsDynamicMemoryOptions(
        memory_flag=config_module.ATTN_MEMORY_FLAG,
        memory_option=config_module.ATTN_MEMORY_OPTION,
        memory_position_option=config_module.ATTN_MEMORY_POSITION_OPTION,
        memory_test_time_training_learning_rate=config_module.ATTN_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
        memory_test_time_training_num_inner_steps=config_module.ATTN_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
        memory_stack_source=_default_controller_stack_source(
            config_module, "attn_memory_stack"
        ),
    )


def _default_feed_forward_dynamic_memory_options(
    config_module: ModuleType,
) -> ExpertsDynamicMemoryOptions:
    return ExpertsDynamicMemoryOptions(
        memory_flag=config_module.FF_MEMORY_FLAG,
        memory_option=config_module.FF_MEMORY_OPTION,
        memory_position_option=config_module.FF_MEMORY_POSITION_OPTION,
        memory_test_time_training_learning_rate=config_module.FF_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
        memory_test_time_training_num_inner_steps=config_module.FF_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
        memory_stack_source=_default_controller_stack_source(
            config_module, "ff_memory_stack"
        ),
    )


def _default_attention_projection_recurrent_controller_options(
    config_module: ModuleType,
) -> ExpertsRecurrentControllerOptions:
    return ExpertsRecurrentControllerOptions(
        recurrent_flag=config_module.ATTN_RECURRENT_FLAG,
        recurrent_max_steps=config_module.ATTN_RECURRENT_MAX_STEPS,
        recurrent_layer_norm_position=config_module.ATTN_RECURRENT_LAYER_NORM_POSITION,
        recurrent_gate_flag=config_module.ATTN_RECURRENT_GATE_FLAG,
        recurrent_gate_option=config_module.ATTN_RECURRENT_GATE_OPTION,
        recurrent_gate_activation=config_module.ATTN_RECURRENT_GATE_ACTIVATION,
        recurrent_gate_stack_source=_default_controller_stack_source(
            config_module, "attn_recurrent_gate_stack"
        ),
        recurrent_halting_flag=config_module.ATTN_RECURRENT_HALTING_FLAG,
        recurrent_halting_threshold=config_module.ATTN_RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout=config_module.ATTN_RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode=config_module.ATTN_RECURRENT_HALTING_HIDDEN_STATE_MODE,
        recurrent_halting_stack_source=_default_controller_stack_source(
            config_module, "attn_recurrent_halting_stack"
        ),
    )


def _default_feed_forward_recurrent_controller_options(
    config_module: ModuleType,
) -> ExpertsRecurrentControllerOptions:
    return ExpertsRecurrentControllerOptions(
        recurrent_flag=config_module.FF_RECURRENT_FLAG,
        recurrent_max_steps=config_module.FF_RECURRENT_MAX_STEPS,
        recurrent_layer_norm_position=config_module.FF_RECURRENT_LAYER_NORM_POSITION,
        recurrent_gate_flag=config_module.FF_RECURRENT_GATE_FLAG,
        recurrent_gate_option=config_module.FF_RECURRENT_GATE_OPTION,
        recurrent_gate_activation=config_module.FF_RECURRENT_GATE_ACTIVATION,
        recurrent_gate_stack_source=_default_controller_stack_source(
            config_module, "ff_recurrent_gate_stack"
        ),
        recurrent_halting_flag=config_module.FF_RECURRENT_HALTING_FLAG,
        recurrent_halting_threshold=config_module.FF_RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout=config_module.FF_RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode=config_module.FF_RECURRENT_HALTING_HIDDEN_STATE_MODE,
        recurrent_halting_stack_source=_default_controller_stack_source(
            config_module, "ff_recurrent_halting_stack"
        ),
    )


def _default_controller_stack_source(
    config_module: ModuleType, prefix: str
) -> ExpertsSubmoduleStackSource:
    config_prefix = prefix.upper()
    return ExpertsSubmoduleStackSource(
        independent_flag=getattr(config_module, f"{config_prefix}_INDEPENDENT_FLAG"),
        hidden_dim=getattr(config_module, f"{config_prefix}_HIDDEN_DIM"),
        num_layers=getattr(config_module, f"{config_prefix}_NUM_LAYERS"),
        last_layer_bias_option=getattr(
            config_module, f"{config_prefix}_LAST_LAYER_BIAS_OPTION"
        ),
        apply_output_pipeline_flag=getattr(
            config_module, f"{config_prefix}_APPLY_OUTPUT_PIPELINE_FLAG"
        ),
        activation=getattr(config_module, f"{config_prefix}_ACTIVATION"),
        layer_norm_position=getattr(
            config_module, f"{config_prefix}_LAYER_NORM_POSITION"
        ),
        residual_connection_option=getattr(
            config_module, f"{config_prefix}_RESIDUAL_CONNECTION_OPTION"
        ),
        dropout_probability=getattr(
            config_module, f"{config_prefix}_DROPOUT_PROBABILITY"
        ),
        bias_flag=getattr(config_module, f"{config_prefix}_BIAS_FLAG"),
    )


def _pop_updates(kwargs: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for flat_key, option_field in mapping.items():
        if flat_key in kwargs:
            updates[option_field] = kwargs.pop(flat_key)
    return updates
