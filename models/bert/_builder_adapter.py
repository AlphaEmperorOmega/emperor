from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from typing import Any

from models.experts._builder_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
)
from models.transformer._builder_options import (
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)

_TOP_LEVEL_KEYS = (
    "batch_size",
    "learning_rate",
    "input_dim",
    "output_dim",
    "sequence_length",
    "embedding_dropout_probability",
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
    "dropout_probability": "dropout_probability",
    "bias_flag": "bias_flag",
}

_SUBMODULE_STACK_FIELD_MAP = {
    field: field for field in _CONTROLLER_STACK_FIELD_MAP if field != "independent_flag"
}


def linear_builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any],
    config_module: ModuleType,
) -> dict[str, Any]:
    kwargs = dict(flat_kwargs)
    builder_kwargs = _pop_top_level_kwargs(kwargs)
    builder_kwargs.update(
        {
            "encoder_options": _encoder_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("encoder_options", None),
            ),
            "positional_embedding_options": _positional_embedding_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("positional_embedding_options", None),
            ),
            "attention_options": _attention_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("attention_options", None),
            ),
            "feed_forward_options": _feed_forward_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("feed_forward_options", None),
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


def _pop_top_level_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: kwargs.pop(key) for key in _TOP_LEVEL_KEYS if key in kwargs}


def _encoder_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: TransformerEncoderOptions | None,
) -> TransformerEncoderOptions:
    options = provided or TransformerEncoderOptions(
        hidden_dim=config_module.STACK_HIDDEN_DIM,
        num_layers=config_module.STACK_NUM_LAYERS,
        activation=config_module.STACK_ACTIVATION,
        dropout_probability=config_module.STACK_DROPOUT_PROBABILITY,
        layer_norm_position=config_module.LAYER_NORM_POSITION,
        causal_attention_mask_flag=config_module.CAUSAL_ATTENTION_MASK_FLAG,
    )
    updates = _pop_updates(
        kwargs,
        {
            "stack_hidden_dim": "hidden_dim",
            "stack_num_layers": "num_layers",
            "stack_activation": "activation",
            "stack_dropout_probability": "dropout_probability",
            "layer_norm_position": "layer_norm_position",
            "stack_layer_norm_position": "layer_norm_position",
            "causal_attention_mask_flag": "causal_attention_mask_flag",
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
        num_layers=config_module.FF_NUM_LAYERS,
        bias_flag=config_module.FF_BIAS_FLAG,
    )
    updates = _pop_updates(
        kwargs,
        {
            "ff_num_layers": "num_layers",
            "ff_bias_flag": "bias_flag",
        },
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
        apply_output_pipeline_flag=(
            config_module.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        activation=config_module.SUBMODULE_STACK_ACTIVATION,
        layer_norm_position=config_module.SUBMODULE_STACK_LAYER_NORM_POSITION,
        residual_connection_option=(
            config_module.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
        ),
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
        gate_stack_source=_default_controller_stack_source(
            config_module,
            "gate_stack",
        ),
        stack_halting_flag=config_module.HALTING_FLAG,
        halting_threshold=config_module.HALTING_THRESHOLD,
        halting_dropout=config_module.HALTING_DROPOUT,
        halting_hidden_state_mode=config_module.HALTING_HIDDEN_STATE_MODE,
        halting_stack_source=_default_controller_stack_source(
            config_module,
            "halting_stack",
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
        kwargs,
        config_module,
        "gate_stack",
        provided=options.gate_stack_source,
    )
    updates["halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "halting_stack",
        provided=options.halting_stack_source,
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
        memory_test_time_training_learning_rate=(
            config_module.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        ),
        memory_test_time_training_num_inner_steps=(
            config_module.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
        ),
        memory_stack_source=_default_controller_stack_source(
            config_module,
            "memory_stack",
        ),
    )
    updates = _pop_updates(
        kwargs,
        {
            "memory_flag": "memory_flag",
            "memory_option": "memory_option",
            "memory_position_option": "memory_position_option",
            "memory_test_time_training_learning_rate": (
                "memory_test_time_training_learning_rate"
            ),
            "memory_test_time_training_num_inner_steps": (
                "memory_test_time_training_num_inner_steps"
            ),
        },
    )
    updates["memory_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "memory_stack",
        provided=options.memory_stack_source,
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
            config_module,
            "recurrent_gate_stack",
        ),
        recurrent_halting_flag=config_module.RECURRENT_HALTING_FLAG,
        recurrent_halting_threshold=config_module.RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout=config_module.RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode=(
            config_module.RECURRENT_HALTING_HIDDEN_STATE_MODE
        ),
        recurrent_halting_stack_source=_default_controller_stack_source(
            config_module,
            "recurrent_halting_stack",
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
            "recurrent_halting_hidden_state_mode": (
                "recurrent_halting_hidden_state_mode"
            ),
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


def _default_controller_stack_source(
    config_module: ModuleType,
    prefix: str,
) -> ExpertsSubmoduleStackSource:
    config_prefix = prefix.upper()
    return ExpertsSubmoduleStackSource(
        independent_flag=getattr(config_module, f"{config_prefix}_INDEPENDENT_FLAG"),
        hidden_dim=getattr(config_module, f"{config_prefix}_HIDDEN_DIM"),
        num_layers=getattr(config_module, f"{config_prefix}_NUM_LAYERS"),
        last_layer_bias_option=getattr(
            config_module,
            f"{config_prefix}_LAST_LAYER_BIAS_OPTION",
        ),
        apply_output_pipeline_flag=getattr(
            config_module,
            f"{config_prefix}_APPLY_OUTPUT_PIPELINE_FLAG",
        ),
        activation=getattr(config_module, f"{config_prefix}_ACTIVATION"),
        layer_norm_position=getattr(
            config_module,
            f"{config_prefix}_LAYER_NORM_POSITION",
        ),
        residual_connection_option=getattr(
            config_module,
            f"{config_prefix}_RESIDUAL_CONNECTION_OPTION",
        ),
        dropout_probability=getattr(
            config_module,
            f"{config_prefix}_DROPOUT_PROBABILITY",
        ),
        bias_flag=getattr(config_module, f"{config_prefix}_BIAS_FLAG"),
    )


def _pop_updates(
    kwargs: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for flat_key, option_field in mapping.items():
        if flat_key in kwargs:
            updates[option_field] = kwargs.pop(flat_key)
    return updates
