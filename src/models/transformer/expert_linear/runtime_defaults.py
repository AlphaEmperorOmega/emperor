from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, fields, replace
from types import ModuleType
from typing import Any, Final

from model_runtime.packages.runtime_values import validate_runtime_default_values

from . import config
from .runtime_options import (
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
        apply_output_postprocessing_flag=getattr(
            config_module, f"{prefix}_APPLY_OUTPUT_POSTPROCESSING_FLAG"
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
            apply_output_postprocessing_flag=(
                getattr(
                    config_module,
                    f"{prefix}_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG",
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
            apply_output_postprocessing_flag=(
                getattr(
                    config_module,
                    f"{prefix}_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG",
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


_STACK_OPTION_FIELDS = (
    "hidden_dim",
    "num_layers",
    "last_layer_bias_option",
    "apply_output_postprocessing_flag",
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


_PATH_COMPONENT_FIELDS: Final[dict[str, str]] = {
    "path": "",
    "stack": "stack_options",
    "controller": "layer_controller_options",
    "memory": "dynamic_memory_options",
    "recurrent": "recurrent_controller_options",
    "gate_stack": "layer_controller_options",
    "halting_stack": "layer_controller_options",
    "memory_stack": "dynamic_memory_options",
    "recurrent_gate_stack": "recurrent_controller_options",
    "recurrent_halting_stack": "recurrent_controller_options",
}


def _replace_dataclass_path(source: Any, dotted_field: str, value: Any) -> Any:
    field_name, separator, nested_field = dotted_field.partition(".")
    if not separator:
        return replace(source, **{field_name: value})
    nested = _replace_dataclass_path(
        getattr(source, field_name),
        nested_field,
        value,
    )
    return replace(source, **{field_name: nested})


def _component_field_path(component: str, field_name: str) -> str:
    try:
        component_field = _PATH_COMPONENT_FIELDS[component]
    except KeyError:
        raise ValueError(component) from None
    return f"{component_field}.{field_name}" if component_field else field_name


def _apply_path_updates(
    options: TransformerAttentionOptions | TransformerFeedForwardOptions,
    updates: dict[str, Any],
    *,
    attention: bool,
) -> TransformerAttentionOptions | TransformerFeedForwardOptions:
    field_map = _ATTENTION_FIELD_MAP if attention else _FEED_FORWARD_FIELD_MAP
    path = options
    for suffix, value in updates.items():
        component, field_name = field_map[suffix]
        path = _replace_dataclass_path(
            path,
            _component_field_path(component, field_name),
            value,
        )
    return replace(path)


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
        recurrent_initial_iterations=config.RECURRENT_INITIAL_ITERATIONS,
        recurrent_gradient_transition_count=config.RECURRENT_GRADIENT_TRANSITION_COUNT,
        recurrent_no_gradient_transition_count=config.RECURRENT_NO_GRADIENT_TRANSITION_COUNT,
        recurrent_iteration_increment=config.RECURRENT_ITERATION_INCREMENT,
        recurrent_forward_calls_before_iteration_increment=(
            config.RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT
        ),
        recurrent_smooth_iteration_growth_flag=(
            config.RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG
        ),
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
        residual_stack_independent_flag=(config.RESIDUAL_STACK_INDEPENDENT_FLAG),
        residual_stack_hidden_dim=config.RESIDUAL_STACK_HIDDEN_DIM,
        residual_stack_layer_norm_position=(config.RESIDUAL_STACK_LAYER_NORM_POSITION),
        residual_stack_num_layers=config.RESIDUAL_STACK_NUM_LAYERS,
        residual_stack_activation=config.RESIDUAL_STACK_ACTIVATION,
        residual_stack_residual_connection_option=(
            config.RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_stack_residual_model_flag=(config.RESIDUAL_STACK_RESIDUAL_MODEL_FLAG),
        residual_stack_dropout_probability=(config.RESIDUAL_STACK_DROPOUT_PROBABILITY),
        residual_stack_last_layer_bias_option=(
            config.RESIDUAL_STACK_LAST_LAYER_BIAS_OPTION
        ),
        residual_stack_apply_output_postprocessing_flag=(
            config.RESIDUAL_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
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


@dataclass(frozen=True)
class _ResolvedTransformerStacks:
    encoder: TransformerStackOptions
    decoder: TransformerStackOptions


@dataclass(frozen=True)
class _ResolvedExpertOptions:
    attention: ExpertOptions
    feed_forward: ExpertOptions


def _resolve_top_level_runtime(
    values: MutableMapping[str, Any],
    runtime: RuntimeOptions,
) -> RuntimeOptions:
    scalar_updates: dict[str, Any] = {}
    model_dim_changed = False
    dropout_changed = False
    for key in list(values):
        if key == "sequence_length":
            length = values.pop(key)
            scalar_updates.update(
                source_sequence_length=length,
                target_sequence_length=length,
            )
        elif key in _TOP_LEVEL_FIELDS - _PATH_FIELDS - {
            "encoder_options",
            "decoder_options",
        }:
            scalar_updates[key] = values.pop(key)
            model_dim_changed |= key == "model_dim"
            dropout_changed |= key == "dropout_probability"

    resolved = replace(runtime, **scalar_updates)
    if model_dim_changed:
        values.setdefault("attn_stack_hidden_dim", resolved.model_dim)
    if dropout_changed:
        values.setdefault(
            "ff_stack_dropout_probability",
            resolved.dropout_probability,
        )
    return resolved


def _resolve_scoped_stack(
    values: MutableMapping[str, Any],
    prefix: str,
    current: TransformerStackOptions,
) -> TransformerStackOptions:
    updates = {}
    for field_name in _STACK_FIELDS:
        key = f"{prefix}{field_name}"
        if key in values:
            updates[field_name] = values.pop(key)
    return replace(current, **updates)


def _resolve_transformer_stacks(
    values: MutableMapping[str, Any],
    runtime: RuntimeOptions,
) -> _ResolvedTransformerStacks:
    broadcast = {key: values.pop(key) for key in list(values) if key in _STACK_FIELDS}
    encoder = replace(
        values.pop("encoder_options", runtime.encoder_options),
        **broadcast,
    )
    decoder = replace(
        values.pop("decoder_options", runtime.decoder_options),
        **broadcast,
    )
    return _ResolvedTransformerStacks(
        encoder=_resolve_scoped_stack(values, "encoder_", encoder),
        decoder=_resolve_scoped_stack(values, "decoder_", decoder),
    )


def _runtime_path_options(runtime: RuntimeOptions) -> TransformerPathOptions:
    return TransformerPathOptions(
        encoder_attention_options=runtime.encoder_attention_options,
        decoder_self_attention_options=runtime.decoder_self_attention_options,
        decoder_cross_attention_options=runtime.decoder_cross_attention_options,
        encoder_feed_forward_options=runtime.encoder_feed_forward_options,
        decoder_feed_forward_options=runtime.decoder_feed_forward_options,
    )


def _resolve_scoped_experts(
    values: MutableMapping[str, Any],
    prefix: str,
    current: ExpertOptions,
) -> ExpertOptions:
    updates = {}
    for field_name in _EXPERT_FIELDS:
        key = f"{prefix}{field_name}"
        if key in values:
            updates[field_name] = values.pop(key)
    return replace(current, **updates)


def _resolve_expert_options(
    values: MutableMapping[str, Any],
    runtime: RuntimeOptions,
) -> _ResolvedExpertOptions:
    if "expert_attention_use_kv_expert_models_flag" in values:
        values["use_kv_expert_models_flag"] = values.pop(
            "expert_attention_use_kv_expert_models_flag"
        )
    broadcast = {key: values.pop(key) for key in list(values) if key in _EXPERT_FIELDS}
    attention = replace(runtime.attention_expert_options, **broadcast)
    feed_forward = replace(runtime.feed_forward_expert_options, **broadcast)
    attention = _resolve_scoped_experts(values, "attention_expert_", attention)
    feed_forward = _resolve_scoped_experts(
        values,
        "feed_forward_expert_",
        feed_forward,
    )

    router_updates = _pop_updates(values, "router_", _FEED_FORWARD_FIELD_MAP)
    expert_path_updates = _pop_updates(values, "expert_", _FEED_FORWARD_FIELD_MAP)

    def with_path_updates(options: ExpertOptions) -> ExpertOptions:
        return replace(
            options,
            router_path_options=_apply_path_updates(
                options.router_path_options,
                router_updates,
                attention=False,
            ),
            expert_path_options=_apply_path_updates(
                options.expert_path_options,
                expert_path_updates,
                attention=False,
            ),
        )

    return _ResolvedExpertOptions(
        attention=with_path_updates(attention),
        feed_forward=with_path_updates(feed_forward),
    )


def _reject_unknown_runtime_default(values: MutableMapping[str, Any]) -> None:
    if values:
        unknown = sorted(values)[0]
        raise TypeError(
            "TransformerExpertLinearConfigBuilder.__init__() got an unexpected "
            f"keyword argument {unknown!r}"
        )


def runtime_from_flat(
    values: dict[str, Any] | None = None,
    base: RuntimeOptions | None = None,
) -> RuntimeOptions:
    values = validate_runtime_default_values(
        values,
        package="models.transformer.expert_linear",
        config_module=config,
    )
    runtime = _resolve_top_level_runtime(
        values,
        DEFAULT_RUNTIME if base is None else base,
    )
    stacks = _resolve_transformer_stacks(values, runtime)
    paths = resolve_transformer_path_options(
        values,
        _runtime_path_options(runtime),
    )
    experts = _resolve_expert_options(values, runtime)
    _reject_unknown_runtime_default(values)
    return replace(
        runtime,
        encoder_options=stacks.encoder,
        decoder_options=stacks.decoder,
        encoder_attention_options=paths.encoder_attention_options,
        decoder_self_attention_options=paths.decoder_self_attention_options,
        decoder_cross_attention_options=paths.decoder_cross_attention_options,
        encoder_feed_forward_options=paths.encoder_feed_forward_options,
        decoder_feed_forward_options=paths.decoder_feed_forward_options,
        attention_expert_options=experts.attention,
        feed_forward_expert_options=experts.feed_forward,
    )


DEFAULT_RUNTIME: Final[RuntimeOptions] = runtime_from_config()

__all__ = ["DEFAULT_RUNTIME", "runtime_from_config", "runtime_from_flat"]
