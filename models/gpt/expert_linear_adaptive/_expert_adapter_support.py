# ruff: noqa: E501

from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from typing import Any

from models.gpt.expert_linear_adaptive import runtime_options as expert_options

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
_CONTROLLER_STACK_FIELD_MAP = {
    "independent_flag": "independent_flag",
    **_SUBMODULE_STACK_FIELD_MAP,
}


def _role_stack_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    prefix: str,
    *,
    defaults: expert_options.ExpertsSubmoduleStackOptions,
    provided: expert_options.ExpertsSubmoduleStackOptions | None,
    extra_mapping: dict[str, str] | None = None,
    default_overrides: dict[str, Any] | None = None,
) -> expert_options.ExpertsSubmoduleStackOptions:
    options = provided or replace(defaults, **default_overrides or {})
    mapping = {
        f"{prefix}_{flat_field}": dataclass_field
        for flat_field, dataclass_field in _SUBMODULE_STACK_FIELD_MAP.items()
    }
    if extra_mapping:
        mapping.update(extra_mapping)
    updates = _pop_updates(kwargs, mapping)
    return replace(options, **updates) if updates else options


def _router_stack_options_from_config(
    config_module: ModuleType,
) -> expert_options.ExpertsSubmoduleStackOptions:
    return expert_options.ExpertsSubmoduleStackOptions(
        hidden_dim=config_module.ROUTER_STACK_HIDDEN_DIM,
        num_layers=config_module.ROUTER_STACK_NUM_LAYERS,
        last_layer_bias_option=config_module.ROUTER_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config_module.ROUTER_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config_module.ROUTER_STACK_ACTIVATION,
        layer_norm_position=config_module.ROUTER_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config_module.ROUTER_STACK_RESIDUAL_CONNECTION_OPTION,
        dropout_probability=config_module.ROUTER_STACK_DROPOUT_PROBABILITY,
        bias_flag=config_module.ROUTER_BIAS_FLAG,
    )


def _mixture_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsMixtureOptions | None,
) -> expert_options.ExpertsMixtureOptions:
    options = provided or expert_options.ExpertsMixtureOptions(
        top_k=config_module.EXPERT_TOP_K,
        num_experts=config_module.EXPERT_NUM_EXPERTS,
        capacity_factor=config_module.EXPERT_CAPACITY_FACTOR,
        dropped_token_behavior=config_module.EXPERT_DROPPED_TOKEN_BEHAVIOR,
        compute_expert_mixture_flag=config_module.EXPERT_COMPUTE_EXPERT_MIXTURE_FLAG,
        weighted_parameters_flag=config_module.EXPERT_WEIGHTED_PARAMETERS_FLAG,
        weighting_position_option=config_module.EXPERT_WEIGHTING_POSITION_OPTION,
        routing_initialization_mode=config_module.EXPERT_ROUTING_INITIALIZATION_MODE,
    )
    updates = _pop_updates(
        kwargs,
        {
            "top_k": "top_k",
            "num_experts": "num_experts",
            "capacity_factor": "capacity_factor",
            "dropped_token_behavior": "dropped_token_behavior",
            "compute_expert_mixture_flag": "compute_expert_mixture_flag",
            "weighted_parameters_flag": "weighted_parameters_flag",
            "weighting_position_option": "weighting_position_option",
            "routing_initialization_mode": "routing_initialization_mode",
        },
    )
    return replace(options, **updates) if updates else options


def _sampler_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsSamplerOptions | None,
) -> expert_options.ExpertsSamplerOptions:
    options = provided or expert_options.ExpertsSamplerOptions(
        threshold=config_module.SAMPLER_THRESHOLD,
        filter_above_threshold=config_module.SAMPLER_FILTER_ABOVE_THRESHOLD,
        num_topk_samples=config_module.SAMPLER_NUM_TOPK_SAMPLES,
        normalize_probabilities_flag=config_module.SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
        noisy_topk_flag=config_module.SAMPLER_NOISY_TOPK_FLAG,
        coefficient_of_variation_loss_weight=config_module.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
        switch_loss_weight=config_module.SAMPLER_SWITCH_LOSS_WEIGHT,
        zero_centred_loss_weight=config_module.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT,
        mutual_information_loss_weight=config_module.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT,
    )
    updates = _pop_updates(
        kwargs,
        {
            "sampler_threshold": "threshold",
            "sampler_filter_above_threshold": "filter_above_threshold",
            "sampler_num_topk_samples": "num_topk_samples",
            "sampler_normalize_probabilities_flag": "normalize_probabilities_flag",
            "sampler_noisy_topk_flag": "noisy_topk_flag",
            "sampler_coefficient_of_variation_loss_weight": "coefficient_of_variation_loss_weight",
            "sampler_switch_loss_weight": "switch_loss_weight",
            "sampler_zero_centred_loss_weight": "zero_centred_loss_weight",
            "sampler_mutual_information_loss_weight": "mutual_information_loss_weight",
        },
    )
    return replace(options, **updates) if updates else options


def _router_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsRouterOptions | None,
) -> expert_options.ExpertsRouterOptions:
    options = provided or expert_options.ExpertsRouterOptions(
        noisy_topk_flag=config_module.ROUTER_NOISY_TOPK_FLAG
    )
    updates = _pop_updates(kwargs, {"router_noisy_topk_flag": "noisy_topk_flag"})
    return replace(options, **updates) if updates else options


def _controller_stack_source_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    prefix: str,
    *,
    provided: expert_options.ExpertsSubmoduleStackSource | None = None,
) -> expert_options.ExpertsSubmoduleStackSource:
    source = provided or _default_controller_stack_source(config_module, prefix)
    updates = _pop_updates(
        kwargs,
        {
            f"{prefix}_{flat_field}": dataclass_field
            for flat_field, dataclass_field in _CONTROLLER_STACK_FIELD_MAP.items()
        },
    )
    return replace(source, **updates) if updates else source


def _layer_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsLayerControllerOptions | None,
) -> expert_options.ExpertsLayerControllerOptions:
    options = provided or expert_options.ExpertsLayerControllerOptions(
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


def _dynamic_memory_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsDynamicMemoryOptions | None,
) -> expert_options.ExpertsDynamicMemoryOptions:
    options = provided or expert_options.ExpertsDynamicMemoryOptions(
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


def _recurrent_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsRecurrentControllerOptions | None,
) -> expert_options.ExpertsRecurrentControllerOptions:
    options = provided or expert_options.ExpertsRecurrentControllerOptions(
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


def _expert_layer_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsLayerControllerOptions | None,
) -> expert_options.ExpertsLayerControllerOptions:
    options = provided or expert_options.ExpertsLayerControllerOptions(
        stack_gate_flag=config_module.EXPERT_GATE_FLAG,
        gate_option=config_module.EXPERT_GATE_OPTION,
        gate_activation=config_module.EXPERT_GATE_ACTIVATION,
        gate_stack_source=_default_controller_stack_source(
            config_module, "expert_gate_stack"
        ),
        stack_halting_flag=config_module.EXPERT_HALTING_FLAG,
        halting_threshold=config_module.EXPERT_HALTING_THRESHOLD,
        halting_dropout=config_module.EXPERT_HALTING_DROPOUT,
        halting_hidden_state_mode=config_module.EXPERT_HALTING_HIDDEN_STATE_MODE,
        halting_stack_source=_default_controller_stack_source(
            config_module, "expert_halting_stack"
        ),
        halting_output_dim=config_module.EXPERT_HALTING_OUTPUT_DIM,
    )
    updates = _pop_updates(
        kwargs,
        {
            "expert_gate_flag": "stack_gate_flag",
            "expert_gate_option": "gate_option",
            "expert_gate_activation": "gate_activation",
            "expert_halting_flag": "stack_halting_flag",
            "expert_halting_threshold": "halting_threshold",
            "expert_halting_dropout": "halting_dropout",
            "expert_halting_hidden_state_mode": "halting_hidden_state_mode",
            "expert_halting_output_dim": "halting_output_dim",
        },
    )
    updates["gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs, config_module, "expert_gate_stack", provided=options.gate_stack_source
    )
    updates["halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "expert_halting_stack",
        provided=options.halting_stack_source,
    )
    return replace(options, **updates)


def _expert_dynamic_memory_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsDynamicMemoryOptions | None,
) -> expert_options.ExpertsDynamicMemoryOptions:
    options = provided or expert_options.ExpertsDynamicMemoryOptions(
        memory_flag=config_module.EXPERT_MEMORY_FLAG,
        memory_option=config_module.EXPERT_MEMORY_OPTION,
        memory_position_option=config_module.EXPERT_MEMORY_POSITION_OPTION,
        memory_test_time_training_learning_rate=config_module.EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
        memory_test_time_training_num_inner_steps=config_module.EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
        memory_stack_source=_default_controller_stack_source(
            config_module, "expert_memory_stack"
        ),
    )
    updates = _pop_updates(
        kwargs,
        {
            "expert_memory_flag": "memory_flag",
            "expert_memory_option": "memory_option",
            "expert_memory_position_option": "memory_position_option",
            "expert_memory_test_time_training_learning_rate": "memory_test_time_training_learning_rate",
            "expert_memory_test_time_training_num_inner_steps": "memory_test_time_training_num_inner_steps",
        },
    )
    updates["memory_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "expert_memory_stack",
        provided=options.memory_stack_source,
    )
    return replace(options, **updates)


def _expert_recurrent_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsRecurrentControllerOptions | None,
) -> expert_options.ExpertsRecurrentControllerOptions:
    options = provided or expert_options.ExpertsRecurrentControllerOptions(
        recurrent_flag=config_module.EXPERT_RECURRENT_FLAG,
        recurrent_max_steps=config_module.EXPERT_RECURRENT_MAX_STEPS,
        recurrent_layer_norm_position=config_module.EXPERT_RECURRENT_LAYER_NORM_POSITION,
        recurrent_gate_flag=config_module.EXPERT_RECURRENT_GATE_FLAG,
        recurrent_gate_option=config_module.EXPERT_RECURRENT_GATE_OPTION,
        recurrent_gate_activation=config_module.EXPERT_RECURRENT_GATE_ACTIVATION,
        recurrent_gate_stack_source=_default_controller_stack_source(
            config_module, "expert_recurrent_gate_stack"
        ),
        recurrent_halting_flag=config_module.EXPERT_RECURRENT_HALTING_FLAG,
        recurrent_halting_threshold=config_module.EXPERT_RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout=config_module.EXPERT_RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode=config_module.EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE,
        recurrent_halting_stack_source=_default_controller_stack_source(
            config_module, "expert_recurrent_halting_stack"
        ),
    )
    updates = _pop_updates(
        kwargs,
        {
            "expert_recurrent_flag": "recurrent_flag",
            "expert_recurrent_max_steps": "recurrent_max_steps",
            "expert_recurrent_layer_norm_position": "recurrent_layer_norm_position",
            "expert_recurrent_gate_flag": "recurrent_gate_flag",
            "expert_recurrent_gate_option": "recurrent_gate_option",
            "expert_recurrent_gate_activation": "recurrent_gate_activation",
            "expert_recurrent_halting_flag": "recurrent_halting_flag",
            "expert_recurrent_halting_threshold": "recurrent_halting_threshold",
            "expert_recurrent_halting_dropout": "recurrent_halting_dropout",
            "expert_recurrent_halting_hidden_state_mode": "recurrent_halting_hidden_state_mode",
        },
    )
    updates["recurrent_gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "expert_recurrent_gate_stack",
        provided=options.recurrent_gate_stack_source,
    )
    updates["recurrent_halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "expert_recurrent_halting_stack",
        provided=options.recurrent_halting_stack_source,
    )
    return replace(options, **updates)


def _router_layer_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsLayerControllerOptions | None,
) -> expert_options.ExpertsLayerControllerOptions:
    options = provided or expert_options.ExpertsLayerControllerOptions(
        stack_gate_flag=config_module.ROUTER_GATE_FLAG,
        gate_option=config_module.ROUTER_GATE_OPTION,
        gate_activation=config_module.ROUTER_GATE_ACTIVATION,
        gate_stack_source=_default_controller_stack_source(
            config_module, "router_gate_stack"
        ),
        stack_halting_flag=config_module.ROUTER_HALTING_FLAG,
        halting_threshold=config_module.ROUTER_HALTING_THRESHOLD,
        halting_dropout=config_module.ROUTER_HALTING_DROPOUT,
        halting_hidden_state_mode=config_module.ROUTER_HALTING_HIDDEN_STATE_MODE,
        halting_stack_source=_default_controller_stack_source(
            config_module, "router_halting_stack"
        ),
        halting_output_dim=config_module.ROUTER_HALTING_OUTPUT_DIM,
    )
    updates = _pop_updates(
        kwargs,
        {
            "router_gate_flag": "stack_gate_flag",
            "router_gate_option": "gate_option",
            "router_gate_activation": "gate_activation",
            "router_halting_flag": "stack_halting_flag",
            "router_halting_threshold": "halting_threshold",
            "router_halting_dropout": "halting_dropout",
            "router_halting_hidden_state_mode": "halting_hidden_state_mode",
            "router_halting_output_dim": "halting_output_dim",
        },
    )
    updates["gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs, config_module, "router_gate_stack", provided=options.gate_stack_source
    )
    updates["halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "router_halting_stack",
        provided=options.halting_stack_source,
    )
    return replace(options, **updates)


def _router_dynamic_memory_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsDynamicMemoryOptions | None,
) -> expert_options.ExpertsDynamicMemoryOptions:
    options = provided or expert_options.ExpertsDynamicMemoryOptions(
        memory_flag=config_module.ROUTER_MEMORY_FLAG,
        memory_option=config_module.ROUTER_MEMORY_OPTION,
        memory_position_option=config_module.ROUTER_MEMORY_POSITION_OPTION,
        memory_test_time_training_learning_rate=config_module.ROUTER_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
        memory_test_time_training_num_inner_steps=config_module.ROUTER_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
        memory_stack_source=_default_controller_stack_source(
            config_module, "router_memory_stack"
        ),
    )
    updates = _pop_updates(
        kwargs,
        {
            "router_memory_flag": "memory_flag",
            "router_memory_option": "memory_option",
            "router_memory_position_option": "memory_position_option",
            "router_memory_test_time_training_learning_rate": "memory_test_time_training_learning_rate",
            "router_memory_test_time_training_num_inner_steps": "memory_test_time_training_num_inner_steps",
        },
    )
    updates["memory_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "router_memory_stack",
        provided=options.memory_stack_source,
    )
    return replace(options, **updates)


def _router_recurrent_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsRecurrentControllerOptions | None,
) -> expert_options.ExpertsRecurrentControllerOptions:
    options = provided or expert_options.ExpertsRecurrentControllerOptions(
        recurrent_flag=config_module.ROUTER_RECURRENT_FLAG,
        recurrent_max_steps=config_module.ROUTER_RECURRENT_MAX_STEPS,
        recurrent_layer_norm_position=config_module.ROUTER_RECURRENT_LAYER_NORM_POSITION,
        recurrent_gate_flag=config_module.ROUTER_RECURRENT_GATE_FLAG,
        recurrent_gate_option=config_module.ROUTER_RECURRENT_GATE_OPTION,
        recurrent_gate_activation=config_module.ROUTER_RECURRENT_GATE_ACTIVATION,
        recurrent_gate_stack_source=_default_controller_stack_source(
            config_module, "router_recurrent_gate_stack"
        ),
        recurrent_halting_flag=config_module.ROUTER_RECURRENT_HALTING_FLAG,
        recurrent_halting_threshold=config_module.ROUTER_RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout=config_module.ROUTER_RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode=config_module.ROUTER_RECURRENT_HALTING_HIDDEN_STATE_MODE,
        recurrent_halting_stack_source=_default_controller_stack_source(
            config_module, "router_recurrent_halting_stack"
        ),
    )
    updates = _pop_updates(
        kwargs,
        {
            "router_recurrent_flag": "recurrent_flag",
            "router_recurrent_max_steps": "recurrent_max_steps",
            "router_recurrent_layer_norm_position": "recurrent_layer_norm_position",
            "router_recurrent_gate_flag": "recurrent_gate_flag",
            "router_recurrent_gate_option": "recurrent_gate_option",
            "router_recurrent_gate_activation": "recurrent_gate_activation",
            "router_recurrent_halting_flag": "recurrent_halting_flag",
            "router_recurrent_halting_threshold": "recurrent_halting_threshold",
            "router_recurrent_halting_dropout": "recurrent_halting_dropout",
            "router_recurrent_halting_hidden_state_mode": "recurrent_halting_hidden_state_mode",
        },
    )
    updates["recurrent_gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "router_recurrent_gate_stack",
        provided=options.recurrent_gate_stack_source,
    )
    updates["recurrent_halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "router_recurrent_halting_stack",
        provided=options.recurrent_halting_stack_source,
    )
    return replace(options, **updates)


def _default_controller_stack_source(
    config_module: ModuleType, prefix: str
) -> expert_options.ExpertsSubmoduleStackSource:
    config_prefix = prefix.upper()
    return expert_options.ExpertsSubmoduleStackSource(
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
