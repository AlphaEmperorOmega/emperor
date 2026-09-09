# ruff: noqa: E501

from __future__ import annotations

from dataclasses import replace
from types import ModuleType
from typing import Any

import models.vit.expert_linear_adaptive.runtime_options as expert_options
from models.vit.expert_linear_adaptive import _config_defaults as config_defaults
from models.vit.expert_linear_adaptive._flat_updates import (
    pop_updates as _pop_updates,
)

_SUBMODULE_STACK_FIELD_MAP = {
    "hidden_dim": "hidden_dim",
    "num_layers": "num_layers",
    "last_layer_bias_option": "last_layer_bias_option",
    "apply_output_postprocessing_flag": "apply_output_postprocessing_flag",
    "activation": "activation",
    "layer_norm_position": "layer_norm_position",
    "normalization": "normalization",
    "residual_connection_option": "residual_connection_option",
    "residual_model_flag": "residual_model_flag",
    "dropout_probability": "dropout_probability",
    "bias_flag": "bias_flag",
}
_CONTROLLER_STACK_FIELD_MAP = {
    "independent_flag": "independent_flag",
    **_SUBMODULE_STACK_FIELD_MAP,
}


def _submodule_stack_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsSubmoduleStackOptions | None,
) -> expert_options.ExpertsSubmoduleStackOptions:
    options = provided or config_defaults.experts_submodule_stack_options(
        config_module, config_defaults.ExpertStackRole.MAIN
    )
    updates = _pop_updates(
        kwargs,
        {
            f"submodule_stack_{flat_field}": dataclass_field
            for flat_field, dataclass_field in _SUBMODULE_STACK_FIELD_MAP.items()
        },
    )
    return replace(options, **updates) if updates else options


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


def _mixture_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsMixtureOptions | None,
) -> expert_options.ExpertsMixtureOptions:
    options = provided or config_defaults.experts_mixture_options(config_module)
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
    options = provided or config_defaults.experts_sampler_options(config_module)
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
    options = provided or config_defaults.experts_router_options(config_module)
    updates = _pop_updates(kwargs, {"router_noisy_topk_flag": "noisy_topk_flag"})
    return replace(options, **updates) if updates else options


def _controller_stack_source_from_kwargs(
    kwargs: dict[str, Any],
    prefix: str,
    *,
    provided: expert_options.ExpertsSubmoduleStackSource,
) -> expert_options.ExpertsSubmoduleStackSource:
    updates = _pop_updates(
        kwargs,
        {
            f"{prefix}_{flat_field}": dataclass_field
            for flat_field, dataclass_field in _CONTROLLER_STACK_FIELD_MAP.items()
        },
    )
    return replace(provided, **updates) if updates else provided


def _layer_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsLayerControllerOptions | None,
) -> expert_options.ExpertsLayerControllerOptions:
    options = provided or config_defaults.experts_layer_controller_options(
        config_module, config_defaults.ExpertControlRole.MAIN
    )
    updates = _pop_updates(
        kwargs,
        {
            "stack_gate_flag": "stack_gate_flag",
            "gate_option": "gate_option",
            "gate_activation": "gate_activation",
            "stack_halting_flag": "stack_halting_flag",
            "halting_option": "halting_option",
            "halting_threshold": "halting_threshold",
            "halting_dropout": "halting_dropout",
            "halting_hidden_state_mode": "halting_hidden_state_mode",
            "halting_output_dim": "halting_output_dim",
            "shared_gate_config": "shared_gate_config",
        },
    )
    updates["gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs, "gate_stack", provided=options.gate_stack_source
    )
    updates["halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs, "halting_stack", provided=options.halting_stack_source
    )
    return replace(options, **updates)


def _dynamic_memory_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsDynamicMemoryOptions | None,
) -> expert_options.ExpertsDynamicMemoryOptions:
    options = provided or config_defaults.experts_dynamic_memory_options(
        config_module, config_defaults.ExpertControlRole.MAIN
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
        kwargs, "memory_stack", provided=options.memory_stack_source
    )
    return replace(options, **updates)


def _recurrent_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsRecurrentControllerOptions | None,
) -> expert_options.ExpertsRecurrentControllerOptions:
    options = provided or config_defaults.experts_recurrent_controller_options(
        config_module, config_defaults.ExpertControlRole.MAIN
    )
    updates = _pop_updates(
        kwargs,
        {
            "recurrent_flag": "recurrent_flag",
            "recurrent_max_steps": "recurrent_max_steps",
            "recurrent_initial_iterations": "recurrent_initial_iterations",
            "recurrent_gradient_transition_count": "recurrent_gradient_transition_count",
            "recurrent_no_gradient_transition_count": "recurrent_no_gradient_transition_count",
            "recurrent_iteration_increment": "recurrent_iteration_increment",
            "recurrent_forward_calls_before_iteration_increment": (
                "recurrent_forward_calls_before_iteration_increment"
            ),
            "recurrent_layer_norm_position": "recurrent_layer_norm_position",
            "recurrent_normalization": "recurrent_normalization",
            "recurrent_stack_gate_flag": "recurrent_stack_gate_flag",
            "recurrent_gate_option": "recurrent_gate_option",
            "recurrent_gate_activation": "recurrent_gate_activation",
            "recurrent_stack_halting_flag": "recurrent_stack_halting_flag",
            "recurrent_halting_option": "recurrent_halting_option",
            "recurrent_halting_threshold": "recurrent_halting_threshold",
            "recurrent_halting_dropout": "recurrent_halting_dropout",
            "recurrent_halting_hidden_state_mode": "recurrent_halting_hidden_state_mode",
        },
    )
    updates["recurrent_gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        "recurrent_gate_stack",
        provided=options.recurrent_gate_stack_source,
    )
    updates["recurrent_halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
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
    options = provided or config_defaults.experts_layer_controller_options(
        config_module, config_defaults.ExpertControlRole.EXPERT
    )
    updates = _pop_updates(
        kwargs,
        {
            "expert_stack_gate_flag": "stack_gate_flag",
            "expert_gate_option": "gate_option",
            "expert_gate_activation": "gate_activation",
            "expert_stack_halting_flag": "stack_halting_flag",
            "expert_halting_option": "halting_option",
            "expert_halting_threshold": "halting_threshold",
            "expert_halting_dropout": "halting_dropout",
            "expert_halting_hidden_state_mode": "halting_hidden_state_mode",
            "expert_halting_output_dim": "halting_output_dim",
        },
    )
    updates["gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs, "expert_gate_stack", provided=options.gate_stack_source
    )
    updates["halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
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
    options = provided or config_defaults.experts_dynamic_memory_options(
        config_module, config_defaults.ExpertControlRole.EXPERT
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
    options = provided or config_defaults.experts_recurrent_controller_options(
        config_module, config_defaults.ExpertControlRole.EXPERT
    )
    updates = _pop_updates(
        kwargs,
        {
            "expert_recurrent_flag": "recurrent_flag",
            "expert_recurrent_max_steps": "recurrent_max_steps",
            "expert_recurrent_layer_norm_position": "recurrent_layer_norm_position",
            "expert_recurrent_normalization": "recurrent_normalization",
            "expert_recurrent_stack_gate_flag": "recurrent_stack_gate_flag",
            "expert_recurrent_gate_option": "recurrent_gate_option",
            "expert_recurrent_gate_activation": "recurrent_gate_activation",
            "expert_recurrent_stack_halting_flag": "recurrent_stack_halting_flag",
            "expert_recurrent_halting_option": "recurrent_halting_option",
            "expert_recurrent_halting_threshold": "recurrent_halting_threshold",
            "expert_recurrent_halting_dropout": "recurrent_halting_dropout",
            "expert_recurrent_halting_hidden_state_mode": "recurrent_halting_hidden_state_mode",
        },
    )
    updates["recurrent_gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        "expert_recurrent_gate_stack",
        provided=options.recurrent_gate_stack_source,
    )
    updates["recurrent_halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
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
    options = provided or config_defaults.experts_layer_controller_options(
        config_module, config_defaults.ExpertControlRole.ROUTER
    )
    updates = _pop_updates(
        kwargs,
        {
            "router_stack_gate_flag": "stack_gate_flag",
            "router_gate_option": "gate_option",
            "router_gate_activation": "gate_activation",
            "router_stack_halting_flag": "stack_halting_flag",
            "router_halting_option": "halting_option",
            "router_halting_threshold": "halting_threshold",
            "router_halting_dropout": "halting_dropout",
            "router_halting_hidden_state_mode": "halting_hidden_state_mode",
            "router_halting_output_dim": "halting_output_dim",
        },
    )
    updates["gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs, "router_gate_stack", provided=options.gate_stack_source
    )
    updates["halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
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
    options = provided or config_defaults.experts_dynamic_memory_options(
        config_module, config_defaults.ExpertControlRole.ROUTER
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
    options = provided or config_defaults.experts_recurrent_controller_options(
        config_module, config_defaults.ExpertControlRole.ROUTER
    )
    updates = _pop_updates(
        kwargs,
        {
            "router_recurrent_flag": "recurrent_flag",
            "router_recurrent_max_steps": "recurrent_max_steps",
            "router_recurrent_layer_norm_position": "recurrent_layer_norm_position",
            "router_recurrent_normalization": "recurrent_normalization",
            "router_recurrent_stack_gate_flag": "recurrent_stack_gate_flag",
            "router_recurrent_gate_option": "recurrent_gate_option",
            "router_recurrent_gate_activation": "recurrent_gate_activation",
            "router_recurrent_stack_halting_flag": "recurrent_stack_halting_flag",
            "router_recurrent_halting_option": "recurrent_halting_option",
            "router_recurrent_halting_threshold": "recurrent_halting_threshold",
            "router_recurrent_halting_dropout": "recurrent_halting_dropout",
            "router_recurrent_halting_hidden_state_mode": "recurrent_halting_hidden_state_mode",
        },
    )
    updates["recurrent_gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        "router_recurrent_gate_stack",
        provided=options.recurrent_gate_stack_source,
    )
    updates["recurrent_halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        "router_recurrent_halting_stack",
        provided=options.recurrent_halting_stack_source,
    )
    return replace(options, **updates)
