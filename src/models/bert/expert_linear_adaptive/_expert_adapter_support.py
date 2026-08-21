from __future__ import annotations

from dataclasses import dataclass, replace
from types import ModuleType

from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions
from models.bert.expert_linear_adaptive import runtime_options as expert_options
from models.bert.expert_linear_adaptive._flat_updates import (
    pop_updates as _pop_updates,
)
from models.bert.expert_linear_adaptive._flat_updates import (
    replace_prefixed_fields as _replace_prefixed_fields,
)

_SUBMODULE_STACK_FIELD_MAP = {
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
_CONTROLLER_STACK_FIELD_MAP = {
    "independent_flag": "independent_flag",
    **_SUBMODULE_STACK_FIELD_MAP,
}


def _role_stack_options_from_kwargs(
    kwargs: dict[str, object],
    prefix: str,
    *,
    defaults: expert_options.ExpertsSubmoduleStackOptions,
    provided: expert_options.ExpertsSubmoduleStackOptions | None,
    extra_mapping: dict[str, str] | None = None,
    default_overrides: dict[str, object] | None = None,
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
        apply_output_postprocessing_flag=config_module.ROUTER_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
        activation=config_module.ROUTER_STACK_ACTIVATION,
        layer_norm_position=config_module.ROUTER_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config_module.ROUTER_STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config_module.ROUTER_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config_module.ROUTER_STACK_DROPOUT_PROBABILITY,
        bias_flag=config_module.ROUTER_BIAS_FLAG,
    )


def _mixture_options_from_kwargs(
    kwargs: dict[str, object],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsMixtureOptions | None,
) -> expert_options.ExpertsMixtureOptions:
    options = provided or expert_options.ExpertsMixtureOptions(
        top_k=config_module.TOP_K,
        num_experts=config_module.NUM_EXPERTS,
        capacity_factor=config_module.CAPACITY_FACTOR,
        dropped_token_behavior=config_module.DROPPED_TOKEN_BEHAVIOR,
        compute_expert_mixture_flag=config_module.COMPUTE_EXPERT_MIXTURE_FLAG,
        weighted_parameters_flag=config_module.WEIGHTED_PARAMETERS_FLAG,
        weighting_position_option=config_module.WEIGHTING_POSITION_OPTION,
        routing_initialization_mode=config_module.ROUTING_INITIALIZATION_MODE,
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
    kwargs: dict[str, object],
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
    kwargs: dict[str, object],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsRouterOptions | None,
) -> expert_options.ExpertsRouterOptions:
    options = provided or expert_options.ExpertsRouterOptions(
        noisy_topk_flag=config_module.ROUTER_NOISY_TOPK_FLAG
    )
    updates = _pop_updates(kwargs, {"router_noisy_topk_flag": "noisy_topk_flag"})
    return replace(options, **updates) if updates else options


def _controller_stack_source(
    *,
    independent_flag: bool,
    hidden_dim: int | None,
    num_layers: int | None,
    last_layer_bias_option: LastLayerBiasOptions | None,
    apply_output_postprocessing_flag: bool | None,
    activation: ActivationOptions | None,
    layer_norm_position: LayerNormPositionOptions | None,
    residual_connection_option: type[ResidualConfig] | None,
    residual_model_flag: bool,
    dropout_probability: float | None,
    bias_flag: bool | None,
) -> expert_options.ExpertsSubmoduleStackSource:
    return expert_options.ExpertsSubmoduleStackSource(
        independent_flag=independent_flag,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=last_layer_bias_option,
        apply_output_postprocessing_flag=apply_output_postprocessing_flag,
        activation=activation,
        layer_norm_position=layer_norm_position,
        residual_connection_option=residual_connection_option,
        residual_model_flag=residual_model_flag,
        dropout_probability=dropout_probability,
        bias_flag=bias_flag,
    )


def _gate_stack_source(
    config_module: ModuleType,
) -> expert_options.ExpertsSubmoduleStackSource:
    return _controller_stack_source(
        independent_flag=config_module.GATE_STACK_INDEPENDENT_FLAG,
        hidden_dim=config_module.GATE_STACK_HIDDEN_DIM,
        num_layers=config_module.GATE_STACK_NUM_LAYERS,
        last_layer_bias_option=config_module.GATE_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_postprocessing_flag=config_module.GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
        activation=config_module.GATE_STACK_ACTIVATION,
        layer_norm_position=config_module.GATE_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config_module.GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config_module.GATE_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config_module.GATE_STACK_DROPOUT_PROBABILITY,
        bias_flag=config_module.GATE_STACK_BIAS_FLAG,
    )


def _halting_stack_source(
    config_module: ModuleType,
) -> expert_options.ExpertsSubmoduleStackSource:
    return _controller_stack_source(
        independent_flag=config_module.HALTING_STACK_INDEPENDENT_FLAG,
        hidden_dim=config_module.HALTING_STACK_HIDDEN_DIM,
        num_layers=config_module.HALTING_STACK_NUM_LAYERS,
        last_layer_bias_option=config_module.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_postprocessing_flag=(
            config_module.HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
        ),
        activation=config_module.HALTING_STACK_ACTIVATION,
        layer_norm_position=config_module.HALTING_STACK_LAYER_NORM_POSITION,
        residual_connection_option=(
            config_module.HALTING_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_model_flag=config_module.HALTING_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config_module.HALTING_STACK_DROPOUT_PROBABILITY,
        bias_flag=config_module.HALTING_STACK_BIAS_FLAG,
    )


def _memory_stack_source(
    config_module: ModuleType,
) -> expert_options.ExpertsSubmoduleStackSource:
    return _controller_stack_source(
        independent_flag=config_module.MEMORY_STACK_INDEPENDENT_FLAG,
        hidden_dim=config_module.MEMORY_STACK_HIDDEN_DIM,
        num_layers=config_module.MEMORY_STACK_NUM_LAYERS,
        last_layer_bias_option=config_module.MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_postprocessing_flag=(
            config_module.MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
        ),
        activation=config_module.MEMORY_STACK_ACTIVATION,
        layer_norm_position=config_module.MEMORY_STACK_LAYER_NORM_POSITION,
        residual_connection_option=(
            config_module.MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_model_flag=config_module.MEMORY_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config_module.MEMORY_STACK_DROPOUT_PROBABILITY,
        bias_flag=config_module.MEMORY_STACK_BIAS_FLAG,
    )


def _recurrent_gate_stack_source(
    config_module: ModuleType,
) -> expert_options.ExpertsSubmoduleStackSource:
    return _controller_stack_source(
        independent_flag=config_module.RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
        hidden_dim=config_module.RECURRENT_GATE_STACK_HIDDEN_DIM,
        num_layers=config_module.RECURRENT_GATE_STACK_NUM_LAYERS,
        last_layer_bias_option=(
            config_module.RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
        ),
        apply_output_postprocessing_flag=(
            config_module.RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
        ),
        activation=config_module.RECURRENT_GATE_STACK_ACTIVATION,
        layer_norm_position=config_module.RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
        residual_connection_option=(
            config_module.RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_model_flag=config_module.RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config_module.RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
        bias_flag=config_module.RECURRENT_GATE_STACK_BIAS_FLAG,
    )


def _recurrent_halting_stack_source(
    config_module: ModuleType,
) -> expert_options.ExpertsSubmoduleStackSource:
    return _controller_stack_source(
        independent_flag=config_module.RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        hidden_dim=config_module.RECURRENT_HALTING_STACK_HIDDEN_DIM,
        num_layers=config_module.RECURRENT_HALTING_STACK_NUM_LAYERS,
        last_layer_bias_option=(
            config_module.RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
        ),
        apply_output_postprocessing_flag=(
            config_module.RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
        ),
        activation=config_module.RECURRENT_HALTING_STACK_ACTIVATION,
        layer_norm_position=config_module.RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        residual_connection_option=(
            config_module.RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_model_flag=config_module.RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=(config_module.RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY),
        bias_flag=config_module.RECURRENT_HALTING_STACK_BIAS_FLAG,
    )


def _controller_stack_source_from_kwargs(
    kwargs: dict[str, object],
    prefix: str,
    *,
    provided: expert_options.ExpertsSubmoduleStackSource,
) -> expert_options.ExpertsSubmoduleStackSource:
    return _replace_prefixed_fields(
        kwargs,
        provided,
        prefix,
        _CONTROLLER_STACK_FIELD_MAP,
    )


def _layer_controller_options_from_kwargs(
    kwargs: dict[str, object],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsLayerControllerOptions | None,
) -> expert_options.ExpertsLayerControllerOptions:
    options = provided or expert_options.ExpertsLayerControllerOptions(
        stack_gate_flag=config_module.STACK_GATE_FLAG,
        gate_option=config_module.GATE_OPTION,
        gate_activation=config_module.GATE_ACTIVATION,
        gate_stack_source=_gate_stack_source(config_module),
        stack_halting_flag=config_module.STACK_HALTING_FLAG,
        halting_option=config_module.HALTING_OPTION,
        halting_threshold=config_module.HALTING_THRESHOLD,
        halting_dropout=config_module.HALTING_DROPOUT,
        halting_hidden_state_mode=config_module.HALTING_HIDDEN_STATE_MODE,
        halting_stack_source=_halting_stack_source(config_module),
        halting_output_dim=config_module.HALTING_OUTPUT_DIM,
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
    kwargs: dict[str, object],
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
        memory_stack_source=_memory_stack_source(config_module),
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
    kwargs: dict[str, object],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsRecurrentControllerOptions | None,
) -> expert_options.ExpertsRecurrentControllerOptions:
    options = provided or expert_options.ExpertsRecurrentControllerOptions(
        recurrent_flag=config_module.RECURRENT_FLAG,
        recurrent_max_steps=config_module.RECURRENT_MAX_STEPS,
        recurrent_initial_iterations=config_module.RECURRENT_INITIAL_ITERATIONS,
        recurrent_gradient_transition_count=config_module.RECURRENT_GRADIENT_TRANSITION_COUNT,
        recurrent_no_gradient_transition_count=config_module.RECURRENT_NO_GRADIENT_TRANSITION_COUNT,
        recurrent_iteration_increment=config_module.RECURRENT_ITERATION_INCREMENT,
        recurrent_forward_calls_before_iteration_increment=(
            config_module.RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT
        ),
        recurrent_layer_norm_position=config_module.RECURRENT_LAYER_NORM_POSITION,
        recurrent_stack_gate_flag=config_module.RECURRENT_STACK_GATE_FLAG,
        recurrent_gate_option=config_module.RECURRENT_GATE_OPTION,
        recurrent_gate_activation=config_module.RECURRENT_GATE_ACTIVATION,
        recurrent_gate_stack_source=_recurrent_gate_stack_source(config_module),
        recurrent_stack_halting_flag=config_module.RECURRENT_STACK_HALTING_FLAG,
        recurrent_halting_option=config_module.RECURRENT_HALTING_OPTION,
        recurrent_halting_threshold=config_module.RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout=config_module.RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode=config_module.RECURRENT_HALTING_HIDDEN_STATE_MODE,
        recurrent_halting_stack_source=_recurrent_halting_stack_source(config_module),
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


@dataclass(frozen=True, slots=True)
class _LayerStackSources:
    gate: expert_options.ExpertsSubmoduleStackSource
    halting: expert_options.ExpertsSubmoduleStackSource


@dataclass(frozen=True, slots=True)
class _RecurrentStackSources:
    gate: expert_options.ExpertsSubmoduleStackSource
    halting: expert_options.ExpertsSubmoduleStackSource


@dataclass(frozen=True, slots=True)
class _LayerControllerDefaults:
    stack_gate_flag: bool
    gate_option: LayerGateOptions | None
    gate_activation: ActivationOptions | None
    stack_sources: _LayerStackSources
    stack_halting_flag: bool
    halting_option: type[HaltingConfig]
    halting_threshold: float
    halting_dropout: float
    halting_hidden_state_mode: HaltingHiddenStateModeOptions
    halting_output_dim: int


@dataclass(frozen=True, slots=True)
class _DynamicMemoryDefaults:
    memory_flag: bool
    memory_option: type[DynamicMemoryConfig]
    memory_position_option: MemoryPositionOptions
    learning_rate: float | None
    num_inner_steps: int | None
    stack_source: expert_options.ExpertsSubmoduleStackSource


@dataclass(frozen=True, slots=True)
class _RecurrentControllerDefaults:
    recurrent_flag: bool
    recurrent_max_steps: int
    recurrent_layer_norm_position: LayerNormPositionOptions
    recurrent_stack_gate_flag: bool
    recurrent_gate_option: LayerGateOptions | None
    recurrent_gate_activation: ActivationOptions | None
    stack_sources: _RecurrentStackSources
    recurrent_stack_halting_flag: bool
    recurrent_halting_option: type[HaltingConfig]
    recurrent_halting_threshold: float
    recurrent_halting_dropout: float
    recurrent_halting_hidden_state_mode: HaltingHiddenStateModeOptions


def _layer_controller_options(
    defaults: _LayerControllerDefaults,
) -> expert_options.ExpertsLayerControllerOptions:
    return expert_options.ExpertsLayerControllerOptions(
        stack_gate_flag=defaults.stack_gate_flag,
        gate_option=defaults.gate_option,
        gate_activation=defaults.gate_activation,
        gate_stack_source=defaults.stack_sources.gate,
        stack_halting_flag=defaults.stack_halting_flag,
        halting_option=defaults.halting_option,
        halting_threshold=defaults.halting_threshold,
        halting_dropout=defaults.halting_dropout,
        halting_hidden_state_mode=defaults.halting_hidden_state_mode,
        halting_stack_source=defaults.stack_sources.halting,
        halting_output_dim=defaults.halting_output_dim,
    )


def _dynamic_memory_options(
    defaults: _DynamicMemoryDefaults,
) -> expert_options.ExpertsDynamicMemoryOptions:
    return expert_options.ExpertsDynamicMemoryOptions(
        memory_flag=defaults.memory_flag,
        memory_option=defaults.memory_option,
        memory_position_option=defaults.memory_position_option,
        memory_test_time_training_learning_rate=defaults.learning_rate,
        memory_test_time_training_num_inner_steps=defaults.num_inner_steps,
        memory_stack_source=defaults.stack_source,
    )


def _recurrent_controller_options(
    defaults: _RecurrentControllerDefaults,
) -> expert_options.ExpertsRecurrentControllerOptions:
    return expert_options.ExpertsRecurrentControllerOptions(
        recurrent_flag=defaults.recurrent_flag,
        recurrent_max_steps=defaults.recurrent_max_steps,
        recurrent_layer_norm_position=defaults.recurrent_layer_norm_position,
        recurrent_stack_gate_flag=defaults.recurrent_stack_gate_flag,
        recurrent_gate_option=defaults.recurrent_gate_option,
        recurrent_gate_activation=defaults.recurrent_gate_activation,
        recurrent_gate_stack_source=defaults.stack_sources.gate,
        recurrent_stack_halting_flag=defaults.recurrent_stack_halting_flag,
        recurrent_halting_option=defaults.recurrent_halting_option,
        recurrent_halting_threshold=defaults.recurrent_halting_threshold,
        recurrent_halting_dropout=defaults.recurrent_halting_dropout,
        recurrent_halting_hidden_state_mode=(
            defaults.recurrent_halting_hidden_state_mode
        ),
        recurrent_halting_stack_source=defaults.stack_sources.halting,
    )


def _role_layer_controller_options_from_kwargs(
    kwargs: dict[str, object],
    *,
    role_prefix: str,
    gate_stack_prefix: str,
    halting_stack_prefix: str,
    options: expert_options.ExpertsLayerControllerOptions,
) -> expert_options.ExpertsLayerControllerOptions:
    updates = _pop_updates(
        kwargs,
        {
            f"{role_prefix}_stack_gate_flag": "stack_gate_flag",
            f"{role_prefix}_gate_option": "gate_option",
            f"{role_prefix}_gate_activation": "gate_activation",
            f"{role_prefix}_stack_halting_flag": "stack_halting_flag",
            f"{role_prefix}_halting_option": "halting_option",
            f"{role_prefix}_halting_threshold": "halting_threshold",
            f"{role_prefix}_halting_dropout": "halting_dropout",
            f"{role_prefix}_halting_hidden_state_mode": "halting_hidden_state_mode",
            f"{role_prefix}_halting_output_dim": "halting_output_dim",
        },
    )
    updates["gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        gate_stack_prefix,
        provided=options.gate_stack_source,
    )
    updates["halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        halting_stack_prefix,
        provided=options.halting_stack_source,
    )
    return replace(options, **updates)


def _role_dynamic_memory_options_from_kwargs(
    kwargs: dict[str, object],
    *,
    role_prefix: str,
    memory_stack_prefix: str,
    options: expert_options.ExpertsDynamicMemoryOptions,
) -> expert_options.ExpertsDynamicMemoryOptions:
    updates = _pop_updates(
        kwargs,
        {
            f"{role_prefix}_memory_flag": "memory_flag",
            f"{role_prefix}_memory_option": "memory_option",
            f"{role_prefix}_memory_position_option": "memory_position_option",
            f"{role_prefix}_memory_test_time_training_learning_rate": (
                "memory_test_time_training_learning_rate"
            ),
            f"{role_prefix}_memory_test_time_training_num_inner_steps": (
                "memory_test_time_training_num_inner_steps"
            ),
        },
    )
    updates["memory_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        memory_stack_prefix,
        provided=options.memory_stack_source,
    )
    return replace(options, **updates)


def _role_recurrent_controller_options_from_kwargs(
    kwargs: dict[str, object],
    *,
    role_prefix: str,
    gate_stack_prefix: str,
    halting_stack_prefix: str,
    options: expert_options.ExpertsRecurrentControllerOptions,
) -> expert_options.ExpertsRecurrentControllerOptions:
    updates = _pop_updates(
        kwargs,
        {
            f"{role_prefix}_recurrent_flag": "recurrent_flag",
            f"{role_prefix}_recurrent_max_steps": "recurrent_max_steps",
            f"{role_prefix}_recurrent_layer_norm_position": (
                "recurrent_layer_norm_position"
            ),
            f"{role_prefix}_recurrent_stack_gate_flag": "recurrent_stack_gate_flag",
            f"{role_prefix}_recurrent_gate_option": "recurrent_gate_option",
            f"{role_prefix}_recurrent_gate_activation": "recurrent_gate_activation",
            f"{role_prefix}_recurrent_stack_halting_flag": (
                "recurrent_stack_halting_flag"
            ),
            f"{role_prefix}_recurrent_halting_option": "recurrent_halting_option",
            f"{role_prefix}_recurrent_halting_threshold": (
                "recurrent_halting_threshold"
            ),
            f"{role_prefix}_recurrent_halting_dropout": "recurrent_halting_dropout",
            f"{role_prefix}_recurrent_halting_hidden_state_mode": (
                "recurrent_halting_hidden_state_mode"
            ),
        },
    )
    updates["recurrent_gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        gate_stack_prefix,
        provided=options.recurrent_gate_stack_source,
    )
    updates["recurrent_halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        halting_stack_prefix,
        provided=options.recurrent_halting_stack_source,
    )
    return replace(options, **updates)


def _expert_layer_stack_sources(config_module: ModuleType) -> _LayerStackSources:
    return _LayerStackSources(
        gate=_controller_stack_source(
            independent_flag=config_module.EXPERT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config_module.EXPERT_GATE_STACK_HIDDEN_DIM,
            num_layers=config_module.EXPERT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config_module.EXPERT_GATE_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config_module.EXPERT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config_module.EXPERT_GATE_STACK_ACTIVATION,
            layer_norm_position=config_module.EXPERT_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config_module.EXPERT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config_module.EXPERT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config_module.EXPERT_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config_module.EXPERT_GATE_STACK_BIAS_FLAG,
        ),
        halting=_controller_stack_source(
            independent_flag=config_module.EXPERT_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config_module.EXPERT_HALTING_STACK_HIDDEN_DIM,
            num_layers=config_module.EXPERT_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config_module.EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config_module.EXPERT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config_module.EXPERT_HALTING_STACK_ACTIVATION,
            layer_norm_position=(
                config_module.EXPERT_HALTING_STACK_LAYER_NORM_POSITION
            ),
            residual_connection_option=(
                config_module.EXPERT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=(
                config_module.EXPERT_HALTING_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config_module.EXPERT_HALTING_STACK_DROPOUT_PROBABILITY
            ),
            bias_flag=config_module.EXPERT_HALTING_STACK_BIAS_FLAG,
        ),
    )


def _expert_memory_stack_source(
    config_module: ModuleType,
) -> expert_options.ExpertsSubmoduleStackSource:
    return _controller_stack_source(
        independent_flag=config_module.EXPERT_MEMORY_STACK_INDEPENDENT_FLAG,
        hidden_dim=config_module.EXPERT_MEMORY_STACK_HIDDEN_DIM,
        num_layers=config_module.EXPERT_MEMORY_STACK_NUM_LAYERS,
        last_layer_bias_option=config_module.EXPERT_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_postprocessing_flag=(
            config_module.EXPERT_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
        ),
        activation=config_module.EXPERT_MEMORY_STACK_ACTIVATION,
        layer_norm_position=config_module.EXPERT_MEMORY_STACK_LAYER_NORM_POSITION,
        residual_connection_option=(
            config_module.EXPERT_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_model_flag=config_module.EXPERT_MEMORY_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config_module.EXPERT_MEMORY_STACK_DROPOUT_PROBABILITY,
        bias_flag=config_module.EXPERT_MEMORY_STACK_BIAS_FLAG,
    )


def _expert_recurrent_stack_sources(
    config_module: ModuleType,
) -> _RecurrentStackSources:
    return _RecurrentStackSources(
        gate=_controller_stack_source(
            independent_flag=(
                config_module.EXPERT_RECURRENT_GATE_STACK_INDEPENDENT_FLAG
            ),
            hidden_dim=config_module.EXPERT_RECURRENT_GATE_STACK_HIDDEN_DIM,
            num_layers=config_module.EXPERT_RECURRENT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config_module.EXPERT_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config_module.EXPERT_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config_module.EXPERT_RECURRENT_GATE_STACK_ACTIVATION,
            layer_norm_position=(
                config_module.EXPERT_RECURRENT_GATE_STACK_LAYER_NORM_POSITION
            ),
            residual_connection_option=(
                config_module.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=(
                config_module.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config_module.EXPERT_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY
            ),
            bias_flag=config_module.EXPERT_RECURRENT_GATE_STACK_BIAS_FLAG,
        ),
        halting=_controller_stack_source(
            independent_flag=(
                config_module.EXPERT_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG
            ),
            hidden_dim=config_module.EXPERT_RECURRENT_HALTING_STACK_HIDDEN_DIM,
            num_layers=config_module.EXPERT_RECURRENT_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config_module.EXPERT_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config_module.EXPERT_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config_module.EXPERT_RECURRENT_HALTING_STACK_ACTIVATION,
            layer_norm_position=(
                config_module.EXPERT_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION
            ),
            residual_connection_option=(
                config_module.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=(
                config_module.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config_module.EXPERT_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY
            ),
            bias_flag=config_module.EXPERT_RECURRENT_HALTING_STACK_BIAS_FLAG,
        ),
    )


def _router_layer_stack_sources(config_module: ModuleType) -> _LayerStackSources:
    return _LayerStackSources(
        gate=_controller_stack_source(
            independent_flag=config_module.ROUTER_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config_module.ROUTER_GATE_STACK_HIDDEN_DIM,
            num_layers=config_module.ROUTER_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config_module.ROUTER_GATE_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config_module.ROUTER_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config_module.ROUTER_GATE_STACK_ACTIVATION,
            layer_norm_position=config_module.ROUTER_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=(
                config_module.ROUTER_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=config_module.ROUTER_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config_module.ROUTER_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config_module.ROUTER_GATE_STACK_BIAS_FLAG,
        ),
        halting=_controller_stack_source(
            independent_flag=config_module.ROUTER_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config_module.ROUTER_HALTING_STACK_HIDDEN_DIM,
            num_layers=config_module.ROUTER_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config_module.ROUTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config_module.ROUTER_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config_module.ROUTER_HALTING_STACK_ACTIVATION,
            layer_norm_position=(
                config_module.ROUTER_HALTING_STACK_LAYER_NORM_POSITION
            ),
            residual_connection_option=(
                config_module.ROUTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=(
                config_module.ROUTER_HALTING_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config_module.ROUTER_HALTING_STACK_DROPOUT_PROBABILITY
            ),
            bias_flag=config_module.ROUTER_HALTING_STACK_BIAS_FLAG,
        ),
    )


def _router_memory_stack_source(
    config_module: ModuleType,
) -> expert_options.ExpertsSubmoduleStackSource:
    return _controller_stack_source(
        independent_flag=config_module.ROUTER_MEMORY_STACK_INDEPENDENT_FLAG,
        hidden_dim=config_module.ROUTER_MEMORY_STACK_HIDDEN_DIM,
        num_layers=config_module.ROUTER_MEMORY_STACK_NUM_LAYERS,
        last_layer_bias_option=config_module.ROUTER_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_postprocessing_flag=(
            config_module.ROUTER_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
        ),
        activation=config_module.ROUTER_MEMORY_STACK_ACTIVATION,
        layer_norm_position=config_module.ROUTER_MEMORY_STACK_LAYER_NORM_POSITION,
        residual_connection_option=(
            config_module.ROUTER_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        residual_model_flag=config_module.ROUTER_MEMORY_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config_module.ROUTER_MEMORY_STACK_DROPOUT_PROBABILITY,
        bias_flag=config_module.ROUTER_MEMORY_STACK_BIAS_FLAG,
    )


def _router_recurrent_stack_sources(
    config_module: ModuleType,
) -> _RecurrentStackSources:
    return _RecurrentStackSources(
        gate=_controller_stack_source(
            independent_flag=(
                config_module.ROUTER_RECURRENT_GATE_STACK_INDEPENDENT_FLAG
            ),
            hidden_dim=config_module.ROUTER_RECURRENT_GATE_STACK_HIDDEN_DIM,
            num_layers=config_module.ROUTER_RECURRENT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config_module.ROUTER_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config_module.ROUTER_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config_module.ROUTER_RECURRENT_GATE_STACK_ACTIVATION,
            layer_norm_position=(
                config_module.ROUTER_RECURRENT_GATE_STACK_LAYER_NORM_POSITION
            ),
            residual_connection_option=(
                config_module.ROUTER_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=(
                config_module.ROUTER_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config_module.ROUTER_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY
            ),
            bias_flag=config_module.ROUTER_RECURRENT_GATE_STACK_BIAS_FLAG,
        ),
        halting=_controller_stack_source(
            independent_flag=(
                config_module.ROUTER_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG
            ),
            hidden_dim=config_module.ROUTER_RECURRENT_HALTING_STACK_HIDDEN_DIM,
            num_layers=config_module.ROUTER_RECURRENT_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=(
                config_module.ROUTER_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_postprocessing_flag=(
                config_module.ROUTER_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
            ),
            activation=config_module.ROUTER_RECURRENT_HALTING_STACK_ACTIVATION,
            layer_norm_position=(
                config_module.ROUTER_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION
            ),
            residual_connection_option=(
                config_module.ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            residual_model_flag=(
                config_module.ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG
            ),
            dropout_probability=(
                config_module.ROUTER_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY
            ),
            bias_flag=config_module.ROUTER_RECURRENT_HALTING_STACK_BIAS_FLAG,
        ),
    )


def _expert_layer_controller_options_from_kwargs(
    kwargs: dict[str, object],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsLayerControllerOptions | None,
) -> expert_options.ExpertsLayerControllerOptions:
    if provided is None:
        provided = _layer_controller_options(
            _LayerControllerDefaults(
                stack_gate_flag=config_module.EXPERT_STACK_GATE_FLAG,
                gate_option=config_module.EXPERT_GATE_OPTION,
                gate_activation=config_module.EXPERT_GATE_ACTIVATION,
                stack_sources=_expert_layer_stack_sources(config_module),
                stack_halting_flag=config_module.EXPERT_STACK_HALTING_FLAG,
                halting_option=config_module.EXPERT_HALTING_OPTION,
                halting_threshold=config_module.EXPERT_HALTING_THRESHOLD,
                halting_dropout=config_module.EXPERT_HALTING_DROPOUT,
                halting_hidden_state_mode=(
                    config_module.EXPERT_HALTING_HIDDEN_STATE_MODE
                ),
                halting_output_dim=config_module.EXPERT_HALTING_OUTPUT_DIM,
            )
        )
    return _role_layer_controller_options_from_kwargs(
        kwargs,
        role_prefix="expert",
        gate_stack_prefix="expert_gate_stack",
        halting_stack_prefix="expert_halting_stack",
        options=provided,
    )


def _expert_dynamic_memory_options_from_kwargs(
    kwargs: dict[str, object],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsDynamicMemoryOptions | None,
) -> expert_options.ExpertsDynamicMemoryOptions:
    if provided is None:
        provided = _dynamic_memory_options(
            _DynamicMemoryDefaults(
                memory_flag=config_module.EXPERT_MEMORY_FLAG,
                memory_option=config_module.EXPERT_MEMORY_OPTION,
                memory_position_option=config_module.EXPERT_MEMORY_POSITION_OPTION,
                learning_rate=(
                    config_module.EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
                ),
                num_inner_steps=(
                    config_module.EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
                ),
                stack_source=_expert_memory_stack_source(config_module),
            )
        )
    return _role_dynamic_memory_options_from_kwargs(
        kwargs,
        role_prefix="expert",
        memory_stack_prefix="expert_memory_stack",
        options=provided,
    )


def _expert_recurrent_controller_options_from_kwargs(
    kwargs: dict[str, object],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsRecurrentControllerOptions | None,
) -> expert_options.ExpertsRecurrentControllerOptions:
    if provided is None:
        provided = _recurrent_controller_options(
            _RecurrentControllerDefaults(
                recurrent_flag=config_module.EXPERT_RECURRENT_FLAG,
                recurrent_max_steps=config_module.EXPERT_RECURRENT_MAX_STEPS,
                recurrent_layer_norm_position=(
                    config_module.EXPERT_RECURRENT_LAYER_NORM_POSITION
                ),
                recurrent_stack_gate_flag=(
                    config_module.EXPERT_RECURRENT_STACK_GATE_FLAG
                ),
                recurrent_gate_option=config_module.EXPERT_RECURRENT_GATE_OPTION,
                recurrent_gate_activation=(
                    config_module.EXPERT_RECURRENT_GATE_ACTIVATION
                ),
                stack_sources=_expert_recurrent_stack_sources(config_module),
                recurrent_stack_halting_flag=(
                    config_module.EXPERT_RECURRENT_STACK_HALTING_FLAG
                ),
                recurrent_halting_option=(
                    config_module.EXPERT_RECURRENT_HALTING_OPTION
                ),
                recurrent_halting_threshold=(
                    config_module.EXPERT_RECURRENT_HALTING_THRESHOLD
                ),
                recurrent_halting_dropout=(
                    config_module.EXPERT_RECURRENT_HALTING_DROPOUT
                ),
                recurrent_halting_hidden_state_mode=(
                    config_module.EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE
                ),
            )
        )
    return _role_recurrent_controller_options_from_kwargs(
        kwargs,
        role_prefix="expert",
        gate_stack_prefix="expert_recurrent_gate_stack",
        halting_stack_prefix="expert_recurrent_halting_stack",
        options=provided,
    )


def _router_layer_controller_options_from_kwargs(
    kwargs: dict[str, object],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsLayerControllerOptions | None,
) -> expert_options.ExpertsLayerControllerOptions:
    if provided is None:
        provided = _layer_controller_options(
            _LayerControllerDefaults(
                stack_gate_flag=config_module.ROUTER_STACK_GATE_FLAG,
                gate_option=config_module.ROUTER_GATE_OPTION,
                gate_activation=config_module.ROUTER_GATE_ACTIVATION,
                stack_sources=_router_layer_stack_sources(config_module),
                stack_halting_flag=config_module.ROUTER_STACK_HALTING_FLAG,
                halting_option=config_module.ROUTER_HALTING_OPTION,
                halting_threshold=config_module.ROUTER_HALTING_THRESHOLD,
                halting_dropout=config_module.ROUTER_HALTING_DROPOUT,
                halting_hidden_state_mode=(
                    config_module.ROUTER_HALTING_HIDDEN_STATE_MODE
                ),
                halting_output_dim=config_module.ROUTER_HALTING_OUTPUT_DIM,
            )
        )
    return _role_layer_controller_options_from_kwargs(
        kwargs,
        role_prefix="router",
        gate_stack_prefix="router_gate_stack",
        halting_stack_prefix="router_halting_stack",
        options=provided,
    )


def _router_dynamic_memory_options_from_kwargs(
    kwargs: dict[str, object],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsDynamicMemoryOptions | None,
) -> expert_options.ExpertsDynamicMemoryOptions:
    if provided is None:
        provided = _dynamic_memory_options(
            _DynamicMemoryDefaults(
                memory_flag=config_module.ROUTER_MEMORY_FLAG,
                memory_option=config_module.ROUTER_MEMORY_OPTION,
                memory_position_option=config_module.ROUTER_MEMORY_POSITION_OPTION,
                learning_rate=(
                    config_module.ROUTER_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
                ),
                num_inner_steps=(
                    config_module.ROUTER_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
                ),
                stack_source=_router_memory_stack_source(config_module),
            )
        )
    return _role_dynamic_memory_options_from_kwargs(
        kwargs,
        role_prefix="router",
        memory_stack_prefix="router_memory_stack",
        options=provided,
    )


def _router_recurrent_controller_options_from_kwargs(
    kwargs: dict[str, object],
    config_module: ModuleType,
    *,
    provided: expert_options.ExpertsRecurrentControllerOptions | None,
) -> expert_options.ExpertsRecurrentControllerOptions:
    if provided is None:
        provided = _recurrent_controller_options(
            _RecurrentControllerDefaults(
                recurrent_flag=config_module.ROUTER_RECURRENT_FLAG,
                recurrent_max_steps=config_module.ROUTER_RECURRENT_MAX_STEPS,
                recurrent_layer_norm_position=(
                    config_module.ROUTER_RECURRENT_LAYER_NORM_POSITION
                ),
                recurrent_stack_gate_flag=(
                    config_module.ROUTER_RECURRENT_STACK_GATE_FLAG
                ),
                recurrent_gate_option=config_module.ROUTER_RECURRENT_GATE_OPTION,
                recurrent_gate_activation=(
                    config_module.ROUTER_RECURRENT_GATE_ACTIVATION
                ),
                stack_sources=_router_recurrent_stack_sources(config_module),
                recurrent_stack_halting_flag=(
                    config_module.ROUTER_RECURRENT_STACK_HALTING_FLAG
                ),
                recurrent_halting_option=(
                    config_module.ROUTER_RECURRENT_HALTING_OPTION
                ),
                recurrent_halting_threshold=(
                    config_module.ROUTER_RECURRENT_HALTING_THRESHOLD
                ),
                recurrent_halting_dropout=(
                    config_module.ROUTER_RECURRENT_HALTING_DROPOUT
                ),
                recurrent_halting_hidden_state_mode=(
                    config_module.ROUTER_RECURRENT_HALTING_HIDDEN_STATE_MODE
                ),
            )
        )
    return _role_recurrent_controller_options_from_kwargs(
        kwargs,
        role_prefix="router",
        gate_stack_prefix="router_recurrent_gate_stack",
        halting_stack_prefix="router_recurrent_halting_stack",
        options=provided,
    )
