from dataclasses import fields, replace
from types import ModuleType
from typing import Any, Final

import models.neuron.expert_linear.config as config
from models.neuron.expert_linear._hidden import runtime_options as options

_TOP_LEVEL_KEYS = ("batch_size", "learning_rate", "input_dim", "output_dim")
_STACK_FIELDS = {
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


def builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any],
    config_module: ModuleType,
) -> dict[str, Any]:
    """Atomically translate flat CLI/search values into local option groups."""

    kwargs = dict(flat_kwargs)
    result = {key: kwargs.pop(key) for key in _TOP_LEVEL_KEYS if key in kwargs}
    stack = _stack_options(kwargs, config_module, kwargs.pop("stack_options", None))
    submodule = _submodule_stack_options(
        kwargs,
        config_module,
        kwargs.pop("submodule_stack_options", None),
    )
    result.update(
        stack_options=stack,
        submodule_stack_options=submodule,
        mixture_options=_mixture_options(
            kwargs, config_module, kwargs.pop("mixture_options", None)
        ),
        expert_stack_options=_role_stack_options(
            kwargs,
            "expert_stack",
            submodule,
            kwargs.pop("expert_stack_options", None),
            extra={"expert_bias_flag": "bias_flag"},
            defaults={
                "layer_norm_position": config_module.EXPERT_STACK_LAYER_NORM_POSITION,
                "apply_output_pipeline_flag": (
                    config_module.EXPERT_STACK_APPLY_OUTPUT_PIPELINE_FLAG
                ),
            },
        ),
        sampler_options=_sampler_options(
            kwargs, config_module, kwargs.pop("sampler_options", None)
        ),
        router_options=_router_options(
            kwargs, config_module, kwargs.pop("router_options", None)
        ),
        router_stack_options=_role_stack_options(
            kwargs,
            "router_stack",
            _router_stack_defaults(config_module),
            kwargs.pop("router_stack_options", None),
            extra={"router_bias_flag": "bias_flag"},
        ),
        layer_controller_options=_layer_controller_options(
            kwargs,
            config_module,
            kwargs.pop("layer_controller_options", None),
            flat_prefix="",
            config_prefix="",
        ),
        dynamic_memory_options=_memory_options(
            kwargs,
            config_module,
            kwargs.pop("dynamic_memory_options", None),
            flat_prefix="",
            config_prefix="",
        ),
        recurrent_controller_options=_recurrent_options(
            kwargs,
            config_module,
            kwargs.pop("recurrent_controller_options", None),
            flat_prefix="",
            config_prefix="",
        ),
        expert_layer_controller_options=_layer_controller_options(
            kwargs,
            config_module,
            kwargs.pop("expert_layer_controller_options", None),
            flat_prefix="expert_",
            config_prefix="EXPERT_",
        ),
        expert_dynamic_memory_options=_memory_options(
            kwargs,
            config_module,
            kwargs.pop("expert_dynamic_memory_options", None),
            flat_prefix="expert_",
            config_prefix="EXPERT_",
        ),
        expert_recurrent_controller_options=_recurrent_options(
            kwargs,
            config_module,
            kwargs.pop("expert_recurrent_controller_options", None),
            flat_prefix="expert_",
            config_prefix="EXPERT_",
        ),
    )
    result.update(kwargs)
    return result


def _stack_options(kwargs, config, provided):
    value = provided or options.ExpertsStackOptions(
        hidden_dim=config.HIDDEN_DIM,
        bias_flag=config.STACK_BIAS_FLAG,
        layer_norm_position=config.STACK_LAYER_NORM_POSITION,
        num_layers=config.STACK_NUM_LAYERS,
        activation=config.STACK_ACTIVATION,
        residual_connection_option=config.STACK_RESIDUAL_CONNECTION_OPTION,
        dropout_probability=config.STACK_DROPOUT_PROBABILITY,
        last_layer_bias_option=config.STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
    )
    updates = _pop_updates(
        kwargs,
        {
            "hidden_dim": "hidden_dim",
            "stack_bias_flag": "bias_flag",
            "layer_norm_position": "layer_norm_position",
            "stack_layer_norm_position": "layer_norm_position",
            "stack_num_layers": "num_layers",
            "stack_activation": "activation",
            "stack_residual_connection_option": "residual_connection_option",
            "stack_dropout_probability": "dropout_probability",
            "stack_last_layer_bias_option": "last_layer_bias_option",
            "stack_apply_output_pipeline_flag": "apply_output_pipeline_flag",
        },
    )
    return replace(value, **updates) if updates else value


def _submodule_stack_options(kwargs, config, provided):
    value = provided or options.ExpertsSubmoduleStackOptions(
        hidden_dim=config.SUBMODULE_STACK_HIDDEN_DIM,
        num_layers=config.SUBMODULE_STACK_NUM_LAYERS,
        last_layer_bias_option=config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.SUBMODULE_STACK_ACTIVATION,
        layer_norm_position=config.SUBMODULE_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION,
        dropout_probability=config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.SUBMODULE_STACK_BIAS_FLAG,
    )
    updates = _pop_updates(
        kwargs,
        {f"submodule_stack_{key}": field for key, field in _STACK_FIELDS.items()},
    )
    return replace(value, **updates) if updates else value


def _role_stack_options(
    kwargs,
    prefix,
    inherited,
    provided,
    *,
    extra=None,
    defaults=None,
):
    value = provided or replace(inherited, **(defaults or {}))
    mapping = {f"{prefix}_{key}": field for key, field in _STACK_FIELDS.items()}
    mapping.update(extra or {})
    updates = _pop_updates(kwargs, mapping)
    return replace(value, **updates) if updates else value


def _router_stack_defaults(config):
    return options.ExpertsSubmoduleStackOptions(
        hidden_dim=config.ROUTER_STACK_HIDDEN_DIM,
        num_layers=config.ROUTER_STACK_NUM_LAYERS,
        last_layer_bias_option=config.ROUTER_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.ROUTER_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.ROUTER_STACK_ACTIVATION,
        layer_norm_position=config.ROUTER_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.ROUTER_STACK_RESIDUAL_CONNECTION_OPTION,
        dropout_probability=config.ROUTER_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.ROUTER_BIAS_FLAG,
    )


def _mixture_options(kwargs, config, provided):
    value = provided or options.ExpertsMixtureOptions(
        top_k=config.EXPERT_TOP_K,
        num_experts=config.EXPERT_NUM_EXPERTS,
        capacity_factor=config.EXPERT_CAPACITY_FACTOR,
        dropped_token_behavior=config.EXPERT_DROPPED_TOKEN_BEHAVIOR,
        compute_expert_mixture_flag=config.EXPERT_COMPUTE_EXPERT_MIXTURE_FLAG,
        weighted_parameters_flag=config.EXPERT_WEIGHTED_PARAMETERS_FLAG,
        weighting_position_option=config.EXPERT_WEIGHTING_POSITION_OPTION,
        routing_initialization_mode=config.EXPERT_ROUTING_INITIALIZATION_MODE,
    )
    fields = (
        "top_k",
        "num_experts",
        "capacity_factor",
        "dropped_token_behavior",
        "compute_expert_mixture_flag",
        "weighted_parameters_flag",
        "weighting_position_option",
        "routing_initialization_mode",
    )
    updates = _pop_updates(kwargs, {name: name for name in fields})
    return replace(value, **updates) if updates else value


def _sampler_options(kwargs, config, provided):
    value = provided or options.ExpertsSamplerOptions(
        threshold=config.SAMPLER_THRESHOLD,
        filter_above_threshold=config.SAMPLER_FILTER_ABOVE_THRESHOLD,
        num_topk_samples=config.SAMPLER_NUM_TOPK_SAMPLES,
        normalize_probabilities_flag=config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
        noisy_topk_flag=config.SAMPLER_NOISY_TOPK_FLAG,
        coefficient_of_variation_loss_weight=(
            config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT
        ),
        switch_loss_weight=config.SAMPLER_SWITCH_LOSS_WEIGHT,
        zero_centred_loss_weight=config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT,
        mutual_information_loss_weight=config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT,
    )
    updates = _pop_updates(
        kwargs,
        {
            "sampler_threshold": "threshold",
            "sampler_filter_above_threshold": "filter_above_threshold",
            "sampler_num_topk_samples": "num_topk_samples",
            "sampler_normalize_probabilities_flag": "normalize_probabilities_flag",
            "sampler_noisy_topk_flag": "noisy_topk_flag",
            "sampler_coefficient_of_variation_loss_weight": (
                "coefficient_of_variation_loss_weight"
            ),
            "sampler_switch_loss_weight": "switch_loss_weight",
            "sampler_zero_centred_loss_weight": "zero_centred_loss_weight",
            "sampler_mutual_information_loss_weight": (
                "mutual_information_loss_weight"
            ),
        },
    )
    return replace(value, **updates) if updates else value


def _router_options(kwargs, config, provided):
    value = provided or options.ExpertsRouterOptions(
        noisy_topk_flag=config.ROUTER_NOISY_TOPK_FLAG
    )
    if "router_noisy_topk_flag" not in kwargs:
        return value
    return replace(value, noisy_topk_flag=kwargs.pop("router_noisy_topk_flag"))


def _stack_source(kwargs, config, flat_prefix, config_prefix, role, provided=None):
    flat = f"{flat_prefix}{role}_stack"
    config_name = f"{config_prefix}{role.upper()}_STACK"
    value = provided or options.ExpertsSubmoduleStackSource(
        independent_flag=getattr(config, f"{config_name}_INDEPENDENT_FLAG"),
        hidden_dim=getattr(config, f"{config_name}_HIDDEN_DIM"),
        num_layers=getattr(config, f"{config_name}_NUM_LAYERS"),
        last_layer_bias_option=getattr(config, f"{config_name}_LAST_LAYER_BIAS_OPTION"),
        apply_output_pipeline_flag=getattr(
            config, f"{config_name}_APPLY_OUTPUT_PIPELINE_FLAG"
        ),
        activation=getattr(config, f"{config_name}_ACTIVATION"),
        layer_norm_position=getattr(config, f"{config_name}_LAYER_NORM_POSITION"),
        residual_connection_option=getattr(
            config, f"{config_name}_RESIDUAL_CONNECTION_OPTION"
        ),
        dropout_probability=getattr(config, f"{config_name}_DROPOUT_PROBABILITY"),
        bias_flag=getattr(config, f"{config_name}_BIAS_FLAG"),
    )
    mapping = {
        f"{flat}_{key}": field
        for key, field in {
            "independent_flag": "independent_flag",
            **_STACK_FIELDS,
        }.items()
    }
    updates = _pop_updates(kwargs, mapping)
    return replace(value, **updates) if updates else value


def _layer_controller_options(kwargs, config, provided, *, flat_prefix, config_prefix):
    gate_flag_key = f"{flat_prefix}gate_flag" if flat_prefix else "stack_gate_flag"
    halting_flag_key = (
        f"{flat_prefix}halting_flag" if flat_prefix else "stack_halting_flag"
    )
    value = provided or options.ExpertsLayerControllerOptions(
        stack_gate_flag=getattr(config, f"{config_prefix}GATE_FLAG"),
        gate_option=getattr(config, f"{config_prefix}GATE_OPTION"),
        gate_activation=getattr(config, f"{config_prefix}GATE_ACTIVATION"),
        gate_stack_source=_stack_source(
            kwargs, config, flat_prefix, config_prefix, "gate"
        ),
        stack_halting_flag=getattr(config, f"{config_prefix}HALTING_FLAG"),
        halting_threshold=getattr(config, f"{config_prefix}HALTING_THRESHOLD"),
        halting_dropout=getattr(config, f"{config_prefix}HALTING_DROPOUT"),
        halting_hidden_state_mode=getattr(
            config, f"{config_prefix}HALTING_HIDDEN_STATE_MODE"
        ),
        halting_stack_source=_stack_source(
            kwargs, config, flat_prefix, config_prefix, "halting"
        ),
        halting_output_dim=getattr(config, f"{config_prefix}HALTING_OUTPUT_DIM"),
    )
    updates = _pop_updates(
        kwargs,
        {
            gate_flag_key: "stack_gate_flag",
            f"{flat_prefix}gate_option": "gate_option",
            f"{flat_prefix}gate_activation": "gate_activation",
            halting_flag_key: "stack_halting_flag",
            f"{flat_prefix}halting_threshold": "halting_threshold",
            f"{flat_prefix}halting_dropout": "halting_dropout",
            f"{flat_prefix}halting_hidden_state_mode": "halting_hidden_state_mode",
            f"{flat_prefix}halting_output_dim": "halting_output_dim",
            **({"shared_gate_config": "shared_gate_config"} if not flat_prefix else {}),
        },
    )
    updates["gate_stack_source"] = _stack_source(
        kwargs, config, flat_prefix, config_prefix, "gate", value.gate_stack_source
    )
    updates["halting_stack_source"] = _stack_source(
        kwargs,
        config,
        flat_prefix,
        config_prefix,
        "halting",
        value.halting_stack_source,
    )
    return replace(value, **updates)


def _memory_options(kwargs, config, provided, *, flat_prefix, config_prefix):
    value = provided or options.ExpertsDynamicMemoryOptions(
        memory_flag=getattr(config, f"{config_prefix}MEMORY_FLAG"),
        memory_option=getattr(config, f"{config_prefix}MEMORY_OPTION"),
        memory_position_option=getattr(
            config, f"{config_prefix}MEMORY_POSITION_OPTION"
        ),
        memory_test_time_training_learning_rate=getattr(
            config, f"{config_prefix}MEMORY_TEST_TIME_TRAINING_LEARNING_RATE"
        ),
        memory_test_time_training_num_inner_steps=getattr(
            config, f"{config_prefix}MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS"
        ),
        memory_stack_source=_stack_source(
            kwargs, config, flat_prefix, config_prefix, "memory"
        ),
    )
    updates = _pop_updates(
        kwargs,
        {
            f"{flat_prefix}memory_flag": "memory_flag",
            f"{flat_prefix}memory_option": "memory_option",
            f"{flat_prefix}memory_position_option": "memory_position_option",
            f"{flat_prefix}memory_test_time_training_learning_rate": (
                "memory_test_time_training_learning_rate"
            ),
            f"{flat_prefix}memory_test_time_training_num_inner_steps": (
                "memory_test_time_training_num_inner_steps"
            ),
        },
    )
    updates["memory_stack_source"] = _stack_source(
        kwargs,
        config,
        flat_prefix,
        config_prefix,
        "memory",
        value.memory_stack_source,
    )
    return replace(value, **updates)


def _recurrent_options(kwargs, config, provided, *, flat_prefix, config_prefix):
    recurrent_flat = f"{flat_prefix}recurrent_"
    recurrent_config = f"{config_prefix}RECURRENT_"
    value = provided or options.ExpertsRecurrentControllerOptions(
        recurrent_flag=getattr(config, f"{recurrent_config}FLAG"),
        recurrent_max_steps=getattr(config, f"{recurrent_config}MAX_STEPS"),
        recurrent_layer_norm_position=getattr(
            config, f"{recurrent_config}LAYER_NORM_POSITION"
        ),
        recurrent_gate_flag=getattr(config, f"{recurrent_config}GATE_FLAG"),
        recurrent_gate_option=getattr(config, f"{recurrent_config}GATE_OPTION"),
        recurrent_gate_activation=getattr(config, f"{recurrent_config}GATE_ACTIVATION"),
        recurrent_gate_stack_source=_stack_source(
            kwargs, config, recurrent_flat, recurrent_config, "gate"
        ),
        recurrent_halting_flag=getattr(config, f"{recurrent_config}HALTING_FLAG"),
        recurrent_halting_threshold=getattr(
            config, f"{recurrent_config}HALTING_THRESHOLD"
        ),
        recurrent_halting_dropout=getattr(config, f"{recurrent_config}HALTING_DROPOUT"),
        recurrent_halting_hidden_state_mode=getattr(
            config, f"{recurrent_config}HALTING_HIDDEN_STATE_MODE"
        ),
        recurrent_halting_stack_source=_stack_source(
            kwargs, config, recurrent_flat, recurrent_config, "halting"
        ),
    )
    updates = _pop_updates(
        kwargs,
        {
            f"{recurrent_flat}flag": "recurrent_flag",
            f"{recurrent_flat}max_steps": "recurrent_max_steps",
            f"{recurrent_flat}layer_norm_position": "recurrent_layer_norm_position",
            f"{recurrent_flat}gate_flag": "recurrent_gate_flag",
            f"{recurrent_flat}gate_option": "recurrent_gate_option",
            f"{recurrent_flat}gate_activation": "recurrent_gate_activation",
            f"{recurrent_flat}halting_flag": "recurrent_halting_flag",
            f"{recurrent_flat}halting_threshold": "recurrent_halting_threshold",
            f"{recurrent_flat}halting_dropout": "recurrent_halting_dropout",
            f"{recurrent_flat}halting_hidden_state_mode": (
                "recurrent_halting_hidden_state_mode"
            ),
        },
    )
    updates["recurrent_gate_stack_source"] = _stack_source(
        kwargs,
        config,
        recurrent_flat,
        recurrent_config,
        "gate",
        value.recurrent_gate_stack_source,
    )
    updates["recurrent_halting_stack_source"] = _stack_source(
        kwargs,
        config,
        recurrent_flat,
        recurrent_config,
        "halting",
        value.recurrent_halting_stack_source,
    )
    return replace(value, **updates)


def _pop_updates(kwargs, mapping):
    return {field: kwargs.pop(key) for key, field in mapping.items() if key in kwargs}


_RUNTIME_FIELDS = {field.name for field in fields(options.RuntimeOptions)}


def runtime_from_builder_options(
    builder_options: dict[str, Any],
    config_module: ModuleType = config,
) -> options.RuntimeOptions:
    values = {
        "batch_size": config_module.BATCH_SIZE,
        "learning_rate": config_module.LEARNING_RATE,
        "input_dim": config_module.INPUT_DIM,
        "hidden_dim": config_module.HIDDEN_DIM,
        "output_dim": config_module.OUTPUT_DIM,
        **builder_kwargs_from_flat({}, config_module),
        **builder_options,
    }
    unknown = set(values) - _RUNTIME_FIELDS
    if unknown:
        name = sorted(unknown)[0]
        raise TypeError(f"unexpected hidden runtime option {name!r}")
    return options.RuntimeOptions(**values)


def runtime_from_flat(
    flat_options: dict[str, Any] | None = None,
    config_module: ModuleType = config,
) -> options.RuntimeOptions:
    return runtime_from_builder_options(
        builder_kwargs_from_flat(flat_options or {}, config_module),
        config_module,
    )


DEFAULT_RUNTIME: Final[options.RuntimeOptions] = runtime_from_flat()
