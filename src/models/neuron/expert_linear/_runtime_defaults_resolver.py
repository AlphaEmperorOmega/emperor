from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TypeVar, cast

import models.neuron.expert_linear.config as config
from models.neuron.expert_linear._hidden.runtime_defaults import runtime_from_flat
from models.neuron.expert_linear._hidden.runtime_options import (
    RuntimeOptions as HiddenRuntimeOptions,
)
from models.neuron.expert_linear.runtime_options import (
    ClusterRouteHaltingOptions,
    NeuronClusterCapacityOptions,
    NeuronSubmoduleStackOptions,
    NeuronTerminalOptions,
    NeuronTerminalRoutingTreeOptions,
    NeuronTerminalSamplerOptions,
)

_CLUSTER_BEAM_WIDTH_DEFAULT = config.CLUSTER_BEAM_WIDTH
_CLUSTER_ESCAPE_DRIVEN_GROWTH_FLAG_DEFAULT = config.CLUSTER_ESCAPE_DRIVEN_GROWTH_FLAG
_CLUSTER_GROWTH_COOLDOWN_STEPS_DEFAULT = config.CLUSTER_GROWTH_COOLDOWN_STEPS
_CLUSTER_GROWTH_THRESHOLD_DEFAULT = config.CLUSTER_GROWTH_THRESHOLD
_CLUSTER_GROWTH_WARMUP_STEPS_DEFAULT = config.CLUSTER_GROWTH_WARMUP_STEPS
_CLUSTER_HALTING_DROPOUT_DEFAULT = config.CLUSTER_HALTING_DROPOUT
_CLUSTER_HALTING_FLAG_DEFAULT = config.CLUSTER_HALTING_FLAG
_CLUSTER_HALTING_HIDDEN_STATE_MODE_DEFAULT = config.CLUSTER_HALTING_HIDDEN_STATE_MODE
_CLUSTER_HALTING_OPTION_DEFAULT = config.CLUSTER_HALTING_OPTION
_CLUSTER_HALTING_OUTPUT_DIM_DEFAULT = config.CLUSTER_HALTING_OUTPUT_DIM
_CLUSTER_HALTING_STACK_ACTIVATION_DEFAULT = config.CLUSTER_HALTING_STACK_ACTIVATION
_CLUSTER_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG_DEFAULT = (
    config.CLUSTER_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
)
_CLUSTER_HALTING_STACK_BIAS_FLAG_DEFAULT = config.CLUSTER_HALTING_STACK_BIAS_FLAG
_CLUSTER_HALTING_STACK_DROPOUT_PROBABILITY_DEFAULT = (
    config.CLUSTER_HALTING_STACK_DROPOUT_PROBABILITY
)
_CLUSTER_HALTING_STACK_HIDDEN_DIM_DEFAULT = config.CLUSTER_HALTING_STACK_HIDDEN_DIM
_CLUSTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION_DEFAULT = (
    config.CLUSTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION
)
_CLUSTER_HALTING_STACK_LAYER_NORM_POSITION_DEFAULT = (
    config.CLUSTER_HALTING_STACK_LAYER_NORM_POSITION
)
_CLUSTER_HALTING_STACK_NORMALIZATION_DEFAULT = (
    config.CLUSTER_HALTING_STACK_NORMALIZATION
)
_CLUSTER_HALTING_STACK_NUM_LAYERS_DEFAULT = config.CLUSTER_HALTING_STACK_NUM_LAYERS
_CLUSTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION_DEFAULT = (
    config.CLUSTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
)
_CLUSTER_HALTING_STACK_RESIDUAL_MODEL_FLAG_DEFAULT = (
    config.CLUSTER_HALTING_STACK_RESIDUAL_MODEL_FLAG
)
_CLUSTER_HALTING_THRESHOLD_DEFAULT = config.CLUSTER_HALTING_THRESHOLD
_CLUSTER_INITIAL_X_AXIS_TOTAL_NEURONS_DEFAULT = (
    config.CLUSTER_INITIAL_X_AXIS_TOTAL_NEURONS
)
_CLUSTER_INITIAL_Y_AXIS_TOTAL_NEURONS_DEFAULT = (
    config.CLUSTER_INITIAL_Y_AXIS_TOTAL_NEURONS
)
_CLUSTER_INITIAL_Z_AXIS_TOTAL_NEURONS_DEFAULT = (
    config.CLUSTER_INITIAL_Z_AXIS_TOTAL_NEURONS
)
_CLUSTER_MAX_STEPS_DEFAULT = config.CLUSTER_MAX_STEPS
_CLUSTER_MAX_TOTAL_GROWTHS_DEFAULT = config.CLUSTER_MAX_TOTAL_GROWTHS
_CLUSTER_MITOSIS_INITIALIZATION_FLAG_DEFAULT = (
    config.CLUSTER_MITOSIS_INITIALIZATION_FLAG
)
_CLUSTER_PRUNING_THRESHOLD_DEFAULT = config.CLUSTER_PRUNING_THRESHOLD
_CLUSTER_TERMINAL_ROUTER_ACTIVATION_DEFAULT = config.CLUSTER_TERMINAL_ROUTER_ACTIVATION
_CLUSTER_TERMINAL_ROUTER_APPLY_OUTPUT_POSTPROCESSING_FLAG_DEFAULT = (
    config.CLUSTER_TERMINAL_ROUTER_APPLY_OUTPUT_POSTPROCESSING_FLAG
)
_CLUSTER_TERMINAL_ROUTER_BIAS_FLAG_DEFAULT = config.CLUSTER_TERMINAL_ROUTER_BIAS_FLAG
_CLUSTER_TERMINAL_ROUTER_DROPOUT_PROBABILITY_DEFAULT = (
    config.CLUSTER_TERMINAL_ROUTER_DROPOUT_PROBABILITY
)
_CLUSTER_TERMINAL_ROUTER_HIDDEN_DIM_DEFAULT = config.CLUSTER_TERMINAL_ROUTER_HIDDEN_DIM
_CLUSTER_TERMINAL_ROUTER_LAST_LAYER_BIAS_OPTION_DEFAULT = (
    config.CLUSTER_TERMINAL_ROUTER_LAST_LAYER_BIAS_OPTION
)
_CLUSTER_TERMINAL_ROUTER_LAYER_NORM_POSITION_DEFAULT = (
    config.CLUSTER_TERMINAL_ROUTER_LAYER_NORM_POSITION
)
_CLUSTER_TERMINAL_ROUTER_NORMALIZATION_DEFAULT = (
    config.CLUSTER_TERMINAL_ROUTER_NORMALIZATION
)
_CLUSTER_TERMINAL_ROUTER_NUM_LAYERS_DEFAULT = config.CLUSTER_TERMINAL_ROUTER_NUM_LAYERS
_CLUSTER_TERMINAL_ROUTER_RESIDUAL_CONNECTION_OPTION_DEFAULT = (
    config.CLUSTER_TERMINAL_ROUTER_RESIDUAL_CONNECTION_OPTION
)
_CLUSTER_TERMINAL_ROUTER_RESIDUAL_MODEL_FLAG_DEFAULT = (
    config.CLUSTER_TERMINAL_ROUTER_RESIDUAL_MODEL_FLAG
)
_CLUSTER_TERMINAL_SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT_DEFAULT = (
    config.CLUSTER_TERMINAL_SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT
)
_CLUSTER_TERMINAL_SAMPLER_FILTER_ABOVE_THRESHOLD_DEFAULT = (
    config.CLUSTER_TERMINAL_SAMPLER_FILTER_ABOVE_THRESHOLD
)
_CLUSTER_TERMINAL_SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT_DEFAULT = (
    config.CLUSTER_TERMINAL_SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT
)
_CLUSTER_TERMINAL_SAMPLER_NOISY_TOPK_FLAG_DEFAULT = (
    config.CLUSTER_TERMINAL_SAMPLER_NOISY_TOPK_FLAG
)
_CLUSTER_TERMINAL_SAMPLER_NORMALIZE_PROBABILITIES_FLAG_DEFAULT = (
    config.CLUSTER_TERMINAL_SAMPLER_NORMALIZE_PROBABILITIES_FLAG
)
_CLUSTER_TERMINAL_SAMPLER_NUM_TOPK_SAMPLES_DEFAULT = (
    config.CLUSTER_TERMINAL_SAMPLER_NUM_TOPK_SAMPLES
)
_CLUSTER_TERMINAL_SAMPLER_SWITCH_LOSS_WEIGHT_DEFAULT = (
    config.CLUSTER_TERMINAL_SAMPLER_SWITCH_LOSS_WEIGHT
)
_CLUSTER_TERMINAL_SAMPLER_THRESHOLD_DEFAULT = config.CLUSTER_TERMINAL_SAMPLER_THRESHOLD
_CLUSTER_TERMINAL_SAMPLER_ZERO_CENTRED_LOSS_WEIGHT_DEFAULT = (
    config.CLUSTER_TERMINAL_SAMPLER_ZERO_CENTRED_LOSS_WEIGHT
)
_CLUSTER_TERMINAL_TOP_K_DEFAULT = config.CLUSTER_TERMINAL_TOP_K
_CLUSTER_TERMINAL_ROUTING_TREE_DEPTH_DEFAULT = (
    config.CLUSTER_TERMINAL_ROUTING_TREE_DEPTH
)
_CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_1_BRANCH_COUNT_DEFAULT = (
    config.CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_1_BRANCH_COUNT
)
_CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_1_TOP_K_DEFAULT = (
    config.CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_1_TOP_K
)
_CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_2_BRANCH_COUNT_DEFAULT = (
    config.CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_2_BRANCH_COUNT
)
_CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_2_TOP_K_DEFAULT = (
    config.CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_2_TOP_K
)
_CLUSTER_TERMINAL_XY_AXIS_RANGE_DEFAULT = config.CLUSTER_TERMINAL_XY_AXIS_RANGE
_CLUSTER_TERMINAL_Z_AXIS_RANGE_DEFAULT = config.CLUSTER_TERMINAL_Z_AXIS_RANGE
_CLUSTER_X_AXIS_TOTAL_NEURONS_DEFAULT = config.CLUSTER_X_AXIS_TOTAL_NEURONS
_CLUSTER_Y_AXIS_TOTAL_NEURONS_DEFAULT = config.CLUSTER_Y_AXIS_TOTAL_NEURONS
_CLUSTER_Z_AXIS_TOTAL_NEURONS_DEFAULT = config.CLUSTER_Z_AXIS_TOTAL_NEURONS
_GATE_ACTIVATION_DEFAULT = config.GATE_ACTIVATION
_GATE_OPTION_DEFAULT = config.GATE_OPTION
_RECURRENT_GATE_ACTIVATION_DEFAULT = config.RECURRENT_GATE_ACTIVATION
_RECURRENT_GATE_OPTION_DEFAULT = config.RECURRENT_GATE_OPTION

_DefaultT = TypeVar("_DefaultT")


def _pop(
    values: dict[str, object],
    key: str,
    default: _DefaultT,
) -> _DefaultT:
    return cast(_DefaultT, values.pop(key, default))


def _cluster_capacity_options(
    values: dict[str, object],
) -> NeuronClusterCapacityOptions:
    provided = cast(
        NeuronClusterCapacityOptions | None,
        values.pop("cluster_capacity_options", None),
    )
    x_axis_total_neurons = _pop(
        values, "cluster_x_axis_total_neurons", _CLUSTER_X_AXIS_TOTAL_NEURONS_DEFAULT
    )
    y_axis_total_neurons = _pop(
        values, "cluster_y_axis_total_neurons", _CLUSTER_Y_AXIS_TOTAL_NEURONS_DEFAULT
    )
    z_axis_total_neurons = _pop(
        values, "cluster_z_axis_total_neurons", _CLUSTER_Z_AXIS_TOTAL_NEURONS_DEFAULT
    )
    initial_x_axis_total_neurons = _pop(
        values,
        "cluster_initial_x_axis_total_neurons",
        _CLUSTER_INITIAL_X_AXIS_TOTAL_NEURONS_DEFAULT,
    )
    initial_y_axis_total_neurons = _pop(
        values,
        "cluster_initial_y_axis_total_neurons",
        _CLUSTER_INITIAL_Y_AXIS_TOTAL_NEURONS_DEFAULT,
    )
    initial_z_axis_total_neurons = _pop(
        values,
        "cluster_initial_z_axis_total_neurons",
        _CLUSTER_INITIAL_Z_AXIS_TOTAL_NEURONS_DEFAULT,
    )
    max_steps = _pop(values, "cluster_max_steps", _CLUSTER_MAX_STEPS_DEFAULT)
    beam_width = _pop(values, "cluster_beam_width", _CLUSTER_BEAM_WIDTH_DEFAULT)
    growth_threshold = _pop(
        values, "cluster_growth_threshold", _CLUSTER_GROWTH_THRESHOLD_DEFAULT
    )
    growth_cooldown_steps = _pop(
        values,
        "cluster_growth_cooldown_steps",
        _CLUSTER_GROWTH_COOLDOWN_STEPS_DEFAULT,
    )
    max_total_growths = _pop(
        values, "cluster_max_total_growths", _CLUSTER_MAX_TOTAL_GROWTHS_DEFAULT
    )
    growth_warmup_steps = _pop(
        values,
        "cluster_growth_warmup_steps",
        _CLUSTER_GROWTH_WARMUP_STEPS_DEFAULT,
    )
    pruning_threshold = _pop(
        values, "cluster_pruning_threshold", _CLUSTER_PRUNING_THRESHOLD_DEFAULT
    )
    escape_driven_growth_flag = _pop(
        values,
        "cluster_escape_driven_growth_flag",
        _CLUSTER_ESCAPE_DRIVEN_GROWTH_FLAG_DEFAULT,
    )
    mitosis_initialization_flag = _pop(
        values,
        "cluster_mitosis_initialization_flag",
        _CLUSTER_MITOSIS_INITIALIZATION_FLAG_DEFAULT,
    )
    if provided:
        return provided
    return NeuronClusterCapacityOptions(
        x_axis_total_neurons=x_axis_total_neurons,
        y_axis_total_neurons=y_axis_total_neurons,
        z_axis_total_neurons=z_axis_total_neurons,
        initial_x_axis_total_neurons=initial_x_axis_total_neurons,
        initial_y_axis_total_neurons=initial_y_axis_total_neurons,
        initial_z_axis_total_neurons=initial_z_axis_total_neurons,
        max_steps=max_steps,
        beam_width=beam_width,
        growth_threshold=growth_threshold,
        growth_cooldown_steps=growth_cooldown_steps,
        max_total_growths=max_total_growths,
        growth_warmup_steps=growth_warmup_steps,
        pruning_threshold=pruning_threshold,
        escape_driven_growth_flag=escape_driven_growth_flag,
        mitosis_initialization_flag=mitosis_initialization_flag,
    )


def _terminal_options(values: dict[str, object]) -> NeuronTerminalOptions:
    provided = cast(
        NeuronTerminalOptions | None,
        values.pop("terminal_options", None),
    )
    xy_axis_range = _pop(
        values,
        "cluster_terminal_xy_axis_range",
        _CLUSTER_TERMINAL_XY_AXIS_RANGE_DEFAULT,
    )
    z_axis_range = _pop(
        values, "cluster_terminal_z_axis_range", _CLUSTER_TERMINAL_Z_AXIS_RANGE_DEFAULT
    )
    top_k = _pop(values, "cluster_terminal_top_k", _CLUSTER_TERMINAL_TOP_K_DEFAULT)
    routing_tree_depth = _pop(
        values,
        "cluster_terminal_routing_tree_depth",
        _CLUSTER_TERMINAL_ROUTING_TREE_DEPTH_DEFAULT,
    )
    level_1_branch_count = _pop(
        values,
        "cluster_terminal_routing_tree_level_1_branch_count",
        _CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_1_BRANCH_COUNT_DEFAULT,
    )
    level_1_top_k = _pop(
        values,
        "cluster_terminal_routing_tree_level_1_top_k",
        _CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_1_TOP_K_DEFAULT,
    )
    level_2_branch_count = _pop(
        values,
        "cluster_terminal_routing_tree_level_2_branch_count",
        _CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_2_BRANCH_COUNT_DEFAULT,
    )
    level_2_top_k = _pop(
        values,
        "cluster_terminal_routing_tree_level_2_top_k",
        _CLUSTER_TERMINAL_ROUTING_TREE_LEVEL_2_TOP_K_DEFAULT,
    )
    if provided:
        return provided
    routing_tree = _terminal_routing_tree_options(
        routing_tree_depth,
        level_1_branch_count,
        level_1_top_k,
        level_2_branch_count,
        level_2_top_k,
    )
    return NeuronTerminalOptions(
        xy_axis_range=xy_axis_range,
        z_axis_range=z_axis_range,
        top_k=top_k,
        routing_tree=routing_tree,
    )


def _terminal_routing_tree_options(
    depth,
    level_1_branch_count,
    level_1_top_k,
    level_2_branch_count,
    level_2_top_k,
) -> NeuronTerminalRoutingTreeOptions | None:
    configured_levels = (
        level_1_branch_count,
        level_1_top_k,
        level_2_branch_count,
        level_2_top_k,
    )
    if depth is None:
        if any(value is not None for value in configured_levels):
            raise ValueError(
                "cluster_terminal_routing_tree_depth must be set when routing "
                "tree level options are provided."
            )
        return None
    required_level_count = depth.value - 1
    required_values = configured_levels[: required_level_count * 2]
    if any(value is None for value in required_values):
        raise ValueError(
            f"Terminal routing tree depth {depth.value} requires branch_count and "
            f"top_k for {required_level_count} direction level(s)."
        )
    unused_values = configured_levels[required_level_count * 2 :]
    if any(value is not None for value in unused_values):
        raise ValueError(
            f"Terminal routing tree depth {depth.value} does not accept level "
            f"{required_level_count + 1} options."
        )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in required_values
    ):
        raise ValueError(
            "Terminal routing tree branch counts and top-k values must be positive integers."
        )
    return NeuronTerminalRoutingTreeOptions(
        depth=depth,
        direction_branch_counts=tuple(required_values[0::2]),
        direction_top_k=tuple(required_values[1::2]),
    )


def _terminal_router_options(
    values: dict[str, object],
) -> NeuronSubmoduleStackOptions:
    provided = cast(
        NeuronSubmoduleStackOptions | None,
        values.pop("terminal_router_options", None),
    )
    hidden_dim = _pop(
        values,
        "cluster_terminal_router_hidden_dim",
        _CLUSTER_TERMINAL_ROUTER_HIDDEN_DIM_DEFAULT,
    )
    num_layers = _pop(
        values,
        "cluster_terminal_router_num_layers",
        _CLUSTER_TERMINAL_ROUTER_NUM_LAYERS_DEFAULT,
    )
    last_layer_bias_option = _pop(
        values,
        "cluster_terminal_router_last_layer_bias_option",
        _CLUSTER_TERMINAL_ROUTER_LAST_LAYER_BIAS_OPTION_DEFAULT,
    )
    apply_output_postprocessing_flag = _pop(
        values,
        "cluster_terminal_router_apply_output_postprocessing_flag",
        _CLUSTER_TERMINAL_ROUTER_APPLY_OUTPUT_POSTPROCESSING_FLAG_DEFAULT,
    )
    activation = _pop(
        values,
        "cluster_terminal_router_activation",
        _CLUSTER_TERMINAL_ROUTER_ACTIVATION_DEFAULT,
    )
    layer_norm_position = _pop(
        values,
        "cluster_terminal_router_layer_norm_position",
        _CLUSTER_TERMINAL_ROUTER_LAYER_NORM_POSITION_DEFAULT,
    )
    normalization = _pop(
        values,
        "cluster_terminal_router_normalization",
        _CLUSTER_TERMINAL_ROUTER_NORMALIZATION_DEFAULT,
    )
    residual_connection_option = _pop(
        values,
        "cluster_terminal_router_residual_connection_option",
        _CLUSTER_TERMINAL_ROUTER_RESIDUAL_CONNECTION_OPTION_DEFAULT,
    )
    residual_model_flag = _pop(
        values,
        "cluster_terminal_router_residual_model_flag",
        _CLUSTER_TERMINAL_ROUTER_RESIDUAL_MODEL_FLAG_DEFAULT,
    )
    dropout_probability = _pop(
        values,
        "cluster_terminal_router_dropout_probability",
        _CLUSTER_TERMINAL_ROUTER_DROPOUT_PROBABILITY_DEFAULT,
    )
    bias_flag = _pop(
        values,
        "cluster_terminal_router_bias_flag",
        _CLUSTER_TERMINAL_ROUTER_BIAS_FLAG_DEFAULT,
    )
    if provided:
        return provided
    return NeuronSubmoduleStackOptions(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=last_layer_bias_option,
        apply_output_postprocessing_flag=apply_output_postprocessing_flag,
        activation=activation,
        layer_norm_position=layer_norm_position,
        normalization=normalization,
        residual_connection_option=residual_connection_option,
        residual_model_flag=residual_model_flag,
        dropout_probability=dropout_probability,
        bias_flag=bias_flag,
    )


def _terminal_sampler_options(
    values: dict[str, object],
) -> NeuronTerminalSamplerOptions:
    provided = cast(
        NeuronTerminalSamplerOptions | None,
        values.pop("terminal_sampler_options", None),
    )
    threshold = _pop(
        values,
        "cluster_terminal_sampler_threshold",
        _CLUSTER_TERMINAL_SAMPLER_THRESHOLD_DEFAULT,
    )
    filter_above_threshold = _pop(
        values,
        "cluster_terminal_sampler_filter_above_threshold",
        _CLUSTER_TERMINAL_SAMPLER_FILTER_ABOVE_THRESHOLD_DEFAULT,
    )
    num_topk_samples = _pop(
        values,
        "cluster_terminal_sampler_num_topk_samples",
        _CLUSTER_TERMINAL_SAMPLER_NUM_TOPK_SAMPLES_DEFAULT,
    )
    normalize_probabilities_flag = _pop(
        values,
        "cluster_terminal_sampler_normalize_probabilities_flag",
        _CLUSTER_TERMINAL_SAMPLER_NORMALIZE_PROBABILITIES_FLAG_DEFAULT,
    )
    noisy_topk_flag = _pop(
        values,
        "cluster_terminal_sampler_noisy_topk_flag",
        _CLUSTER_TERMINAL_SAMPLER_NOISY_TOPK_FLAG_DEFAULT,
    )
    coefficient_of_variation_loss_weight = _pop(
        values,
        "cluster_terminal_sampler_coefficient_of_variation_loss_weight",
        _CLUSTER_TERMINAL_SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT_DEFAULT,
    )
    switch_loss_weight = _pop(
        values,
        "cluster_terminal_sampler_switch_loss_weight",
        _CLUSTER_TERMINAL_SAMPLER_SWITCH_LOSS_WEIGHT_DEFAULT,
    )
    zero_centred_loss_weight = _pop(
        values,
        "cluster_terminal_sampler_zero_centred_loss_weight",
        _CLUSTER_TERMINAL_SAMPLER_ZERO_CENTRED_LOSS_WEIGHT_DEFAULT,
    )
    mutual_information_loss_weight = _pop(
        values,
        "cluster_terminal_sampler_mutual_information_loss_weight",
        _CLUSTER_TERMINAL_SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT_DEFAULT,
    )
    if provided:
        return provided
    return NeuronTerminalSamplerOptions(
        threshold=threshold,
        filter_above_threshold=filter_above_threshold,
        num_topk_samples=num_topk_samples,
        normalize_probabilities_flag=normalize_probabilities_flag,
        noisy_topk_flag=noisy_topk_flag,
        coefficient_of_variation_loss_weight=coefficient_of_variation_loss_weight,
        switch_loss_weight=switch_loss_weight,
        zero_centred_loss_weight=zero_centred_loss_weight,
        mutual_information_loss_weight=mutual_information_loss_weight,
    )


def _cluster_halting_options(
    values: dict[str, object],
) -> ClusterRouteHaltingOptions:
    provided = cast(
        ClusterRouteHaltingOptions | None,
        values.pop("cluster_halting_options", None),
    )
    enabled = _pop(values, "cluster_halting_flag", _CLUSTER_HALTING_FLAG_DEFAULT)
    halting_option = _pop(
        values, "cluster_halting_option", _CLUSTER_HALTING_OPTION_DEFAULT
    )
    threshold = _pop(
        values, "cluster_halting_threshold", _CLUSTER_HALTING_THRESHOLD_DEFAULT
    )
    dropout = _pop(values, "cluster_halting_dropout", _CLUSTER_HALTING_DROPOUT_DEFAULT)
    hidden_state_mode = _pop(
        values,
        "cluster_halting_hidden_state_mode",
        _CLUSTER_HALTING_HIDDEN_STATE_MODE_DEFAULT,
    )
    output_dim = _pop(
        values, "cluster_halting_output_dim", _CLUSTER_HALTING_OUTPUT_DIM_DEFAULT
    )
    hidden_dim = _pop(
        values,
        "cluster_halting_stack_hidden_dim",
        _CLUSTER_HALTING_STACK_HIDDEN_DIM_DEFAULT,
    )
    num_layers = _pop(
        values,
        "cluster_halting_stack_num_layers",
        _CLUSTER_HALTING_STACK_NUM_LAYERS_DEFAULT,
    )
    last_layer_bias_option = _pop(
        values,
        "cluster_halting_stack_last_layer_bias_option",
        _CLUSTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION_DEFAULT,
    )
    apply_output_postprocessing_flag = _pop(
        values,
        "cluster_halting_stack_apply_output_postprocessing_flag",
        _CLUSTER_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG_DEFAULT,
    )
    activation = _pop(
        values,
        "cluster_halting_stack_activation",
        _CLUSTER_HALTING_STACK_ACTIVATION_DEFAULT,
    )
    layer_norm_position = _pop(
        values,
        "cluster_halting_stack_layer_norm_position",
        _CLUSTER_HALTING_STACK_LAYER_NORM_POSITION_DEFAULT,
    )
    normalization = _pop(
        values,
        "cluster_halting_stack_normalization",
        _CLUSTER_HALTING_STACK_NORMALIZATION_DEFAULT,
    )
    residual_connection_option = _pop(
        values,
        "cluster_halting_stack_residual_connection_option",
        _CLUSTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION_DEFAULT,
    )
    residual_model_flag = _pop(
        values,
        "cluster_halting_stack_residual_model_flag",
        _CLUSTER_HALTING_STACK_RESIDUAL_MODEL_FLAG_DEFAULT,
    )
    dropout_probability = _pop(
        values,
        "cluster_halting_stack_dropout_probability",
        _CLUSTER_HALTING_STACK_DROPOUT_PROBABILITY_DEFAULT,
    )
    bias_flag = _pop(
        values,
        "cluster_halting_stack_bias_flag",
        _CLUSTER_HALTING_STACK_BIAS_FLAG_DEFAULT,
    )
    if provided:
        return provided
    return ClusterRouteHaltingOptions(
        enabled=enabled,
        halting_option=halting_option,
        threshold=threshold,
        dropout=dropout,
        hidden_state_mode=hidden_state_mode,
        stack_options=NeuronSubmoduleStackOptions(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_postprocessing_flag=apply_output_postprocessing_flag,
            activation=activation,
            layer_norm_position=layer_norm_position,
            normalization=normalization,
            residual_connection_option=residual_connection_option,
            residual_model_flag=residual_model_flag,
            dropout_probability=dropout_probability,
            bias_flag=bias_flag,
        ),
        output_dim=output_dim,
    )


def _hidden_flat_options(values: dict[str, object]) -> dict[str, object]:
    gate_option = _pop(values, "gate_option", _GATE_OPTION_DEFAULT)
    gate_activation = _pop(values, "gate_activation", _GATE_ACTIVATION_DEFAULT)
    recurrent_gate_option = _pop(
        values, "recurrent_gate_option", _RECURRENT_GATE_OPTION_DEFAULT
    )
    recurrent_gate_activation = _pop(
        values,
        "recurrent_gate_activation",
        _RECURRENT_GATE_ACTIVATION_DEFAULT,
    )
    shared_gate_config = values.pop("shared_gate_config", None)
    hidden_options = {
        "gate_option": gate_option,
        "gate_activation": gate_activation,
        "recurrent_gate_option": recurrent_gate_option,
        "recurrent_gate_activation": recurrent_gate_activation,
        **values,
    }
    if shared_gate_config is not None:
        hidden_options["shared_gate_config"] = shared_gate_config
    return hidden_options


class _NeuronExpertLinearRuntimeDefaultsResolver:
    hidden_runtime: HiddenRuntimeOptions
    cluster_capacity_options: NeuronClusterCapacityOptions
    terminal_options: NeuronTerminalOptions
    terminal_router_options: NeuronSubmoduleStackOptions
    terminal_sampler_options: NeuronTerminalSamplerOptions
    cluster_halting_options: ClusterRouteHaltingOptions

    def __init__(self, values: Mapping[str, object] | None = None) -> None:
        flat_values = dict(values or {})
        cluster_capacity_options = _cluster_capacity_options(flat_values)
        terminal_options = _terminal_options(flat_values)
        terminal_router_options = _terminal_router_options(flat_values)
        terminal_sampler_options = _terminal_sampler_options(flat_values)
        cluster_halting_options = _cluster_halting_options(flat_values)

        hidden_runtime = runtime_from_flat(_hidden_flat_options(flat_values), config)
        residual_stack_options = hidden_runtime.stack_options.residual_stack_options
        terminal_router_options = replace(
            terminal_router_options,
            residual_stack_options=residual_stack_options,
        )
        cluster_halting_options = replace(
            cluster_halting_options,
            stack_options=replace(
                cluster_halting_options.stack_options,
                residual_stack_options=residual_stack_options,
            ),
        )

        self.hidden_runtime = hidden_runtime
        self.cluster_capacity_options = cluster_capacity_options
        self.terminal_options = terminal_options
        self.terminal_router_options = terminal_router_options
        self.terminal_sampler_options = terminal_sampler_options
        self.cluster_halting_options = cluster_halting_options
