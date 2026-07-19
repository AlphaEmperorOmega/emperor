# ruff: noqa: E501

from typing import Any

import models.neuron.linear_adaptive.config as config
from emperor.halting import HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    ResidualConnectionOptions,
)
from emperor.neuron import TerminalRangeOptions, TerminalZAxisOffsetOptions
from models.neuron.linear_adaptive._hidden.runtime_defaults import runtime_from_flat
from models.neuron.linear_adaptive._neuron_config_builder import (
    NeuronConfigBuilder,
)
from models.neuron.linear_adaptive.runtime_options import (
    ClusterRouteHaltingOptions,
    NeuronClusterCapacityOptions,
    NeuronSubmoduleStackOptions,
    NeuronTerminalOptions,
    NeuronTerminalSamplerOptions,
)


class NeuronLinearAdaptiveConfigBuilder(NeuronConfigBuilder):
    def __init__(
        self,
        cluster_x_axis_total_neurons: int = config.CLUSTER_X_AXIS_TOTAL_NEURONS,
        cluster_y_axis_total_neurons: int = config.CLUSTER_Y_AXIS_TOTAL_NEURONS,
        cluster_z_axis_total_neurons: int = config.CLUSTER_Z_AXIS_TOTAL_NEURONS,
        cluster_initial_x_axis_total_neurons: int | None = (
            config.CLUSTER_INITIAL_X_AXIS_TOTAL_NEURONS
        ),
        cluster_initial_y_axis_total_neurons: int | None = (
            config.CLUSTER_INITIAL_Y_AXIS_TOTAL_NEURONS
        ),
        cluster_initial_z_axis_total_neurons: int | None = (
            config.CLUSTER_INITIAL_Z_AXIS_TOTAL_NEURONS
        ),
        cluster_max_steps: int = config.CLUSTER_MAX_STEPS,
        cluster_growth_threshold: int | None = config.CLUSTER_GROWTH_THRESHOLD,
        cluster_terminal_xy_axis_range: TerminalRangeOptions = (
            config.CLUSTER_TERMINAL_XY_AXIS_RANGE
        ),
        cluster_terminal_z_axis_range: TerminalRangeOptions = (
            config.CLUSTER_TERMINAL_Z_AXIS_RANGE
        ),
        cluster_terminal_z_axis_offset: TerminalZAxisOffsetOptions = (
            config.CLUSTER_TERMINAL_Z_AXIS_OFFSET
        ),
        cluster_terminal_top_k: int = config.CLUSTER_TERMINAL_TOP_K,
        cluster_terminal_router_num_layers: int = (
            config.CLUSTER_TERMINAL_ROUTER_NUM_LAYERS
        ),
        cluster_terminal_router_hidden_dim: int = (
            config.CLUSTER_TERMINAL_ROUTER_HIDDEN_DIM
        ),
        cluster_terminal_router_activation: ActivationOptions = (
            config.CLUSTER_TERMINAL_ROUTER_ACTIVATION
        ),
        cluster_terminal_router_layer_norm_position: LayerNormPositionOptions = (
            config.CLUSTER_TERMINAL_ROUTER_LAYER_NORM_POSITION
        ),
        cluster_terminal_router_residual_connection_option: ResidualConnectionOptions = (
            config.CLUSTER_TERMINAL_ROUTER_RESIDUAL_CONNECTION_OPTION
        ),
        cluster_terminal_router_dropout_probability: float = (
            config.CLUSTER_TERMINAL_ROUTER_DROPOUT_PROBABILITY
        ),
        cluster_terminal_router_last_layer_bias_option: LastLayerBiasOptions = (
            config.CLUSTER_TERMINAL_ROUTER_LAST_LAYER_BIAS_OPTION
        ),
        cluster_terminal_router_apply_output_pipeline_flag: bool = (
            config.CLUSTER_TERMINAL_ROUTER_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        cluster_terminal_router_bias_flag: bool = (
            config.CLUSTER_TERMINAL_ROUTER_BIAS_FLAG
        ),
        cluster_terminal_sampler_threshold: float = (
            config.CLUSTER_TERMINAL_SAMPLER_THRESHOLD
        ),
        cluster_terminal_sampler_filter_above_threshold: bool = (
            config.CLUSTER_TERMINAL_SAMPLER_FILTER_ABOVE_THRESHOLD
        ),
        cluster_terminal_sampler_num_topk_samples: int = (
            config.CLUSTER_TERMINAL_SAMPLER_NUM_TOPK_SAMPLES
        ),
        cluster_terminal_sampler_normalize_probabilities_flag: bool = (
            config.CLUSTER_TERMINAL_SAMPLER_NORMALIZE_PROBABILITIES_FLAG
        ),
        cluster_terminal_sampler_noisy_topk_flag: bool = (
            config.CLUSTER_TERMINAL_SAMPLER_NOISY_TOPK_FLAG
        ),
        cluster_terminal_sampler_coefficient_of_variation_loss_weight: float = (
            config.CLUSTER_TERMINAL_SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT
        ),
        cluster_terminal_sampler_switch_loss_weight: float = (
            config.CLUSTER_TERMINAL_SAMPLER_SWITCH_LOSS_WEIGHT
        ),
        cluster_terminal_sampler_zero_centred_loss_weight: float = (
            config.CLUSTER_TERMINAL_SAMPLER_ZERO_CENTRED_LOSS_WEIGHT
        ),
        cluster_terminal_sampler_mutual_information_loss_weight: float = (
            config.CLUSTER_TERMINAL_SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT
        ),
        cluster_halting_flag: bool = config.CLUSTER_HALTING_FLAG,
        cluster_halting_threshold: float = config.CLUSTER_HALTING_THRESHOLD,
        cluster_halting_dropout: float = config.CLUSTER_HALTING_DROPOUT,
        cluster_halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
            config.CLUSTER_HALTING_HIDDEN_STATE_MODE
        ),
        cluster_halting_stack_hidden_dim: int = (
            config.CLUSTER_HALTING_STACK_HIDDEN_DIM
        ),
        cluster_halting_output_dim: int = config.CLUSTER_HALTING_OUTPUT_DIM,
        cluster_halting_stack_layer_norm_position: LayerNormPositionOptions = (
            config.CLUSTER_HALTING_STACK_LAYER_NORM_POSITION
        ),
        cluster_halting_stack_num_layers: int = (
            config.CLUSTER_HALTING_STACK_NUM_LAYERS
        ),
        cluster_halting_stack_activation: ActivationOptions = (
            config.CLUSTER_HALTING_STACK_ACTIVATION
        ),
        cluster_halting_stack_residual_connection_option: ResidualConnectionOptions = (
            config.CLUSTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        cluster_halting_stack_dropout_probability: float = (
            config.CLUSTER_HALTING_STACK_DROPOUT_PROBABILITY
        ),
        cluster_halting_stack_last_layer_bias_option: LastLayerBiasOptions = (
            config.CLUSTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION
        ),
        cluster_halting_stack_apply_output_pipeline_flag: bool = (
            config.CLUSTER_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        cluster_halting_stack_bias_flag: bool = (
            config.CLUSTER_HALTING_STACK_BIAS_FLAG
        ),
        gate_option: LayerGateOptions | None = config.GATE_OPTION,
        gate_activation: ActivationOptions | None = config.GATE_ACTIVATION,
        recurrent_gate_option: LayerGateOptions | None = config.RECURRENT_GATE_OPTION,
        recurrent_gate_activation: ActivationOptions | None = (
            config.RECURRENT_GATE_ACTIVATION
        ),
        shared_gate_config: GateConfig | None = None,
        cluster_capacity_options: NeuronClusterCapacityOptions | None = None,
        terminal_options: NeuronTerminalOptions | None = None,
        terminal_router_options: NeuronSubmoduleStackOptions | None = None,
        terminal_sampler_options: NeuronTerminalSamplerOptions | None = None,
        cluster_halting_options: ClusterRouteHaltingOptions | None = None,
        **hidden_options: Any,
    ) -> None:
        cluster_capacity_options = (
            cluster_capacity_options
            or NeuronClusterCapacityOptions(
                x_axis_total_neurons=cluster_x_axis_total_neurons,
                y_axis_total_neurons=cluster_y_axis_total_neurons,
                z_axis_total_neurons=cluster_z_axis_total_neurons,
                initial_x_axis_total_neurons=(cluster_initial_x_axis_total_neurons),
                initial_y_axis_total_neurons=(cluster_initial_y_axis_total_neurons),
                initial_z_axis_total_neurons=(cluster_initial_z_axis_total_neurons),
                max_steps=cluster_max_steps,
                growth_threshold=cluster_growth_threshold,
            )
        )
        terminal_options = terminal_options or NeuronTerminalOptions(
            xy_axis_range=cluster_terminal_xy_axis_range,
            z_axis_range=cluster_terminal_z_axis_range,
            z_axis_offset=cluster_terminal_z_axis_offset,
            top_k=cluster_terminal_top_k,
        )
        terminal_router_options = (
            terminal_router_options
            or NeuronSubmoduleStackOptions(
                hidden_dim=cluster_terminal_router_hidden_dim,
                num_layers=cluster_terminal_router_num_layers,
                last_layer_bias_option=(cluster_terminal_router_last_layer_bias_option),
                apply_output_pipeline_flag=(
                    cluster_terminal_router_apply_output_pipeline_flag
                ),
                activation=cluster_terminal_router_activation,
                layer_norm_position=cluster_terminal_router_layer_norm_position,
                residual_connection_option=(
                    cluster_terminal_router_residual_connection_option
                ),
                dropout_probability=cluster_terminal_router_dropout_probability,
                bias_flag=cluster_terminal_router_bias_flag,
            )
        )
        terminal_sampler_options = (
            terminal_sampler_options
            or NeuronTerminalSamplerOptions(
                threshold=cluster_terminal_sampler_threshold,
                filter_above_threshold=(
                    cluster_terminal_sampler_filter_above_threshold
                ),
                num_topk_samples=cluster_terminal_sampler_num_topk_samples,
                normalize_probabilities_flag=(
                    cluster_terminal_sampler_normalize_probabilities_flag
                ),
                noisy_topk_flag=cluster_terminal_sampler_noisy_topk_flag,
                coefficient_of_variation_loss_weight=(
                    cluster_terminal_sampler_coefficient_of_variation_loss_weight
                ),
                switch_loss_weight=cluster_terminal_sampler_switch_loss_weight,
                zero_centred_loss_weight=(
                    cluster_terminal_sampler_zero_centred_loss_weight
                ),
                mutual_information_loss_weight=(
                    cluster_terminal_sampler_mutual_information_loss_weight
                ),
            )
        )
        cluster_halting_options = cluster_halting_options or ClusterRouteHaltingOptions(
            enabled=cluster_halting_flag,
            threshold=cluster_halting_threshold,
            dropout=cluster_halting_dropout,
            hidden_state_mode=cluster_halting_hidden_state_mode,
            stack_options=NeuronSubmoduleStackOptions(
                hidden_dim=cluster_halting_stack_hidden_dim,
                num_layers=cluster_halting_stack_num_layers,
                last_layer_bias_option=(cluster_halting_stack_last_layer_bias_option),
                apply_output_pipeline_flag=(
                    cluster_halting_stack_apply_output_pipeline_flag
                ),
                activation=cluster_halting_stack_activation,
                layer_norm_position=cluster_halting_stack_layer_norm_position,
                residual_connection_option=(
                    cluster_halting_stack_residual_connection_option
                ),
                dropout_probability=cluster_halting_stack_dropout_probability,
                bias_flag=cluster_halting_stack_bias_flag,
            ),
            output_dim=cluster_halting_output_dim,
        )
        hidden_flat_options = {
            "gate_option": gate_option,
            "gate_activation": gate_activation,
            "recurrent_gate_option": recurrent_gate_option,
            "recurrent_gate_activation": recurrent_gate_activation,
            **hidden_options,
        }
        if shared_gate_config is not None:
            hidden_flat_options["shared_gate_config"] = shared_gate_config

        super().__init__(
            hidden_runtime=runtime_from_flat(hidden_flat_options),
            cluster_capacity_options=cluster_capacity_options,
            terminal_options=terminal_options,
            terminal_router_options=terminal_router_options,
            terminal_sampler_options=terminal_sampler_options,
            cluster_halting_options=cluster_halting_options,
        )
