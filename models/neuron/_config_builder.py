from __future__ import annotations

from typing import Any

from emperor.config import ModelConfig

from models.neuron._builder_options import (
    ClusterRouteHaltingOptions,
    NeuronClusterCapacityOptions,
    NeuronSubmoduleStackOptions,
    NeuronTerminalOptions,
    NeuronTerminalSamplerOptions,
)
from models.neuron._control_config_factory import (
    NeuronControlConfigDependencies,
    NeuronControlConfigFactory,
)
from models.neuron.experiment_config import ExperimentConfig
from models.neuron._source_adapter import SourcePackageAdapter


class NeuronWrapperConfigBuilder:
    def __init__(
        self,
        *,
        source_adapter: SourcePackageAdapter,
        cluster_capacity_options: NeuronClusterCapacityOptions,
        terminal_options: NeuronTerminalOptions,
        terminal_router_options: NeuronSubmoduleStackOptions,
        terminal_sampler_options: NeuronTerminalSamplerOptions,
        cluster_halting_options: ClusterRouteHaltingOptions,
        source_kwargs: dict[str, Any],
        shared_gate_config=None,
        gate_option=None,
        gate_activation=None,
        recurrent_gate_option=None,
        recurrent_gate_activation=None,
        experiment_config_type: type[ExperimentConfig] = ExperimentConfig,
    ) -> None:
        self.source_adapter = source_adapter
        self.source_kwargs = source_adapter.normalize_source_kwargs(source_kwargs)
        self.experiment_config_type = experiment_config_type
        self.shared_gate_config = shared_gate_config
        self.gate_option = gate_option
        self.gate_activation = gate_activation
        self.recurrent_gate_option = recurrent_gate_option
        self.recurrent_gate_activation = recurrent_gate_activation
        self.cluster_capacity_options = cluster_capacity_options
        self.cluster_x_axis_total_neurons = (
            cluster_capacity_options.x_axis_total_neurons
        )
        self.cluster_y_axis_total_neurons = (
            cluster_capacity_options.y_axis_total_neurons
        )
        self.cluster_z_axis_total_neurons = (
            cluster_capacity_options.z_axis_total_neurons
        )
        self.cluster_initial_x_axis_total_neurons = (
            cluster_capacity_options.initial_x_axis_total_neurons
        )
        self.cluster_initial_y_axis_total_neurons = (
            cluster_capacity_options.initial_y_axis_total_neurons
        )
        self.cluster_initial_z_axis_total_neurons = (
            cluster_capacity_options.initial_z_axis_total_neurons
        )
        self.cluster_max_steps = cluster_capacity_options.max_steps
        self.cluster_growth_threshold = cluster_capacity_options.growth_threshold
        self.terminal_options = terminal_options
        self.cluster_terminal_xy_axis_range = terminal_options.xy_axis_range
        self.cluster_terminal_z_axis_range = terminal_options.z_axis_range
        self.cluster_terminal_z_axis_offset = terminal_options.z_axis_offset
        self.cluster_terminal_top_k = terminal_options.top_k
        self.terminal_router_options = terminal_router_options
        self.cluster_terminal_router_num_layers = terminal_router_options.num_layers
        self.cluster_terminal_router_hidden_dim = terminal_router_options.hidden_dim
        self.cluster_terminal_router_activation = terminal_router_options.activation
        self.cluster_terminal_router_layer_norm_position = (
            terminal_router_options.layer_norm_position
        )
        self.cluster_terminal_router_residual_connection_option = (
            terminal_router_options.residual_connection_option
        )
        self.cluster_terminal_router_dropout_probability = (
            terminal_router_options.dropout_probability
        )
        self.cluster_terminal_router_last_layer_bias_option = (
            terminal_router_options.last_layer_bias_option
        )
        self.cluster_terminal_router_apply_output_pipeline_flag = (
            terminal_router_options.apply_output_pipeline_flag
        )
        self.cluster_terminal_router_bias_flag = terminal_router_options.bias_flag
        self.terminal_sampler_options = terminal_sampler_options
        self.cluster_terminal_sampler_threshold = terminal_sampler_options.threshold
        self.cluster_terminal_sampler_filter_above_threshold = (
            terminal_sampler_options.filter_above_threshold
        )
        self.cluster_terminal_sampler_num_topk_samples = (
            terminal_sampler_options.num_topk_samples
        )
        self.cluster_terminal_sampler_normalize_probabilities_flag = (
            terminal_sampler_options.normalize_probabilities_flag
        )
        self.cluster_terminal_sampler_noisy_topk_flag = (
            terminal_sampler_options.noisy_topk_flag
        )
        self.cluster_terminal_sampler_coefficient_of_variation_loss_weight = (
            terminal_sampler_options.coefficient_of_variation_loss_weight
        )
        self.cluster_terminal_sampler_switch_loss_weight = (
            terminal_sampler_options.switch_loss_weight
        )
        self.cluster_terminal_sampler_zero_centred_loss_weight = (
            terminal_sampler_options.zero_centred_loss_weight
        )
        self.cluster_terminal_sampler_mutual_information_loss_weight = (
            terminal_sampler_options.mutual_information_loss_weight
        )
        self.cluster_halting_options = cluster_halting_options
        self.cluster_halting_flag = cluster_halting_options.enabled
        self.cluster_halting_threshold = cluster_halting_options.threshold
        self.cluster_halting_dropout = cluster_halting_options.dropout
        self.cluster_halting_hidden_state_mode = (
            cluster_halting_options.hidden_state_mode
        )
        self.cluster_halting_stack_options = cluster_halting_options.stack_options
        self.cluster_halting_stack_hidden_dim = (
            self.cluster_halting_stack_options.hidden_dim
        )
        self.cluster_halting_output_dim = cluster_halting_options.output_dim
        self.cluster_halting_stack_layer_norm_position = (
            self.cluster_halting_stack_options.layer_norm_position
        )
        self.cluster_halting_stack_num_layers = (
            self.cluster_halting_stack_options.num_layers
        )
        self.cluster_halting_stack_activation = (
            self.cluster_halting_stack_options.activation
        )
        self.cluster_halting_stack_residual_connection_option = (
            self.cluster_halting_stack_options.residual_connection_option
        )
        self.cluster_halting_stack_dropout_probability = (
            self.cluster_halting_stack_options.dropout_probability
        )
        self.cluster_halting_stack_last_layer_bias_option = (
            self.cluster_halting_stack_options.last_layer_bias_option
        )
        self.cluster_halting_stack_apply_output_pipeline_flag = (
            self.cluster_halting_stack_options.apply_output_pipeline_flag
        )
        self.cluster_halting_stack_bias_flag = (
            self.cluster_halting_stack_options.bias_flag
        )

    def build(self) -> ModelConfig:
        source_flat_kwargs = {
            "gate_option": self.gate_option,
            "gate_activation": self.gate_activation,
            "recurrent_gate_option": self.recurrent_gate_option,
            "recurrent_gate_activation": self.recurrent_gate_activation,
            **self.source_kwargs,
        }
        if self.shared_gate_config is not None:
            source_flat_kwargs["shared_gate_config"] = self.shared_gate_config

        source_cfg = self.source_adapter.build_source_config(source_flat_kwargs)
        source_experiment_cfg = source_cfg.experiment_config
        self._validate_source_experiment_config(source_experiment_cfg)

        neuron_dependencies = self.__neuron_control_config_dependencies()
        neuron_control_factory = NeuronControlConfigFactory(neuron_dependencies)
        neuron_cluster_config = neuron_control_factory.build(
            source_experiment_cfg.model_config,
            source_cfg.hidden_dim,
        )

        return ModelConfig(
            learning_rate=source_cfg.learning_rate,
            batch_size=source_cfg.batch_size,
            input_dim=source_cfg.input_dim,
            hidden_dim=source_cfg.hidden_dim,
            output_dim=source_cfg.output_dim,
            experiment_config=self.experiment_config_type(
                input_model_config=source_experiment_cfg.input_model_config,
                neuron_cluster_config=neuron_cluster_config,
                output_model_config=source_experiment_cfg.output_model_config,
            ),
        )

    def __neuron_control_config_dependencies(
        self,
    ) -> NeuronControlConfigDependencies:
        return NeuronControlConfigDependencies(
            cluster_capacity_options=self.cluster_capacity_options,
            terminal_options=self.terminal_options,
            terminal_router_options=self.terminal_router_options,
            terminal_sampler_options=self.terminal_sampler_options,
            cluster_halting_options=self.cluster_halting_options,
        )

    def _validate_source_experiment_config(self, source_experiment_cfg) -> None:
        required_fields = {
            "input_model_config",
            "model_config",
            "output_model_config",
        }
        missing_fields = [
            field
            for field in sorted(required_fields)
            if not hasattr(source_experiment_cfg, field)
        ]
        if missing_fields:
            raise TypeError(
                "The Neuron source model must use the boundary_classifier "
                f"experiment config fields. Missing: {missing_fields}"
            )
