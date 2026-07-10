from __future__ import annotations

from emperor.config import ModelConfig

from models.neuron.expert_linear_adaptive._hidden._hidden_model_config_factory import (
    HiddenModelConfigFactory,
)
from models.neuron.expert_linear_adaptive._hidden.runtime_options import RuntimeOptions
from models.neuron.expert_linear_adaptive._neuron_control_config_factory import (
    NeuronControlConfigDependencies,
    NeuronControlConfigFactory,
)
from models.neuron.expert_linear_adaptive.experiment_config import ExperimentConfig
from models.neuron.expert_linear_adaptive.runtime_options import (
    ClusterRouteHaltingOptions,
    NeuronClusterCapacityOptions,
    NeuronSubmoduleStackOptions,
    NeuronTerminalOptions,
    NeuronTerminalSamplerOptions,
)


class NeuronConfigBuilder:
    def __init__(
        self,
        *,
        hidden_runtime: RuntimeOptions,
        cluster_capacity_options: NeuronClusterCapacityOptions,
        terminal_options: NeuronTerminalOptions,
        terminal_router_options: NeuronSubmoduleStackOptions,
        terminal_sampler_options: NeuronTerminalSamplerOptions,
        cluster_halting_options: ClusterRouteHaltingOptions,
    ) -> None:
        self.hidden_runtime = hidden_runtime
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
        hidden_factory = HiddenModelConfigFactory(self.hidden_runtime)
        neuron_dependencies = self.__neuron_control_config_dependencies()
        neuron_control_factory = NeuronControlConfigFactory(neuron_dependencies)
        neuron_cluster_config = neuron_control_factory.build(
            hidden_factory.build_hidden_model_config(),
            hidden_factory.hidden_dim,
        )

        return ModelConfig(
            learning_rate=self.hidden_runtime.learning_rate,
            batch_size=self.hidden_runtime.batch_size,
            input_dim=self.hidden_runtime.input_dim,
            hidden_dim=hidden_factory.hidden_dim,
            output_dim=self.hidden_runtime.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=hidden_factory.build_input_model_config(),
                neuron_cluster_config=neuron_cluster_config,
                output_model_config=hidden_factory.build_output_model_config(),
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
