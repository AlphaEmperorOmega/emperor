from typing import Any

from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.config import ModelConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.neuron.core.options import TerminalRangeOptions, TerminalZAxisOffsetOptions

import models.neuron.neuron_linear.config as config
from models.neuron.neuron_linear._builder_options import (
    ClusterRouteHaltingOptions,
    NeuronClusterCapacityOptions,
    NeuronControllerStackOptions,
    NeuronTerminalOptions,
    NeuronTerminalSamplerOptions,
)
from models.neuron.neuron_linear._control_config_factory import (
    NeuronControlConfigFactory,
)
from models.neuron.neuron_linear._source_linear_adapter import (
    normalize_source_kwargs,
    source_linear_default_kwargs,
)
from models.neuron.neuron_linear.experiment_config import ExperimentConfig


class NeuronLinearConfigBuilder:
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
        recurrent_gate_activation: ActivationOptions | None = config.RECURRENT_GATE_ACTIVATION,
        shared_gate_config: GateConfig | None = None,
        cluster_capacity_options: NeuronClusterCapacityOptions | None = None,
        terminal_options: NeuronTerminalOptions | None = None,
        terminal_router_options: NeuronControllerStackOptions | None = None,
        terminal_sampler_options: NeuronTerminalSamplerOptions | None = None,
        cluster_halting_options: ClusterRouteHaltingOptions | None = None,
        **source_kwargs: Any,
    ) -> None:
        cluster_capacity_options = (
            cluster_capacity_options
            or NeuronClusterCapacityOptions(
                x_axis_total_neurons=cluster_x_axis_total_neurons,
                y_axis_total_neurons=cluster_y_axis_total_neurons,
                z_axis_total_neurons=cluster_z_axis_total_neurons,
                initial_x_axis_total_neurons=(
                    cluster_initial_x_axis_total_neurons
                ),
                initial_y_axis_total_neurons=(
                    cluster_initial_y_axis_total_neurons
                ),
                initial_z_axis_total_neurons=(
                    cluster_initial_z_axis_total_neurons
                ),
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
            or NeuronControllerStackOptions(
                hidden_dim=cluster_terminal_router_hidden_dim,
                num_layers=cluster_terminal_router_num_layers,
                last_layer_bias_option=(
                    cluster_terminal_router_last_layer_bias_option
                ),
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
        cluster_halting_options = (
            cluster_halting_options
            or ClusterRouteHaltingOptions(
                enabled=cluster_halting_flag,
                threshold=cluster_halting_threshold,
                dropout=cluster_halting_dropout,
                hidden_state_mode=cluster_halting_hidden_state_mode,
                stack_options=NeuronControllerStackOptions(
                    hidden_dim=cluster_halting_stack_hidden_dim,
                    num_layers=cluster_halting_stack_num_layers,
                    last_layer_bias_option=(
                        cluster_halting_stack_last_layer_bias_option
                    ),
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
        )
        self.source_kwargs = normalize_source_kwargs(source_kwargs)
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
        from models.linears.linear.config_builder import LinearConfigBuilder

        source_kwargs = {
            **source_linear_default_kwargs(),
            "gate_option": self.gate_option,
            "gate_activation": self.gate_activation,
            "recurrent_gate_option": self.recurrent_gate_option,
            "recurrent_gate_activation": self.recurrent_gate_activation,
            **self.source_kwargs,
        }
        if self.shared_gate_config is not None:
            source_kwargs["shared_gate_config"] = self.shared_gate_config
        source_cfg = LinearConfigBuilder(**source_kwargs).build()
        source_experiment_cfg = source_cfg.experiment_config
        self._validate_source_experiment_config(source_experiment_cfg)

        neuron_cluster_config = NeuronControlConfigFactory(self).build(
            source_experiment_cfg.model_config,
            source_cfg.hidden_dim,
        )

        return ModelConfig(
            learning_rate=source_cfg.learning_rate,
            batch_size=source_cfg.batch_size,
            input_dim=source_cfg.input_dim,
            hidden_dim=source_cfg.hidden_dim,
            output_dim=source_cfg.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=source_experiment_cfg.input_model_config,
                neuron_cluster_config=neuron_cluster_config,
                output_model_config=source_experiment_cfg.output_model_config,
            ),
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
                "The linear source model must use the boundary_classifier "
                f"experiment config fields. Missing: {missing_fields}"
            )
