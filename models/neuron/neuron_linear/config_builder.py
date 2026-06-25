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
from models.neuron.neuron_linear._control_config_factory import (
    NeuronControlConfigFactory,
)
from models.neuron.neuron_linear.experiment_config import ExperimentConfig

SOURCE_LINEAR_KWARG_ALIASES = {
    "gate_hidden_dim": "gate_stack_hidden_dim",
    "gate_layer_norm_position": "gate_stack_layer_norm_position",
    "gate_bias_flag": "gate_stack_bias_flag",
    "halting_hidden_dim": "halting_stack_hidden_dim",
    "halting_layer_norm_position": "halting_stack_layer_norm_position",
    "halting_bias_flag": "halting_stack_bias_flag",
}


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
        **source_kwargs: Any,
    ) -> None:
        self.source_kwargs = self._normalize_source_kwargs(source_kwargs)
        self.shared_gate_config = shared_gate_config
        self.gate_option = gate_option
        self.gate_activation = gate_activation
        self.recurrent_gate_option = recurrent_gate_option
        self.recurrent_gate_activation = recurrent_gate_activation
        self.cluster_x_axis_total_neurons = cluster_x_axis_total_neurons
        self.cluster_y_axis_total_neurons = cluster_y_axis_total_neurons
        self.cluster_z_axis_total_neurons = cluster_z_axis_total_neurons
        self.cluster_initial_x_axis_total_neurons = cluster_initial_x_axis_total_neurons
        self.cluster_initial_y_axis_total_neurons = cluster_initial_y_axis_total_neurons
        self.cluster_initial_z_axis_total_neurons = cluster_initial_z_axis_total_neurons
        self.cluster_max_steps = cluster_max_steps
        self.cluster_growth_threshold = cluster_growth_threshold
        self.cluster_terminal_xy_axis_range = cluster_terminal_xy_axis_range
        self.cluster_terminal_z_axis_range = cluster_terminal_z_axis_range
        self.cluster_terminal_z_axis_offset = cluster_terminal_z_axis_offset
        self.cluster_terminal_top_k = cluster_terminal_top_k
        self.cluster_terminal_router_num_layers = cluster_terminal_router_num_layers
        self.cluster_terminal_router_hidden_dim = cluster_terminal_router_hidden_dim
        self.cluster_terminal_router_activation = cluster_terminal_router_activation
        self.cluster_terminal_router_layer_norm_position = (
            cluster_terminal_router_layer_norm_position
        )
        self.cluster_terminal_router_residual_connection_option = (
            cluster_terminal_router_residual_connection_option
        )
        self.cluster_terminal_router_dropout_probability = (
            cluster_terminal_router_dropout_probability
        )
        self.cluster_terminal_router_last_layer_bias_option = (
            cluster_terminal_router_last_layer_bias_option
        )
        self.cluster_terminal_router_apply_output_pipeline_flag = (
            cluster_terminal_router_apply_output_pipeline_flag
        )
        self.cluster_terminal_router_bias_flag = cluster_terminal_router_bias_flag
        self.cluster_terminal_sampler_threshold = cluster_terminal_sampler_threshold
        self.cluster_terminal_sampler_filter_above_threshold = (
            cluster_terminal_sampler_filter_above_threshold
        )
        self.cluster_terminal_sampler_num_topk_samples = (
            cluster_terminal_sampler_num_topk_samples
        )
        self.cluster_terminal_sampler_normalize_probabilities_flag = (
            cluster_terminal_sampler_normalize_probabilities_flag
        )
        self.cluster_terminal_sampler_noisy_topk_flag = (
            cluster_terminal_sampler_noisy_topk_flag
        )
        self.cluster_terminal_sampler_coefficient_of_variation_loss_weight = (
            cluster_terminal_sampler_coefficient_of_variation_loss_weight
        )
        self.cluster_terminal_sampler_switch_loss_weight = (
            cluster_terminal_sampler_switch_loss_weight
        )
        self.cluster_terminal_sampler_zero_centred_loss_weight = (
            cluster_terminal_sampler_zero_centred_loss_weight
        )
        self.cluster_terminal_sampler_mutual_information_loss_weight = (
            cluster_terminal_sampler_mutual_information_loss_weight
        )
        self.cluster_halting_flag = cluster_halting_flag
        self.cluster_halting_threshold = cluster_halting_threshold
        self.cluster_halting_dropout = cluster_halting_dropout
        self.cluster_halting_hidden_state_mode = cluster_halting_hidden_state_mode
        self.cluster_halting_stack_hidden_dim = cluster_halting_stack_hidden_dim
        self.cluster_halting_output_dim = cluster_halting_output_dim
        self.cluster_halting_stack_layer_norm_position = (
            cluster_halting_stack_layer_norm_position
        )
        self.cluster_halting_stack_num_layers = cluster_halting_stack_num_layers
        self.cluster_halting_stack_activation = cluster_halting_stack_activation
        self.cluster_halting_stack_residual_connection_option = (
            cluster_halting_stack_residual_connection_option
        )
        self.cluster_halting_stack_dropout_probability = (
            cluster_halting_stack_dropout_probability
        )
        self.cluster_halting_stack_last_layer_bias_option = (
            cluster_halting_stack_last_layer_bias_option
        )
        self.cluster_halting_stack_apply_output_pipeline_flag = (
            cluster_halting_stack_apply_output_pipeline_flag
        )
        self.cluster_halting_stack_bias_flag = cluster_halting_stack_bias_flag

    def build(self) -> ModelConfig:
        from models.linears.linear.config_builder import LinearConfigBuilder

        source_kwargs = {
            **self._source_linear_defaults(),
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

    def _source_linear_defaults(self) -> dict[str, Any]:
        return {
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "input_dim": config.INPUT_DIM,
            "hidden_dim": config.HIDDEN_DIM,
            "output_dim": config.OUTPUT_DIM,
            "bias_flag": config.BIAS_FLAG,
            "layer_norm_position": config.LAYER_NORM_POSITION,
            "stack_num_layers": config.STACK_NUM_LAYERS,
            "stack_activation": config.STACK_ACTIVATION,
            "stack_residual_connection_option": config.STACK_RESIDUAL_CONNECTION_OPTION,
            "stack_dropout_probability": config.STACK_DROPOUT_PROBABILITY,
            "stack_last_layer_bias_option": config.STACK_LAST_LAYER_BIAS_OPTION,
            "stack_apply_output_pipeline_flag": config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            "stack_gate_flag": config.GATE_FLAG,
            "gate_option": config.GATE_OPTION,
            "gate_activation": config.GATE_ACTIVATION,
            "gate_stack_hidden_dim": config.GATE_STACK_HIDDEN_DIM,
            "gate_stack_layer_norm_position": config.GATE_STACK_LAYER_NORM_POSITION,
            "gate_stack_num_layers": config.GATE_STACK_NUM_LAYERS,
            "gate_stack_activation": config.GATE_STACK_ACTIVATION,
            "gate_stack_residual_connection_option": config.GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            "gate_stack_dropout_probability": config.GATE_STACK_DROPOUT_PROBABILITY,
            "gate_stack_last_layer_bias_option": (
                config.GATE_STACK_LAST_LAYER_BIAS_OPTION
            ),
            "gate_stack_apply_output_pipeline_flag": (
                config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            "gate_stack_bias_flag": config.GATE_STACK_BIAS_FLAG,
            "stack_halting_flag": config.HALTING_FLAG,
            "halting_threshold": config.HALTING_THRESHOLD,
            "halting_dropout": config.HALTING_DROPOUT,
            "halting_hidden_state_mode": config.HALTING_HIDDEN_STATE_MODE,
            "halting_stack_hidden_dim": config.HALTING_STACK_HIDDEN_DIM,
            "halting_output_dim": config.HALTING_OUTPUT_DIM,
            "halting_stack_layer_norm_position": (
                config.HALTING_STACK_LAYER_NORM_POSITION
            ),
            "halting_stack_num_layers": config.HALTING_STACK_NUM_LAYERS,
            "halting_stack_activation": config.HALTING_STACK_ACTIVATION,
            "halting_stack_residual_connection_option": config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
            "halting_stack_dropout_probability": (
                config.HALTING_STACK_DROPOUT_PROBABILITY
            ),
            "halting_stack_last_layer_bias_option": (
                config.HALTING_STACK_LAST_LAYER_BIAS_OPTION
            ),
            "halting_stack_apply_output_pipeline_flag": (
                config.HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            "halting_stack_bias_flag": config.HALTING_STACK_BIAS_FLAG,
            "recurrent_flag": config.RECURRENT_FLAG,
            "recurrent_max_steps": config.RECURRENT_MAX_STEPS,
            "recurrent_gate_flag": config.RECURRENT_GATE_FLAG,
            "recurrent_gate_option": config.RECURRENT_GATE_OPTION,
            "recurrent_gate_activation": config.RECURRENT_GATE_ACTIVATION,
            "recurrent_halting_flag": config.RECURRENT_HALTING_FLAG,
        }

    @staticmethod
    def _normalize_source_kwargs(source_kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            SOURCE_LINEAR_KWARG_ALIASES.get(key, key): value
            for key, value in source_kwargs.items()
        }

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
