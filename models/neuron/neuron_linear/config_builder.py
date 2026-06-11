import copy
from typing import Any

from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.config import ModelConfig
from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import LinearLayerConfig
from emperor.neuron.config import NeuronClusterConfig, NeuronConfig
from emperor.neuron.core.config import AxonsConfig, NucleusConfig, TerminalConfig
from emperor.neuron.core.options import TerminalRangeOptions, TerminalZAxisOffsetOptions
from emperor.sampler.core.config import RouterConfig, SamplerConfig

import models.neuron.neuron_linear.config as config
from models.linears.linear.config_builder import LinearConfigBuilder
from models.neuron.neuron_linear.experiment_config import (
    ExperimentConfig,
    HiddenBlockConfig,
)


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
        cluster_terminal_router_residual_flag: bool = (
            config.CLUSTER_TERMINAL_ROUTER_RESIDUAL_FLAG
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
        cluster_halting_hidden_dim: int = config.CLUSTER_HALTING_HIDDEN_DIM,
        cluster_halting_output_dim: int = config.CLUSTER_HALTING_OUTPUT_DIM,
        cluster_halting_layer_norm_position: LayerNormPositionOptions = (
            config.CLUSTER_HALTING_LAYER_NORM_POSITION
        ),
        cluster_halting_stack_num_layers: int = (
            config.CLUSTER_HALTING_STACK_NUM_LAYERS
        ),
        cluster_halting_stack_activation: ActivationOptions = (
            config.CLUSTER_HALTING_STACK_ACTIVATION
        ),
        cluster_halting_stack_residual_flag: bool = (
            config.CLUSTER_HALTING_STACK_RESIDUAL_FLAG
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
        cluster_halting_bias_flag: bool = config.CLUSTER_HALTING_BIAS_FLAG,
        **source_kwargs: Any,
    ) -> None:
        self.source_kwargs = source_kwargs
        self.cluster_x_axis_total_neurons = cluster_x_axis_total_neurons
        self.cluster_y_axis_total_neurons = cluster_y_axis_total_neurons
        self.cluster_z_axis_total_neurons = cluster_z_axis_total_neurons
        self.cluster_initial_x_axis_total_neurons = (
            cluster_initial_x_axis_total_neurons
        )
        self.cluster_initial_y_axis_total_neurons = (
            cluster_initial_y_axis_total_neurons
        )
        self.cluster_initial_z_axis_total_neurons = (
            cluster_initial_z_axis_total_neurons
        )
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
        self.cluster_terminal_router_residual_flag = (
            cluster_terminal_router_residual_flag
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
        self.cluster_halting_hidden_dim = cluster_halting_hidden_dim
        self.cluster_halting_output_dim = cluster_halting_output_dim
        self.cluster_halting_layer_norm_position = cluster_halting_layer_norm_position
        self.cluster_halting_stack_num_layers = cluster_halting_stack_num_layers
        self.cluster_halting_stack_activation = cluster_halting_stack_activation
        self.cluster_halting_stack_residual_flag = cluster_halting_stack_residual_flag
        self.cluster_halting_stack_dropout_probability = (
            cluster_halting_stack_dropout_probability
        )
        self.cluster_halting_stack_last_layer_bias_option = (
            cluster_halting_stack_last_layer_bias_option
        )
        self.cluster_halting_stack_apply_output_pipeline_flag = (
            cluster_halting_stack_apply_output_pipeline_flag
        )
        self.cluster_halting_bias_flag = cluster_halting_bias_flag

    def build(self) -> ModelConfig:
        source_cfg = LinearConfigBuilder(
            **{
                **self._source_linear_defaults(),
                **self.source_kwargs,
            }
        ).build()
        source_experiment_cfg = source_cfg.experiment_config
        self._validate_source_experiment_config(source_experiment_cfg)

        hidden_block_config = self._build_hidden_block_config(
            source_experiment_cfg.model_config,
            source_cfg.hidden_dim,
        )
        neuron_cluster_config = self._build_neuron_cluster_config(
            hidden_block_config,
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
            "stack_residual_flag": config.STACK_RESIDUAL_FLAG,
            "stack_dropout_probability": config.STACK_DROPOUT_PROBABILITY,
            "stack_last_layer_bias_option": config.STACK_LAST_LAYER_BIAS_OPTION,
            "stack_apply_output_pipeline_flag": config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
            "stack_gate_flag": config.GATE_FLAG,
            "gate_hidden_dim": config.GATE_HIDDEN_DIM,
            "gate_layer_norm_position": config.GATE_LAYER_NORM_POSITION,
            "gate_stack_num_layers": config.GATE_STACK_NUM_LAYERS,
            "gate_stack_activation": config.GATE_STACK_ACTIVATION,
            "gate_stack_residual_flag": config.GATE_STACK_RESIDUAL_FLAG,
            "gate_stack_dropout_probability": config.GATE_STACK_DROPOUT_PROBABILITY,
            "gate_stack_last_layer_bias_option": (
                config.GATE_STACK_LAST_LAYER_BIAS_OPTION
            ),
            "gate_stack_apply_output_pipeline_flag": (
                config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            "gate_bias_flag": config.GATE_BIAS_FLAG,
            "stack_halting_flag": config.HALTING_FLAG,
            "halting_threshold": config.HALTING_THRESHOLD,
            "halting_dropout": config.HALTING_DROPOUT,
            "halting_hidden_state_mode": config.HALTING_HIDDEN_STATE_MODE,
            "halting_hidden_dim": config.HALTING_HIDDEN_DIM,
            "halting_output_dim": config.HALTING_OUTPUT_DIM,
            "halting_layer_norm_position": config.HALTING_LAYER_NORM_POSITION,
            "halting_stack_num_layers": config.HALTING_STACK_NUM_LAYERS,
            "halting_stack_activation": config.HALTING_STACK_ACTIVATION,
            "halting_stack_residual_flag": config.HALTING_STACK_RESIDUAL_FLAG,
            "halting_stack_dropout_probability": (
                config.HALTING_STACK_DROPOUT_PROBABILITY
            ),
            "halting_stack_last_layer_bias_option": (
                config.HALTING_STACK_LAST_LAYER_BIAS_OPTION
            ),
            "halting_stack_apply_output_pipeline_flag": (
                config.HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            "halting_bias_flag": config.HALTING_BIAS_FLAG,
            "recurrent_flag": config.RECURRENT_FLAG,
            "recurrent_max_steps": config.RECURRENT_MAX_STEPS,
            "recurrent_gate_flag": config.RECURRENT_GATE_FLAG,
            "recurrent_halting_flag": config.RECURRENT_HALTING_FLAG,
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

    def _build_hidden_block_config(
        self,
        source_hidden_model_config,
        hidden_dim: int,
    ) -> HiddenBlockConfig:
        return HiddenBlockConfig(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            model_config=copy.deepcopy(source_hidden_model_config),
        )

    def _build_neuron_cluster_config(
        self,
        hidden_block_config: HiddenBlockConfig,
        hidden_dim: int,
    ) -> NeuronClusterConfig:
        cluster_terminal_sampler_config = self._build_cluster_terminal_sampler_config(
            hidden_dim
        )
        neuron_config = NeuronConfig(
            nucleus_config=NucleusConfig(model_config=hidden_block_config),
            axons_config=AxonsConfig(memory_config=None),
            terminal_config=TerminalConfig(
                input_dim=hidden_dim,
                xy_axis_range=self.cluster_terminal_xy_axis_range,
                z_axis_range=self.cluster_terminal_z_axis_range,
                z_axis_offset=self.cluster_terminal_z_axis_offset,
                sampler_config=cluster_terminal_sampler_config,
            ),
        )
        return NeuronClusterConfig(
            x_axis_total_neurons=self.cluster_x_axis_total_neurons,
            y_axis_total_neurons=self.cluster_y_axis_total_neurons,
            z_axis_total_neurons=self.cluster_z_axis_total_neurons,
            initial_x_axis_total_neurons=self.cluster_initial_x_axis_total_neurons,
            initial_y_axis_total_neurons=self.cluster_initial_y_axis_total_neurons,
            initial_z_axis_total_neurons=self.cluster_initial_z_axis_total_neurons,
            entry_sampler_config=None,
            max_steps=self.cluster_max_steps,
            growth_threshold=self.cluster_growth_threshold,
            halting_config=self._build_cluster_halting_config(hidden_dim),
            neuron_config=neuron_config,
        )

    def _build_cluster_terminal_sampler_config(self, hidden_dim: int) -> SamplerConfig:
        num_experts = self._cluster_terminal_num_experts()
        top_k = min(max(1, self.cluster_terminal_top_k), num_experts)
        return SamplerConfig(
            top_k=top_k,
            threshold=self.cluster_terminal_sampler_threshold,
            filter_above_threshold=self.cluster_terminal_sampler_filter_above_threshold,
            num_topk_samples=min(self.cluster_terminal_sampler_num_topk_samples, top_k),
            normalize_probabilities_flag=(
                self.cluster_terminal_sampler_normalize_probabilities_flag
            ),
            noisy_topk_flag=self.cluster_terminal_sampler_noisy_topk_flag,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=(
                self.cluster_terminal_sampler_coefficient_of_variation_loss_weight
            ),
            switch_loss_weight=self.cluster_terminal_sampler_switch_loss_weight,
            zero_centred_loss_weight=(
                self.cluster_terminal_sampler_zero_centred_loss_weight
            ),
            mutual_information_loss_weight=(
                self.cluster_terminal_sampler_mutual_information_loss_weight
            ),
            router_config=self._build_router_config(hidden_dim, num_experts),
        )

    def _build_router_config(
        self,
        hidden_dim: int,
        num_experts: int,
    ) -> RouterConfig:
        router_hidden_dim = self.cluster_terminal_router_hidden_dim or max(
            hidden_dim,
            num_experts,
        )
        return RouterConfig(
            input_dim=hidden_dim,
            num_experts=num_experts,
            noisy_topk_flag=self.cluster_terminal_sampler_noisy_topk_flag,
            model_config=LayerStackConfig(
                input_dim=hidden_dim,
                hidden_dim=router_hidden_dim,
                output_dim=num_experts,
                num_layers=self.cluster_terminal_router_num_layers,
                last_layer_bias_option=(
                    self.cluster_terminal_router_last_layer_bias_option
                ),
                apply_output_pipeline_flag=(
                    self.cluster_terminal_router_apply_output_pipeline_flag
                ),
                layer_config=LayerConfig(
                    activation=self.cluster_terminal_router_activation,
                    residual_flag=self.cluster_terminal_router_residual_flag,
                    dropout_probability=(
                        self.cluster_terminal_router_dropout_probability
                    ),
                    layer_norm_position=(
                        self.cluster_terminal_router_layer_norm_position
                    ),
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    shared_halting_flag=False,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=self.cluster_terminal_router_bias_flag
                    ),
                ),
            ),
        )

    def _build_cluster_halting_config(
        self,
        hidden_dim: int,
    ) -> StickBreakingConfig | None:
        if not self.cluster_halting_flag:
            return None
        return StickBreakingConfig(
            input_dim=hidden_dim,
            threshold=self.cluster_halting_threshold,
            halting_dropout=self.cluster_halting_dropout,
            hidden_state_mode=self.cluster_halting_hidden_state_mode,
            halting_gate_config=LayerStackConfig(
                input_dim=hidden_dim,
                hidden_dim=self.cluster_halting_hidden_dim,
                output_dim=self.cluster_halting_output_dim,
                num_layers=self.cluster_halting_stack_num_layers,
                last_layer_bias_option=(
                    self.cluster_halting_stack_last_layer_bias_option
                ),
                apply_output_pipeline_flag=(
                    self.cluster_halting_stack_apply_output_pipeline_flag
                ),
                layer_config=LayerConfig(
                    activation=self.cluster_halting_stack_activation,
                    residual_flag=self.cluster_halting_stack_residual_flag,
                    dropout_probability=(
                        self.cluster_halting_stack_dropout_probability
                    ),
                    layer_norm_position=self.cluster_halting_layer_norm_position,
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    shared_halting_flag=False,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=self.cluster_halting_bias_flag
                    ),
                ),
            ),
        )

    def _cluster_terminal_num_experts(self) -> int:
        xy_range = self._enum_or_int_value(self.cluster_terminal_xy_axis_range)
        z_range = self._enum_or_int_value(self.cluster_terminal_z_axis_range)
        return (xy_range * 2 + 1) ** 2 * (z_range + 1)

    def _enum_or_int_value(self, value) -> int:
        return int(value.value if hasattr(value, "value") else value)
