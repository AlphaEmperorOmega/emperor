from emperor.base.layer.residual import ResidualConnectionOptions
import models.experts.experts_linear.config as config

from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer.config import (
    LayerConfig,
)
from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.linears.core.config import LinearLayerConfig
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.halting.options import HaltingHiddenStateModeOptions
from models.experts._builder_options import (
    ExpertsControllerStackOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
)
from models.experts.experts_linear._control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
)
from models.experts.experts_linear.experiment_config import ExperimentConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExpertsLinearConfigBuilder:
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        stack_hidden_dim: int = config.STACK_HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        stack_bias_flag: bool = config.STACK_BIAS_FLAG,
        layer_norm_position: LayerNormPositionOptions = config.LAYER_NORM_POSITION,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_residual_connection_option: ResidualConnectionOptions = config.STACK_RESIDUAL_CONNECTION_OPTION,
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
        stack_last_layer_bias_option: LastLayerBiasOptions = config.STACK_LAST_LAYER_BIAS_OPTION,
        stack_apply_output_pipeline_flag: bool = config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        top_k: int = config.EXPERT_TOP_K,
        num_experts: int = config.EXPERT_NUM_EXPERTS,
        capacity_factor: float = config.EXPERT_CAPACITY_FACTOR,
        dropped_token_behavior: DroppedTokenOptions = config.EXPERT_DROPPED_TOKEN_BEHAVIOR,
        compute_expert_mixture_flag: bool = config.EXPERT_COMPUTE_EXPERT_MIXTURE_FLAG,
        weighted_parameters_flag: bool = config.EXPERT_WEIGHTED_PARAMETERS_FLAG,
        weighting_position_option: ExpertWeightingPositionOptions = config.EXPERT_WEIGHTING_POSITION_OPTION,
        routing_initialization_mode: RoutingInitializationMode = config.EXPERT_ROUTING_INITIALIZATION_MODE,
        expert_stack_num_layers: int = config.EXPERT_STACK_NUM_LAYERS,
        expert_stack_activation: ActivationOptions = config.EXPERT_STACK_ACTIVATION,
        expert_stack_residual_connection_option: ResidualConnectionOptions = config.EXPERT_STACK_RESIDUAL_CONNECTION_OPTION,
        expert_stack_dropout_probability: float = config.EXPERT_STACK_DROPOUT_PROBABILITY,
        expert_stack_layer_norm_position: LayerNormPositionOptions = config.EXPERT_STACK_LAYER_NORM_POSITION,
        expert_stack_last_layer_bias_option: LastLayerBiasOptions = config.EXPERT_STACK_LAST_LAYER_BIAS_OPTION,
        expert_stack_apply_output_pipeline_flag: bool = config.EXPERT_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        expert_bias_flag: bool = config.EXPERT_BIAS_FLAG,
        sampler_threshold: float = config.SAMPLER_THRESHOLD,
        sampler_filter_above_threshold: bool = config.SAMPLER_FILTER_ABOVE_THRESHOLD,
        sampler_num_topk_samples: int = config.SAMPLER_NUM_TOPK_SAMPLES,
        sampler_normalize_probabilities_flag: bool = config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
        sampler_noisy_topk_flag: bool = config.SAMPLER_NOISY_TOPK_FLAG,
        sampler_coefficient_of_variation_loss_weight: float = config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
        sampler_switch_loss_weight: float = config.SAMPLER_SWITCH_LOSS_WEIGHT,
        sampler_zero_centred_loss_weight: float = config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT,
        sampler_mutual_information_loss_weight: float = config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT,
        router_noisy_topk_flag: bool = config.ROUTER_NOISY_TOPK_FLAG,
        sampler_stack_num_layers: int = config.SAMPLER_STACK_NUM_LAYERS,
        sampler_stack_activation: ActivationOptions = config.SAMPLER_STACK_ACTIVATION,
        sampler_stack_residual_connection_option: ResidualConnectionOptions = config.SAMPLER_STACK_RESIDUAL_CONNECTION_OPTION,
        sampler_stack_dropout_probability: float = config.SAMPLER_STACK_DROPOUT_PROBABILITY,
        sampler_stack_layer_norm_position: LayerNormPositionOptions = config.SAMPLER_STACK_LAYER_NORM_POSITION,
        sampler_stack_last_layer_bias_option: LastLayerBiasOptions = config.SAMPLER_STACK_LAST_LAYER_BIAS_OPTION,
        sampler_stack_apply_output_pipeline_flag: bool = config.SAMPLER_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        sampler_bias_flag: bool = config.SAMPLER_BIAS_FLAG,
        stack_gate_flag: bool = config.GATE_FLAG,
        gate_option: LayerGateOptions | None = config.GATE_OPTION,
        gate_activation: ActivationOptions | None = config.GATE_ACTIVATION,
        gate_stack_hidden_dim: int = config.GATE_STACK_HIDDEN_DIM,
        gate_stack_layer_norm_position: LayerNormPositionOptions = config.GATE_STACK_LAYER_NORM_POSITION,
        gate_stack_num_layers: int = config.GATE_STACK_NUM_LAYERS,
        gate_stack_activation: ActivationOptions = config.GATE_STACK_ACTIVATION,
        gate_stack_residual_connection_option: ResidualConnectionOptions = config.GATE_STACK_RESIDUAL_CONNECTION_OPTION,
        gate_stack_dropout_probability: float = config.GATE_STACK_DROPOUT_PROBABILITY,
        gate_stack_last_layer_bias_option: LastLayerBiasOptions = config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
        gate_stack_apply_output_pipeline_flag: bool = config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        gate_stack_bias_flag: bool = config.GATE_STACK_BIAS_FLAG,
        stack_halting_flag: bool = config.HALTING_FLAG,
        halting_threshold: float = config.HALTING_THRESHOLD,
        halting_dropout: float = config.HALTING_DROPOUT,
        halting_hidden_state_mode: HaltingHiddenStateModeOptions = config.HALTING_HIDDEN_STATE_MODE,
        halting_stack_hidden_dim: int = config.HALTING_STACK_HIDDEN_DIM,
        halting_output_dim: int = config.HALTING_OUTPUT_DIM,
        halting_stack_layer_norm_position: LayerNormPositionOptions = config.HALTING_STACK_LAYER_NORM_POSITION,
        halting_stack_num_layers: int = config.HALTING_STACK_NUM_LAYERS,
        halting_stack_activation: ActivationOptions = config.HALTING_STACK_ACTIVATION,
        halting_stack_residual_connection_option: ResidualConnectionOptions = config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        halting_stack_dropout_probability: float = config.HALTING_STACK_DROPOUT_PROBABILITY,
        halting_stack_last_layer_bias_option: LastLayerBiasOptions = config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        halting_stack_apply_output_pipeline_flag: bool = config.HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        halting_stack_bias_flag: bool = config.HALTING_STACK_BIAS_FLAG,
        recurrent_flag: bool = config.RECURRENT_FLAG,
        recurrent_max_steps: int = config.RECURRENT_MAX_STEPS,
        recurrent_layer_norm_position: LayerNormPositionOptions = config.RECURRENT_LAYER_NORM_POSITION,
        recurrent_gate_flag: bool = config.RECURRENT_GATE_FLAG,
        recurrent_gate_option: LayerGateOptions | None = config.RECURRENT_GATE_OPTION,
        recurrent_gate_activation: ActivationOptions | None = config.RECURRENT_GATE_ACTIVATION,
        recurrent_halting_flag: bool = config.RECURRENT_HALTING_FLAG,
        shared_gate_config: GateConfig | None = None,
        stack_options: ExpertsStackOptions | None = None,
        mixture_options: ExpertsMixtureOptions | None = None,
        expert_stack_options: ExpertsControllerStackOptions | None = None,
        sampler_options: ExpertsSamplerOptions | None = None,
        router_options: ExpertsRouterOptions | None = None,
        sampler_stack_options: ExpertsControllerStackOptions | None = None,
        layer_controller_options: ExpertsLayerControllerOptions | None = None,
        recurrent_controller_options: ExpertsRecurrentControllerOptions | None = None,
    ) -> None:
        stack_options = stack_options or ExpertsStackOptions(
            hidden_dim=stack_hidden_dim,
            bias_flag=stack_bias_flag,
            layer_norm_position=layer_norm_position,
            num_layers=stack_num_layers,
            activation=stack_activation,
            residual_connection_option=stack_residual_connection_option,
            dropout_probability=stack_dropout_probability,
            last_layer_bias_option=stack_last_layer_bias_option,
            apply_output_pipeline_flag=stack_apply_output_pipeline_flag,
        )
        mixture_options = mixture_options or ExpertsMixtureOptions(
            top_k=top_k,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            dropped_token_behavior=dropped_token_behavior,
            compute_expert_mixture_flag=compute_expert_mixture_flag,
            weighted_parameters_flag=weighted_parameters_flag,
            weighting_position_option=weighting_position_option,
            routing_initialization_mode=routing_initialization_mode,
        )
        expert_stack_options = expert_stack_options or ExpertsControllerStackOptions(
            hidden_dim=stack_options.hidden_dim,
            num_layers=expert_stack_num_layers,
            last_layer_bias_option=expert_stack_last_layer_bias_option,
            apply_output_pipeline_flag=expert_stack_apply_output_pipeline_flag,
            activation=expert_stack_activation,
            layer_norm_position=expert_stack_layer_norm_position,
            residual_connection_option=expert_stack_residual_connection_option,
            dropout_probability=expert_stack_dropout_probability,
            bias_flag=expert_bias_flag,
        )
        sampler_options = sampler_options or ExpertsSamplerOptions(
            threshold=sampler_threshold,
            filter_above_threshold=sampler_filter_above_threshold,
            num_topk_samples=sampler_num_topk_samples,
            normalize_probabilities_flag=sampler_normalize_probabilities_flag,
            noisy_topk_flag=sampler_noisy_topk_flag,
            coefficient_of_variation_loss_weight=(
                sampler_coefficient_of_variation_loss_weight
            ),
            switch_loss_weight=sampler_switch_loss_weight,
            zero_centred_loss_weight=sampler_zero_centred_loss_weight,
            mutual_information_loss_weight=sampler_mutual_information_loss_weight,
        )
        router_options = router_options or ExpertsRouterOptions(
            noisy_topk_flag=router_noisy_topk_flag,
        )
        sampler_stack_options = sampler_stack_options or ExpertsControllerStackOptions(
            hidden_dim=stack_options.hidden_dim,
            num_layers=sampler_stack_num_layers,
            last_layer_bias_option=sampler_stack_last_layer_bias_option,
            apply_output_pipeline_flag=sampler_stack_apply_output_pipeline_flag,
            activation=sampler_stack_activation,
            layer_norm_position=sampler_stack_layer_norm_position,
            residual_connection_option=sampler_stack_residual_connection_option,
            dropout_probability=sampler_stack_dropout_probability,
            bias_flag=sampler_bias_flag,
        )
        layer_controller_options = (
            layer_controller_options
            or ExpertsLayerControllerOptions(
                stack_gate_flag=stack_gate_flag,
                gate_option=gate_option,
                gate_activation=gate_activation,
                gate_stack_options=ExpertsControllerStackOptions(
                    hidden_dim=gate_stack_hidden_dim,
                    num_layers=gate_stack_num_layers,
                    last_layer_bias_option=gate_stack_last_layer_bias_option,
                    apply_output_pipeline_flag=gate_stack_apply_output_pipeline_flag,
                    activation=gate_stack_activation,
                    layer_norm_position=gate_stack_layer_norm_position,
                    residual_connection_option=gate_stack_residual_connection_option,
                    dropout_probability=gate_stack_dropout_probability,
                    bias_flag=gate_stack_bias_flag,
                ),
                stack_halting_flag=stack_halting_flag,
                halting_threshold=halting_threshold,
                halting_dropout=halting_dropout,
                halting_hidden_state_mode=halting_hidden_state_mode,
                halting_stack_options=ExpertsControllerStackOptions(
                    hidden_dim=halting_stack_hidden_dim,
                    num_layers=halting_stack_num_layers,
                    last_layer_bias_option=halting_stack_last_layer_bias_option,
                    apply_output_pipeline_flag=(
                        halting_stack_apply_output_pipeline_flag
                    ),
                    activation=halting_stack_activation,
                    layer_norm_position=halting_stack_layer_norm_position,
                    residual_connection_option=(
                        halting_stack_residual_connection_option
                    ),
                    dropout_probability=halting_stack_dropout_probability,
                    bias_flag=halting_stack_bias_flag,
                ),
                halting_output_dim=halting_output_dim,
                shared_gate_config=shared_gate_config,
            )
        )
        recurrent_controller_options = (
            recurrent_controller_options
            or ExpertsRecurrentControllerOptions(
                recurrent_flag=recurrent_flag,
                recurrent_max_steps=recurrent_max_steps,
                recurrent_layer_norm_position=recurrent_layer_norm_position,
                recurrent_gate_flag=recurrent_gate_flag,
                recurrent_gate_option=recurrent_gate_option,
                recurrent_gate_activation=recurrent_gate_activation,
                recurrent_halting_flag=recurrent_halting_flag,
            )
        )
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.stack_options = stack_options
        self.hidden_dim = stack_options.hidden_dim
        self.output_dim = output_dim
        self.bias_flag = stack_options.bias_flag
        self.layer_norm_position = stack_options.layer_norm_position
        self.stack_num_layers = stack_options.num_layers
        self.stack_activation = stack_options.activation
        self.stack_residual_connection_option = (
            stack_options.residual_connection_option
        )
        self.stack_dropout_probability = stack_options.dropout_probability
        self.stack_last_layer_bias_option = stack_options.last_layer_bias_option
        self.stack_apply_output_pipeline_flag = (
            stack_options.apply_output_pipeline_flag
        )
        self.mixture_options = mixture_options
        self.top_k = mixture_options.top_k
        self.num_experts = mixture_options.num_experts
        self.capacity_factor = mixture_options.capacity_factor
        self.dropped_token_behavior = mixture_options.dropped_token_behavior
        self.compute_expert_mixture_flag = (
            mixture_options.compute_expert_mixture_flag
        )
        self.weighted_parameters_flag = mixture_options.weighted_parameters_flag
        self.weighting_position_option = mixture_options.weighting_position_option
        self.routing_initialization_mode = (
            mixture_options.routing_initialization_mode
        )
        self.expert_stack_options = expert_stack_options
        self.expert_stack_num_layers = expert_stack_options.num_layers
        self.expert_stack_activation = expert_stack_options.activation
        self.expert_stack_residual_connection_option = (
            expert_stack_options.residual_connection_option
        )
        self.expert_stack_dropout_probability = (
            expert_stack_options.dropout_probability
        )
        self.expert_stack_layer_norm_position = (
            expert_stack_options.layer_norm_position
        )
        self.expert_stack_last_layer_bias_option = (
            expert_stack_options.last_layer_bias_option
        )
        self.expert_stack_apply_output_pipeline_flag = (
            expert_stack_options.apply_output_pipeline_flag
        )
        self.expert_bias_flag = expert_stack_options.bias_flag
        self.sampler_options = sampler_options
        self.sampler_threshold = sampler_options.threshold
        self.sampler_filter_above_threshold = sampler_options.filter_above_threshold
        self.sampler_num_topk_samples = sampler_options.num_topk_samples
        self.sampler_normalize_probabilities_flag = (
            sampler_options.normalize_probabilities_flag
        )
        self.sampler_noisy_topk_flag = sampler_options.noisy_topk_flag
        self.sampler_coefficient_of_variation_loss_weight = (
            sampler_options.coefficient_of_variation_loss_weight
        )
        self.sampler_switch_loss_weight = sampler_options.switch_loss_weight
        self.sampler_zero_centred_loss_weight = (
            sampler_options.zero_centred_loss_weight
        )
        self.sampler_mutual_information_loss_weight = (
            sampler_options.mutual_information_loss_weight
        )
        self.router_options = router_options
        self.router_noisy_topk_flag = router_options.noisy_topk_flag
        self.sampler_stack_options = sampler_stack_options
        self.sampler_stack_num_layers = sampler_stack_options.num_layers
        self.sampler_stack_activation = sampler_stack_options.activation
        self.sampler_stack_residual_connection_option = (
            sampler_stack_options.residual_connection_option
        )
        self.sampler_stack_dropout_probability = (
            sampler_stack_options.dropout_probability
        )
        self.sampler_stack_layer_norm_position = (
            sampler_stack_options.layer_norm_position
        )
        self.sampler_stack_last_layer_bias_option = (
            sampler_stack_options.last_layer_bias_option
        )
        self.sampler_stack_apply_output_pipeline_flag = (
            sampler_stack_options.apply_output_pipeline_flag
        )
        self.sampler_bias_flag = sampler_stack_options.bias_flag
        self.layer_controller_options = layer_controller_options
        self.stack_gate_flag = layer_controller_options.stack_gate_flag
        self.gate_option = layer_controller_options.gate_option
        self.gate_activation = layer_controller_options.gate_activation
        self.gate_stack_options = layer_controller_options.gate_stack_options
        self.gate_stack_hidden_dim = self.gate_stack_options.hidden_dim
        self.gate_stack_layer_norm_position = (
            self.gate_stack_options.layer_norm_position
        )
        self.gate_stack_num_layers = self.gate_stack_options.num_layers
        self.gate_stack_activation = self.gate_stack_options.activation
        self.gate_stack_residual_connection_option = (
            self.gate_stack_options.residual_connection_option
        )
        self.gate_stack_dropout_probability = (
            self.gate_stack_options.dropout_probability
        )
        self.gate_stack_last_layer_bias_option = (
            self.gate_stack_options.last_layer_bias_option
        )
        self.gate_stack_apply_output_pipeline_flag = (
            self.gate_stack_options.apply_output_pipeline_flag
        )
        self.gate_stack_bias_flag = self.gate_stack_options.bias_flag
        self.shared_gate_config = layer_controller_options.shared_gate_config
        self.stack_halting_flag = layer_controller_options.stack_halting_flag
        self.halting_threshold = layer_controller_options.halting_threshold
        self.halting_dropout = layer_controller_options.halting_dropout
        self.halting_hidden_state_mode = (
            layer_controller_options.halting_hidden_state_mode
        )
        self.halting_stack_options = layer_controller_options.halting_stack_options
        self.halting_stack_hidden_dim = self.halting_stack_options.hidden_dim
        self.halting_output_dim = layer_controller_options.halting_output_dim
        self.halting_stack_layer_norm_position = (
            self.halting_stack_options.layer_norm_position
        )
        self.halting_stack_num_layers = self.halting_stack_options.num_layers
        self.halting_stack_activation = self.halting_stack_options.activation
        self.halting_stack_residual_connection_option = (
            self.halting_stack_options.residual_connection_option
        )
        self.halting_stack_dropout_probability = (
            self.halting_stack_options.dropout_probability
        )
        self.halting_stack_last_layer_bias_option = (
            self.halting_stack_options.last_layer_bias_option
        )
        self.halting_stack_apply_output_pipeline_flag = (
            self.halting_stack_options.apply_output_pipeline_flag
        )
        self.halting_stack_bias_flag = self.halting_stack_options.bias_flag
        self.recurrent_controller_options = recurrent_controller_options
        self.recurrent_flag = recurrent_controller_options.recurrent_flag
        self.recurrent_max_steps = recurrent_controller_options.recurrent_max_steps
        self.recurrent_layer_norm_position = (
            recurrent_controller_options.recurrent_layer_norm_position
        )
        self.recurrent_gate_flag = recurrent_controller_options.recurrent_gate_flag
        self.recurrent_gate_option = (
            recurrent_controller_options.recurrent_gate_option
        )
        self.recurrent_gate_activation = (
            recurrent_controller_options.recurrent_gate_activation
        )
        self.recurrent_halting_flag = (
            recurrent_controller_options.recurrent_halting_flag
        )

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        control_dependencies = self.__control_config_dependencies()
        control_factory = ControlConfigFactory(control_dependencies)

        input_model_config = LayerConfig(
            activation=self.stack_activation,
            layer_norm_position=self.layer_norm_position,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=self.stack_dropout_probability,
            gate_config=None,
            halting_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=self.bias_flag,
            ),
        )

        model_config = control_factory.build()

        output_model_config = LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=self.bias_flag,
            ),
        )

        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=input_model_config,
                model_config=model_config,
                output_model_config=output_model_config,
            ),
        )

    def __control_config_dependencies(self) -> ControlConfigDependencies:
        return ControlConfigDependencies(
            stack_options=self.stack_options,
            mixture_options=self.mixture_options,
            expert_stack_options=self.expert_stack_options,
            sampler_options=self.sampler_options,
            router_options=self.router_options,
            sampler_stack_options=self.sampler_stack_options,
            layer_controller_options=self.layer_controller_options,
            recurrent_controller_options=self.recurrent_controller_options,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        )
