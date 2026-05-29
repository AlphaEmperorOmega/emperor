import models.experts_linear.config as config

from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer import LayerStackConfig
from emperor.base.layer.config import LayerConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.experts.core.config import (
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
)
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.sampler.core.config import RouterConfig, SamplerConfig
from models.experts_linear.experiment_config import ExperimentConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExpertsLinearConfigBuilder:
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        bias_flag: bool = config.BIAS_FLAG,
        layer_norm_position: LayerNormPositionOptions = config.LAYER_NORM_POSITION,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_residual_flag: bool = config.STACK_RESIDUAL_FLAG,
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
        expert_stack_residual_flag: bool = config.EXPERT_STACK_RESIDUAL_FLAG,
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
        sampler_stack_residual_flag: bool = config.SAMPLER_STACK_RESIDUAL_FLAG,
        sampler_stack_dropout_probability: float = config.SAMPLER_STACK_DROPOUT_PROBABILITY,
        sampler_stack_layer_norm_position: LayerNormPositionOptions = config.SAMPLER_STACK_LAYER_NORM_POSITION,
        sampler_stack_last_layer_bias_option: LastLayerBiasOptions = config.SAMPLER_STACK_LAST_LAYER_BIAS_OPTION,
        sampler_stack_apply_output_pipeline_flag: bool = config.SAMPLER_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        sampler_bias_flag: bool = config.SAMPLER_BIAS_FLAG,
        stack_gate_flag: bool = config.GATE_FLAG,
        gate_hidden_dim: int = config.GATE_HIDDEN_DIM,
        gate_layer_norm_position: LayerNormPositionOptions = config.GATE_LAYER_NORM_POSITION,
        gate_stack_num_layers: int = config.GATE_STACK_NUM_LAYERS,
        gate_stack_activation: ActivationOptions = config.GATE_STACK_ACTIVATION,
        gate_stack_residual_flag: bool = config.GATE_STACK_RESIDUAL_FLAG,
        gate_stack_dropout_probability: float = config.GATE_STACK_DROPOUT_PROBABILITY,
        gate_stack_last_layer_bias_option: LastLayerBiasOptions = config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
        gate_stack_apply_output_pipeline_flag: bool = config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        gate_bias_flag: bool = config.GATE_BIAS_FLAG,
        stack_halting_flag: bool = config.HALTING_FLAG,
        halting_threshold: float = config.HALTING_THRESHOLD,
        halting_dropout: float = config.HALTING_DROPOUT,
        halting_hidden_state_mode: HaltingHiddenStateModeOptions = config.HALTING_HIDDEN_STATE_MODE,
        halting_hidden_dim: int = config.HALTING_HIDDEN_DIM,
        halting_output_dim: int = config.HALTING_OUTPUT_DIM,
        halting_layer_norm_position: LayerNormPositionOptions = config.HALTING_LAYER_NORM_POSITION,
        halting_stack_num_layers: int = config.HALTING_STACK_NUM_LAYERS,
        halting_stack_activation: ActivationOptions = config.HALTING_STACK_ACTIVATION,
        halting_stack_residual_flag: bool = config.HALTING_STACK_RESIDUAL_FLAG,
        halting_stack_dropout_probability: float = config.HALTING_STACK_DROPOUT_PROBABILITY,
        halting_stack_last_layer_bias_option: LastLayerBiasOptions = config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        halting_stack_apply_output_pipeline_flag: bool = config.HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        halting_bias_flag: bool = config.HALTING_BIAS_FLAG,
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bias_flag = bias_flag
        self.layer_norm_position = layer_norm_position
        self.stack_num_layers = stack_num_layers
        self.stack_activation = stack_activation
        self.stack_residual_flag = stack_residual_flag
        self.stack_dropout_probability = stack_dropout_probability
        self.stack_last_layer_bias_option = stack_last_layer_bias_option
        self.stack_apply_output_pipeline_flag = stack_apply_output_pipeline_flag
        self.top_k = top_k
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.dropped_token_behavior = dropped_token_behavior
        self.compute_expert_mixture_flag = compute_expert_mixture_flag
        self.weighted_parameters_flag = weighted_parameters_flag
        self.weighting_position_option = weighting_position_option
        self.routing_initialization_mode = routing_initialization_mode
        self.expert_stack_num_layers = expert_stack_num_layers
        self.expert_stack_activation = expert_stack_activation
        self.expert_stack_residual_flag = expert_stack_residual_flag
        self.expert_stack_dropout_probability = expert_stack_dropout_probability
        self.expert_stack_layer_norm_position = expert_stack_layer_norm_position
        self.expert_stack_last_layer_bias_option = expert_stack_last_layer_bias_option
        self.expert_stack_apply_output_pipeline_flag = (
            expert_stack_apply_output_pipeline_flag
        )
        self.expert_bias_flag = expert_bias_flag
        self.sampler_threshold = sampler_threshold
        self.sampler_filter_above_threshold = sampler_filter_above_threshold
        self.sampler_num_topk_samples = sampler_num_topk_samples
        self.sampler_normalize_probabilities_flag = sampler_normalize_probabilities_flag
        self.sampler_noisy_topk_flag = sampler_noisy_topk_flag
        self.sampler_coefficient_of_variation_loss_weight = (
            sampler_coefficient_of_variation_loss_weight
        )
        self.sampler_switch_loss_weight = sampler_switch_loss_weight
        self.sampler_zero_centred_loss_weight = sampler_zero_centred_loss_weight
        self.sampler_mutual_information_loss_weight = (
            sampler_mutual_information_loss_weight
        )
        self.router_noisy_topk_flag = router_noisy_topk_flag
        self.sampler_stack_num_layers = sampler_stack_num_layers
        self.sampler_stack_activation = sampler_stack_activation
        self.sampler_stack_residual_flag = sampler_stack_residual_flag
        self.sampler_stack_dropout_probability = sampler_stack_dropout_probability
        self.sampler_stack_layer_norm_position = sampler_stack_layer_norm_position
        self.sampler_stack_last_layer_bias_option = sampler_stack_last_layer_bias_option
        self.sampler_stack_apply_output_pipeline_flag = (
            sampler_stack_apply_output_pipeline_flag
        )
        self.sampler_bias_flag = sampler_bias_flag
        self.stack_gate_flag = stack_gate_flag
        self.gate_hidden_dim = gate_hidden_dim
        self.gate_layer_norm_position = gate_layer_norm_position
        self.gate_stack_num_layers = gate_stack_num_layers
        self.gate_stack_activation = gate_stack_activation
        self.gate_stack_residual_flag = gate_stack_residual_flag
        self.gate_stack_dropout_probability = gate_stack_dropout_probability
        self.gate_stack_last_layer_bias_option = gate_stack_last_layer_bias_option
        self.gate_stack_apply_output_pipeline_flag = (
            gate_stack_apply_output_pipeline_flag
        )
        self.gate_bias_flag = gate_bias_flag
        self.stack_halting_flag = stack_halting_flag
        self.halting_threshold = halting_threshold
        self.halting_dropout = halting_dropout
        self.halting_hidden_state_mode = halting_hidden_state_mode
        self.halting_hidden_dim = halting_hidden_dim
        self.halting_output_dim = halting_output_dim
        self.halting_layer_norm_position = halting_layer_norm_position
        self.halting_stack_num_layers = halting_stack_num_layers
        self.halting_stack_activation = halting_stack_activation
        self.halting_stack_residual_flag = halting_stack_residual_flag
        self.halting_stack_dropout_probability = halting_stack_dropout_probability
        self.halting_stack_last_layer_bias_option = halting_stack_last_layer_bias_option
        self.halting_stack_apply_output_pipeline_flag = (
            halting_stack_apply_output_pipeline_flag
        )
        self.halting_bias_flag = halting_bias_flag

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        input_model_config = LayerConfig(
            activation=self.stack_activation,
            layer_norm_position=self.layer_norm_position,
            residual_flag=False,
            dropout_probability=self.stack_dropout_probability,
            gate_config=None,
            halting_config=None,
            shared_halting_flag=False,
            layer_model_config=LinearLayerConfig(
                bias_flag=self.bias_flag,
            ),
        )

        model_config = self._build_main_model_config()

        output_model_config = LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_flag=False,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            shared_halting_flag=False,
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

    def _build_main_model_config(self) -> MixtureOfExpertsModelConfig:
        return MixtureOfExpertsModelConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            top_k=self.top_k,
            routing_initialization_mode=self.routing_initialization_mode,
            sampler_config=self._build_sampler_config(),
            stack_config=self._build_main_stack_config(),
        )

    def _build_main_stack_config(self) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=self.stack_num_layers,
            last_layer_bias_option=self.stack_last_layer_bias_option,
            apply_output_pipeline_flag=self.stack_apply_output_pipeline_flag,
            layer_config=MixtureOfExpertsLayerConfig(
                activation=self.stack_activation,
                layer_norm_position=self.layer_norm_position,
                residual_flag=self.stack_residual_flag,
                dropout_probability=self.stack_dropout_probability,
                gate_config=self._build_gate_config(),
                halting_config=self._build_halting_config(),
                shared_halting_flag=False,
                layer_model_config=self._build_mixture_of_experts_config(),
            ),
        )

    def _build_mixture_of_experts_config(self) -> MixtureOfExpertsConfig:
        return MixtureOfExpertsConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            top_k=self.top_k,
            num_experts=self.num_experts,
            capacity_factor=self.capacity_factor,
            dropped_token_behavior=self.dropped_token_behavior,
            compute_expert_mixture_flag=self.compute_expert_mixture_flag,
            weighted_parameters_flag=self.weighted_parameters_flag,
            weighting_position_option=self.weighting_position_option,
            routing_initialization_mode=self.routing_initialization_mode,
            sampler_config=self._build_sampler_config(),
            expert_model_config=self._build_expert_model_config(),
        )

    def _build_gate_config(self) -> LayerStackConfig | None:
        if not self.stack_gate_flag:
            return None
        return LayerStackConfig(
            hidden_dim=self.gate_hidden_dim,
            num_layers=self.gate_stack_num_layers,
            last_layer_bias_option=self.gate_stack_last_layer_bias_option,
            apply_output_pipeline_flag=self.gate_stack_apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=self.gate_stack_activation,
                layer_norm_position=self.gate_layer_norm_position,
                residual_flag=self.gate_stack_residual_flag,
                dropout_probability=self.gate_stack_dropout_probability,
                halting_config=None,
                shared_halting_flag=False,
                gate_config=None,
                layer_model_config=LinearLayerConfig(
                    bias_flag=self.gate_bias_flag,
                ),
            ),
        )

    def _build_halting_config(self) -> StickBreakingConfig | None:
        if not self.stack_halting_flag:
            return None
        return StickBreakingConfig(
            threshold=self.halting_threshold,
            halting_dropout=self.halting_dropout,
            hidden_state_mode=self.halting_hidden_state_mode,
            halting_gate_config=LayerStackConfig(
                hidden_dim=self.halting_hidden_dim or self.output_dim,
                output_dim=self.halting_output_dim,
                num_layers=self.halting_stack_num_layers,
                last_layer_bias_option=self.halting_stack_last_layer_bias_option,
                apply_output_pipeline_flag=self.halting_stack_apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    activation=self.halting_stack_activation,
                    layer_norm_position=self.halting_layer_norm_position,
                    residual_flag=self.halting_stack_residual_flag,
                    dropout_probability=self.halting_stack_dropout_probability,
                    halting_config=None,
                    shared_halting_flag=False,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=self.halting_bias_flag,
                    ),
                ),
            ),
        )

    def _build_expert_model_config(self) -> LayerStackConfig:
        return LayerStackConfig(
            hidden_dim=self.hidden_dim,
            num_layers=self.expert_stack_num_layers,
            last_layer_bias_option=self.expert_stack_last_layer_bias_option,
            apply_output_pipeline_flag=self.expert_stack_apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=self.expert_stack_activation,
                layer_norm_position=self.expert_stack_layer_norm_position,
                residual_flag=self.expert_stack_residual_flag,
                dropout_probability=self.expert_stack_dropout_probability,
                gate_config=None,
                halting_config=None,
                shared_halting_flag=False,
                layer_model_config=LinearLayerConfig(
                    bias_flag=self.expert_bias_flag,
                ),
            ),
        )

    def _build_sampler_config(self) -> SamplerConfig:
        return SamplerConfig(
            top_k=self.top_k,
            threshold=self.sampler_threshold,
            filter_above_threshold=self.sampler_filter_above_threshold,
            num_topk_samples=self.sampler_num_topk_samples,
            normalize_probabilities_flag=self.sampler_normalize_probabilities_flag,
            noisy_topk_flag=self.sampler_noisy_topk_flag,
            num_experts=self.num_experts,
            coefficient_of_variation_loss_weight=self.sampler_coefficient_of_variation_loss_weight,
            switch_loss_weight=self.sampler_switch_loss_weight,
            zero_centred_loss_weight=self.sampler_zero_centred_loss_weight,
            mutual_information_loss_weight=self.sampler_mutual_information_loss_weight,
            router_config=self._build_router_config(),
        )

    def _build_router_config(self) -> RouterConfig:
        return RouterConfig(
            input_dim=self.hidden_dim,
            num_experts=self.num_experts,
            noisy_topk_flag=self.router_noisy_topk_flag,
            model_config=LayerStackConfig(
                hidden_dim=self.hidden_dim,
                num_layers=self.sampler_stack_num_layers,
                last_layer_bias_option=self.sampler_stack_last_layer_bias_option,
                apply_output_pipeline_flag=self.sampler_stack_apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    activation=self.sampler_stack_activation,
                    layer_norm_position=self.sampler_stack_layer_norm_position,
                    residual_flag=self.sampler_stack_residual_flag,
                    dropout_probability=self.sampler_stack_dropout_probability,
                    gate_config=None,
                    halting_config=None,
                    shared_halting_flag=False,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=self.sampler_bias_flag,
                    ),
                ),
            ),
        )
