from emperor.base.layer import LayerConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.config import ModelConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.parametric.core.mixtures.options import ClipParameterOptions
from models.parametric.parametric_vector import config
from models.parametric.parametric_vector._control_config_factory import (
    build_parametric_stack_config,
)
from models.parametric.parametric_vector.experiment_config import ExperimentConfig


class ParametricVectorConfigBuilder:
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_residual_connection_option: ResidualConnectionOptions = (
            config.STACK_RESIDUAL_CONNECTION_OPTION
        ),
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
        adaptive_mixture_top_k: int = config.ADAPTIVE_MIXTURE_TOP_K,
        adaptive_mixture_num_experts: int = config.ADAPTIVE_MIXTURE_NUM_EXPERTS,
        adaptive_mixture_weighted_parameters_flag: bool = (
            config.ADAPTIVE_MIXTURE_WEIGHTED_PARAMETERS_FLAG
        ),
        adaptive_mixture_clip_parameter_option: ClipParameterOptions = (
            config.ADAPTIVE_MIXTURE_CLIP_PARAMETER_OPTION
        ),
        adaptive_mixture_clip_range: float = config.ADAPTIVE_MIXTURE_CLIP_RANGE,
        sampler_threshold: float = config.SAMPLER_THRESHOLD,
        sampler_filter_above_threshold: bool = config.SAMPLER_FILTER_ABOVE_THRESHOLD,
        sampler_num_topk_samples: int = config.SAMPLER_NUM_TOPK_SAMPLES,
        sampler_normalize_probabilities_flag: bool = (
            config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG
        ),
        sampler_noisy_topk_flag: bool = config.SAMPLER_NOISY_TOPK_FLAG,
        sampler_coefficient_of_variation_loss_weight: float = (
            config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT
        ),
        sampler_switch_loss_weight: float = config.SAMPLER_SWITCH_LOSS_WEIGHT,
        sampler_zero_centred_loss_weight: float = (
            config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT
        ),
        sampler_mutual_information_loss_weight: float = (
            config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT
        ),
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.stack_num_layers = stack_num_layers
        self.stack_activation = stack_activation
        self.stack_residual_connection_option = stack_residual_connection_option
        self.stack_dropout_probability = stack_dropout_probability
        self.adaptive_mixture_top_k = adaptive_mixture_top_k
        self.adaptive_mixture_num_experts = adaptive_mixture_num_experts
        self.adaptive_mixture_weighted_parameters_flag = (
            adaptive_mixture_weighted_parameters_flag
        )
        self.adaptive_mixture_clip_parameter_option = (
            adaptive_mixture_clip_parameter_option
        )
        self.adaptive_mixture_clip_range = adaptive_mixture_clip_range
        self.sampler_threshold = sampler_threshold
        self.sampler_filter_above_threshold = sampler_filter_above_threshold
        self.sampler_num_topk_samples = sampler_num_topk_samples
        self.sampler_normalize_probabilities_flag = (
            sampler_normalize_probabilities_flag
        )
        self.sampler_noisy_topk_flag = sampler_noisy_topk_flag
        self.sampler_coefficient_of_variation_loss_weight = (
            sampler_coefficient_of_variation_loss_weight
        )
        self.sampler_switch_loss_weight = sampler_switch_loss_weight
        self.sampler_zero_centred_loss_weight = sampler_zero_centred_loss_weight
        self.sampler_mutual_information_loss_weight = (
            sampler_mutual_information_loss_weight
        )

    def build(self) -> ModelConfig:
        return ModelConfig(
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            learning_rate=self.learning_rate,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=build_linear_layer_config(
                    activation=self.stack_activation,
                ),
                model_config=build_parametric_stack_config(
                    input_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    num_layers=self.stack_num_layers,
                    activation=self.stack_activation,
                    residual_connection_option=self.stack_residual_connection_option,
                    dropout_probability=self.stack_dropout_probability,
                    adaptive_mixture_top_k=self.adaptive_mixture_top_k,
                    adaptive_mixture_num_experts=self.adaptive_mixture_num_experts,
                    adaptive_mixture_weighted_parameters_flag=(
                        self.adaptive_mixture_weighted_parameters_flag
                    ),
                    adaptive_mixture_clip_parameter_option=(
                        self.adaptive_mixture_clip_parameter_option
                    ),
                    adaptive_mixture_clip_range=self.adaptive_mixture_clip_range,
                    sampler_threshold=self.sampler_threshold,
                    sampler_filter_above_threshold=(
                        self.sampler_filter_above_threshold
                    ),
                    sampler_num_topk_samples=self.sampler_num_topk_samples,
                    sampler_normalize_probabilities_flag=(
                        self.sampler_normalize_probabilities_flag
                    ),
                    sampler_noisy_topk_flag=self.sampler_noisy_topk_flag,
                    sampler_coefficient_of_variation_loss_weight=(
                        self.sampler_coefficient_of_variation_loss_weight
                    ),
                    sampler_switch_loss_weight=self.sampler_switch_loss_weight,
                    sampler_zero_centred_loss_weight=(
                        self.sampler_zero_centred_loss_weight
                    ),
                    sampler_mutual_information_loss_weight=(
                        self.sampler_mutual_information_loss_weight
                    ),
                ),
                output_model_config=build_linear_layer_config(
                    activation=ActivationOptions.DISABLED,
                ),
            ),
        )


def build_linear_layer_config(
    *,
    activation: ActivationOptions,
) -> LayerConfig:
    return LayerConfig(
        activation=activation,
        residual_connection_option=ResidualConnectionOptions.DISABLED,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=LinearLayerConfig(
            bias_flag=True,
        ),
    )
