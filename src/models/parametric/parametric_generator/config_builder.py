from emperor.config import ModelConfig
from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig
from emperor.parametric import ClipParameterOptions, GeneratorBiasMixtureConfig
from models.parametric.parametric_generator import config
from models.parametric.parametric_generator._control_config_factory import (
    build_parametric_stack_config,
)
from models.parametric.parametric_generator.experiment_config import ExperimentConfig
from models.parametric.parametric_generator.runtime_options import (
    ParametricGeneratorStackOptions,
    ParametricMixtureOptions,
    ParametricRouterOptions,
    ParametricSamplerOptions,
    ParametricStackOptions,
)


class ParametricGeneratorConfigBuilder:
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
        adaptive_bias_option: type[GeneratorBiasMixtureConfig] | None = (
            config.ADAPTIVE_BIAS_OPTION
        ),
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
        generator_stack_num_layers: int = config.GENERATOR_STACK_NUM_LAYERS,
        generator_stack_hidden_dim: int = config.GENERATOR_STACK_HIDDEN_DIM,
        generator_stack_activation: ActivationOptions = (
            config.GENERATOR_STACK_ACTIVATION
        ),
        generator_stack_dropout_probability: float = (
            config.GENERATOR_STACK_DROPOUT_PROBABILITY
        ),
        stack_options: ParametricStackOptions | None = None,
        mixture_options: ParametricMixtureOptions | None = None,
        sampler_options: ParametricSamplerOptions | None = None,
        router_options: ParametricRouterOptions | None = None,
        generator_stack_options: ParametricGeneratorStackOptions | None = None,
    ) -> None:
        stack_options = stack_options or ParametricStackOptions(
            hidden_dim=hidden_dim,
            num_layers=stack_num_layers,
            activation=stack_activation,
            residual_connection_option=stack_residual_connection_option,
            dropout_probability=stack_dropout_probability,
        )
        mixture_options = mixture_options or ParametricMixtureOptions(
            top_k=adaptive_mixture_top_k,
            num_experts=adaptive_mixture_num_experts,
            weighted_parameters_flag=adaptive_mixture_weighted_parameters_flag,
            clip_parameter_option=adaptive_mixture_clip_parameter_option,
            clip_range=adaptive_mixture_clip_range,
        )
        sampler_options = sampler_options or ParametricSamplerOptions(
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
        router_options = router_options or ParametricRouterOptions(
            activation=stack_options.activation,
        )
        generator_stack_options = (
            generator_stack_options
            or ParametricGeneratorStackOptions(
                hidden_dim=generator_stack_hidden_dim,
                num_layers=generator_stack_num_layers,
                activation=generator_stack_activation,
                dropout_probability=generator_stack_dropout_probability,
            )
        )
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.stack_options = stack_options
        self.hidden_dim = stack_options.hidden_dim
        self.output_dim = output_dim
        self.stack_num_layers = stack_options.num_layers
        self.stack_activation = stack_options.activation
        self.stack_residual_connection_option = stack_options.residual_connection_option
        self.stack_dropout_probability = stack_options.dropout_probability
        self.mixture_options = mixture_options
        self.adaptive_mixture_top_k = mixture_options.top_k
        self.adaptive_mixture_num_experts = mixture_options.num_experts
        self.adaptive_mixture_weighted_parameters_flag = (
            mixture_options.weighted_parameters_flag
        )
        self.adaptive_mixture_clip_parameter_option = (
            mixture_options.clip_parameter_option
        )
        self.adaptive_mixture_clip_range = mixture_options.clip_range
        self.adaptive_bias_option = adaptive_bias_option
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
        self.sampler_zero_centred_loss_weight = sampler_options.zero_centred_loss_weight
        self.sampler_mutual_information_loss_weight = (
            sampler_options.mutual_information_loss_weight
        )
        self.router_options = router_options
        self.generator_stack_options = generator_stack_options
        self.generator_stack_num_layers = generator_stack_options.num_layers
        self.generator_stack_hidden_dim = generator_stack_options.hidden_dim
        self.generator_stack_activation = generator_stack_options.activation
        self.generator_stack_dropout_probability = (
            generator_stack_options.dropout_probability
        )

    def build(self) -> ModelConfig:
        input_model_config = build_linear_layer_config(
            activation=self.stack_activation,
        )
        model_config = build_parametric_stack_config(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            stack_options=self.stack_options,
            mixture_options=self.mixture_options,
            sampler_options=self.sampler_options,
            router_options=self.router_options,
            adaptive_bias_option=self.adaptive_bias_option,
            generator_stack_options=self.generator_stack_options,
        )
        output_model_config = build_linear_layer_config(
            activation=ActivationOptions.DISABLED,
        )
        return ModelConfig(
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            learning_rate=self.learning_rate,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=input_model_config,
                model_config=model_config,
                output_model_config=output_model_config,
            ),
        )


def build_linear_layer_config(
    *,
    activation: ActivationOptions,
) -> LayerConfig:
    layer_model_config = LinearLayerConfig(
        bias_flag=True,
    )
    return LayerConfig(
        activation=activation,
        residual_config=None,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=layer_model_config,
    )
