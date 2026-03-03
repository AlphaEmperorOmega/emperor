from emperor.base.enums import BaseOptions, ActivationOptions, LayerNormPositionOptions
from emperor.datasets.image.mnist import Mnist
from emperor.base.layer import LayerStack, LayerStackConfig
from emperor.linears.utils.layers import LinearLayerConfig
from emperor.sampler.utils.routers import RouterConfig
from emperor.sampler.utils.samplers import SamplerConfig
from emperor.parametric.options import AdaptiveLayerOptions
from emperor.parametric.utils.layers import AdaptiveParameterLayerConfig, AdaptiveRouterOptions
from emperor.parametric.utils.mixtures.base import AdaptiveMixtureConfig
from emperor.parametric.utils.mixtures.options import AdaptiveBiasOptions, AdaptiveWeightOptions
from emperor.parametric.utils.mixtures.types.utils.enums import ClipParameterOptions
from emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
from emperor.behaviours.model import AdaptiveParameterBehaviourConfig
from emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)
from emperor.experiments.base import ExperimentPresetsBase
import models.parametric_vector.config as config
from models.parametric_vector.config import ExperimentConfig
from emperor.experiments.base import SearchMode

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    DEFAULT = 0
    BASE = 1


class ExperimentPresets(ExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__()

    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.DEFAULT,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.DEFAULT:
                return self._default_config(dataset)
            case ExperimentOptions.BASE:
                return self._create_search_space_configs(dataset, search_mode)
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `ExperimentOptions`."
                )

    def _preset(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_residual_flag: bool = config.STACK_RESIDUAL_FLAG,
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
        adaptive_mixture_top_k: int = config.ADAPTIVE_MIXTURE_TOP_K,
        adaptive_mixture_num_experts: int = config.ADAPTIVE_MIXTURE_NUM_EXPERTS,
        adaptive_mixture_weighted_parameters_flag: bool = config.ADAPTIVE_MIXTURE_WEIGHTED_PARAMETERS_FLAG,
        adaptive_mixture_clip_parameter_option: ClipParameterOptions = config.ADAPTIVE_MIXTURE_CLIP_PARAMETER_OPTION,
        adaptive_mixture_clip_range: float = config.ADAPTIVE_MIXTURE_CLIP_RANGE,
        router_layer_stack_option: LinearLayerStackOptions = config.ROUTER_LAYER_STACK_OPTION,
        router_hidden_dim: int = config.ROUTER_HIDDEN_DIM,
        router_num_layers: int = config.ROUTER_NUM_LAYERS,
        router_activation: ActivationOptions = config.ROUTER_ACTIVATION,
        router_layer_norm_position: LayerNormPositionOptions = config.ROUTER_LAYER_NORM_POSITION,
        router_residual_flag: bool = config.ROUTER_RESIDUAL_FLAG,
        router_dropout_probability: float = config.ROUTER_DROPOUT_PROBABILITY,
        router_bias_flag: bool = config.ROUTER_BIAS_FLAG,
        router_noisy_topk_flag: bool = config.ROUTER_NOISY_TOPK_FLAG,
        router_generator_depth: DynamicDepthOptions = config.ROUTER_GENERATOR_DEPTH,
        router_diagonal_option: DynamicDiagonalOptions = config.ROUTER_DIAGONAL_OPTION,
        router_bias_option: DynamicBiasOptions = config.ROUTER_BIAS_OPTION,
        router_memory_option: LinearMemoryOptions = config.ROUTER_MEMORY_OPTION,
        router_memory_size_option: LinearMemorySizeOptions = config.ROUTER_MEMORY_SIZE_OPTION,
        router_memory_position_option: LinearMemoryPositionOptions = config.ROUTER_MEMORY_POSITION_OPTION,
        router_adaptive_generator_stack_hidden_dim: int = config.ROUTER_ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM,
        router_adaptive_generator_stack_num_layers: int = config.ROUTER_ADAPTIVE_GENERATOR_STACK_NUM_LAYERS,
        router_adaptive_generator_stack_activation: ActivationOptions = config.ROUTER_ADAPTIVE_GENERATOR_STACK_ACTIVATION,
        router_adaptive_generator_stack_residual_flag: bool = config.ROUTER_ADAPTIVE_GENERATOR_STACK_RESIDUAL_FLAG,
        router_adaptive_generator_stack_dropout_probability: float = config.ROUTER_ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY,
        sampler_num_experts: int = config.SAMPLER_NUM_EXPERTS,
        sampler_top_k: int = config.SAMPLER_TOP_K,
        sampler_threshold: float = config.SAMPLER_THRESHOLD,
        sampler_filter_above_threshold: bool = config.SAMPLER_FILTER_ABOVE_THRESHOLD,
        sampler_num_topk_samples: int = config.SAMPLER_NUM_TOPK_SAMPLES,
        sampler_normalize_probabilities_flag: bool = config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
        sampler_noisy_topk_flag: bool = config.SAMPLER_NOISY_TOPK_FLAG,
        sampler_coefficient_of_variation_loss_weight: float = config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
        sampler_switch_loss_weight: float = config.SAMPLER_SWITCH_LOSS_WEIGHT,
        sampler_zero_centred_loss_weight: float = config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT,
        sampler_mutual_information_loss_weight: float = config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT,
    ) -> "ModelConfig":
        from emperor.config import ModelConfig

        _hidden_dim = max(input_dim, output_dim)

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            override_config=ExperimentConfig(
                model_config=LayerStackConfig(
                    model_type=AdaptiveLayerOptions.BASE,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=stack_num_layers,
                    activation=stack_activation,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    residual_flag=stack_residual_flag,
                    adaptive_computation_flag=False,
                    dropout_probability=stack_dropout_probability,
                    override_config=AdaptiveParameterLayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        adaptive_weight_option=AdaptiveWeightOptions.VECTOR,
                        adaptive_bias_option=AdaptiveBiasOptions.DISABLED,
                        init_sampler_model_option=AdaptiveRouterOptions.INDEPENTENT_ROUTER,
                        time_tracker_flag=False,
                        adaptive_behaviour_config=AdaptiveParameterBehaviourConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            override_config=LayerStackConfig(
                                model_type=LinearLayerOptions.BASE,
                                input_dim=input_dim,
                                hidden_dim=_hidden_dim,
                                output_dim=output_dim,
                                num_layers=2,
                                activation=stack_activation,
                                layer_norm_position=LayerNormPositionOptions.NONE,
                                residual_flag=False,
                                adaptive_computation_flag=False,
                                dropout_probability=0.0,
                                override_config=LinearLayerConfig(
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    bias_flag=False,
                                    data_monitor=None,
                                    parameter_monitor=None,
                                ),
                            ),
                        ),
                        router_config=RouterConfig(
                            input_dim=input_dim,
                            layer_stack_option=router_layer_stack_option,
                            num_experts=adaptive_mixture_num_experts,
                            noisy_topk_flag=router_noisy_topk_flag,
                            override_config=self.__build_linear_layer_stack_config(
                                layer_stack_option=router_layer_stack_option,
                                input_dim=input_dim,
                                hidden_dim=(
                                    router_hidden_dim
                                    if router_hidden_dim > 0
                                    else max(input_dim, adaptive_mixture_num_experts)
                                ),
                                output_dim=adaptive_mixture_num_experts,
                                stack_num_layers=router_num_layers,
                                stack_activation=router_activation,
                                layer_norm_position=router_layer_norm_position,
                                stack_residual_flag=router_residual_flag,
                                stack_dropout_probability=router_dropout_probability,
                                bias_flag=router_bias_flag,
                                generator_depth=router_generator_depth,
                                diagonal_option=router_diagonal_option,
                                bias_option=router_bias_option,
                                memory_option=router_memory_option,
                                memory_size_option=router_memory_size_option,
                                memory_position_option=router_memory_position_option,
                                adaptive_generator_stack_hidden_dim=router_adaptive_generator_stack_hidden_dim,
                                adaptive_generator_stack_num_layers=router_adaptive_generator_stack_num_layers,
                                adaptive_generator_stack_activation=router_adaptive_generator_stack_activation,
                                adaptive_generator_stack_residual_flag=router_adaptive_generator_stack_residual_flag,
                                adaptive_generator_stack_dropout_probability=router_adaptive_generator_stack_dropout_probability,
                            ),
                        ),
                        sampler_config=SamplerConfig(
                            top_k=sampler_top_k,
                            threshold=sampler_threshold,
                            filter_above_threshold=sampler_filter_above_threshold,
                            num_topk_samples=sampler_num_topk_samples,
                            normalize_probabilities_flag=sampler_normalize_probabilities_flag,
                            noisy_topk_flag=sampler_noisy_topk_flag,
                            num_experts=sampler_num_experts,
                            coefficient_of_variation_loss_weight=sampler_coefficient_of_variation_loss_weight,
                            switch_loss_weight=sampler_switch_loss_weight,
                            zero_centred_loss_weight=sampler_zero_centred_loss_weight,
                            mutual_information_loss_weight=sampler_mutual_information_loss_weight,
                        ),
                        override_config=AdaptiveMixtureConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            top_k=adaptive_mixture_top_k,
                            num_experts=adaptive_mixture_num_experts,
                            weighted_parameters_flag=adaptive_mixture_weighted_parameters_flag,
                            clip_parameter_option=adaptive_mixture_clip_parameter_option,
                            clip_range=adaptive_mixture_clip_range,
                        ),
                    ),
                ),
            ),
        )

    def __build_linear_layer_stack_config(
        self,
        layer_stack_option: LinearLayerStackOptions,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        stack_num_layers: int,
        stack_activation: ActivationOptions,
        layer_norm_position: LayerNormPositionOptions,
        stack_residual_flag: bool,
        stack_dropout_probability: float,
        bias_flag: bool,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        adaptive_generator_stack_hidden_dim: int = 0,
        adaptive_generator_stack_num_layers: int = 2,
        adaptive_generator_stack_activation: ActivationOptions = ActivationOptions.RELU,
        adaptive_generator_stack_residual_flag: bool = False,
        adaptive_generator_stack_dropout_probability: float = 0.0,
    ) -> "LayerStackConfig":
        if layer_stack_option == LinearLayerStackOptions.ADAPTIVE:
            return LayerStackConfig(
                model_type=LinearLayerOptions.ADAPTIVE,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=stack_num_layers,
                activation=stack_activation,
                layer_norm_position=layer_norm_position,
                residual_flag=stack_residual_flag,
                adaptive_computation_flag=False,
                dropout_probability=stack_dropout_probability,
                override_config=LinearLayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias_flag=bias_flag,
                    data_monitor=None,
                    parameter_monitor=None,
                    override_config=AdaptiveParameterBehaviourConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        generator_depth=generator_depth,
                        diagonal_option=diagonal_option,
                        bias_option=bias_option,
                        memory_option=memory_option,
                        memory_size_option=memory_size_option,
                        memory_position_option=memory_position_option,
                        override_config=LayerStackConfig(
                            model_type=LinearLayerOptions.BASE,
                            input_dim=input_dim,
                            hidden_dim=adaptive_generator_stack_hidden_dim,
                            output_dim=output_dim,
                            num_layers=adaptive_generator_stack_num_layers,
                            activation=adaptive_generator_stack_activation,
                            layer_norm_position=layer_norm_position,
                            residual_flag=adaptive_generator_stack_residual_flag,
                            adaptive_computation_flag=False,
                            dropout_probability=adaptive_generator_stack_dropout_probability,
                            override_config=LinearLayerConfig(
                                input_dim=input_dim,
                                output_dim=output_dim,
                                bias_flag=bias_flag,
                                data_monitor=None,
                                parameter_monitor=None,
                                override_config=AdaptiveParameterBehaviourConfig(
                                    generator_depth=generator_depth,
                                ),
                            ),
                        ),
                    ),
                ),
            )

        return LayerStackConfig(
            model_type=LinearLayerOptions.BASE,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            activation=stack_activation,
            layer_norm_position=layer_norm_position,
            residual_flag=stack_residual_flag,
            adaptive_computation_flag=False,
            dropout_probability=stack_dropout_probability,
            override_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
            ),
        )
