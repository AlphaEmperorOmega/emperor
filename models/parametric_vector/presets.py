from emperor.base.enums import BaseOptions, ActivationOptions, LayerNormPositionOptions
from emperor.augmentations.adaptive_parameters.config import AdaptiveParameterAugmentationConfig
from emperor.augmentations.adaptive_parameters.options import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)
from emperor.datasets.image.classification.mnist import Mnist
from emperor.base.layer import LayerStackConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.sampler.utils.routers import RouterConfig
from emperor.sampler.utils.samplers import SamplerConfig
from emperor.parametric.options import AdaptiveLayerOptions
from emperor.parametric.utils.config import ParametricLayerConfig, AdaptiveRouterOptions
from emperor.parametric.utils.mixtures.base import AdaptiveMixtureConfig
from emperor.parametric.utils.mixtures.options import (
    AdaptiveBiasOptions,
    AdaptiveWeightOptions,
)
from emperor.parametric.utils.mixtures.types.utils.enums import ClipParameterOptions
from emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
from emperor.experiments.base import ExperimentBase, ExperimentPresetsBase
import models.parametric_vector.config as config
from models.parametric_vector.config import ExperimentConfig
from models.parametric_vector.model import Model
from emperor.experiments.base import SearchMode

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    PRESET = 0
    CONFIG = 1


class ExperimentPresets(ExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__()

    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.PRESET,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.PRESET:
                return self._create_default_preset_configs(dataset)
            case ExperimentOptions.CONFIG:
                return self._create_default_search_space_configs(dataset, search_mode)
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
        bias_flag: bool = config.BIAS_FLAG,
        layer_norm_position: LayerNormPositionOptions = config.LAYER_NORM_POSITION,
        generator_depth: DynamicDepthOptions = config.GENERATOR_DEPTH,
        diagonal_option: DynamicDiagonalOptions = config.DIAGONAL_OPTION,
        bias_option: DynamicBiasOptions = config.BIAS_OPTION,
        memory_option: LinearMemoryOptions = config.MEMORY_OPTION,
        memory_size_option: LinearMemorySizeOptions = config.MEMORY_SIZE_OPTION,
        memory_position_option: LinearMemoryPositionOptions = config.MEMORY_POSITION_OPTION,
        adaptive_generator_stack_num_layers: int = config.ADAPTIVE_GENERATOR_STACK_NUM_LAYERS,
        adaptive_generator_stack_hidden_dim: int = config.ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM,
        adaptive_generator_stack_activation: ActivationOptions = config.ADAPTIVE_GENERATOR_STACK_ACTIVATION,
        adaptive_generator_stack_residual_flag: bool = config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_FLAG,
        adaptive_generator_stack_dropout_probability: float = config.ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY,
        adaptive_generator_stack_layer_norm_position: LayerNormPositionOptions = config.LAYER_NORM_POSITION,
    ) -> "ModelConfig":
        from emperor.config import ModelConfig

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            override_config=ExperimentConfig(
                model_config=LayerStackConfig(
                    model_type=AdaptiveLayerOptions.BASE,
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_layers=stack_num_layers,
                    activation=stack_activation,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_flag=stack_residual_flag,
                    adaptive_computation_flag=False,
                    dropout_probability=stack_dropout_probability,
                    override_config=ParametricLayerConfig(
                        input_dim=hidden_dim,
                        output_dim=hidden_dim,
                        adaptive_weight_option=AdaptiveWeightOptions.VECTOR,
                        adaptive_bias_option=AdaptiveBiasOptions.DISABLED,
                        init_sampler_model_option=AdaptiveRouterOptions.INDEPENTENT_ROUTER,
                        time_tracker_flag=False,
                        adaptive_behaviour_config=AdaptiveParameterAugmentationConfig(
                            input_dim=hidden_dim,
                            output_dim=hidden_dim,
                            generator_depth=DynamicDepthOptions.DISABLED,
                            diagonal_option=DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL,
                            bias_option=DynamicBiasOptions.DISABLED,
                            memory_option=LinearMemoryOptions.DISABLED,
                            memory_size_option=LinearMemorySizeOptions.DISABLED,
                            memory_position_option=LinearMemoryPositionOptions.BEFORE_AFFINE,
                            override_config=LayerStackConfig(
                                model_type=LinearLayerOptions.BASE,
                                input_dim=hidden_dim,
                                hidden_dim=hidden_dim,
                                output_dim=hidden_dim,
                                num_layers=2,
                                activation=stack_activation,
                                layer_norm_position=LayerNormPositionOptions.DISABLED,
                                residual_flag=False,
                                adaptive_computation_flag=False,
                                dropout_probability=0.0,
                                override_config=LinearLayerConfig(
                                    input_dim=hidden_dim,
                                    output_dim=hidden_dim,
                                    bias_flag=False,
                                    data_monitor=None,
                                    parameter_monitor=None,
                                ),
                            ),
                        ),
                        router_config=RouterConfig(
                            input_dim=hidden_dim,
                            layer_stack_option=LinearLayerStackOptions.ADAPTIVE,
                            num_experts=adaptive_mixture_num_experts,
                            noisy_topk_flag=False,
                            override_config=LayerStackConfig(
                                input_dim=hidden_dim,
                                hidden_dim=hidden_dim,
                                output_dim=adaptive_mixture_num_experts,
                                num_layers=1,
                                activation=stack_activation,
                                layer_norm_position=LayerNormPositionOptions.DISABLED,
                                residual_flag=False,
                                adaptive_computation_flag=False,
                                dropout_probability=0.0,
                                override_config=LinearLayerConfig(
                                    input_dim=hidden_dim,
                                    output_dim=adaptive_mixture_num_experts,
                                    bias_flag=False,
                                    data_monitor=None,
                                    parameter_monitor=None,
                                ),
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
                            input_dim=hidden_dim,
                            output_dim=hidden_dim,
                            top_k=adaptive_mixture_top_k,
                            num_experts=adaptive_mixture_num_experts,
                            weighted_parameters_flag=adaptive_mixture_weighted_parameters_flag,
                            clip_parameter_option=adaptive_mixture_clip_parameter_option,
                            clip_range=adaptive_mixture_clip_range,
                        ),
                    ),
                ),
                input_model_config=self.__adaptive_stack_preset(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    bias_flag=bias_flag,
                    layer_norm_position=layer_norm_position,
                    generator_depth=generator_depth,
                    diagonal_option=diagonal_option,
                    bias_option=bias_option,
                    memory_option=memory_option,
                    memory_size_option=memory_size_option,
                    memory_position_option=memory_position_option,
                    stack_num_layers=stack_num_layers,
                    stack_activation=stack_activation,
                    stack_residual_flag=stack_residual_flag,
                    stack_dropout_probability=stack_dropout_probability,
                    adaptive_generator_stack_num_layers=adaptive_generator_stack_num_layers,
                    adaptive_generator_stack_hidden_dim=adaptive_generator_stack_hidden_dim,
                    adaptive_generator_stack_activation=adaptive_generator_stack_activation,
                    adaptive_generator_stack_residual_flag=adaptive_generator_stack_residual_flag,
                    adaptive_generator_stack_dropout_probability=adaptive_generator_stack_dropout_probability,
                    adaptive_generator_stack_layer_norm_position=adaptive_generator_stack_layer_norm_position,
                ),
                output_model_config=self.__adaptive_stack_preset(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    bias_flag=bias_flag,
                    layer_norm_position=layer_norm_position,
                    generator_depth=generator_depth,
                    diagonal_option=diagonal_option,
                    bias_option=bias_option,
                    memory_option=memory_option,
                    memory_size_option=memory_size_option,
                    memory_position_option=memory_position_option,
                    stack_num_layers=stack_num_layers,
                    stack_activation=stack_activation,
                    stack_residual_flag=stack_residual_flag,
                    stack_dropout_probability=stack_dropout_probability,
                    adaptive_generator_stack_num_layers=adaptive_generator_stack_num_layers,
                    adaptive_generator_stack_hidden_dim=adaptive_generator_stack_hidden_dim,
                    adaptive_generator_stack_activation=adaptive_generator_stack_activation,
                    adaptive_generator_stack_residual_flag=adaptive_generator_stack_residual_flag,
                    adaptive_generator_stack_dropout_probability=adaptive_generator_stack_dropout_probability,
                    adaptive_generator_stack_layer_norm_position=adaptive_generator_stack_layer_norm_position,
                ),
            ),
        )

    def __adaptive_stack_preset(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bias_flag: bool,
        layer_norm_position: LayerNormPositionOptions,
        generator_depth: DynamicDepthOptions,
        diagonal_option: DynamicDiagonalOptions,
        bias_option: DynamicBiasOptions,
        memory_option: LinearMemoryOptions,
        memory_size_option: LinearMemorySizeOptions,
        memory_position_option: LinearMemoryPositionOptions,
        stack_num_layers: int,
        stack_activation: ActivationOptions,
        stack_residual_flag: bool,
        stack_dropout_probability: float,
        adaptive_generator_stack_num_layers: int,
        adaptive_generator_stack_hidden_dim: int,
        adaptive_generator_stack_activation: ActivationOptions,
        adaptive_generator_stack_residual_flag: bool,
        adaptive_generator_stack_dropout_probability: float,
        adaptive_generator_stack_layer_norm_position: LayerNormPositionOptions,
    ) -> "LayerStackConfig":
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
                override_config=AdaptiveParameterAugmentationConfig(
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
                        layer_norm_position=adaptive_generator_stack_layer_norm_position,
                        residual_flag=adaptive_generator_stack_residual_flag,
                        adaptive_computation_flag=False,
                        dropout_probability=adaptive_generator_stack_dropout_probability,
                        override_config=LinearLayerConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            bias_flag=bias_flag,
                            data_monitor=None,
                            parameter_monitor=None,
                            override_config=AdaptiveParameterAugmentationConfig(
                                generator_depth=generator_depth,
                            ),
                        ),
                    ),
                ),
            ),
        )


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_option: ExperimentOptions | None = None,
    ) -> None:
        super().__init__(experiment_option)
        # self.accelerator = "cpu"

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return config.DATASET_OPTIONS

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_enumeration(self) -> type[BaseOptions]:
        return ExperimentOptions
