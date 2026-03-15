import models.experts.config as config

from emperor.base.enums import BaseOptions, ActivationOptions, LayerNormPositionOptions
from emperor.datasets.image.classification.mnist import Mnist
from emperor.linears.utils.config import LinearLayerConfig
from emperor.base.layer import LayerStackConfig
from emperor.experts.utils.layers import MixtureOfExpertsConfig
from emperor.sampler.utils.routers import RouterConfig
from emperor.sampler.utils.samplers import SamplerConfig
from emperor.experiments.base import (
    ExperimentBase,
    ExperimentPresetsBase,
    create_search_space,
    SearchMode,
)
from emperor.experts.utils.enums import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    InitSamplerOptions,
)
from emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)
from models.experts.config import ExperimentConfig
from models.experts.model import Model

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    PRESET = 0
    CONFIG = 1
    ADAPTIVE = 2


class ExperimentPresets(ExperimentPresetsBase):
    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.PRESET,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.PRESET:
                return self._create_default_preset_configs(dataset)
            case ExperimentOptions.CONFIG:
                return self._create_default_search_space_configs(dataset, search_mode)
            case ExperimentOptions.ADAPTIVE:
                return self.__adaptive_search_space_configs(dataset, search_mode)
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `ExperimentOptions`."
                )

    def __adaptive_search_space_configs(
        self,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
    ) -> list["ModelConfig"]:
        base_config = self._dataset_config(dataset)

        search_space = {
            **self._extract_search_space_from_config(search_mode),
            "experts_model_generator_depth": [
                DynamicDepthOptions.DEPTH_OF_ONE,
                DynamicDepthOptions.DEPTH_OF_TWO,
                DynamicDepthOptions.DEPTH_OF_THREE,
            ],
            "experts_model_diagonal_option": [
                DynamicDiagonalOptions.DISABLED,
                DynamicDiagonalOptions.DIAGONAL,
                DynamicDiagonalOptions.ANTI_DIAGONAL,
                DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL,
            ],
            "experts_model_bias_option": [
                DynamicBiasOptions.DISABLED,
                DynamicBiasOptions.SCALE_AND_OFFSET,
                DynamicBiasOptions.ELEMENT_WISE_OFFSET,
                DynamicBiasOptions.DYNAMIC_PARAMETERS,
            ],
        }

        return create_search_space(self._preset, base_config, search_space, search_mode)

    def _preset(
        self,
        batch_size: int = config.BATCH_SIZE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        learning_rate: float = config.LEARNING_RATE,
        bias_flag: bool = config.BIAS_FLAG,
        output_num_layers: int = config.OUTPUT_NUM_LAYERS,
        output_activation: ActivationOptions = config.OUTPUT_ACTIVATION,
        output_dropout_probability: float = config.OUTPUT_DROPOUT_PROBABILITY,
        router_noisy_topk_flag: bool = config.ROUTER_NOISY_TOPK_FLAG,
        sampler_threshold: float = config.SAMPLER_THRESHOLD,
        sampler_filter_above_threshold: bool = config.SAMPLER_FILTER_ABOVE_THRESHOLD,
        sampler_num_topk_samples: int = config.SAMPLER_NUM_TOPK_SAMPLES,
        sampler_normalize_probabilities_flag: bool = config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
        sampler_coefficient_of_variation_loss_weight: float = config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
        sampler_switch_loss_weight: float = config.SAMPLER_SWITCH_LOSS_WEIGHT,
        sampler_zero_centred_loss_weight: float = config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT,
        sampler_mutual_information_loss_weight: float = config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT,
        experts_top_k: int = config.EXPERTS_TOP_K,
        experts_num_experts: int = config.EXPERTS_NUM_EXPERTS,
        experts_compute_expert_mixture_flag: bool = config.EXPERTS_COMPUTE_EXPERT_MIXTURE_FLAG,
        experts_weighted_parameters_flag: bool = config.EXPERTS_WEIGHTED_PARAMETERS_FLAG,
        experts_weighting_position_option: ExpertWeightingPositionOptions = config.EXPERTS_WEIGHTING_POSITION_OPTION,
        experts_init_sampler_option: InitSamplerOptions = config.EXPERTS_INIT_SAMPLER_OPTION,
        experts_capacity_factor: float = config.EXPERTS_CAPACITY_FACTOR,
        experts_dropped_token_behavior: DroppedTokenOptions = config.EXPERTS_DROPPED_TOKEN_BEHAVIOR,
        experts_model_generator_depth: DynamicDepthOptions = config.EXPERTS_MODEL_GENERATOR_DEPTH,
        experts_model_diagonal_option: DynamicDiagonalOptions = config.EXPERTS_MODEL_DIAGONAL_OPTION,
        experts_model_bias_option: DynamicBiasOptions = config.EXPERTS_MODEL_BIAS_OPTION,
        experts_model_memory_option: LinearMemoryOptions = config.EXPERTS_MODEL_MEMORY_OPTION,
        experts_model_memory_size_option: LinearMemorySizeOptions = config.EXPERTS_MODEL_MEMORY_SIZE_OPTION,
        experts_model_memory_position_option: LinearMemoryPositionOptions = config.EXPERTS_MODEL_MEMORY_POSITION_OPTION,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_residual_flag: bool = config.STACK_RESIDUAL_FLAG,
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
    ) -> "ModelConfig":
        from emperor.config import ModelConfig
        from emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
        from emperor.behaviours.model import AdaptiveParameterBehaviourConfig

        experts_layer_stack_option = LinearLayerStackOptions.BASE
        if experts_model_generator_depth != DynamicDepthOptions.DISABLED:
            experts_layer_stack_option = LinearLayerStackOptions.ADAPTIVE

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            learning_rate=learning_rate,
            override_config=ExperimentConfig(
                experts_config=LayerStackConfig(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=stack_num_layers,
                    activation=stack_activation,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    residual_flag=stack_residual_flag,
                    adaptive_computation_flag=False,
                    dropout_probability=stack_dropout_probability,
                    override_config=MixtureOfExpertsConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        top_k=experts_top_k,
                        num_experts=experts_num_experts,
                        layer_stack_option=experts_layer_stack_option,
                        compute_expert_mixture_flag=experts_compute_expert_mixture_flag,
                        weighted_parameters_flag=experts_weighted_parameters_flag,
                        weighting_position_option=experts_weighting_position_option,
                        init_sampler_option=experts_init_sampler_option,
                        capacity_factor=experts_capacity_factor,
                        dropped_token_behavior=experts_dropped_token_behavior,
                        override_config=LayerStackConfig(
                            model_type=LinearLayerOptions.BASE,
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            num_layers=stack_num_layers,
                            activation=stack_activation,
                            layer_norm_position=LayerNormPositionOptions.NONE,
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
                                    generator_depth=experts_model_generator_depth,
                                    diagonal_option=experts_model_diagonal_option,
                                    bias_option=experts_model_bias_option,
                                    memory_option=experts_model_memory_option,
                                    memory_size_option=experts_model_memory_size_option,
                                    memory_position_option=experts_model_memory_position_option,
                                    override_config=LayerStackConfig(
                                        model_type=LinearLayerOptions.BASE,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        output_dim=output_dim,
                                        num_layers=stack_num_layers,
                                        activation=stack_activation,
                                        layer_norm_position=LayerNormPositionOptions.NONE,
                                        residual_flag=stack_residual_flag,
                                        adaptive_computation_flag=False,
                                        dropout_probability=stack_dropout_probability,
                                        override_config=LinearLayerConfig(
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            bias_flag=bias_flag,
                                            data_monitor=None,
                                            parameter_monitor=None,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        router_model_config=RouterConfig(
                            input_dim=input_dim,
                            layer_stack_option=LinearLayerStackOptions.BASE,
                            num_experts=experts_num_experts,
                            noisy_topk_flag=router_noisy_topk_flag,
                            override_config=LayerStackConfig(
                                model_type=LinearLayerOptions.BASE,
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                output_dim=experts_num_experts,
                                num_layers=stack_num_layers,
                                activation=stack_activation,
                                layer_norm_position=LayerNormPositionOptions.NONE,
                                residual_flag=stack_residual_flag,
                                adaptive_computation_flag=False,
                                dropout_probability=stack_dropout_probability,
                                override_config=LinearLayerConfig(
                                    input_dim=input_dim,
                                    output_dim=experts_num_experts,
                                    bias_flag=bias_flag,
                                    data_monitor=None,
                                    parameter_monitor=None,
                                ),
                            ),
                        ),
                        sampler_model_config=SamplerConfig(
                            top_k=experts_top_k,
                            threshold=sampler_threshold,
                            filter_above_threshold=sampler_filter_above_threshold,
                            num_topk_samples=sampler_num_topk_samples,
                            normalize_probabilities_flag=sampler_normalize_probabilities_flag,
                            noisy_topk_flag=router_noisy_topk_flag,
                            num_experts=experts_num_experts,
                            coefficient_of_variation_loss_weight=sampler_coefficient_of_variation_loss_weight,
                            switch_loss_weight=sampler_switch_loss_weight,
                            zero_centred_loss_weight=sampler_zero_centred_loss_weight,
                            mutual_information_loss_weight=sampler_mutual_information_loss_weight,
                        ),
                    ),
                ),
                output_config=LayerStackConfig(
                    model_type=LinearLayerOptions.BASE,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=output_num_layers,
                    activation=output_activation,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    residual_flag=False,
                    adaptive_computation_flag=False,
                    dropout_probability=output_dropout_probability,
                    override_config=LinearLayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                        data_monitor=None,
                        parameter_monitor=None,
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
