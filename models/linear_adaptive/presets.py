import models.linear_adaptive.config as config

from models.linear_adaptive.model import Model
from emperor.base.layer import LayerStackConfig
from emperor.linears.utils.config import LinearLayerConfig
from models.linear_adaptive.config import ExperimentConfig
from emperor.datasets.image.classification.mnist import Mnist
from emperor.augmentations.adaptive_parameters.config import AdaptiveParameterAugmentationConfig
from emperor.base.enums import BaseOptions, ActivationOptions, LayerNormPositionOptions
from emperor.augmentations.adaptive_parameters.options import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)
from emperor.experiments.base import (
    ExperimentBase,
    ExperimentPresetsBase,
    create_search_space,
    SearchMode,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    PRESET = 0
    CONFIG = 1
    GENERATOR_DEPTH = 2
    DIAGONAL = 3
    BIAS = 4
    MEMORY = 5
    COMBINED = 6


class ExperimentPresets(ExperimentPresetsBase):
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
                return self._create_default_search_space_configs(
                    dataset, search_mode, log_folder
                )
            case ExperimentOptions.GENERATOR_DEPTH:
                return self.__generator_depth_search_space_configs(
                    dataset, search_mode, log_folder
                )
            case ExperimentOptions.DIAGONAL:
                return self.__diagonal_search_space_configs(
                    dataset, search_mode, log_folder
                )
            case ExperimentOptions.BIAS:
                return self.__bias_search_space_configs(
                    dataset, search_mode, log_folder
                )
            case ExperimentOptions.MEMORY:
                return self.__memory_search_space_configs(
                    dataset, search_mode, log_folder
                )
            case ExperimentOptions.COMBINED:
                return self.__combined_search_space_configs(
                    dataset, search_mode, log_folder
                )
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `ExperimentOptions`."
                )

    def __generator_depth_search_space_configs(
        self,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
    ) -> list["ModelConfig"]:
        base_config = {
            **self._dataset_config(dataset),
            **self._best_params(dataset, log_folder),
        }

        search_space = {
            **self._extract_search_space_from_config(search_mode),
            "generator_depth": [
                DynamicDepthOptions.DISABLED,
                DynamicDepthOptions.DEPTH_OF_ONE,
                DynamicDepthOptions.DEPTH_OF_TWO,
                DynamicDepthOptions.DEPTH_OF_THREE,
            ],
        }

        return create_search_space(self._preset, base_config, search_space, search_mode)

    def __diagonal_search_space_configs(
        self,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
    ) -> list["ModelConfig"]:
        base_config = {
            **self._dataset_config(dataset),
            **self._best_params(dataset, log_folder),
        }

        search_space = {
            **self._extract_search_space_from_config(search_mode),
            "diagonal_option": [
                DynamicDiagonalOptions.DISABLED,
                DynamicDiagonalOptions.DIAGONAL,
                DynamicDiagonalOptions.ANTI_DIAGONAL,
                DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL,
            ],
        }

        return create_search_space(self._preset, base_config, search_space, search_mode)

    def __bias_search_space_configs(
        self,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
    ) -> list["ModelConfig"]:
        base_config = {
            **self._dataset_config(dataset),
            **self._best_params(dataset, log_folder),
        }

        search_space = {
            **self._extract_search_space_from_config(search_mode),
            "bias_option": [
                DynamicBiasOptions.DISABLED,
                DynamicBiasOptions.SCALE_AND_OFFSET,
                DynamicBiasOptions.ELEMENT_WISE_OFFSET,
                DynamicBiasOptions.DYNAMIC_PARAMETERS,
            ],
        }

        return create_search_space(self._preset, base_config, search_space, search_mode)

    def __memory_search_space_configs(
        self,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
    ) -> list["ModelConfig"]:
        base_config = {
            **self._dataset_config(dataset),
            **self._best_params(dataset, log_folder),
        }

        search_space = {
            **self._extract_search_space_from_config(search_mode),
            "memory_option": [
                LinearMemoryOptions.DISABLED,
                LinearMemoryOptions.FUSION,
                LinearMemoryOptions.WEIGHTED,
            ],
            "memory_size_option": [
                LinearMemorySizeOptions.SMALL,
                LinearMemorySizeOptions.MEDIUM,
                LinearMemorySizeOptions.LARGE,
                LinearMemorySizeOptions.MAX,
            ],
            "memory_position_option": [
                LinearMemoryPositionOptions.BEFORE_AFFINE,
                LinearMemoryPositionOptions.AFTER_AFFINE,
            ],
        }

        return create_search_space(self._preset, base_config, search_space, search_mode)

    def __combined_search_space_configs(
        self,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
    ) -> list["ModelConfig"]:
        base_config = {
            **self._dataset_config(dataset),
            **self._best_params(dataset, log_folder),
        }

        search_space = {
            **self._extract_search_space_from_config(search_mode),
            "generator_depth": [
                DynamicDepthOptions.DISABLED,
                DynamicDepthOptions.DEPTH_OF_ONE,
                DynamicDepthOptions.DEPTH_OF_TWO,
                DynamicDepthOptions.DEPTH_OF_THREE,
            ],
            "diagonal_option": [
                DynamicDiagonalOptions.DISABLED,
                DynamicDiagonalOptions.DIAGONAL,
                DynamicDiagonalOptions.ANTI_DIAGONAL,
                DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL,
            ],
            "bias_option": [
                DynamicBiasOptions.DISABLED,
                DynamicBiasOptions.SCALE_AND_OFFSET,
                DynamicBiasOptions.ELEMENT_WISE_OFFSET,
                DynamicBiasOptions.DYNAMIC_PARAMETERS,
            ],
            "memory_option": [
                LinearMemoryOptions.DISABLED,
                LinearMemoryOptions.FUSION,
                LinearMemoryOptions.WEIGHTED,
            ],
            "memory_size_option": [
                LinearMemorySizeOptions.SMALL,
                LinearMemorySizeOptions.MEDIUM,
                LinearMemorySizeOptions.LARGE,
                LinearMemorySizeOptions.MAX,
            ],
            "memory_position_option": [
                LinearMemoryPositionOptions.BEFORE_AFFINE,
                LinearMemoryPositionOptions.AFTER_AFFINE,
            ],
        }

        return create_search_space(self._preset, base_config, search_space, search_mode)

    def _preset(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        bias_flag: bool = config.BIAS_FLAG,
        layer_norm_position: LayerNormPositionOptions = config.LAYER_NORM_POSITION,
        generator_depth: DynamicDepthOptions = config.GENERATOR_DEPTH,
        diagonal_option: DynamicDiagonalOptions = config.DIAGONAL_OPTION,
        bias_option: DynamicBiasOptions = config.BIAS_OPTION,
        memory_option: LinearMemoryOptions = config.MEMORY_OPTION,
        memory_size_option: LinearMemorySizeOptions = config.MEMORY_SIZE_OPTION,
        memory_position_option: LinearMemoryPositionOptions = config.MEMORY_POSITION_OPTION,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_residual_flag: bool = config.STACK_RESIDUAL_FLAG,
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
        adaptive_generator_stack_num_layers: int = config.ADAPTIVE_GENERATOR_STACK_NUM_LAYERS,
        adaptive_generator_stack_hidden_dim: int = config.ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM,
        adaptive_generator_stack_activation: ActivationOptions = config.ADAPTIVE_GENERATOR_STACK_ACTIVATION,
        adaptive_generator_stack_residual_flag: bool = config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_FLAG,
        adaptive_generator_stack_dropout_probability: float = config.ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY,
        adaptive_generator_stack_layer_norm_position: LayerNormPositionOptions = config.LAYER_NORM_POSITION,
    ) -> "ModelConfig":
        from emperor.config import ModelConfig
        from emperor.linears.options import LinearLayerOptions

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            learning_rate=learning_rate,
            override_config=ExperimentConfig(
                model_config=LayerStackConfig(
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
