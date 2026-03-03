from emperor.base.enums import BaseOptions, ActivationOptions, LayerNormPositionOptions
from emperor.datasets.image.mnist import Mnist
from emperor.linears.utils.layers import LinearLayerConfig
from emperor.base.layer import LayerStackConfig
from emperor.experiments.base import ExperimentPresetsBase, create_search_space
from emperor.behaviours.model import AdaptiveParameterBehaviourConfig
from emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)
from models.linear_adaptive.config import (
    ExperimentConfig,
    BATCH_SIZE,
    INPUT_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    BIAS_FLAG,
    LAYER_NORM_POSITION,
    GENERATOR_DEPTH,
    DIAGONAL_OPTION,
    BIAS_OPTION,
    MEMORY_OPTION,
    MEMORY_SIZE_OPTION,
    MEMORY_POSITION_OPTION,
    STACK_NUM_LAYERS,
    STACK_ACTIVATION,
    STACK_RESIDUAL_FLAG,
    STACK_DROPOUT_PROBABILITY,
    ADAPTIVE_GENERATOR_STACK_NUM_LAYERS,
    ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM,
    ADAPTIVE_GENERATOR_STACK_ACTIVATION,
    ADAPTIVE_GENERATOR_STACK_RESIDUAL_FLAG,
    ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
    DEFAULT = 0
    BASE = 1
    GENERATOR_DEPTH = 2
    DIAGONAL = 3
    BIAS = 4
    MEMORY = 5
    COMBINED = 6


class ExperimentPresets(ExperimentPresetsBase):
    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.DEFAULT,
        dataset: type = Mnist,
        num_samples: int | None = None,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.DEFAULT:
                return self._default_config(dataset)
            case ExperimentOptions.GENERATOR_DEPTH:
                return self.__generator_depth_grid_search_config(dataset, num_samples)
            case ExperimentOptions.BASE:
                return self.__base_grid_search_config(dataset, num_samples)
            case ExperimentOptions.DIAGONAL:
                return self.__diagonal_grid_search_config(dataset, num_samples)
            case ExperimentOptions.BIAS:
                return self.__bias_grid_search_config(dataset, num_samples)
            case ExperimentOptions.MEMORY:
                return self.__memory_grid_search_config(dataset, num_samples)
            case ExperimentOptions.COMBINED:
                return self.__combined_grid_search_config(dataset, num_samples)
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `ExperimentOptions`."
                )

    def __base_grid_search_config(
        self,
        dataset: type = Mnist,
        num_random_search_samples: int | None = None,
    ) -> list["ModelConfig"]:
        base_config = self._dataset_config(dataset)

        return create_search_space(
            self._preset,
            base_config,
            self.__base_search_space(),
            num_random_search_samples,
        )

    def __generator_depth_grid_search_config(
        self,
        dataset: type = Mnist,
        num_random_search_samples: int | None = None,
    ) -> list["ModelConfig"]:
        base_config = self._dataset_config(dataset)

        search_space = {
            **self.__base_search_space(),
            "generator_depth": [
                DynamicDepthOptions.DEPTH_OF_ONE,
                DynamicDepthOptions.DEPTH_OF_TWO,
                DynamicDepthOptions.DEPTH_OF_THREE,
            ],
        }

        return create_search_space(
            self._preset, base_config, search_space, num_random_search_samples
        )

    def __diagonal_grid_search_config(
        self,
        dataset: type = Mnist,
        num_random_search_samples: int | None = None,
    ) -> list["ModelConfig"]:
        base_config = self._dataset_config(dataset)

        search_space = {
            **self.__base_search_space(),
            "diagonal_option": [
                DynamicDiagonalOptions.DISABLED,
                DynamicDiagonalOptions.DIAGONAL,
                DynamicDiagonalOptions.ANTI_DIAGONAL,
                DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL,
            ],
        }

        return create_search_space(
            self._preset, base_config, search_space, num_random_search_samples
        )

    def __bias_grid_search_config(
        self,
        dataset: type = Mnist,
        num_random_search_samples: int | None = None,
    ) -> list["ModelConfig"]:
        base_config = self._dataset_config(dataset)

        search_space = {
            **self.__base_search_space(),
            "bias_option": [
                DynamicBiasOptions.DISABLED,
                DynamicBiasOptions.SCALE_AND_OFFSET,
                DynamicBiasOptions.ELEMENT_WISE_OFFSET,
                DynamicBiasOptions.DYNAMIC_PARAMETERS,
            ],
        }

        return create_search_space(
            self._preset, base_config, search_space, num_random_search_samples
        )

    def __memory_grid_search_config(
        self,
        dataset: type = Mnist,
        num_random_search_samples: int | None = None,
    ) -> list["ModelConfig"]:
        base_config = self._dataset_config(dataset)

        search_space = {
            **self.__base_search_space(),
            "memory_option": [
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

        return create_search_space(
            self._preset, base_config, search_space, num_random_search_samples
        )

    def __combined_grid_search_config(
        self,
        dataset: type = Mnist,
        num_random_search_samples: int | None = None,
    ) -> list["ModelConfig"]:
        base_config = self._dataset_config(dataset)

        search_space = {
            **self.__base_search_space(),
            "generator_depth": [
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

        return create_search_space(
            self._preset, base_config, search_space, num_random_search_samples
        )

    def __base_search_space(self) -> dict:
        return {
            "learning_rate": [1e-4, 1e-3, 1e-2],
            "hidden_dim": [64, 128, 256],
            "stack_num_layers": [3, 6],
            "stack_dropout_probability": [0.0, 0.1],
            "stack_activation": [
                ActivationOptions.RELU,
                ActivationOptions.SILU,
                ActivationOptions.GELU,
                ActivationOptions.LEAKY_RELU,
            ],
            "adaptive_generator_stack_num_layers": [1, 2, 3],
            "adaptive_generator_stack_hidden_dim": [64, 128, 256],
            "adaptive_generator_stack_dropout_probability": [0.0, 0.1],
        }

    def _preset(
        self,
        batch_size: int = BATCH_SIZE,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
        bias_flag: bool = BIAS_FLAG,
        layer_norm_position: LayerNormPositionOptions = LAYER_NORM_POSITION,
        generator_depth: DynamicDepthOptions = GENERATOR_DEPTH,
        diagonal_option: DynamicDiagonalOptions = DIAGONAL_OPTION,
        bias_option: DynamicBiasOptions = BIAS_OPTION,
        memory_option: LinearMemoryOptions = MEMORY_OPTION,
        memory_size_option: LinearMemorySizeOptions = MEMORY_SIZE_OPTION,
        memory_position_option: LinearMemoryPositionOptions = MEMORY_POSITION_OPTION,
        stack_num_layers: int = STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = STACK_ACTIVATION,
        stack_residual_flag: bool = STACK_RESIDUAL_FLAG,
        stack_dropout_probability: float = STACK_DROPOUT_PROBABILITY,
        adaptive_generator_stack_num_layers: int = ADAPTIVE_GENERATOR_STACK_NUM_LAYERS,
        adaptive_generator_stack_hidden_dim: int = ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM,
        adaptive_generator_stack_activation: ActivationOptions = ADAPTIVE_GENERATOR_STACK_ACTIVATION,
        adaptive_generator_stack_residual_flag: bool = ADAPTIVE_GENERATOR_STACK_RESIDUAL_FLAG,
        adaptive_generator_stack_dropout_probability: float = ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY,
    ) -> "ModelConfig":
        from emperor.config import ModelConfig
        from emperor.linears.options import LinearLayerOptions

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
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
            ),
        )
