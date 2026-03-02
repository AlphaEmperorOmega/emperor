import torch

from torch import Tensor
from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.base.enums import BaseOptions
from emperor.datasets.image.mnist import Mnist
from models.parser import get_experiment_parser
from emperor.linears.utils.layers import LinearLayerConfig
from emperor.base.layer import LayerStack, LayerStackConfig
from emperor.experiments.classifier import ClassifierExperiment
from emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from emperor.behaviours.model import AdaptiveParameterBehaviourConfig
from emperor.experiments.base import (
    ExperimentBase,
    ExperimentPresetsBase,
    create_search_space,
)
from emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


@dataclass
class ExperimentConfig(ConfigBase):
    model_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )


class Model(ClassifierExperiment):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__(cfg)
        self.main_cfg: ExperimentConfig = self._resolve_main_config(self.cfg, cfg)
        self.model_config: LayerStackConfig = self.main_cfg.model_config
        self.model = LayerStack(self.model_config).build_model()

    def _resolve_main_config(
        self, sub_config: "ConfigBase", main_cfg: "ConfigBase"
    ) -> None:
        if sub_config.override_config is not None:
            return sub_config.override_config
        return main_cfg

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = X.to(self.device)
        X = torch.flatten(X, start_dim=1)
        X = self.model(X)
        return X


class ExperimentOptions(BaseOptions):
    DEFAULT = 0
    BASE = 1
    GENERATOR_DEPTH = 2
    DIAGONAL = 3
    BIAS = 4
    MEMORY = 5
    COMBINED = 6


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_option: ExperimentOptions | None = None,
    ) -> None:
        super().__init__(experiment_option)
        self.accelerator = "cpu"

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_enumeration(self) -> type[BaseOptions]:
        return ExperimentOptions


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
        batch_size: int = 64,
        input_dim: int = 28**2,
        hidden_dim: int = 256,
        output_dim: int = 10,
        bias_flag: bool = True,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DEFAULT,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_THREE,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL,
        bias_option: DynamicBiasOptions = DynamicBiasOptions.DYNAMIC_PARAMETERS,
        memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        stack_num_layers: int = 3,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
        adaptive_generator_stack_num_layers: int = 2,
        adaptive_generator_stack_hidden_dim: int = 256,
        adaptive_generator_stack_activation: ActivationOptions = ActivationOptions.RELU,
        adaptive_generator_stack_residual_flag: bool = False,
        adaptive_generator_stack_dropout_probability: float = 0.0,
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


if __name__ == "__main__":
    parser = get_experiment_parser(ExperimentOptions.names())
    args = parser.parse_args()
    config_option = ExperimentOptions.get_option(args.name)

    experiment = Experiment(config_option)
    experiment.train_model(num_samples=args.num_samples, log_folder=args.log_folder)
