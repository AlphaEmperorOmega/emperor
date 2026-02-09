import copy
import torch
import random
import itertools

import torch.nn as nn

from torch import Tensor
from Emperor.behaviours.model import AdaptiveParameterBehaviourConfig
from Emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)
from models.parser import get_parser
from Emperor.config import ModelConfig
from dataclasses import asdict, dataclass, field
from Emperor.base.enums import BaseOptions
from Emperor.datasets.image.mnist import Mnist
from Emperor.base.utils import ConfigBase, Module
from Emperor.datasets.image.cifar_10 import Cifar10
from Emperor.datasets.image.cifar_100 import Cifar100
from Emperor.linears.options import LinearLayerOptions
from Emperor.linears.utils.layers import LinearLayerConfig
from Emperor.base.layer import LayerStack, LayerStackConfig
from Emperor.experiments.utils.factories import Experiments
from Emperor.datasets.image.fashion_mnist import FashionMNIST
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions


@dataclass
class LinearExperimentConfig(ConfigBase):
    model_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )


@dataclass
class ExperimentConfig(ConfigBase):
    dataset: type | None = field(
        default=None,
    )

    grid_search_configs: list["ModelConfig"] | None = field(
        default=None,
    )


class Model(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.main_cfg: LinearExperimentConfig = self._resolve_main_config(self.cfg, cfg)
        self.model_config: LayerStackConfig = self.main_cfg.model_config
        self.model = LayerStack(self.model_config).build_model()

    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        X = X.to(self.device)
        X = torch.flatten(X, start_dim=1)
        X = self.model(X)
        return X


class LinearExperimentOptions(BaseOptions):
    BASE = 0


class LinearExperiment(Experiments):
    def __init__(
        self,
        model_config_option: LinearExperimentOptions | None = None,
        mini_datasetset_flag: bool = False,
    ) -> None:
        self.print_frequency = 50
        self.model_config_option = model_config_option
        super().__init__(mini_datasetset_flag, self.print_frequency)

    def _get_num_epochs(self) -> int:
        return 20

    def _get_learning_rates(self) -> list:
        return [1e-3]

    def _get_dataset_options(self) -> list:
        return [Mnist, FashionMNIST, Cifar10, Cifar100]

    def _get_model_config(self):
        if self.model_config_option is None:
            return None
        return LinearExperimentPresets().get_config(self.model_config_option)

    def _get_model_type(self) -> type:
        return Model

    def train_model(self) -> None:
        if self.model_config_option is not None:
            super().train_model()
            return None

        for config_option in LinearExperimentOptions:
            self.model_config_option = config_option
            for dataset_type in self.dataset_options:
                config_options = LinearExperimentPresets().get_config(
                    config_option, dataset_type
                )
                self._set_dataset_option(dataset_type)
                for config in config_options:
                    self._set_model_config(config)
                    super().train_model()


class LinearExperimentPresets:
    def __init__(self) -> None:
        self.dataset_specs = {
            Mnist: {
                "input_dim": 28 * 28,
                "output_dim": 10,
            },
            FashionMNIST: {
                "input_dim": 28 * 28,
                "output_dim": 10,
            },
            Cifar10: {
                "input_dim": 32 * 32 * 3,
                "output_dim": 10,
            },
            Cifar100: {
                "input_dim": 32 * 32 * 3,
                "output_dim": 100,
            },
        }

    def get_config(
        self,
        dataset: type = Mnist,
        num_random_search_samples: int | None = 0,
    ) -> list["ModelConfig"]:
        spec = self.dataset_specs[dataset]

        base_config = self.__default_preset(
            input_dim=spec["input_dim"],
            output_dim=spec["output_dim"],
        )

        search_space = {
            "hidden_dim": [128, 256],
            "stack_num_layers": [3, 6],
            "stack_dropout_probability": [0.0, 0.1],
            "stack_activation": [ActivationOptions.RELU, ActivationOptions.SILU],
        }

        return self.__create_search(
            base_config, search_space, num_random_search_samples
        )

    def __create_search(
        self,
        base_params: ModelConfig,
        search_space: dict,
        num_samples: int | None = None,
    ) -> list["ModelConfig"]:
        experiments = []
        parameter_names = list(search_space.keys())
        parameter_value_options = list(search_space.values())

        is_grid_search = num_samples is None
        if is_grid_search:
            all_combinations = itertools.product(*parameter_value_options)
        else:
            random_combinations = (
                tuple(
                    random.choice(value_options)
                    for value_options in parameter_value_options
                )
                for _ in range(num_samples)
            )
            all_combinations = random_combinations

        for parameter_values in all_combinations:
            base_params_dict = asdict(base_params)
            updated_params = copy.deepcopy(base_params_dict)
            for param_name, param_value in zip(parameter_names, parameter_values):
                updated_params[param_name] = param_value
            experiments.append(self.__default_preset(**updated_params))

        return experiments

    def __default_preset(
        self,
        batch_size: int = 64,
        input_dim: int = 28**2,
        hidden_dim: int = 256,
        output_dim: int = 10,
        bias_flag: bool = True,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DEFAULT,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DIAGONAL,
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
    ) -> ModelConfig:
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            override_config=LinearExperimentConfig(
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
    parser = get_parser(LinearExperimentOptions.names())
    args = parser.parse_args()
    config_option = LinearExperimentOptions.get_option(args.config_name)

    experiment = LinearExperiment(config_option)
    experiment.train_model()
