import torch

from torch import Tensor
from Emperor.experiments.utils.models import ClassifierExperiment
from models.parser import get_parser
from lightning import Trainer
from dataclasses import dataclass, field
from Emperor.base.enums import BaseOptions
from Emperor.datasets.image.mnist import Mnist
from Emperor.base.utils import ConfigBase
from Emperor.datasets.image.cifar_10 import Cifar10
from Emperor.datasets.image.cifar_100 import Cifar100
from Emperor.linears.utils.layers import LinearLayerConfig
from Emperor.base.layer import LayerStack, LayerStackConfig
from Emperor.experiments.utils.factories import ExperimentBase, create_search_space
from Emperor.datasets.image.fashion_mnist import FashionMNIST
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


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
    BASE = 0


class Experiment(ExperimentBase):
    def __init__(
        self,
        model_config_option: ExperimentOptions | None = None,
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
        return ExperimentPresets().get_config(self.model_config_option)

    def _get_model_type(self) -> type:
        return Model

    def train_model(self) -> None:
        if self.model_config_option is not None:
            self._run_experiment()
            return None

        for config_option in ExperimentOptions:
            self.model_config_option = config_option
            for dataset_type in self.dataset_options:
                config_options = ExperimentPresets().get_config(
                    config_option, dataset_type
                )
                self._set_dataset_option(dataset_type)
                for config in config_options:
                    self._set_model_config(config)
                    self._run_experiment()

    def _run_experiment(self) -> None:
        for dataset_type in self.dataset_options:
            dataset = dataset_type(batch_size=self.model_config.batch_size)
            model = Model(cfg=self.model_config)
            trainer = Trainer(max_epochs=self.num_epochs, accelerator="auto")
            trainer.fit(model, datamodule=dataset)


class ExperimentPresets:
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
        model_config_options: ExperimentOptions = ExperimentOptions.BASE,
        dataset: type = Mnist,
    ) -> list["ModelConfig"]:
        match model_config_options:
            case ExperimentOptions.BASE:
                return self.__base_grid_search_config(dataset)
            case _:
                raise ValueError(
                    "The specified option is not supported. Please choose a valid `LinearExperimentOptions`."
                )

    def __base_grid_search_config(
        self,
        dataset: type = Mnist,
        num_random_search_samples: int | None = None,
    ) -> list["ModelConfig"]:
        spec = self.dataset_specs[dataset]

        base_config = {
            "input_dim": spec["input_dim"],
            "output_dim": spec["output_dim"],
        }

        search_space = {
            "hidden_dim": [128, 256],
            "stack_num_layers": [3, 6],
            "stack_dropout_probability": [0.0, 0.1],
            "stack_activation": [ActivationOptions.RELU, ActivationOptions.SILU],
        }

        return create_search_space(
            self.__preset, base_config, search_space, num_random_search_samples
        )

    def __preset(
        self,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        input_dim: int = 28**2,
        hidden_dim: int = 256,
        output_dim: int = 10,
        bias_flag: bool = True,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DEFAULT,
        stack_num_layers: int = 3,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
    ) -> "ModelConfig":
        from Emperor.config import ModelConfig
        from Emperor.linears.options import LinearLayerOptions

        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            override_config=ExperimentConfig(
                model_config=LayerStackConfig(
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
            ),
        )


if __name__ == "__main__":
    parser = get_parser(ExperimentOptions.names())
    args = parser.parse_args()
    config_option = ExperimentOptions.get_option(args.config_name)

    experiment = Experiment(config_option)
    experiment.train_model()
