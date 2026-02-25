import itertools

from typing import Callable
from lightning import Trainer
from Emperor.config import ModelConfig
from Emperor.base.enums import BaseOptions
from Emperor.datasets.image.mnist import Mnist
from Emperor.datasets.image.cifar_10 import Cifar10
from Emperor.datasets.image.cifar_100 import Cifar100
from Emperor.datasets.image.fashion_mnist import FashionMNIST


def create_search_space(
    base_preset_callback: Callable,
    base_config: dict,
    search_space: dict,
    num_samples: int | None = None,
) -> list["ModelConfig"]:
    if search_space == {}:
        return [base_preset_callback(**base_config)]

    experiments = []
    parameter_names = list(search_space.keys())
    parameter_value_options = list(search_space.values())

    is_grid_search = num_samples is None
    if is_grid_search:
        all_combinations = itertools.product(*parameter_value_options)
    else:
        create_single_sample = lambda: (
            random.choice(value_options) for value_options in parameter_value_options
        )
        all_combinations = (create_single_sample() for _ in range(num_samples))

    for parameter_values in all_combinations:
        updated_params = base_config
        for param_name, param_value in zip(parameter_names, parameter_values):
            updated_params[param_name] = param_value
        preset = base_preset_callback(**updated_params)
        experiments.append(preset)

    return experiments


class ExperimentPresetsBase:
    def get_config(self, model_config_options, dataset) -> list["ModelConfig"]:
        raise NotImplementedError(
            "The method 'train_model' must be implemented in the subclass."
        )


class ExperimentBase:
    def __init__(self) -> None:
        self.num_epochs = self._get_num_epochs()
        self.dataset_options = self._get_dataset_options()
        self.model_type = self._get_model_type()
        self.experiment_preset_generator = self._get_experiment_preset_generator()
        self.accelerator = "auto"

    def _get_num_epochs(self) -> int:
        return 10

    def _get_dataset_options(self) -> list:
        return [Mnist, FashionMNIST, Cifar10, Cifar100]

    def _set_dataset_option(self, dataset_option) -> None:
        self.dataset_options = [dataset_option]

    def _get_model_type(self) -> type:
        raise NotImplementedError(
            "The method '_get_model_type' must be implemented in the subclass."
        )

    def _get_experiment_preset_generator(self) -> ExperimentPresetsBase:
        raise NotImplementedError(
            "The method '_get_experiment_preset_generator' must be implemented in the subclass."
        )

    def _run_experiment(
        self,
        experiment_option: BaseOptions,
    ) -> None:
        for dataset_type in self.dataset_options:
            for config in self.experiment_preset_generator.get_config(
                experiment_option, dataset_type
            ):
                dataset = dataset_type(batch_size=config.batch_size)
                model = self.model_type(cfg=config)
                trainer = Trainer(
                    max_epochs=self.num_epochs, accelerator=self.accelerator
                )
                trainer.fit(model, datamodule=dataset)

    def train_model(self) -> None:
        raise NotImplementedError(
            "The method 'train_model' must be implemented in the subclass."
        )
