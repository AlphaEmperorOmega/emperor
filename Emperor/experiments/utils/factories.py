import os
import datetime

from pprint import pprint
import itertools
from copy import deepcopy
from dataclasses import asdict
from Emperor.base.utils import Trainer
from Emperor.config import ModelConfig
from Emperor.datasets.image.mnist import Mnist
from Emperor.datasets.image.cifar_10 import Cifar10
from Emperor.datasets.image.cifar_100 import Cifar100
from Emperor.datasets.image.fashion_mnist import FashionMNIST
from Emperor.experiments.utils.models import ClassifierExperiment

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from Emperor.experiments.utils.models import ClassifierExperiment


def create_search_space(
    base_preset_callback: Callable,
    base_config: dict,
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


class ExperimentBase:
    def __init__(
        self,
        mini_datasetset_flag: bool = True,
        print_loss_frequency: int = 50,
    ) -> None:
        self.mini_datasetset_flag = mini_datasetset_flag
        self.print_loss_flag = True
        self.print_loss_frequency = print_loss_frequency

        self.num_epochs = self._get_num_epochs()
        self.learning_rates = self._get_learning_rates()
        self.dataset_options = self._get_dataset_options()
        self.model_config = self._get_model_config()
        self.experiment_type = self._get_experiment_type()
        self.model_type = self._get_model_type()

    def _get_num_epochs(self) -> int:
        return 10

    def _get_learning_rates(self) -> list:
        return [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    def _get_dataset_options(self) -> list:
        return [Mnist, FashionMNIST, Cifar10, Cifar100]

    def _get_model_config(self) -> ModelConfig:
        raise NotImplementedError(
            "The method '_get_model_config' must be implemented in the subclass."
        )

    def _set_model_config(self, model_config: "ModelConfig") -> None:
        self.model_config = model_config

    def _set_dataset_option(self, dataset_option) -> None:
        self.dataset_options = [dataset_option]

    def _get_experiment_type(self):
        return ClassifierExperiment

    def _get_model_type(self) -> type:
        raise NotImplementedError(
            "The method '_get_model_type' must be implemented in the subclass."
        )

    def train_model(self) -> None:
        for dataset_option in self.dataset_options:
            for learning_rate in self.learning_rates:
                batch_size = self.model_config.batch_size
                dataset = dataset_option(batch_size=batch_size)
                model = self.experiment_type(
                    self.model_config, self.model_type, learning_rate
                )
                trainer = Trainer(max_epochs=self.num_epochs)
                parameter_count = self.__print_model_parameter_count(model)
                self.__print_model_title(dataset_option, learning_rate, parameter_count)
                self._initialize_monitor(dataset, model)
                trainer.fit(
                    model,
                    dataset,
                    self.print_loss_flag,
                    self.print_loss_frequency,
                )

    def __print_model_parameter_count(self, model) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def __print_model_title(
        self,
        dataset,
        learning_rate: float,
        parameter_count: int,
    ) -> None:
        print("\n" * 2 + "#" * 50)
        message_parts = [
            f"# Model type: {self.model_type.__name__} \n",
            f"# Dataset type: {dataset.__name__} \n",
            f"# Learning rate: {learning_rate} \n",
            f"# Experiment type: {self.experiment_type.__name__} \n",
            f"# Model Parameter count: {parameter_count} \n",
            f"# Config option: {self.model_config_option} \n",
        ]
        message = "\n" + "".join(message_parts) + "\n"
        print(message)
        print("#" * 50 + "\n" * 2)

    def _initialize_monitor(
        self,
        dataset,
        model,
    ) -> None:
        dataset_name = dataset.__class__.__name__
        experiment_type = model.__class__.__name__
        current_date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        learning_rate = "lr_" + str(model.lr)
        model_config_option = "config_" + self.model_config_option.name
        log_dir_parts = [
            experiment_type,
            model_config_option,
            learning_rate,
            dataset_name,
            current_date_time,
        ]
        log_file_name = "_".join(log_dir_parts)
        log_dir_path = os.path.join(log_file_name)
        model.initialize_monitor(log_dir=log_dir_path)
