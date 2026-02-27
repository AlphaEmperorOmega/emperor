import hashlib
import inspect
import itertools
from datetime import datetime
from pathlib import Path

from typing import Callable
from lightning import Trainer
from Emperor.config import ModelConfig
from Emperor.base.enums import BaseOptions
from Emperor.datasets.image.mnist import Mnist
from lightning.pytorch.loggers import TensorBoardLogger
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
    def __init__(self, option: BaseOptions | None = None) -> None:
        self.option = option
        self.num_epochs = self._num_epochs()
        self.dataset_options = self._dataset_options()
        self.model_type = self._model_type()
        self.preset_generator = self._preset_generator_instance()
        self.options_enumeration = self._experiment_enumeration()
        self.accelerator = "auto"

    def _num_epochs(self) -> int:
        return 10

    def _dataset_options(self) -> list:
        return [Mnist, FashionMNIST, Cifar10, Cifar100]

    def _model_type(self) -> type:
        raise NotImplementedError(
            "The method '_model_type' must be implemented in the subclass."
        )

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        raise NotImplementedError(
            "The method '_preset_generator_instance' must be implemented in the subclass."
        )

    def _experiment_enumeration(self) -> type[BaseOptions]:
        raise NotImplementedError(
            "The method '_experiment_enumeration' must be implemented in the subclass."
        )

    def train_model(self) -> None:
        options = [self.option] if self.option else self.options_enumeration
        for option in options:
            for dataset_type in self.dataset_options:
                for config in self.preset_generator.get_config(option, dataset_type):
                    dataset = dataset_type(batch_size=config.batch_size)
                    model = self.model_type(cfg=config)
                    logger = TensorBoardLogger(
                        save_dir="logs",
                        name=self._build_log_path(option, dataset_type, config),
                    )
                    trainer = Trainer(
                        max_epochs=self.num_epochs,
                        accelerator=self.accelerator,
                        logger=logger,
                    )
                    trainer.fit(model, datamodule=dataset)

    def _build_log_path(
        self, option: BaseOptions, dataset_type: type, config: "ModelConfig"
    ) -> str:
        params = config.get_custom_parameters()
        param_str = "_".join(f"{k}={v}" for k, v in params.items())
        param_id = (
            hashlib.md5(param_str.encode()).hexdigest()[:8] if param_str else "default"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_file = Path(inspect.getfile(type(self))).stem
        return f"{source_file}/{option.name}/{dataset_type.__name__}/{param_id}_{timestamp}"
