import json
import random
import hashlib
import inspect
import itertools
import importlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from typing import Callable
from lightning import Trainer, seed_everything
from emperor.config import ModelConfig
from emperor.base.enums import BaseOptions
from emperor.datasets.image.classification.mnist import Mnist
from lightning.pytorch.loggers import TensorBoardLogger
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST


@dataclass
class GridSearch:
    pass


@dataclass
class RandomSearch:
    num_samples: int


SearchMode = GridSearch | RandomSearch | None


def create_search_space(
    base_preset_callback: Callable,
    base_config: dict,
    search_space: dict = {},
    search_mode: SearchMode = None,
) -> list["ModelConfig"]:
    if search_space == {}:
        return [base_preset_callback(**base_config)]

    experiments = []
    parameter_names = list(search_space.keys())
    parameter_value_options = list(search_space.values())

    if isinstance(search_mode, RandomSearch):
        all_combinations_list = list(itertools.product(*parameter_value_options))
        all_combinations = random.sample(
            all_combinations_list,
            min(search_mode.num_samples, len(all_combinations_list)),
        )
    else:
        all_combinations = itertools.product(*parameter_value_options)

    for parameter_values in all_combinations:
        updated_params = {**base_config}
        for param_name, param_value in zip(parameter_names, parameter_values):
            updated_params[param_name] = param_value
        preset = base_preset_callback(**updated_params)
        experiments.append(preset)

    return experiments


class ExperimentPresetsBase:
    def get_config(
        self, model_config_options, dataset, search_mode: SearchMode = None,
        log_folder: str | None = None,
    ) -> list["ModelConfig"]:
        raise NotImplementedError(
            "The method 'train_model' must be implemented in the subclass."
        )

    def _preset(self, *args, **kwargs) -> "ModelConfig":
        raise NotImplementedError(
            "The method '_preset' must be implemented in the subclass."
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            "input_dim": dataset.flattened_input_dim,
            "output_dim": dataset.num_classes,
        }

    def _create_default_preset_configs(
        self,
        dataset: type = Mnist,
    ) -> list["ModelConfig"]:
        base_config = self._dataset_config(dataset)
        return create_search_space(self._preset, base_config)

    def _create_default_search_space_configs(
        self,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
    ) -> list["ModelConfig"]:
        base_config = {
            **self._dataset_config(dataset),
            **self._best_params(dataset, log_folder),
        }
        return create_search_space(
            self._preset,
            base_config,
            self._extract_search_space_from_config(search_mode),
            search_mode,
        )

    def _best_params(self, dataset: type, log_folder: str | None = None) -> dict:
        package = type(self).__module__.rsplit(".", 1)[0]
        source_name = package.rsplit(".", 1)[-1]
        folder = f"{log_folder}/{source_name}" if log_folder else source_name
        path = Path("logs") / folder / "best_results.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        runs = data.get(dataset.__name__, [])
        if not runs:
            return {}
        best = min(runs, key=lambda r: r.get("rank", 999))
        return {k: v for k, v in best.get("params", {}).items()
                if type(v) in (int, float, bool)}

    def _extract_search_space_from_config(self, search_mode: SearchMode = None) -> dict:
        if search_mode is None:
            return {}
        package = type(self).__module__.rsplit(".", 1)[0]
        config = importlib.import_module(f"{package}.config")
        prefix = "SEARCH_SPACE_"
        return {
            key[len(prefix) :].lower(): value
            for key, value in vars(config).items()
            if key.startswith(prefix)
        }


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

    def train_model(
        self, search_mode: SearchMode = None, log_folder: str | None = None
    ) -> None:
        options = [self.option] if self.option else self.options_enumeration
        top5 = self._load_best_results(log_folder)
        for option in options:
            for dataset_type in self.dataset_options:
                for config in self.preset_generator.get_config(
                    option, dataset_type, search_mode, log_folder
                ):
                    seed_everything(42, workers=True)
                    dataset = dataset_type(batch_size=config.batch_size)
                    model = self.model_type(cfg=config)
                    logger = TensorBoardLogger(
                        save_dir="logs",
                        name=self._build_log_path(
                            option, dataset_type, config, log_folder
                        ),
                    )
                    trainer = Trainer(
                        max_epochs=self.num_epochs,
                        accelerator=self.accelerator,
                        logger=logger,
                    )
                    trainer.fit(model, datamodule=dataset)

                    result = {
                        "dataset": dataset_type.__name__,
                        "option": option.name,
                        "params": config.get_custom_parameters(),
                        "metrics": {
                            k: v.item()
                            for k, v in trainer.callback_metrics.items()
                        },
                    }
                    Path(logger.log_dir).mkdir(parents=True, exist_ok=True)
                    (Path(logger.log_dir) / "result.json").write_text(
                        json.dumps(result, indent=2, default=str)
                    )
                    self._update_best_results(result, top5, log_folder)

    def _load_best_results(self, log_folder: str | None = None) -> dict:
        source_file = Path(inspect.getfile(type(self))).parent.name
        folder = f"{log_folder}/{source_file}" if log_folder is not None else source_file
        summary_path = Path("logs") / folder / "best_results.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text())
        return {}

    def _update_best_results(
        self, result: dict, top5: dict, log_folder: str | None = None
    ) -> None:
        dataset = result["dataset"]
        runs = top5.get(dataset, [])
        new_acc = result["metrics"].get("val_accuracy", 0)
        worst_acc = min((r["metrics"].get("val_accuracy", 0) for r in runs), default=-1)

        if len(runs) < 5 or new_acc > worst_acc:
            runs.append(result)
            top5[dataset] = [
                {**run, "rank": i + 1}
                for i, run in enumerate(
                    sorted(
                        runs,
                        key=lambda r: r["metrics"].get("val_accuracy", 0),
                        reverse=True,
                    )[:5]
                )
            ]

            source_file = Path(inspect.getfile(type(self))).parent.name
            folder = f"{log_folder}/{source_file}" if log_folder is not None else source_file
            summary_path = Path("logs") / folder / "best_results.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(top5, indent=2, default=str))

    def _build_log_path(
        self,
        option: BaseOptions,
        dataset_type: type,
        config: "ModelConfig",
        log_folder: str | None = None,
    ) -> str:
        params = config.get_custom_parameters()
        param_str = "_".join(f"{k}={v}" for k, v in params.items())
        param_id = (
            hashlib.md5(param_str.encode()).hexdigest()[:8] if param_str else "default"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_file = Path(inspect.getfile(type(self))).parent.name
        folder = (
            f"{log_folder}/{source_file}" if log_folder is not None else source_file
        )
        return f"{folder}/{option.name}/{dataset_type.__name__}/{param_id}_{timestamp}"
