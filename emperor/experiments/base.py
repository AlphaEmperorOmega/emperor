import fcntl
import hashlib
import importlib
import itertools
import json
import os
import random
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from typing import Callable
from emperor.config import ModelConfig
from emperor.base.options import BaseOptions
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from emperor.datasets.image.classification.mnist import Mnist
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST
from models.catalog import public_id_for_module
from emperor.experiments.progress import sanitize_metric_payload

DEFAULT_RESULT_METRIC_KEY_LIMIT = 512
DEFAULT_RESULT_STRING_VALUE_LIMIT = 20_000


@dataclass
class GridSearch:
    pass


@dataclass
class RandomSearch:
    num_samples: int


@dataclass(frozen=True)
class PresetLock:
    value: object
    reason: str


SearchMode = GridSearch | RandomSearch | None


@dataclass
class _TrainingRun:
    option: BaseOptions
    dataset_type: type
    config: ModelConfig
    config_overrides: dict
    num_epochs: int
    run_id: str | None = None
    run_index: int | None = None
    run_total: int | None = None


def _public_model_id_from_package(package: str) -> str:
    public_id = public_id_for_module(package)
    if public_id is not None:
        return public_id
    return package.rsplit(".", 1)[-1]


def _validate_log_folder(log_folder: str | None) -> str | None:
    if log_folder is None:
        return None
    folder = str(log_folder)
    path = Path(folder)
    if (
        not folder
        or folder in {".", ".."}
        or "\\" in folder
        or path.is_absolute()
        or len(path.parts) != 1
    ):
        raise ValueError(
            "log_folder must be a single relative folder name without path separators"
        )
    return folder


def _result_metrics_payload(metrics: dict) -> dict:
    sanitized, original_count, dropped_count = sanitize_metric_payload(
        metrics,
        metric_key_limit=DEFAULT_RESULT_METRIC_KEY_LIMIT,
        string_value_limit=DEFAULT_RESULT_STRING_VALUE_LIMIT,
    )
    payload = {"metrics": sanitized}
    if dropped_count > 0:
        payload["metricsOriginalCount"] = original_count
        payload["metricsDroppedCount"] = dropped_count
    return payload


@contextmanager
def _best_results_lock(summary_path: Path):
    lock_path = summary_path.with_suffix(summary_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _read_best_results_path(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _write_json_atomic(summary_path: Path, payload: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=summary_path.parent,
            encoding="utf-8",
            prefix=f".{summary_path.name}.",
            suffix=".tmp",
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(payload, temp_file, indent=2, default=str)
            temp_file.write("\n")
        os.replace(temp_path, summary_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


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
    PRESET_LOCKS: dict[object, dict[str, PresetLock | object]] = {}

    def get_config(
        self,
        model_config_options,
        dataset,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfig"]:
        raise NotImplementedError(
            "The method 'train_model' must be implemented in the subclass."
        )

    def _preset(self, *args, **kwargs) -> "ModelConfig":
        raise NotImplementedError(
            "The method '_preset' must be implemented in the subclass."
        )

    def locked_fields(self, model_config_options) -> dict[str, PresetLock]:
        locks = self.PRESET_LOCKS.get(model_config_options, {})
        return {
            key: value
            if isinstance(value, PresetLock)
            else PresetLock(
                value=value,
                reason=(
                    f"Locked by the {model_config_options.name} preset because this "
                    "preset owns this model behavior."
                ),
            )
            for key, value in locks.items()
        }

    def _create_default_preset_configs(
        self,
        dataset: type = Mnist,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfig"]:
        base_config = {
            **self._dataset_config(dataset),
            **self._model_config_overrides(config_overrides),
        }
        return create_search_space(
            self._preset,
            base_config,
            search_overrides or {},
        )

    def _create_preset_search_space_configs(
        self,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        preset_callback: Callable | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfig"]:
        base_config = {
            **self._dataset_config(dataset),
            **self._model_config_overrides(config_overrides),
        }
        if search_overrides and search_keys is None:
            search_space = {**search_overrides}
        else:
            search_space = self._extract_search_space_from_config(
                search_mode, search_keys
            )
            search_space.update(search_overrides or {})
        return create_search_space(
            preset_callback or self._preset,
            base_config,
            search_space,
            search_mode,
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            "input_dim": dataset.flattened_input_dim,
            "output_dim": dataset.num_classes,
        }

    def _model_config_overrides(self, config_overrides: dict | None = None) -> dict:
        ignored_prefixes = ("trainer_", "callback_")
        ignored_keys = {"num_epochs"}
        return {
            key: value
            for key, value in (config_overrides or {}).items()
            if key not in ignored_keys
            and not any(key.startswith(prefix) for prefix in ignored_prefixes)
        }

    def _best_params(self, dataset: type, log_folder: str | None = None) -> dict:
        package = type(self).__module__.rsplit(".", 1)[0]
        model_id = _public_model_id_from_package(package)
        log_folder = _validate_log_folder(log_folder)
        folder = Path(log_folder) / model_id if log_folder else Path(model_id)
        path = Path("logs") / folder / "best_results.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        runs = data.get(dataset.__name__, [])
        if not runs:
            return {}
        best = min(runs, key=lambda r: r.get("rank", 999))
        return {
            k: v
            for k, v in best.get("params", {}).items()
            if type(v) in (int, float, bool)
        }

    def _extract_search_space_from_config(
        self,
        search_mode: SearchMode = None,
        search_keys: list[str] | None = None,
    ) -> dict:
        if search_mode is None:
            return {}
        package = type(self).__module__.rsplit(".", 1)[0]
        config = importlib.import_module(f"{package}.config")
        prefix = "SEARCH_SPACE_"
        full_space = {
            key[len(prefix) :].lower(): value
            for key, value in vars(config).items()
            if key.startswith(prefix)
        }
        if search_keys is None:
            return full_space
        unknown_keys = set(search_keys) - set(full_space)
        if unknown_keys:
            raise ValueError(
                f"Unknown --search-keys: {sorted(unknown_keys)}. "
                f"Valid keys: {sorted(full_space)}"
            )
        return {key: full_space[key] for key in search_keys}


class ExperimentBase:
    def __init__(self, option: BaseOptions | None = None) -> None:
        self.option = option
        self.num_epochs = self._num_epochs()
        self.dataset_options = self._dataset_options()
        self.model_type = self._model_type()
        self.preset_generator = self._preset_generator_instance()
        self.options_enumeration = self._experiment_enumeration()

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

    def _load_trainer_config(self, config_overrides: dict | None = None) -> dict:
        package = type(self.preset_generator).__module__.rsplit(".", 1)[0]
        config = importlib.import_module(f"{package}.config")
        config_overrides = config_overrides or {}
        return {
            "trainer_args": self._trainer_args(config, config_overrides),
            "callbacks": self._trainer_callbacks(config, config_overrides),
        }

    def _trainer_callbacks(self, config, config_overrides: dict) -> list[Callback]:
        callbacks = []
        early_stopping = self._early_stopping_callback(config, config_overrides)
        if early_stopping is not None:
            callbacks.append(early_stopping)
        checkpoint = self._checkpoint_callback(config, config_overrides)
        if checkpoint is not None:
            callbacks.append(checkpoint)

        for key, value in vars(config).items():
            if key.startswith("CALLBACK_") and isinstance(value, Callback):
                callbacks.append(value)
        return callbacks

    def _early_stopping_callback(
        self,
        config,
        config_overrides: dict,
    ) -> EarlyStopping | None:
        early_stopping_patience = self._trainer_config_value(
            config,
            config_overrides,
            "CALLBACK_EARLY_STOPPING_PATIENCE",
            0,
        )
        if early_stopping_patience <= 0:
            return None
        early_stopping_metric = self._trainer_config_value(
            config,
            config_overrides,
            "CALLBACK_EARLY_STOPPING_METRIC", "validation/loss"
        )
        return EarlyStopping(
            monitor=early_stopping_metric,
            patience=early_stopping_patience,
            min_delta=self._trainer_config_value(
                config,
                config_overrides,
                "CALLBACK_EARLY_STOPPING_MIN_DELTA",
                0.0,
            ),
            strict=self._trainer_config_value(
                config,
                config_overrides,
                "CALLBACK_EARLY_STOPPING_STRICT",
                True,
            ),
            check_finite=self._trainer_config_value(
                config,
                config_overrides,
                "CALLBACK_EARLY_STOPPING_CHECK_FINITE",
                True,
            ),
            mode="min" if "loss" in early_stopping_metric else "max",
        )

    def _checkpoint_callback(
        self,
        config,
        config_overrides: dict,
    ) -> ModelCheckpoint | None:
        checkpoint_flag = self._trainer_config_value(
            config,
            config_overrides,
            "CALLBACK_CHECKPOINT_FLAG",
            False,
        )
        if not checkpoint_flag:
            return None
        early_stopping_metric = self._trainer_config_value(
            config,
            config_overrides,
            "CALLBACK_EARLY_STOPPING_METRIC",
            "validation/loss",
        )
        return ModelCheckpoint(
            monitor=early_stopping_metric,
            save_top_k=1,
            mode="min" if "loss" in early_stopping_metric else "max",
        )

    def _trainer_config_value(self, config, config_overrides: dict, key: str, default=None):
        return config_overrides.get(key.lower(), getattr(config, key, default))

    def _trainer_args(self, config, config_overrides: dict) -> dict:
        trainer_args = {}
        for key, value in vars(config).items():
            if not key.startswith("TRAINER_"):
                continue
            if value is None:
                continue
            clean_key = key[len("TRAINER_") :].lower()
            trainer_args[clean_key] = config_overrides.get(key.lower(), value)

        for key, value in config_overrides.items():
            if not key.startswith("trainer_"):
                continue
            clean_key = key[len("trainer_") :]
            if value is not None:
                trainer_args[clean_key] = value
        return trainer_args

    def train_model(
        self,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
        selected_datasets: list[type] | None = None,
        selected_options: list[BaseOptions] | None = None,
        callbacks: list[Callback] | None = None,
        materialized_runs: list[dict] | None = None,
    ) -> None:
        log_folder = _validate_log_folder(log_folder)
        config_overrides = config_overrides or {}
        search_overrides = search_overrides or {}
        callbacks = callbacks or []
        best_results = self._load_best_results(log_folder)
        training_run_plan = self._build_training_run_plan(
            search_mode=search_mode,
            log_folder=log_folder,
            search_keys=search_keys,
            config_overrides=config_overrides,
            search_overrides=search_overrides,
            selected_datasets=selected_datasets,
            selected_options=selected_options,
            materialized_runs=materialized_runs,
        )
        for training_run in training_run_plan:
            self._execute_training_run(
                training_run,
                log_folder=log_folder,
                callbacks=callbacks,
                best_results=best_results,
            )

    def _build_training_run_plan(
        self,
        *,
        search_mode: SearchMode,
        log_folder: str | None,
        search_keys: list[str] | None,
        config_overrides: dict,
        search_overrides: dict,
        selected_datasets: list[type] | None,
        selected_options: list[BaseOptions] | None,
        materialized_runs: list[dict] | None,
    ) -> list[_TrainingRun]:
        options = (
            selected_options
            if selected_options is not None
            else [self.option]
            if self.option
            else self.options_enumeration
        )
        dataset_options = selected_datasets or self.dataset_options
        if materialized_runs is None:
            return self._planned_training_runs(
                options=options,
                dataset_options=dataset_options,
                search_mode=search_mode,
                log_folder=log_folder,
                search_keys=search_keys,
                config_overrides=config_overrides,
                search_overrides=search_overrides,
            )
        return self._materialized_training_runs(materialized_runs, log_folder)

    def _planned_training_runs(
        self,
        *,
        options,
        dataset_options: list[type],
        search_mode: SearchMode,
        log_folder: str | None,
        search_keys: list[str] | None,
        config_overrides: dict,
        search_overrides: dict,
    ) -> list[_TrainingRun]:
        training_runs = []
        for option in options:
            for dataset_type in dataset_options:
                run_overrides = config_overrides
                run_epochs = run_overrides.get("num_epochs", self.num_epochs)
                for config in self.preset_generator.get_config(
                    option,
                    dataset_type,
                    search_mode,
                    log_folder,
                    search_keys,
                    config_overrides=run_overrides,
                    search_overrides=search_overrides,
                ):
                    training_runs.append(
                        _TrainingRun(
                            option=option,
                            dataset_type=dataset_type,
                            config=config,
                            config_overrides=run_overrides,
                            num_epochs=run_epochs,
                        )
                    )
        return training_runs

    def _materialized_training_runs(
        self,
        materialized_runs: list[dict],
        log_folder: str | None,
    ) -> list[_TrainingRun]:
        training_runs = []
        run_total = len(materialized_runs)
        for run_index, run in enumerate(materialized_runs, start=1):
            option = run["option"]
            dataset_type = run["dataset_type"]
            run_overrides = run.get("config_overrides") or {}
            run_epochs = run_overrides.get("num_epochs", self.num_epochs)
            for config in self.preset_generator.get_config(
                option,
                dataset_type,
                None,
                log_folder,
                None,
                config_overrides=run_overrides,
                search_overrides={},
            ):
                training_runs.append(
                    _TrainingRun(
                        option=option,
                        dataset_type=dataset_type,
                        config=config,
                        config_overrides=run_overrides,
                        num_epochs=run_epochs,
                        run_id=run.get("id"),
                        run_index=run.get("index", run_index),
                        run_total=run.get("run_total", run_total),
                    )
                )
        return training_runs

    def _execute_training_run(
        self,
        training_run: _TrainingRun,
        *,
        log_folder: str | None,
        callbacks: list[Callback],
        best_results: dict,
    ) -> None:
        trainer_config = self._load_trainer_config(training_run.config_overrides)
        dataset = training_run.dataset_type(batch_size=training_run.config.batch_size)
        model = self.model_type(cfg=training_run.config)
        logger = TensorBoardLogger(
            save_dir="logs",
            name=self._build_log_path(
                training_run.option,
                training_run.dataset_type,
                training_run.config,
                log_folder,
            ),
        )
        self._set_training_run_context(training_run, logger, callbacks)
        self._emit_dataset_started(training_run, logger, callbacks)
        trainer = Trainer(
            max_epochs=training_run.num_epochs,
            logger=logger,
            callbacks=[*trainer_config["callbacks"], *callbacks],
            **trainer_config["trainer_args"],
        )
        try:
            trainer.fit(model, datamodule=dataset)
            trainer.test(model, datamodule=dataset)
        except Exception as exc:
            self._emit_training_error(training_run, exc, callbacks)
            raise

        result = self._training_result(training_run, trainer)
        self._write_training_result(logger.log_dir, result)
        self._update_best_results(result, best_results, log_folder)
        self._emit_dataset_completed(training_run, logger, result, callbacks)

    def _set_training_run_context(
        self,
        training_run: _TrainingRun,
        logger,
        callbacks: list[Callback],
    ) -> None:
        for callback in callbacks:
            set_run_context = getattr(callback, "set_run_context", None)
            if callable(set_run_context):
                set_run_context(
                    training_run.dataset_type.__name__,
                    logger.log_dir,
                    self._option_cli_name(training_run.option),
                    training_run.option.name,
                    run_id=training_run.run_id,
                    run_index=training_run.run_index,
                    run_total=training_run.run_total,
                    total_epochs=training_run.num_epochs,
                )

    def _emit_dataset_started(
        self,
        training_run: _TrainingRun,
        logger,
        callbacks: list[Callback],
    ) -> None:
        self._write_progress_event(
            callbacks,
            {
                **self._training_run_event_fields(training_run),
                "type": "dataset_started",
                "status": "running",
                "logDir": logger.log_dir,
                "params": training_run.config.get_custom_parameters(),
            },
        )

    def _emit_training_error(
        self,
        training_run: _TrainingRun,
        exc: Exception,
        callbacks: list[Callback],
    ) -> None:
        self._write_progress_event(
            callbacks,
            {
                **self._training_run_event_fields(training_run),
                "type": "error",
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )

    def _emit_dataset_completed(
        self,
        training_run: _TrainingRun,
        logger,
        result: dict,
        callbacks: list[Callback],
    ) -> None:
        self._write_progress_event(
            callbacks,
            {
                **self._training_run_event_fields(training_run),
                "type": "dataset_completed",
                "status": "running",
                "metrics": result["metrics"],
                "logDir": logger.log_dir,
            },
        )

    def _training_run_event_fields(self, training_run: _TrainingRun) -> dict:
        return {
            "dataset": training_run.dataset_type.__name__,
            "preset": self._option_cli_name(training_run.option),
            "option": training_run.option.name,
            "runId": training_run.run_id,
            "runIndex": training_run.run_index,
            "runTotal": training_run.run_total,
            "totalEpochs": training_run.num_epochs,
        }

    def _write_progress_event(
        self,
        callbacks: list[Callback],
        event: dict,
    ) -> None:
        for callback in callbacks:
            write_event = getattr(callback, "write_event", None)
            if callable(write_event):
                write_event(event)

    def _training_result(self, training_run: _TrainingRun, trainer) -> dict:
        return {
            "model": self._public_model_id(),
            "dataset": training_run.dataset_type.__name__,
            "preset": self._option_cli_name(training_run.option),
            "option": training_run.option.name,
            "params": training_run.config.get_custom_parameters(),
            **_result_metrics_payload(trainer.callback_metrics),
        }

    def _write_training_result(self, log_dir: str, result: dict) -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        (Path(log_dir) / "result.json").write_text(
            json.dumps(result, indent=2, default=str)
        )

    def _option_cli_name(self, option: BaseOptions) -> str:
        cli_name = getattr(type(option), "cli_name", None)
        if callable(cli_name):
            return cli_name(option.name)
        return option.name.lower().replace("_", "-")

    def _load_best_results(self, log_folder: str | None = None) -> dict:
        return _read_best_results_path(self._best_results_path(log_folder))

    def _update_best_results(
        self, result: dict, top5: dict, log_folder: str | None = None
    ) -> None:
        summary_path = self._best_results_path(log_folder)
        with _best_results_lock(summary_path):
            merged_top5 = _read_best_results_path(summary_path)
            dataset = result["dataset"]
            runs = list(merged_top5.get(dataset, []))
            new_acc = result["metrics"].get("validation_accuracy", 0)
            worst_acc = min(
                (r["metrics"].get("validation_accuracy", 0) for r in runs),
                default=-1,
            )

            if len(runs) < 5 or new_acc > worst_acc:
                runs.append(result)
                merged_top5[dataset] = [
                    {**run, "rank": i + 1}
                    for i, run in enumerate(
                        sorted(
                            runs,
                            key=lambda r: r["metrics"].get(
                                "validation_accuracy", 0
                            ),
                            reverse=True,
                        )[:5]
                    )
                ]
                _write_json_atomic(summary_path, merged_top5)

            top5.clear()
            top5.update(merged_top5)

    def _best_results_path(self, log_folder: str | None = None) -> Path:
        log_folder = _validate_log_folder(log_folder)
        model_id = self._public_model_id()
        folder = (
            Path(log_folder) / model_id if log_folder is not None else Path(model_id)
        )
        return Path("logs") / folder / "best_results.json"

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
        model_id = self._public_model_id()
        log_folder = _validate_log_folder(log_folder)
        folder = f"{log_folder}/{model_id}" if log_folder is not None else model_id
        return f"{folder}/{option.name}/{dataset_type.__name__}/{param_id}_{timestamp}"

    def _public_model_id(self) -> str:
        package = type(self).__module__.rsplit(".", 1)[0]
        return _public_model_id_from_package(package)
