from __future__ import annotations

import importlib
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from emperor.config import BaseOptions, ModelConfig
from emperor.datasets.image.classification import Cifar10, Cifar100, FashionMNIST, Mnist
from emperor.experiments import (
    ExperimentTask,
    experiment_task_name,
    resolve_experiment_task,
)
from model_runtime.packages import ModelPackage
from model_runtime.packages.definition import model_package_from_module_path
from model_runtime.packages.identity import split_model_id
from model_runtime.packages.metadata import load_model_metadata_from_module_path
from model_runtime.packages.presets import (
    ExperimentPresetsBase,
    SearchMode,
)
from model_runtime.runs.artifacts import (
    FilesystemRunArtifacts,
    result_metrics_payload,
    result_ranking_score,
    validate_artifact_namespace,
    write_run_result,
)


def _public_model_id_from_package(package: str) -> str:
    model_package = model_package_from_module_path(package)
    if model_package is not None:
        return model_package.catalog_key
    return package.rsplit(".", 1)[-1]


def _validate_log_folder(log_folder: str | None) -> str | None:
    return validate_artifact_namespace(log_folder)


def _result_metrics_payload(metrics: dict) -> dict:
    return result_metrics_payload(metrics)


@dataclass
class TrainingRun:
    experiment_task: ExperimentTask | None
    preset: BaseOptions
    dataset_type: type
    config: ModelConfig
    config_overrides: dict
    num_epochs: int
    run_id: str | None = None
    run_index: int | None = None
    run_total: int | None = None


class ExperimentBase:
    def __init__(
        self,
        preset: BaseOptions | None = None,
        experiment_task: ExperimentTask | str | None = None,
        *,
        model_package: ModelPackage | None = None,
        run_artifacts: FilesystemRunArtifacts | None = None,
    ) -> None:
        self.model_package = model_package
        self.run_artifacts = run_artifacts
        self.preset = preset
        self.num_epochs = self._num_epochs()
        self.experiment_task = self._resolve_experiment_task(experiment_task)
        self.dataset_options = self._dataset_options_for_task(self.experiment_task)
        self.model_type = self._model_type()
        self.preset_generator = self._preset_generator_instance()
        self.preset_enum = self._experiment_preset_enum()

    def _num_epochs(self) -> int:
        return 10

    def _dataset_options(self) -> list:
        return [Mnist, FashionMNIST, Cifar10, Cifar100]

    def _model_package(self) -> ModelPackage | None:
        return self.model_package or model_package_from_module_path(
            type(self).__module__
        )

    def _resolve_experiment_task(
        self,
        experiment_task: ExperimentTask | str | None,
    ) -> ExperimentTask | None:
        metadata = self._model_metadata()
        if metadata is None:
            return resolve_experiment_task(experiment_task)
        if experiment_task is None:
            return metadata.default_experiment_task
        return resolve_experiment_task(experiment_task)

    def _dataset_options_for_task(
        self,
        experiment_task: ExperimentTask | None,
    ) -> list:
        metadata = self._model_metadata()
        if metadata is None:
            return self._dataset_options()
        return metadata.dataset_options_for_task(experiment_task)

    def _model_metadata(self):
        model_package = self._model_package()
        if model_package is not None:
            return model_package.metadata
        package = type(self).__module__.rsplit(".", 1)[0]
        try:
            return load_model_metadata_from_module_path(package)
        except ModuleNotFoundError as exc:
            expected_metadata_modules = {
                f"{package}.config",
                f"{package}.dataset_options",
                f"{package}.monitor_options",
                f"{package}.search_space",
            }
            if (
                model_package_from_module_path(package) is not None
                or exc.name not in expected_metadata_modules
            ):
                raise
            return None

    def _model_type(self) -> type:
        raise NotImplementedError(
            "The method '_model_type' must be implemented in the subclass."
        )

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        raise NotImplementedError(
            "The method '_preset_generator_instance' must be implemented in the "
            "subclass."
        )

    def _experiment_preset_enum(self) -> type[BaseOptions]:
        raise NotImplementedError(
            "The method '_experiment_preset_enum' must be implemented in the subclass."
        )

    def _load_trainer_config(self, config_overrides: dict | None = None) -> dict:
        package = type(self.preset_generator).__module__.rsplit(".", 1)[0]
        model_package = model_package_from_module_path(package)
        config = (
            model_package.runtime_defaults
            if model_package is not None
            else importlib.import_module(f"{package}.config")
        )
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
            "CALLBACK_EARLY_STOPPING_METRIC",
            "validation/loss",
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
            save_last=True,
            mode="min" if "loss" in early_stopping_metric else "max",
        )

    def _trainer_config_value(
        self,
        config,
        config_overrides: dict,
        key: str,
        default=None,
    ):
        return config_overrides.get(key.lower(), getattr(config, key, default))

    def _load_runtime_config(self, config_overrides: dict | None = None) -> dict:
        package = type(self.preset_generator).__module__.rsplit(".", 1)[0]
        model_package = model_package_from_module_path(package)
        if model_package is not None:
            config = model_package.runtime_defaults
        else:
            try:
                config = importlib.import_module(f"{package}.config")
            except ModuleNotFoundError as exc:
                if exc.name != f"{package}.config":
                    raise
                config = None
        config_overrides = config_overrides or {}

        def runtime_value(key: str, default):
            if config is None:
                return config_overrides.get(key.lower(), default)
            return self._trainer_config_value(config, config_overrides, key, default)

        return {
            "data_num_workers": runtime_value("DATA_NUM_WORKERS", None),
            "run_test_after_fit": runtime_value("RUN_TEST_AFTER_FIT", True),
            "seed": runtime_value("SEED", None),
        }

    def _configure_dataset(self, dataset, runtime_config: dict) -> None:
        data_num_workers = runtime_config.get("data_num_workers")
        if data_num_workers is None or not hasattr(dataset, "num_workers"):
            pass
        else:
            dataset.num_workers = int(data_num_workers)
        seed = runtime_config.get("seed")
        if seed is not None and hasattr(dataset, "seed"):
            dataset.seed = int(seed)

    def _dataset_constructor_kwargs(self, training_run: TrainingRun) -> dict:
        """Return package-specific keyword arguments for a data module."""

        kwargs = {"batch_size": training_run.config.batch_size}
        if training_run.experiment_task == ExperimentTask.CAUSAL_LANGUAGE_MODELING:
            kwargs["sequence_length"] = training_run.config.sequence_length
        return kwargs

    def _build_dataset(self, training_run: TrainingRun):
        return training_run.dataset_type(
            **self._dataset_constructor_kwargs(training_run)
        )

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
        selected_presets: list[BaseOptions] | None = None,
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
            selected_presets=selected_presets,
            materialized_runs=materialized_runs,
        )
        for training_run in training_run_plan:
            self._execute_training_run(
                training_run,
                log_folder=log_folder,
                callbacks=callbacks,
                best_results=best_results,
            )

    def load_best_results(self, log_folder: str | None = None) -> dict:
        """Load ranking state through the configured Run Artifact store."""

        return self._load_best_results(log_folder)

    def materialize_training_runs(
        self,
        materialized_runs: list[dict],
        log_folder: str | None,
    ) -> list[TrainingRun]:
        """Materialize an accepted semantic Run Plan for execution."""

        return self._materialized_training_runs(materialized_runs, log_folder)

    def execute_training_run(
        self,
        training_run: TrainingRun,
        *,
        log_folder: str | None,
        callbacks: list[Callback],
        best_results: dict,
        ckpt_path: Path | None = None,
        model_validator: Callable[[object], None] | None = None,
        resumed_from: Mapping[str, object] | None = None,
    ) -> tuple[dict, str]:
        """Execute one materialized Run through the public runtime Interface."""

        return self._execute_training_run(
            training_run,
            log_folder=log_folder,
            callbacks=callbacks,
            best_results=best_results,
            ckpt_path=ckpt_path,
            model_validator=model_validator,
            resumed_from=resumed_from,
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
        selected_presets: list[BaseOptions] | None,
        materialized_runs: list[dict] | None,
    ) -> list[TrainingRun]:
        presets = (
            selected_presets
            if selected_presets is not None
            else [self.preset]
            if self.preset
            else self.preset_enum
        )
        dataset_options = selected_datasets or self.dataset_options
        if materialized_runs is None:
            return self._planned_training_runs(
                presets=presets,
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
        presets,
        dataset_options: list[type],
        search_mode: SearchMode,
        log_folder: str | None,
        search_keys: list[str] | None,
        config_overrides: dict,
        search_overrides: dict,
    ) -> list[TrainingRun]:
        training_runs = []
        for preset in presets:
            for dataset_type in dataset_options:
                run_overrides = config_overrides
                run_epochs = run_overrides.get("num_epochs", self.num_epochs)
                for config in self.preset_generator.get_config(
                    preset,
                    dataset_type,
                    search_mode,
                    log_folder,
                    search_keys,
                    config_overrides=run_overrides,
                    search_overrides=search_overrides,
                ):
                    training_runs.append(
                        TrainingRun(
                            experiment_task=self.experiment_task,
                            preset=preset,
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
    ) -> list[TrainingRun]:
        training_runs = []
        run_total = len(materialized_runs)
        for run_index, run in enumerate(materialized_runs, start=1):
            preset = run["preset"]
            dataset_type = run["dataset_type"]
            run_overrides = run.get("config_overrides") or {}
            run_epochs = run_overrides.get("num_epochs", self.num_epochs)
            for config in self.preset_generator.get_config(
                preset,
                dataset_type,
                None,
                log_folder,
                None,
                config_overrides=run_overrides,
                search_overrides={},
            ):
                training_runs.append(
                    TrainingRun(
                        experiment_task=self.experiment_task,
                        preset=preset,
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
        training_run: TrainingRun,
        *,
        log_folder: str | None,
        callbacks: list[Callback],
        best_results: dict,
        ckpt_path: Path | None = None,
        model_validator: Callable[[object], None] | None = None,
        resumed_from: Mapping[str, object] | None = None,
    ) -> tuple[dict, str]:
        trainer_config = self._load_trainer_config(training_run.config_overrides)
        runtime_config = self._load_runtime_config(training_run.config_overrides)
        if runtime_config["seed"] is not None:
            seed_everything(int(runtime_config["seed"]), workers=True)
        dataset = self._build_dataset(training_run)
        self._configure_dataset(dataset, runtime_config)
        model_package = self._model_package()
        model = (
            model_package.build_model(training_run.config)
            if model_package is not None
            else self.model_type(training_run.config)
        )
        if model_validator is not None:
            model_validator(model)
        artifact_store = self._artifact_store(log_folder)
        logger = TensorBoardLogger(
            save_dir=str(artifact_store.root),
            name=self._build_log_path(
                training_run.preset,
                training_run.dataset_type,
                training_run.config,
                log_folder,
            ),
        )
        self._set_training_run_context(training_run, logger, callbacks)
        self._emit_dataset_started(
            training_run,
            logger,
            callbacks,
            resumed_from=resumed_from,
        )
        trainer = Trainer(
            max_epochs=training_run.num_epochs,
            logger=logger,
            callbacks=[*trainer_config["callbacks"], *callbacks],
            **trainer_config["trainer_args"],
        )
        try:
            if ckpt_path is None:
                trainer.fit(model, datamodule=dataset)
            else:
                trainer.fit(model, datamodule=dataset, ckpt_path=ckpt_path)
            if runtime_config["run_test_after_fit"]:
                trainer.test(model, datamodule=dataset)
        except Exception as exc:
            self._emit_training_error(training_run, exc, callbacks)
            raise

        result = self._training_result(
            training_run,
            trainer,
            resumed_from=resumed_from,
        )
        self._write_training_result(logger.log_dir, result)
        self._update_best_results(result, best_results, log_folder)
        self._emit_dataset_completed(
            training_run,
            logger,
            result,
            callbacks,
            resumed_from=resumed_from,
        )
        return result, logger.log_dir

    def _set_training_run_context(
        self,
        training_run: TrainingRun,
        logger,
        callbacks: list[Callback],
    ) -> None:
        for callback in callbacks:
            set_run_context = getattr(callback, "set_run_context", None)
            if callable(set_run_context):
                set_run_context(
                    training_run.dataset_type.__name__,
                    logger.log_dir,
                    self._preset_cli_name(training_run.preset),
                    training_run.preset.name,
                    run_id=training_run.run_id,
                    run_index=training_run.run_index,
                    run_total=training_run.run_total,
                    total_epochs=training_run.num_epochs,
                )

    def _emit_dataset_started(
        self,
        training_run: TrainingRun,
        logger,
        callbacks: list[Callback],
        *,
        resumed_from: Mapping[str, object] | None = None,
    ) -> None:
        self._write_progress_event(
            callbacks,
            {
                **self._training_run_event_fields(training_run),
                "type": "dataset_started",
                "status": "running",
                "logDir": logger.log_dir,
                "params": training_run.config.get_custom_parameters(),
                **(
                    {"resumedFrom": dict(resumed_from)}
                    if resumed_from is not None
                    else {}
                ),
            },
        )

    def _emit_training_error(
        self,
        training_run: TrainingRun,
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
        training_run: TrainingRun,
        logger,
        result: dict,
        callbacks: list[Callback],
        *,
        resumed_from: Mapping[str, object] | None = None,
    ) -> None:
        self._write_progress_event(
            callbacks,
            {
                **self._training_run_event_fields(training_run),
                "type": "dataset_completed",
                "status": "running",
                "metrics": result["metrics"],
                "logDir": logger.log_dir,
                **(
                    {"resumedFrom": dict(resumed_from)}
                    if resumed_from is not None
                    else {}
                ),
            },
        )

    def _training_run_event_fields(self, training_run: TrainingRun) -> dict:
        experiment_task = (
            experiment_task_name(training_run.experiment_task)
            if training_run.experiment_task is not None
            else None
        )
        return {
            "experimentTask": experiment_task,
            "dataset": training_run.dataset_type.__name__,
            "preset": self._preset_cli_name(training_run.preset),
            "presetKey": training_run.preset.name,
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

    def _training_result(
        self,
        training_run: TrainingRun,
        trainer,
        *,
        resumed_from: Mapping[str, object] | None = None,
    ) -> dict:
        experiment_task = (
            experiment_task_name(training_run.experiment_task)
            if training_run.experiment_task is not None
            else None
        )
        return {
            **self._public_model_identity_payload(),
            "experimentTask": experiment_task,
            "dataset": training_run.dataset_type.__name__,
            "preset": self._preset_cli_name(training_run.preset),
            "presetKey": training_run.preset.name,
            "params": training_run.config.get_custom_parameters(),
            **_result_metrics_payload(trainer.callback_metrics),
            **({"resumedFrom": dict(resumed_from)} if resumed_from is not None else {}),
        }

    def _write_training_result(self, log_dir: str, result: dict) -> None:
        write_run_result(log_dir, result)

    def _preset_cli_name(self, preset: BaseOptions) -> str:
        cli_name = getattr(type(preset), "cli_name", None)
        if callable(cli_name):
            return cli_name(preset.name)
        return preset.name.lower().replace("_", "-")

    def _load_best_results(self, log_folder: str | None = None) -> dict:
        return self._artifact_store(log_folder).read_best_results(
            self._model_identity()
        )

    def _update_best_results(
        self, result: dict, top5: dict, log_folder: str | None = None
    ) -> None:
        self._artifact_store(log_folder).update_best_results(
            self._model_identity(),
            self.experiment_task,
            result,
            top5,
        )

    def _result_ranking_score(self, result: dict) -> tuple[float, float]:
        return result_ranking_score(self.experiment_task, result)

    def _best_results_path(self, log_folder: str | None = None) -> Path:
        return self._artifact_store(log_folder).best_results_path(
            self._model_identity()
        )

    def _build_log_path(
        self,
        preset: BaseOptions,
        dataset_type: type,
        config: ModelConfig,
        log_folder: str | None = None,
    ) -> str:
        return self._artifact_store(log_folder).run_name(
            self._model_identity(),
            preset.name,
            dataset_type.__name__,
            config.get_custom_parameters(),
        )

    def _artifact_store(
        self,
        log_folder: str | None = None,
    ) -> FilesystemRunArtifacts:
        if self.run_artifacts is not None:
            if self.run_artifacts.namespace == log_folder:
                return self.run_artifacts
            return FilesystemRunArtifacts(
                root=self.run_artifacts.root,
                namespace=log_folder,
                clock=self.run_artifacts.clock,
            )
        return FilesystemRunArtifacts(root=Path("logs"), namespace=log_folder)

    def _model_identity(self):
        model_package = self._model_package()
        if model_package is not None:
            return model_package.identity
        return self._public_model_id()

    def _public_model_id(self) -> str:
        package = type(self).__module__.rsplit(".", 1)[0]
        return _public_model_id_from_package(package)

    def _public_model_identity_payload(self) -> dict[str, str]:
        model_id = self._public_model_id()
        identity = split_model_id(model_id)
        if identity is not None:
            return identity.to_payload()
        return {"modelType": "models", "model": model_id}


__all__ = ["ExperimentBase", "TrainingRun"]
