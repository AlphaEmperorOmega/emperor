from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from emperor.config import BaseOptions, ModelConfig
from emperor.experiments import (
    ExperimentTask,
    experiment_task_name,
)
from model_runtime.packages import ModelPackage
from model_runtime.runs._lightning_progress import lightning_progress_adapter
from model_runtime.runs.artifacts import FilesystemRunArtifacts, RunArtifacts
from model_runtime.runs.progress import (
    ContextualRunProgress,
    RunProgress,
    RunProgressContext,
    contextual_run_progress,
)
from model_runtime.task_behavior import experiment_task_behavior


@dataclass
class TrainingRun:
    experiment_task: ExperimentTask | None
    preset: BaseOptions
    dataset_type: type
    config: ModelConfig
    config_overrides: dict
    num_epochs: int
    parameters: dict[str, object] = field(default_factory=dict)
    run_id: str | None = None
    run_index: int | None = None
    run_total: int | None = None


class ExperimentBase:
    def __init__(
        self,
        preset: BaseOptions | None = None,
        experiment_task: ExperimentTask | str | None = None,
        *,
        model_package: ModelPackage,
        run_artifacts: RunArtifacts | None = None,
    ) -> None:
        if not isinstance(model_package, ModelPackage):
            raise TypeError("Runs require an explicit ModelPackage.")
        self.model_package = model_package
        self.run_artifacts = (
            run_artifacts if run_artifacts is not None else FilesystemRunArtifacts()
        )
        self.preset = preset
        self.num_epochs = self._num_epochs()
        self.experiment_task = self._resolve_experiment_task(experiment_task)
        self.dataset_options = self._dataset_options_for_task(self.experiment_task)
        self.preset_generator = model_package.presets
        self.preset_enum = model_package.preset_type

    def _num_epochs(self) -> int:
        return 10

    def _resolve_experiment_task(
        self,
        experiment_task: ExperimentTask | str | None,
    ) -> ExperimentTask:
        return self.model_package.resolve_experiment_task(experiment_task)

    def _dataset_options_for_task(
        self,
        experiment_task: ExperimentTask,
    ) -> list:
        return self.model_package.metadata.dataset_options_for_task(experiment_task)

    def _load_trainer_config(self, config_overrides: dict | None = None) -> dict:
        config = self.model_package.runtime_defaults
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
        config = self.model_package.runtime_defaults
        config_overrides = config_overrides or {}

        def runtime_value(key: str, default):
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
        """Return Experiment Task arguments for a data module."""

        task = getattr(training_run, "experiment_task", None) or self.experiment_task
        return experiment_task_behavior(task).dataset_constructor_kwargs(
            training_run.config
        )

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

    def materialize_training_runs(
        self,
        materialized_runs: list[dict],
    ) -> list[TrainingRun]:
        """Materialize an accepted semantic Run Plan for execution."""

        training_runs = []
        run_total = len(materialized_runs)
        for run_index, run in enumerate(materialized_runs, start=1):
            preset = run["preset"]
            dataset_type = run["dataset_type"]
            run_overrides = run.get("config_overrides") or {}
            run_parameters = run.get("parameters") or {}
            run_epochs = run_overrides.get("num_epochs", self.num_epochs)
            configs = self.preset_generator.get_config(
                preset,
                dataset_type,
                config_overrides=run_overrides,
            )
            if len(configs) != 1:
                raise ValueError(
                    "Accepted Run materialization must produce exactly one "
                    f"configuration, got {len(configs)}."
                )
            training_runs.append(
                TrainingRun(
                    experiment_task=self.experiment_task,
                    preset=preset,
                    dataset_type=dataset_type,
                    config=configs[0],
                    config_overrides=run_overrides,
                    num_epochs=run_epochs,
                    parameters=dict(run_parameters),
                    run_id=run.get("id"),
                    run_index=run.get("index", run_index),
                    run_total=run.get("run_total", run_total),
                )
            )
        return training_runs

    def execute_training_run(
        self,
        training_run: TrainingRun,
        *,
        callbacks: list[Callback],
        progress: RunProgress | None = None,
        progress_step_interval: int = 1,
        ckpt_path: Path | None = None,
        model_validator: Callable[[object], None] | None = None,
        resumed_from: Mapping[str, object] | None = None,
    ) -> tuple[dict, str]:
        """Execute one materialized Run through the public runtime Interface."""

        return self._execute_training_run(
            training_run,
            callbacks=callbacks,
            progress=progress,
            progress_step_interval=progress_step_interval,
            ckpt_path=ckpt_path,
            model_validator=model_validator,
            resumed_from=resumed_from,
        )

    def _execute_training_run(
        self,
        training_run: TrainingRun,
        *,
        callbacks: list[Callback],
        progress: RunProgress | None = None,
        progress_step_interval: int = 1,
        ckpt_path: Path | None = None,
        model_validator: Callable[[object], None] | None = None,
        resumed_from: Mapping[str, object] | None = None,
    ) -> tuple[dict, str]:
        run_progress = contextual_run_progress(
            progress,
            self._run_progress_context(training_run),
        )
        try:
            trainer_config = self._load_trainer_config(training_run.config_overrides)
            runtime_config = self._load_runtime_config(training_run.config_overrides)
            if runtime_config["seed"] is not None:
                seed_everything(int(runtime_config["seed"]), workers=True)
            dataset = self._build_dataset(training_run)
            self._configure_dataset(dataset, runtime_config)
            model = self.model_package.build_model(training_run.config)
            if model_validator is not None:
                model_validator(model)
            logger = TensorBoardLogger(
                save_dir=str(self.run_artifacts.root),
                name=self.run_artifacts.run_name(
                    self.model_package.identity,
                    training_run.preset.name,
                    training_run.dataset_type.__name__,
                    training_run.parameters,
                ),
            )
            if run_progress is not None:
                run_progress = run_progress.with_log_dir(logger.log_dir)
            self._emit_dataset_started(
                training_run,
                run_progress,
                resumed_from=resumed_from,
            )
            progress_callbacks = (
                [
                    lightning_progress_adapter(
                        run_progress,
                        step_interval=progress_step_interval,
                    )
                ]
                if run_progress is not None
                else []
            )
            trainer = Trainer(
                max_epochs=training_run.num_epochs,
                logger=logger,
                callbacks=[
                    *trainer_config["callbacks"],
                    *callbacks,
                    *progress_callbacks,
                ],
                **trainer_config["trainer_args"],
            )
            if ckpt_path is None:
                trainer.fit(model, datamodule=dataset)
            else:
                trainer.fit(model, datamodule=dataset, ckpt_path=ckpt_path)
            if runtime_config["run_test_after_fit"]:
                trainer.test(model, datamodule=dataset)
            result = self._training_result(
                training_run,
                trainer,
                resumed_from=resumed_from,
            )
            self.run_artifacts.write_result(logger.log_dir, result)
            self.run_artifacts.update_best_results(
                self.model_package.identity,
                self.experiment_task,
                result,
            )
            self._emit_dataset_completed(
                result,
                run_progress,
                resumed_from=resumed_from,
            )
            return result, logger.log_dir
        except Exception as exc:
            self._emit_training_error(exc, run_progress)
            raise

    def _run_progress_context(
        self,
        training_run: TrainingRun,
    ) -> RunProgressContext:
        experiment_task = (
            experiment_task_name(training_run.experiment_task)
            if training_run.experiment_task is not None
            else None
        )
        return RunProgressContext(
            experiment_task=experiment_task,
            dataset=training_run.dataset_type.__name__,
            preset=self._preset_cli_name(training_run.preset),
            preset_key=training_run.preset.name,
            log_dir=None,
            run_id=training_run.run_id,
            run_index=training_run.run_index,
            run_total=training_run.run_total,
            total_epochs=training_run.num_epochs,
        )

    def _emit_dataset_started(
        self,
        training_run: TrainingRun,
        progress: ContextualRunProgress | None,
        *,
        resumed_from: Mapping[str, object] | None = None,
    ) -> None:
        self._write_progress_event(
            progress,
            {
                "type": "dataset_started",
                "status": "running",
                "params": training_run.parameters,
                **(
                    {"resumedFrom": dict(resumed_from)}
                    if resumed_from is not None
                    else {}
                ),
            },
        )

    def _emit_training_error(
        self,
        exc: Exception,
        progress: ContextualRunProgress | None,
    ) -> None:
        self._write_progress_event(
            progress,
            {
                "type": "error",
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )

    def _emit_dataset_completed(
        self,
        result: dict,
        progress: ContextualRunProgress | None,
        *,
        resumed_from: Mapping[str, object] | None = None,
    ) -> None:
        self._write_progress_event(
            progress,
            {
                "type": "dataset_completed",
                "status": "running",
                "metrics": result["metrics"],
                **(
                    {"resumedFrom": dict(resumed_from)}
                    if resumed_from is not None
                    else {}
                ),
            },
        )

    @staticmethod
    def _write_progress_event(
        progress: ContextualRunProgress | None,
        event: dict,
    ) -> None:
        if progress is not None:
            progress.write_event(event)

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
            **self.model_package.identity.to_payload(),
            "experimentTask": experiment_task,
            "dataset": training_run.dataset_type.__name__,
            "preset": self._preset_cli_name(training_run.preset),
            "presetKey": training_run.preset.name,
            "params": training_run.parameters,
            **self.run_artifacts.result_metrics_payload(trainer.callback_metrics),
            **({"resumedFrom": dict(resumed_from)} if resumed_from is not None else {}),
        }

    def _preset_cli_name(self, preset: BaseOptions) -> str:
        cli_name = getattr(type(preset), "cli_name", None)
        if callable(cli_name):
            return cli_name(preset.name)
        return preset.name.lower().replace("_", "-")


__all__ = ["ExperimentBase"]
