from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NewType, cast

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from emperor.config import BaseOptions
from emperor.experiments import ExperimentTask, experiment_task_name
from model_runtime.packages import ModelPackage, RuntimeDefaultsSpec
from model_runtime.runs._handoff import (
    TrainingExecutionRequest,
    TrainingRun,
    TrainingRunRequest,
)
from model_runtime.runs._lightning_progress import lightning_progress_adapter
from model_runtime.runs._outcomes import TrainingOutcomeObserver
from model_runtime.runs._progress_events import (
    DatasetCompletedEvent,
    DatasetStartedEvent,
    RunProgressEvent,
    TrainingErrorEvent,
)
from model_runtime.runs.artifacts import FilesystemRunArtifacts, RunArtifacts
from model_runtime.runs.progress import (
    ContextualRunProgress,
    RunProgress,
    RunProgressContext,
    contextual_run_progress,
)
from model_runtime.task_behavior import experiment_task_behavior

_ExperimentPresetUnset = NewType("_ExperimentPresetUnset", object)
_EXPERIMENT_PRESET_UNSET = _ExperimentPresetUnset(object())
_ExperimentPresetInput = BaseOptions | None | _ExperimentPresetUnset


@dataclass(frozen=True, slots=True)
class _TrainingRuntime:
    trainer_config: dict[str, Any]
    runtime_config: dict[str, Any]
    dataset: Any
    model: Any


@dataclass(frozen=True, slots=True)
class _StartedTrainingRun:
    logger: Any
    progress_callbacks: list[Callback]


@dataclass(slots=True)
class _TrainingExecutionState:
    training_run: TrainingRun
    options: TrainingExecutionRequest
    run_progress: ContextualRunProgress | None
    outcome: TrainingOutcomeObserver


class ExperimentBase:
    def __init__(
        self,
        preset: BaseOptions | None = None,
        experiment_task: ExperimentTask | str | None = None,
        *,
        model_package: ModelPackage,
        run_artifacts: RunArtifacts | None = None,
        experiment_preset: _ExperimentPresetInput = _EXPERIMENT_PRESET_UNSET,
    ) -> None:
        if not isinstance(cast(object, model_package), ModelPackage):
            raise TypeError("Runs require an explicit ModelPackage.")
        if experiment_preset is not _EXPERIMENT_PRESET_UNSET:
            if preset is not None:
                raise TypeError("Pass only 'preset' or 'experiment_preset'.")
            preset = cast(BaseOptions | None, experiment_preset)
        self.model_package = model_package
        self.run_artifacts = (
            run_artifacts if run_artifacts is not None else FilesystemRunArtifacts()
        )
        self.preset = preset
        self.experiment_task = model_package.resolve_experiment_task(experiment_task)
        self.dataset_options = model_package.dataset_options_for_task(
            self.experiment_task
        )
        runtime_defaults = model_package.runtime_defaults_spec
        default_epochs = runtime_defaults.current_value_or("NUM_EPOCHS", 10)
        self.num_epochs = cast(int, default_epochs)
        self.preset_generator = model_package.presets
        self.preset_enum = model_package.preset_type

    def _load_trainer_config(
        self,
        config_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime_defaults = self.model_package.runtime_defaults_spec
        config_overrides = config_overrides or {}
        return {
            "trainer_args": self._trainer_args(runtime_defaults, config_overrides),
            "callbacks": self._trainer_callbacks(runtime_defaults, config_overrides),
        }

    def _trainer_callbacks(
        self,
        runtime_defaults: RuntimeDefaultsSpec,
        config_overrides: dict[str, Any],
    ) -> list[Callback]:
        callbacks: list[Callback] = []
        early_stopping = self._early_stopping_callback(
            runtime_defaults,
            config_overrides,
        )
        if early_stopping is not None:
            callbacks.append(early_stopping)
        checkpoint = self._checkpoint_callback(runtime_defaults, config_overrides)
        if checkpoint is not None:
            callbacks.append(checkpoint)

        for _key, value in runtime_defaults.items_with_prefix("CALLBACK_"):
            if isinstance(value, Callback):
                callbacks.append(value)
        return callbacks

    def _early_stopping_callback(
        self,
        runtime_defaults: RuntimeDefaultsSpec,
        config_overrides: dict[str, Any],
    ) -> EarlyStopping | None:
        early_stopping_patience = self._trainer_config_value(
            runtime_defaults,
            config_overrides,
            "CALLBACK_EARLY_STOPPING_PATIENCE",
            0,
        )
        if early_stopping_patience <= 0:
            return None
        early_stopping_metric = self._trainer_config_value(
            runtime_defaults,
            config_overrides,
            "CALLBACK_EARLY_STOPPING_METRIC",
            "validation/loss",
        )
        return EarlyStopping(
            monitor=early_stopping_metric,
            patience=early_stopping_patience,
            min_delta=self._trainer_config_value(
                runtime_defaults,
                config_overrides,
                "CALLBACK_EARLY_STOPPING_MIN_DELTA",
                0.0,
            ),
            strict=self._trainer_config_value(
                runtime_defaults,
                config_overrides,
                "CALLBACK_EARLY_STOPPING_STRICT",
                True,
            ),
            check_finite=self._trainer_config_value(
                runtime_defaults,
                config_overrides,
                "CALLBACK_EARLY_STOPPING_CHECK_FINITE",
                True,
            ),
            mode="min" if "loss" in early_stopping_metric else "max",
        )

    def _checkpoint_callback(
        self,
        runtime_defaults: RuntimeDefaultsSpec,
        config_overrides: dict[str, Any],
    ) -> ModelCheckpoint | None:
        checkpoint_flag = self._trainer_config_value(
            runtime_defaults,
            config_overrides,
            "CALLBACK_CHECKPOINT_FLAG",
            False,
        )
        if not checkpoint_flag:
            return None
        early_stopping_metric = self._trainer_config_value(
            runtime_defaults,
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
        runtime_defaults: RuntimeDefaultsSpec,
        config_overrides: dict[str, Any],
        key: str,
        default: Any = None,
    ) -> Any:
        return config_overrides.get(
            key.lower(),
            runtime_defaults.current_value_or(key, default),
        )

    def _load_runtime_config(
        self,
        config_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime_defaults = self.model_package.runtime_defaults_spec
        config_overrides = config_overrides or {}

        def runtime_value(key: str, default: Any) -> Any:
            return self._trainer_config_value(
                runtime_defaults, config_overrides, key, default
            )

        return {
            "data_num_workers": runtime_value("DATA_NUM_WORKERS", None),
            "run_test_after_fit": runtime_value("RUN_TEST_AFTER_FIT", True),
            "seed": runtime_value("SEED", None),
        }

    def _configure_dataset(
        self,
        dataset: Any,
        runtime_config: dict[str, Any],
    ) -> None:
        data_num_workers = runtime_config.get("data_num_workers")
        if data_num_workers is None or not hasattr(dataset, "num_workers"):
            pass
        else:
            dataset.num_workers = int(data_num_workers)
        seed = runtime_config.get("seed")
        if seed is not None and hasattr(dataset, "seed"):
            dataset.seed = int(seed)

    def _dataset_constructor_kwargs(
        self,
        training_run: TrainingRun,
    ) -> dict[str, Any]:
        """Return Experiment Task arguments for a data module."""

        task = getattr(training_run, "experiment_task", None) or self.experiment_task
        return experiment_task_behavior(task).dataset_constructor_kwargs(
            training_run.config
        )

    def _build_dataset(self, training_run: TrainingRun) -> Any:
        return training_run.dataset_type(
            **self._dataset_constructor_kwargs(training_run)
        )

    def _trainer_args(
        self,
        runtime_defaults: RuntimeDefaultsSpec,
        config_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        trainer_args: dict[str, Any] = {}
        for key, value in runtime_defaults.items_with_prefix("TRAINER_"):
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
        materialized_runs: Sequence[TrainingRunRequest],
    ) -> list[TrainingRun]:
        """Materialize an accepted semantic Run Plan for execution."""
        untrusted_runs = cast(Sequence[object], materialized_runs)
        if any(not isinstance(run, TrainingRunRequest) for run in untrusted_runs):
            raise TypeError(
                "Run Experiment materialization requires TrainingRunRequest values."
            )
        training_runs: list[TrainingRun] = []
        for run in materialized_runs:
            preset = cast(BaseOptions, run.preset)
            dataset_type = run.dataset_type
            run_overrides = dict(run.config_overrides)
            run_parameters = run.parameters
            run_epochs = cast(
                int,
                run_overrides.get("num_epochs", self.num_epochs),
            )
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
                    run_id=run.run_id,
                    run_index=run.run_index,
                    run_total=run.run_total,
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
    ) -> tuple[dict[str, Any], str]:
        """Execute one materialized Run through the public runtime Interface."""

        request = TrainingExecutionRequest(
            training_run=training_run,
            callbacks=callbacks,
            progress=progress,
            progress_step_interval=progress_step_interval,
            ckpt_path=ckpt_path,
            model_validator=model_validator,
            resumed_from=resumed_from,
        )
        return self.execute_training(request)

    def execute_training(
        self,
        request: TrainingExecutionRequest,
    ) -> tuple[dict[str, Any], str]:
        if not isinstance(cast(object, request), TrainingExecutionRequest):
            raise TypeError(
                "Run Experiment execution requires a TrainingExecutionRequest."
            )
        run_progress = contextual_run_progress(
            request.progress,
            self._run_progress_context(request.training_run),
        )
        state = _TrainingExecutionState(
            request.training_run,
            request,
            run_progress,
            request.outcome_observer or TrainingOutcomeObserver(),
        )
        try:
            runtime = self._prepare_training_runtime(state)
            started = self._start_training_execution(state)
            trainer = self._build_training_trainer(state, runtime, started)
            self._fit_and_test_training(state, trainer, runtime)
            return self._complete_training_execution(state, trainer, started)
        except Exception as exc:
            if not state.outcome.is_committed:
                self._emit_training_error_preserving_primary(exc, state.run_progress)
            raise

    def _prepare_training_runtime(
        self,
        state: _TrainingExecutionState,
    ) -> _TrainingRuntime:
        trainer_config = self._load_trainer_config(state.training_run.config_overrides)
        runtime_config = self._load_runtime_config(state.training_run.config_overrides)
        if runtime_config["seed"] is not None:
            seed_everything(int(runtime_config["seed"]), workers=True)
        dataset = self._build_dataset(state.training_run)
        self._configure_dataset(dataset, runtime_config)
        model = self.model_package.build_model(state.training_run.config)
        if state.options.model_validator is not None:
            state.options.model_validator(model)
        return _TrainingRuntime(
            trainer_config=trainer_config,
            runtime_config=runtime_config,
            dataset=dataset,
            model=model,
        )

    def _start_training_execution(
        self,
        state: _TrainingExecutionState,
    ) -> _StartedTrainingRun:
        if isinstance(self.run_artifacts, FilesystemRunArtifacts):
            reservation = self.run_artifacts.reserve_run(
                self.model_package.identity,
                state.training_run.preset.name,
                state.training_run.dataset_type.__name__,
                state.training_run.parameters,
            )
            logger = TensorBoardLogger(
                save_dir=str(self.run_artifacts.root),
                name=reservation.name,
                version=reservation.version,
            )
            if Path(logger.log_dir).resolve() != reservation.log_dir:
                raise RuntimeError(
                    "Logger did not consume its reserved Run Artifact directory."
                )
        else:
            logger = TensorBoardLogger(
                save_dir=str(self.run_artifacts.root),
                name=self.run_artifacts.run_name(
                    self.model_package.identity,
                    state.training_run.preset.name,
                    state.training_run.dataset_type.__name__,
                    state.training_run.parameters,
                ),
            )
        if state.run_progress is not None:
            state.run_progress = state.run_progress.with_log_dir(logger.log_dir)
        self._emit_dataset_started(
            state.training_run,
            state.run_progress,
            resumed_from=state.options.resumed_from,
        )
        progress_callbacks = (
            [
                lightning_progress_adapter(
                    state.run_progress,
                    step_interval=state.options.progress_step_interval,
                )
            ]
            if state.run_progress is not None
            else []
        )
        return _StartedTrainingRun(logger, progress_callbacks)

    def _build_training_trainer(
        self,
        state: _TrainingExecutionState,
        runtime: _TrainingRuntime,
        started: _StartedTrainingRun,
    ) -> Trainer:
        return Trainer(
            max_epochs=state.training_run.num_epochs,
            logger=started.logger,
            callbacks=[
                *runtime.trainer_config["callbacks"],
                *state.options.callbacks,
                *started.progress_callbacks,
            ],
            **runtime.trainer_config["trainer_args"],
        )

    @staticmethod
    def _fit_and_test_training(
        state: _TrainingExecutionState,
        trainer: Trainer,
        runtime: _TrainingRuntime,
    ) -> None:
        if state.options.ckpt_path is None:
            trainer.fit(runtime.model, datamodule=runtime.dataset)
        else:
            trainer.fit(
                runtime.model,
                datamodule=runtime.dataset,
                ckpt_path=state.options.ckpt_path,
                weights_only=True,
            )
        if runtime.runtime_config["run_test_after_fit"]:
            trainer.test(runtime.model, datamodule=runtime.dataset)

    def _complete_training_execution(
        self,
        state: _TrainingExecutionState,
        trainer: Trainer,
        started: _StartedTrainingRun,
    ) -> tuple[dict[str, Any], str]:
        state.outcome.begin_result_commit()
        result = self._training_result(
            state.training_run,
            trainer,
            resumed_from=state.options.resumed_from,
        )
        result.update(state.outcome.receipt_fields(state.training_run.run_id))
        log_dir = started.logger.log_dir
        committed_result = state.outcome.prepare_commit(result)
        self.run_artifacts.write_result(log_dir, result)
        state.outcome.commit(committed_result, log_dir)
        state.outcome.project(
            lambda: self.run_artifacts.update_best_results(
                self.model_package.identity,
                self.experiment_task,
                result,
            ),
            lambda: self._emit_dataset_completed(
                result,
                state.run_progress,
                resumed_from=state.options.resumed_from,
            ),
        )
        return result, log_dir

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
            preset=self.model_package.preset_name(training_run.preset),
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
            DatasetStartedEvent(
                params=training_run.parameters,
                resumed_from=resumed_from,
            ),
        )

    def _emit_training_error(
        self,
        exc: Exception,
        progress: ContextualRunProgress | None,
    ) -> None:
        self._write_progress_event(
            progress,
            TrainingErrorEvent(
                error=str(exc),
                traceback=traceback.format_exc(),
            ),
        )

    def _emit_training_error_preserving_primary(
        self,
        exc: Exception,
        progress: ContextualRunProgress | None,
    ) -> None:
        try:
            self._emit_training_error(exc, progress)
        except Exception as progress_exc:
            exc.add_note(
                "Failed to persist the training error progress event: "
                f"{type(progress_exc).__name__}: {progress_exc}"
            )

    def _emit_dataset_completed(
        self,
        result: dict[str, Any],
        progress: ContextualRunProgress | None,
        *,
        resumed_from: Mapping[str, object] | None = None,
    ) -> None:
        self._write_progress_event(
            progress,
            DatasetCompletedEvent(
                metrics=result["metrics"],
                resumed_from=resumed_from,
            ),
        )

    @staticmethod
    def _write_progress_event(
        progress: ContextualRunProgress | None,
        event: RunProgressEvent,
    ) -> None:
        if progress is not None:
            progress.write_event(event)

    def _training_result(
        self,
        training_run: TrainingRun,
        trainer: Trainer,
        *,
        resumed_from: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        experiment_task = (
            experiment_task_name(training_run.experiment_task)
            if training_run.experiment_task is not None
            else None
        )
        return {
            **self.model_package.identity.to_payload(),
            "experimentTask": experiment_task,
            "dataset": training_run.dataset_type.__name__,
            "preset": self.model_package.preset_name(training_run.preset),
            "presetKey": training_run.preset.name,
            "params": training_run.parameters,
            **self.run_artifacts.result_metrics_payload(trainer.callback_metrics),
            **({"resumedFrom": dict(resumed_from)} if resumed_from is not None else {}),
        }


__all__ = ["ExperimentBase"]
