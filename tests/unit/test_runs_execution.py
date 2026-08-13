from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from inspect import Parameter, signature
from pathlib import Path
from types import ModuleType
from unittest.mock import PropertyMock, patch

import torch
from filelock import FileLock

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from emperor.experiments import ExperimentTask
from emperor.monitoring import MonitorOption
from model_runtime.packages import ModelIdentity, ModelPackage
from model_runtime.runs import (
    DEFAULT_CHECKPOINT_ADMISSION_POLICY,
    CheckpointContinuation,
    InvalidCheckpointContinuation,
    InvalidRunPlan,
    InvalidRunRequest,
    PlanningBudget,
    RunParameter,
    RunPlanExecutionError,
    RunPlanRetry,
    RunRequest,
    SubmittedRun,
    accept_run_plan,
    execute_runs,
    execution,
    plan_runs,
)
from model_runtime.runs.artifacts import FilesystemRunArtifacts, RunArtifacts
from model_runtime.runs.experiment import ExperimentBase
from models.catalog import model_package


class _Metric:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _Logger:
    instances: list[_Logger] = []

    def __init__(
        self,
        save_dir: str,
        name: str,
        version: int | None = None,
    ) -> None:
        selected_version = 0 if version is None else version
        self.log_dir = str(Path(save_dir) / name / f"version_{selected_version}")
        type(self).instances.append(self)


class _ReservationConsumingLogger(TensorBoardLogger):
    instances: list[_ReservationConsumingLogger] = []

    def __init__(
        self,
        save_dir: str,
        name: str,
        version: int | None = None,
    ) -> None:
        self.requested_version = version
        self.reservation_existed = (
            version is not None
            and (Path(save_dir) / name / f"version_{version}").is_dir()
        )
        super().__init__(save_dir=save_dir, name=name, version=version)
        type(self).instances.append(self)


class _Trainer:
    instances: list[_Trainer] = []

    def __init__(self, max_epochs, logger, callbacks, **kwargs) -> None:
        self.max_epochs = max_epochs
        self.logger = logger
        self.callbacks = callbacks
        self.callback_metrics = {"validation_accuracy": _Metric(0.75)}
        self.current_epoch = 1
        self.global_step = 2
        self.is_global_zero = False
        self.loggers: tuple[object, ...] = ()
        self.save_checkpoint_calls: list[tuple[str, bool]] = []
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint) and callback.dirpath is None:
                callback.dirpath = str(Path(logger.log_dir) / "checkpoints")
        type(self).instances.append(self)

    def fit(self, model, datamodule, **kwargs) -> None:
        self.model = model
        self.fit_datamodule = datamodule
        self.fit_kwargs = kwargs

    def test(self, model, datamodule) -> None:
        self.test_datamodule = datamodule

    def save_checkpoint(self, path: str, weights_only: bool) -> None:
        self.save_checkpoint_calls.append((path, weights_only))


class _FailingTrainer(_Trainer):
    def fit(self, model, datamodule) -> None:
        exception = RuntimeError("training exploded")
        for callback in self.callbacks:
            on_exception = getattr(callback, "on_exception", None)
            if callable(on_exception):
                on_exception(self, model, exception)
        raise exception


class _Progress:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def write_event(self, event) -> None:
        self.events.append(dict(event))


class _PresetBoundRunExperiment:
    def __init__(
        self,
        package,
        preset,
        experiment_task,
        artifacts,
    ) -> None:
        self.package = package
        self.preset = preset
        self.delegate = ExperimentBase(
            preset,
            experiment_task,
            model_package=package,
            run_artifacts=artifacts,
        )
        self.materialized_run_ids: list[str] = []
        self.executed_run_ids: list[str] = []
        self.executed_callback_intervals: list[list[int]] = []

    def materialize_training_runs(self, requests):
        self.materialized_run_ids = [request.run_id for request in requests]
        if any(request.preset is not self.preset for request in requests):
            raise AssertionError("received a request for another preset")
        return self.delegate.materialize_training_runs(requests)

    def execute_training(self, request):
        training_run = request.training_run
        if training_run.preset is not self.preset:
            raise AssertionError("executed a Run through another preset")
        assert training_run.run_id is not None
        self.executed_run_ids.append(training_run.run_id)
        self.executed_callback_intervals.append(
            [callback.log_every_n_steps for callback in request.callbacks]
        )
        return (
            {"boundPreset": self.package.preset_name(self.preset)},
            f"logs/{training_run.run_id}",
        )


def _linears_linear():
    package = model_package("linears/linear")
    if package is None:
        raise AssertionError("Expected the linears/linear Model Package.")
    return package


class RunsExecutionTests(unittest.TestCase):
    def test_secondary_progress_failure_is_attached_to_primary_error(self) -> None:
        experiment = object.__new__(ExperimentBase)
        primary = RuntimeError("primary training failure")
        with patch.object(
            ExperimentBase,
            "_emit_training_error",
            side_effect=OSError("progress sink failure"),
        ):
            experiment._emit_training_error_preserving_primary(primary, None)

        self.assertEqual(
            primary.__notes__,
            [
                "Failed to persist the training error progress event: "
                "OSError: progress sink failure"
            ],
        )

    def setUp(self) -> None:
        _Trainer.instances.clear()
        _Logger.instances.clear()
        _ReservationConsumingLogger.instances.clear()

    def test_filesystem_execution_consumes_exact_reserved_logger_versions(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(
                root=Path(tmp) / "logs",
                namespace="runs_fixture",
                clock=lambda: datetime(2026, 6, 1, 1, 2, 3),
            )
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch(
                    "model_runtime.runs.experiment.TensorBoardLogger",
                    _ReservationConsumingLogger,
                ),
                patch("model_runtime.runs.experiment.seed_everything"),
            ):
                first_result = execute_runs(package, plan, artifacts=artifacts)[0]
                second_result = execute_runs(package, plan, artifacts=artifacts)[0]

            self.assertEqual(
                [
                    logger.requested_version
                    for logger in _ReservationConsumingLogger.instances
                ],
                [0, 1],
            )
            self.assertTrue(
                all(
                    logger.reservation_existed
                    for logger in _ReservationConsumingLogger.instances
                )
            )
            self.assertEqual(
                [Path(first_result.log_dir).name, Path(second_result.log_dir).name],
                ["version_0", "version_1"],
            )
            self.assertNotEqual(
                first_result.payload["executionId"],
                second_result.payload["executionId"],
            )
            self.assertNotEqual(
                first_result.payload["artifactId"],
                second_result.payload["artifactId"],
            )

    def test_no_search_plan_executes_exact_run_and_writes_portable_artifacts(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        progress = _Progress()
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(
                root=Path(tmp) / "logs",
                namespace="runs_fixture",
                clock=lambda: datetime(2026, 6, 1, 1, 2, 3),
            )
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
            ):
                results = execute_runs(
                    package,
                    plan,
                    artifacts=artifacts,
                    progress=progress,
                )

            result = results[0]
            self.assertEqual(result.run_id, "run-0001")
            self.assertEqual(result.experiment_task, "image-classification")
            self.assertEqual(result.preset, "baseline")
            self.assertEqual(result.dataset, "Mnist")
            self.assertEqual(
                result.payload["metrics"],
                {"validation_accuracy": 0.75},
            )
            self.assertEqual(result.payload["params"], {})
            self.assertEqual(progress.events[0]["params"], {})
            self.assertIn("/default_20260601_010203/", result.log_dir)
            self.assertNotIn("resumedFrom", result.payload)
            self.assertTrue(Path(result.log_dir, "result.json").is_file())
            self.assertTrue(artifacts.best_results_path(package.identity).is_file())
            best = json.loads(
                artifacts.best_results_path(package.identity).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(best["Mnist"][0]["rank"], 1)
            self.assertEqual(
                [event["type"] for event in progress.events],
                ["dataset_started", "dataset_completed"],
            )
            self.assertEqual(progress.events[0]["runId"], "run-0001")
            self.assertEqual(progress.events[0]["runIndex"], 1)
            self.assertEqual(progress.events[0]["runTotal"], 1)
            self.assertEqual(
                progress.events[0]["experimentTask"],
                "image-classification",
            )
            self.assertEqual(len(_Trainer.instances), 1)
            self.assertNotIn(progress, _Trainer.instances[0].callbacks)
            self.assertEqual(_Trainer.instances[0].fit_kwargs, {})
            self.assertFalse(
                any(
                    isinstance(callback, ModelCheckpoint)
                    for callback in _Trainer.instances[0].callbacks
                )
            )

    def test_explicit_run_seed_seeds_lightning_and_the_dataset(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                overrides={"SEED": 17, "RUN_TEST_AFTER_FIT": False},
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch(
                    "model_runtime.runs.experiment.seed_everything"
                ) as seed_everything_mock,
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                )

        seed_everything_mock.assert_called_once_with(17, workers=True)
        self.assertEqual(_Trainer.instances[0].fit_datamodule.seed, 17)

    def test_unset_run_seed_does_not_seed_lightning(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                overrides={"RUN_TEST_AFTER_FIT": False},
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch(
                    "model_runtime.runs.experiment.seed_everything"
                ) as seed_everything_mock,
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                )

        seed_everything_mock.assert_not_called()

    def test_direct_execution_rejects_oversized_plan_before_package_side_effects(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        oversized_plan = replace(
            plan,
            runs=tuple(
                replace(plan.runs[0], id=f"run-{index:04d}")
                for index in range(1, 2_002)
            ),
        )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(ModelPackage, "build_experiment") as build_experiment,
            self.assertRaisesRegex(
                InvalidRunPlan,
                "2001 Runs.*maximum of 2000",
            ),
        ):
            execute_runs(
                package,
                oversized_plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
            )

        build_experiment.assert_not_called()

    def test_execution_budget_can_be_tightened_explicitly(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist", "FashionMNIST"),
            ),
        )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(ModelPackage, "build_experiment") as build_experiment,
            self.assertRaisesRegex(InvalidRunPlan, "maximum of 1"),
        ):
            execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                budget=PlanningBudget(max_materialized_runs=1),
            )

        build_experiment.assert_not_called()

    def test_public_execution_signatures_remain_keyword_compatible(self) -> None:
        execute_parameters = signature(execute_runs).parameters
        self.assertEqual(
            list(execute_parameters),
            [
                "package",
                "plan",
                "artifacts",
                "progress",
                "progress_step_interval",
                "monitors",
                "continuation",
                "budget",
                "checkpoint_admission",
                "retry",
            ],
        )
        self.assertEqual(
            [parameter.kind for parameter in execute_parameters.values()],
            [
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.KEYWORD_ONLY,
                Parameter.KEYWORD_ONLY,
                Parameter.KEYWORD_ONLY,
                Parameter.KEYWORD_ONLY,
                Parameter.KEYWORD_ONLY,
                Parameter.KEYWORD_ONLY,
                Parameter.KEYWORD_ONLY,
                Parameter.KEYWORD_ONLY,
            ],
        )
        self.assertIs(execute_parameters["artifacts"].default, Parameter.empty)
        self.assertEqual(
            [
                execute_parameters[name].default
                for name in (
                    "progress",
                    "progress_step_interval",
                    "monitors",
                    "continuation",
                    "budget",
                    "checkpoint_admission",
                    "retry",
                )
            ],
            [
                None,
                1,
                (),
                None,
                None,
                DEFAULT_CHECKPOINT_ADMISSION_POLICY,
                None,
            ],
        )

        training_parameters = signature(ExperimentBase.execute_training_run).parameters
        self.assertEqual(
            list(training_parameters),
            [
                "self",
                "training_run",
                "callbacks",
                "progress",
                "progress_step_interval",
                "ckpt_path",
                "model_validator",
                "resumed_from",
            ],
        )
        self.assertEqual(
            [parameter.kind for parameter in training_parameters.values()],
            [
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.KEYWORD_ONLY,
                Parameter.KEYWORD_ONLY,
                Parameter.KEYWORD_ONLY,
                Parameter.KEYWORD_ONLY,
                Parameter.KEYWORD_ONLY,
                Parameter.KEYWORD_ONLY,
            ],
        )
        self.assertIs(training_parameters["callbacks"].default, Parameter.empty)
        self.assertEqual(
            [
                training_parameters[name].default
                for name in (
                    "progress",
                    "progress_step_interval",
                    "ckpt_path",
                    "model_validator",
                    "resumed_from",
                )
            ],
            [None, 1, None, None, None],
        )

    def test_execution_plan_validation_precedence_is_stable(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        run = plan.runs[0]
        duplicate_parameters = (
            RunParameter("HIDDEN_DIM", 64, "override"),
            RunParameter("HIDDEN_DIM", 128, "override"),
        )
        cases = (
            (
                replace(
                    plan,
                    identity=ModelIdentity("gpt", "linear"),
                    presets=(),
                    datasets=(),
                    runs=(),
                    preset_searches=(),
                ),
                "Run plan model 'gpt/linear' does not match selected model "
                "'linears/linear'.",
            ),
            (
                replace(
                    plan,
                    presets=(),
                    datasets=(),
                    runs=(),
                    preset_searches=(),
                ),
                "Run plan requires at least one selected preset.",
            ),
            (
                replace(plan, datasets=(), runs=()),
                "Run plan requires at least one selected dataset.",
            ),
            (
                replace(plan, runs=()),
                "Run plan requires at least one training run.",
            ),
            (
                replace(
                    plan,
                    experiment_task="missing-task",
                    presets=("missing-preset",),
                    datasets=("missing-dataset",),
                    preset_searches=(
                        replace(
                            plan.preset_searches[0],
                            preset="missing-preset",
                        ),
                    ),
                ),
                "Unknown experiment task 'missing-task' for model "
                "'linears/linear'. Valid tasks: image-classification.",
            ),
            (
                replace(
                    plan,
                    presets=("missing-preset",),
                    datasets=("missing-dataset",),
                    preset_searches=(
                        replace(
                            plan.preset_searches[0],
                            preset="missing-preset",
                        ),
                    ),
                ),
                "Unknown preset 'missing-preset' for model 'linears/linear'.",
            ),
            (
                replace(plan, datasets=("missing-dataset",)),
                "Unknown dataset 'missing-dataset' for model 'linears/linear'. "
                "Valid datasets: Mnist, FashionMNIST, Cifar10, Cifar100.",
            ),
            (
                replace(
                    plan,
                    runs=(
                        replace(
                            run,
                            id="",
                            experiment_task="foreign",
                            preset="missing",
                            dataset="missing",
                            parameters=duplicate_parameters,
                        ),
                    ),
                ),
                "Run plan contains empty run id.",
            ),
            (
                replace(
                    plan,
                    runs=(
                        replace(
                            run,
                            id="bad",
                            experiment_task="foreign",
                            preset="missing",
                            dataset="missing",
                            parameters=duplicate_parameters,
                        ),
                    ),
                ),
                "Run 'bad' experiment task 'foreign' does not match plan task "
                "'image-classification'.",
            ),
            (
                replace(
                    plan,
                    runs=(
                        replace(
                            run,
                            id="bad",
                            preset="missing",
                            dataset="missing",
                            parameters=duplicate_parameters,
                        ),
                    ),
                ),
                "Run plan contains unknown preset 'missing'.",
            ),
            (
                replace(
                    plan,
                    runs=(
                        replace(
                            run,
                            id="bad",
                            dataset="missing",
                            parameters=duplicate_parameters,
                        ),
                    ),
                ),
                "Run plan contains unknown dataset 'missing'.",
            ),
            (
                replace(
                    plan,
                    runs=(replace(run, id="bad", parameters=duplicate_parameters),),
                ),
                "Run 'bad' contains duplicate Runtime Defaults assignments.",
            ),
        )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(ModelPackage, "build_experiment") as build_experiment,
        ):
            artifacts = FilesystemRunArtifacts(root=Path(tmp))
            for invalid_plan, message in cases:
                with self.subTest(message=message):
                    with self.assertRaises(InvalidRunPlan) as raised:
                        execute_runs(
                            package,
                            invalid_plan,
                            artifacts=artifacts,
                        )
                    self.assertEqual(str(raised.exception), message)

        build_experiment.assert_not_called()

    def test_accepted_runs_build_fresh_monitors_from_exact_run_overrides(
        self,
    ) -> None:
        package = _linears_linear()
        plan = accept_run_plan(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
            (
                SubmittedRun(
                    "cadence-37",
                    "baseline",
                    "Mnist",
                    {"MONITOR_LOG_EVERY_N_STEPS": 37},
                ),
                SubmittedRun(
                    "cadence-73",
                    "baseline",
                    "Mnist",
                    {"MONITOR_LOG_EVERY_N_STEPS": 73},
                ),
            ),
        )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                ExperimentBase,
                "execute_training",
                autospec=True,
                side_effect=[({}, "logs/cadence-37"), ({}, "logs/cadence-73")],
            ) as execute_training,
        ):
            execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                monitors=("layer-controller",),
            )

        first_callbacks = execute_training.call_args_list[0].args[1].callbacks
        second_callbacks = execute_training.call_args_list[1].args[1].callbacks
        self.assertEqual(first_callbacks[0].log_every_n_steps, 37)
        self.assertEqual(second_callbacks[0].log_every_n_steps, 73)
        self.assertIsNot(first_callbacks[0], second_callbacks[0])

    def test_all_exact_run_monitors_build_before_any_run_executes(self) -> None:
        package = _linears_linear()
        plan = accept_run_plan(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
            (
                SubmittedRun(
                    "cadence-37",
                    "baseline",
                    "Mnist",
                    {"MONITOR_LOG_EVERY_N_STEPS": 37},
                ),
                SubmittedRun(
                    "cadence-73",
                    "baseline",
                    "Mnist",
                    {"MONITOR_LOG_EVERY_N_STEPS": 73},
                ),
            ),
        )

        def callback_factory(settings):
            if settings.log_every_n_steps == 73:
                raise ValueError("unsupported exact Run cadence")
            return Callback()

        option = MonitorOption(
            name="fixture-monitor",
            label="Fixture monitor",
            description="Exercises pre-execution callback construction.",
            kinds=("scalar",),
            callback_factory=callback_factory,
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                ModelPackage,
                "resolve_monitors",
                return_value=[option],
            ),
            patch.object(
                ExperimentBase,
                "execute_training",
                autospec=True,
            ) as execute_training,
            self.assertRaisesRegex(
                InvalidRunPlan,
                "unsupported exact Run cadence",
            ),
        ):
            execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                monitors=("fixture-monitor",),
            )

        execute_training.assert_not_called()

    def test_invalid_run_experiment_port_fails_before_materialization(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                ModelPackage,
                "build_experiment",
                return_value=object(),
            ),
            patch.object(
                ExperimentBase,
                "materialize_training_runs",
                autospec=True,
            ) as materialize,
            self.assertRaisesRegex(
                TypeError,
                "Model Package 'linears/linear' returned an invalid Run Experiment",
            ),
        ):
            execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
            )

        materialize.assert_not_called()

    def test_interleaved_presets_use_one_scoped_experiment_per_preset(self) -> None:
        package = _linears_linear()
        plan = accept_run_plan(
            package,
            RunRequest(
                presets=("baseline", "gating"),
                datasets=("Mnist",),
            ),
            (
                SubmittedRun(
                    "baseline-1",
                    "baseline",
                    "Mnist",
                    {"MONITOR_LOG_EVERY_N_STEPS": 11},
                ),
                SubmittedRun(
                    "gating-1",
                    "gating",
                    "Mnist",
                    {"MONITOR_LOG_EVERY_N_STEPS": 22},
                ),
                SubmittedRun(
                    "baseline-2",
                    "baseline",
                    "Mnist",
                    {"MONITOR_LOG_EVERY_N_STEPS": 33},
                ),
            ),
        )
        experiments: list[_PresetBoundRunExperiment] = []
        monitor = MonitorOption(
            name="fixture-monitor",
            label="Fixture monitor",
            description="Pins Run-to-Experiment callback association.",
            kinds=("scalar",),
            callback_factory=lambda settings: type(
                "FixtureCallback",
                (Callback,),
                {"log_every_n_steps": settings.log_every_n_steps},
            )(),
        )

        def build_experiment(
            selected_package,
            preset,
            *,
            experiment_task,
            run_artifacts,
        ):
            scoped = _PresetBoundRunExperiment(
                selected_package,
                preset,
                experiment_task,
                run_artifacts,
            )
            experiments.append(scoped)
            return scoped

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                ModelPackage,
                "build_experiment",
                autospec=True,
                side_effect=build_experiment,
            ),
            patch.object(ModelPackage, "resolve_monitors", return_value=[monitor]),
        ):
            results = execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                monitors=("fixture-monitor",),
            )

        self.assertEqual(
            [package.preset_name(experiment.preset) for experiment in experiments],
            ["baseline", "gating"],
        )
        self.assertEqual(
            [experiment.materialized_run_ids for experiment in experiments],
            [["baseline-1", "baseline-2"], ["gating-1"]],
        )
        self.assertEqual(
            [experiment.executed_callback_intervals for experiment in experiments],
            [[[11], [33]], [[22]]],
        )
        self.assertEqual(
            [experiment.executed_run_ids for experiment in experiments],
            [["baseline-1", "baseline-2"], ["gating-1"]],
        )
        self.assertEqual(
            [result.run_id for result in results],
            ["baseline-1", "gating-1", "baseline-2"],
        )

    def test_execution_does_not_build_an_unused_selected_preset(self) -> None:
        package = _linears_linear()
        plan = accept_run_plan(
            package,
            RunRequest(
                presets=("baseline", "gating"),
                datasets=("Mnist",),
            ),
            (SubmittedRun("gating-only", "gating", "Mnist", {}),),
        )
        built_presets: list[str] = []

        def build_experiment(
            selected_package,
            preset,
            *,
            experiment_task,
            run_artifacts,
        ):
            built_presets.append(selected_package.preset_name(preset))
            return _PresetBoundRunExperiment(
                selected_package,
                preset,
                experiment_task,
                run_artifacts,
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                ModelPackage,
                "build_experiment",
                autospec=True,
                side_effect=build_experiment,
            ),
        ):
            results = execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
            )

        self.assertEqual(built_presets, ["gating"])
        self.assertEqual([result.run_id for result in results], ["gating-only"])

    def test_single_preset_runs_still_share_one_materialization_batch(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist", "FashionMNIST"),
            ),
        )
        experiments: list[_PresetBoundRunExperiment] = []

        def build_experiment(
            selected_package,
            preset,
            *,
            experiment_task,
            run_artifacts,
        ):
            scoped = _PresetBoundRunExperiment(
                selected_package,
                preset,
                experiment_task,
                run_artifacts,
            )
            experiments.append(scoped)
            return scoped

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                ModelPackage,
                "build_experiment",
                autospec=True,
                side_effect=build_experiment,
            ),
        ):
            results = execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
            )

        self.assertEqual(len(experiments), 1)
        self.assertEqual(
            experiments[0].materialized_run_ids,
            [run.id for run in plan.runs],
        )
        self.assertEqual(
            experiments[0].executed_run_ids,
            [run.id for run in plan.runs],
        )
        self.assertEqual(
            [result.run_id for result in results], [run.id for run in plan.runs]
        )

    def test_invalid_later_preset_experiment_fails_before_bind_or_training(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline", "gating"),
                datasets=("Mnist",),
            ),
        )
        baseline = package.resolve_preset("baseline")

        def build_experiment(
            selected_package,
            preset,
            *,
            experiment_task,
            run_artifacts,
        ):
            if preset is not baseline:
                return object()
            return ExperimentBase(
                preset,
                experiment_task,
                model_package=selected_package,
                run_artifacts=run_artifacts,
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                ModelPackage,
                "build_experiment",
                autospec=True,
                side_effect=build_experiment,
            ),
            patch(
                "model_runtime.runs.execution."
                "CheckpointContinuationLifecycle.bind_training_runs",
                autospec=True,
            ) as bind_checkpoint,
            patch.object(
                ExperimentBase,
                "execute_training",
                autospec=True,
                return_value=({}, "logs/unused"),
            ) as execute_training,
            self.assertRaisesRegex(
                TypeError,
                "Model Package 'linears/linear' returned an invalid Run Experiment",
            ),
        ):
            execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
            )

        bind_checkpoint.assert_not_called()
        execute_training.assert_not_called()

    def test_later_preset_handoff_uses_original_pending_position(self) -> None:
        package = _linears_linear()
        plan = accept_run_plan(
            package,
            RunRequest(
                presets=("baseline", "gating"),
                datasets=("Mnist",),
            ),
            (
                SubmittedRun("baseline-1", "baseline", "Mnist", {}),
                SubmittedRun("gating-1", "gating", "Mnist", {}),
                SubmittedRun("baseline-2", "baseline", "Mnist", {}),
            ),
        )
        gating = package.resolve_preset("gating")

        class CorruptingExperiment(_PresetBoundRunExperiment):
            def materialize_training_runs(self, requests):
                training_runs = super().materialize_training_runs(requests)
                training_runs[0].run_id = "corrupted"
                return training_runs

        def build_experiment(
            selected_package,
            preset,
            *,
            experiment_task,
            run_artifacts,
        ):
            experiment_type = (
                CorruptingExperiment if preset is gating else _PresetBoundRunExperiment
            )
            return experiment_type(
                selected_package,
                preset,
                experiment_task,
                run_artifacts,
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                ModelPackage,
                "build_experiment",
                autospec=True,
                side_effect=build_experiment,
            ),
            patch(
                "model_runtime.runs.execution."
                "CheckpointContinuationLifecycle.bind_training_runs",
                autospec=True,
            ) as bind_checkpoint,
            self.assertRaisesRegex(
                InvalidRunPlan,
                "position 2.*expected run id 'gating-1'.*got 'corrupted'",
            ),
        ):
            execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
            )

        bind_checkpoint.assert_not_called()

    def test_later_preset_construction_failures_preserve_exception_identity(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline", "gating"),
                datasets=("Mnist",),
            ),
        )
        baseline = package.resolve_preset("baseline")

        for failure_phase in ("build", "materialize"):
            with self.subTest(failure_phase=failure_phase):
                failure = RuntimeError(f"{failure_phase} failed")

                class FailingMaterialization:
                    @staticmethod
                    def materialize_training_runs(
                        _requests,
                        _failure=failure,
                    ):
                        raise _failure

                    @staticmethod
                    def execute_training(_request):
                        raise AssertionError("training must not begin")

                def build_experiment(
                    selected_package,
                    preset,
                    *,
                    experiment_task,
                    run_artifacts,
                    selected_failure_phase=failure_phase,
                    selected_failure=failure,
                ):
                    if preset is not baseline:
                        if selected_failure_phase == "build":
                            raise selected_failure
                        return FailingMaterialization()
                    return ExperimentBase(
                        preset,
                        experiment_task,
                        model_package=selected_package,
                        run_artifacts=run_artifacts,
                    )

                with (
                    tempfile.TemporaryDirectory() as tmp,
                    patch.object(
                        ModelPackage,
                        "build_experiment",
                        autospec=True,
                        side_effect=build_experiment,
                    ),
                    patch(
                        "model_runtime.runs.execution."
                        "CheckpointContinuationLifecycle.bind_training_runs",
                        autospec=True,
                    ) as bind_checkpoint,
                    self.assertRaises(RuntimeError) as raised,
                ):
                    execute_runs(
                        package,
                        plan,
                        artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                    )

                self.assertIs(raised.exception, failure)
                bind_checkpoint.assert_not_called()

    def test_reordered_training_runs_fail_before_checkpoint_binding_or_training(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist", "FashionMNIST"),
            ),
        )
        original_materialize = ExperimentBase.materialize_training_runs

        def materialize_in_reverse(experiment, requests):
            return list(reversed(original_materialize(experiment, requests)))

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                ExperimentBase,
                "materialize_training_runs",
                autospec=True,
                side_effect=materialize_in_reverse,
            ),
            patch(
                "model_runtime.runs.execution."
                "CheckpointContinuationLifecycle.bind_training_runs",
                autospec=True,
            ) as bind_checkpoint,
            patch.object(
                ExperimentBase,
                "execute_training",
                autospec=True,
                return_value=({}, "logs/unused"),
            ) as execute_training,
            self.assertRaisesRegex(
                InvalidRunPlan,
                "position 1.*expected run id 'run-0001'.*got 'run-0002'",
            ),
        ):
            execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
            )

        bind_checkpoint.assert_not_called()
        execute_training.assert_not_called()

    def test_training_run_handoff_rejects_missing_duplicate_and_mismatched_ids(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist", "FashionMNIST"),
            ),
        )
        cases = (
            ((None, "run-0002"), "expected run id 'run-0001', got None"),
            (
                ("run-0001", "run-0001"),
                "expected run id 'run-0002', got 'run-0001'",
            ),
            (
                ("external-run", "run-0002"),
                "expected run id 'run-0001', got 'external-run'",
            ),
        )
        original_materialize = ExperimentBase.materialize_training_runs

        for run_ids, message in cases:
            with self.subTest(run_ids=run_ids):

                def materialize_with_run_ids(
                    experiment,
                    requests,
                    selected_run_ids=run_ids,
                ):
                    training_runs = original_materialize(experiment, requests)
                    for training_run, run_id in zip(
                        training_runs,
                        selected_run_ids,
                        strict=True,
                    ):
                        training_run.run_id = run_id
                    return training_runs

                with (
                    tempfile.TemporaryDirectory() as tmp,
                    patch.object(
                        ExperimentBase,
                        "materialize_training_runs",
                        autospec=True,
                        side_effect=materialize_with_run_ids,
                    ),
                    self.assertRaisesRegex(InvalidRunPlan, message),
                ):
                    execute_runs(
                        package,
                        plan,
                        artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                    )

    def test_training_run_handoff_rejects_identity_corruption(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        experiment_task = package.resolve_experiment_task(plan.experiment_task)
        cases = (
            ("run_index", 2, "expected run index 1, got 2"),
            ("run_total", 2, "expected run total 1, got 2"),
            (
                "preset",
                package.resolve_preset("gating"),
                "expected preset 'baseline', got 'gating'",
            ),
            (
                "dataset_type",
                package.resolve_dataset("FashionMNIST", experiment_task),
                "expected Dataset 'Mnist', got 'FashionMNIST'",
            ),
            (
                "experiment_task",
                ExperimentTask.TEXT_TRANSLATION,
                "expected Experiment Task 'image-classification', got "
                "'text-translation'",
            ),
        )
        original_materialize = ExperimentBase.materialize_training_runs

        for field_name, replacement, message in cases:
            with self.subTest(field=field_name):

                def materialize_with_corruption(
                    experiment,
                    requests,
                    selected_field=field_name,
                    selected_replacement=replacement,
                ):
                    training_runs = original_materialize(experiment, requests)
                    setattr(
                        training_runs[0],
                        selected_field,
                        selected_replacement,
                    )
                    return training_runs

                with (
                    tempfile.TemporaryDirectory() as tmp,
                    patch.object(
                        ExperimentBase,
                        "materialize_training_runs",
                        autospec=True,
                        side_effect=materialize_with_corruption,
                    ),
                    patch(
                        "model_runtime.runs.execution."
                        "CheckpointContinuationLifecycle.bind_training_runs",
                        autospec=True,
                    ) as bind_checkpoint,
                    patch.object(
                        ExperimentBase,
                        "execute_training",
                        autospec=True,
                        return_value=({}, "logs/unused"),
                    ) as execute_training,
                    self.assertRaisesRegex(InvalidRunPlan, message),
                ):
                    execute_runs(
                        package,
                        plan,
                        artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                    )

                bind_checkpoint.assert_not_called()
                execute_training.assert_not_called()

    def test_training_run_handoff_rejects_nested_semantic_corruption(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        experiment_task, selected_presets, requests = (
            execution._validated_materialized_runs(package, plan)
        )
        nested = {"nested": {"values": [1, 2]}}
        nested_request = replace(
            requests[0],
            parameters=nested,
            config_overrides=nested,
        )
        cases = (
            ("parameters", "did not preserve requested parameters"),
            (
                "config_overrides",
                "did not preserve Runtime Defaults overrides",
            ),
        )
        original_materialize = ExperimentBase.materialize_training_runs

        for field_name, message in cases:
            with self.subTest(field=field_name):

                def materialize_with_nested_corruption(
                    experiment,
                    materialized_requests,
                    selected_field=field_name,
                ):
                    sanitized_request = replace(
                        materialized_requests[0],
                        parameters={},
                        config_overrides={},
                    )
                    training_run = original_materialize(
                        experiment,
                        [sanitized_request],
                    )[0]
                    training_run.parameters = dict(nested_request.parameters)
                    training_run.config_overrides = dict(
                        nested_request.config_overrides
                    )
                    setattr(
                        training_run,
                        selected_field,
                        {"nested": {"values": (1, 3)}},
                    )
                    return [training_run]

                with (
                    tempfile.TemporaryDirectory() as tmp,
                    patch.object(
                        execution,
                        "_validated_materialized_runs",
                        return_value=(
                            experiment_task,
                            selected_presets,
                            [nested_request],
                        ),
                    ),
                    patch.object(
                        ExperimentBase,
                        "materialize_training_runs",
                        autospec=True,
                        side_effect=materialize_with_nested_corruption,
                    ),
                    patch(
                        "model_runtime.runs.execution."
                        "CheckpointContinuationLifecycle.bind_training_runs",
                        autospec=True,
                    ) as bind_checkpoint,
                    patch.object(
                        ExperimentBase,
                        "execute_training",
                        autospec=True,
                        return_value=({}, "logs/unused"),
                    ) as execute_training,
                    self.assertRaisesRegex(InvalidRunPlan, message),
                ):
                    execute_runs(
                        package,
                        plan,
                        artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                    )

                bind_checkpoint.assert_not_called()
                execute_training.assert_not_called()

    def test_training_run_handoff_rejects_corrupted_epoch_count(self) -> None:
        package = _linears_linear()
        default_epochs = package.runtime_defaults_spec.current_value_or(
            "NUM_EPOCHS",
            10,
        )
        cases = (
            ({}, default_epochs, default_epochs + 1),
            ({"NUM_EPOCHS": 2}, 2, 3),
        )
        original_materialize = ExperimentBase.materialize_training_runs

        for overrides, expected_epochs, corrupted_epochs in cases:
            with self.subTest(overrides=overrides):
                plan = plan_runs(
                    package,
                    RunRequest(
                        presets=("baseline",),
                        datasets=("Mnist",),
                        overrides=overrides,
                    ),
                )

                def materialize_with_epoch_corruption(
                    experiment,
                    requests,
                    selected_epochs=corrupted_epochs,
                ):
                    training_runs = original_materialize(experiment, requests)
                    training_runs[0].num_epochs = selected_epochs
                    return training_runs

                with (
                    tempfile.TemporaryDirectory() as tmp,
                    patch.object(
                        ExperimentBase,
                        "materialize_training_runs",
                        autospec=True,
                        side_effect=materialize_with_epoch_corruption,
                    ),
                    patch(
                        "model_runtime.runs.execution."
                        "CheckpointContinuationLifecycle.bind_training_runs",
                        autospec=True,
                    ) as bind_checkpoint,
                    patch.object(
                        ExperimentBase,
                        "execute_training",
                        autospec=True,
                        return_value=({}, "logs/unused"),
                    ) as execute_training,
                    self.assertRaisesRegex(
                        InvalidRunPlan,
                        f"expected epoch count {expected_epochs}, got "
                        f"{corrupted_epochs}",
                    ),
                ):
                    execute_runs(
                        package,
                        plan,
                        artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                    )

                bind_checkpoint.assert_not_called()
                execute_training.assert_not_called()

    def test_training_run_handoff_preserves_historical_epoch_fallback(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        runtime_defaults_without_epochs = replace(
            package.runtime_defaults_spec,
            _config_module=ModuleType("runtime_defaults_without_epochs"),
        )
        original_materialize = ExperimentBase.materialize_training_runs

        def materialize_with_epoch_corruption(experiment, requests):
            training_runs = original_materialize(experiment, requests)
            self.assertEqual(training_runs[0].num_epochs, 10)
            training_runs[0].num_epochs = 11
            return training_runs

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                ModelPackage,
                "runtime_defaults_spec",
                new_callable=PropertyMock,
                return_value=runtime_defaults_without_epochs,
            ),
            patch.object(
                ExperimentBase,
                "materialize_training_runs",
                autospec=True,
                side_effect=materialize_with_epoch_corruption,
            ),
            patch(
                "model_runtime.runs.execution."
                "CheckpointContinuationLifecycle.bind_training_runs",
                autospec=True,
            ) as bind_checkpoint,
            patch.object(
                ExperimentBase,
                "execute_training",
                autospec=True,
                return_value=({}, "logs/unused"),
            ) as execute_training,
            self.assertRaisesRegex(
                InvalidRunPlan,
                "expected epoch count 10, got 11",
            ),
        ):
            execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
            )

        bind_checkpoint.assert_not_called()
        execute_training.assert_not_called()

    def test_requested_checkpointing_keeps_best_and_last_checkpoints(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                overrides={
                    "CALLBACK_CHECKPOINT_FLAG": True,
                    "RUN_TEST_AFTER_FIT": False,
                },
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
            ):
                results = execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                )

        self.assertEqual(
            results[0].payload["params"],
            {
                "RUN_TEST_AFTER_FIT": False,
                "CALLBACK_CHECKPOINT_FLAG": True,
            },
        )
        self.assertNotIn("/default_", results[0].log_dir)

        checkpoints = [
            callback
            for callback in _Trainer.instances[0].callbacks
            if isinstance(callback, ModelCheckpoint)
        ]
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].save_top_k, 1)
        self.assertIs(checkpoints[0].save_last, True)
        self.assertFalse(checkpoints[0].save_weights_only)
        self.assertEqual(
            _Trainer.instances[0].save_checkpoint_calls,
            [(checkpoints[0].last_model_path, False)],
        )
        self.assertTrue(checkpoints[0].last_model_path.endswith("/last.ckpt"))

    def test_terminal_save_uses_only_the_runtime_owned_checkpoint_callback(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                overrides={
                    "CALLBACK_CHECKPOINT_FLAG": True,
                    "RUN_TEST_AFTER_FIT": False,
                },
            ),
        )
        caller_checkpoint = ModelCheckpoint(save_last=True)

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
                patch.object(
                    execution._RunExecutor,
                    "_callback_groups",
                    return_value=([caller_checkpoint],),
                ),
                patch(
                    "model_runtime.runs.experiment.save_terminal_last_checkpoint"
                ) as terminal_save,
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                )

        trainer, runtime_checkpoint = terminal_save.call_args.args
        self.assertIs(trainer, _Trainer.instances[0])
        self.assertIsNot(runtime_checkpoint, caller_checkpoint)
        self.assertIn(runtime_checkpoint, trainer.callbacks)
        self.assertIn(caller_checkpoint, trainer.callbacks)

    def test_fit_terminal_save_test_and_completion_are_strictly_ordered(
        self,
    ) -> None:
        events: list[str] = []

        class TracingTrainer(_Trainer):
            def fit(self, model, datamodule, **kwargs) -> None:
                events.append("fit")
                super().fit(model, datamodule, **kwargs)

            def test(self, model, datamodule) -> None:
                events.append("test")
                super().test(model, datamodule)

        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                overrides={"CALLBACK_CHECKPOINT_FLAG": True},
            ),
        )
        complete_training = ExperimentBase._complete_training_execution

        def complete(experiment, state, trainer, started):
            events.append("complete")
            return complete_training(experiment, state, trainer, started)

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("model_runtime.runs.experiment.Trainer", TracingTrainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
                patch(
                    "model_runtime.runs.experiment.save_terminal_last_checkpoint",
                    side_effect=lambda *_args: events.append("terminal_save"),
                ),
                patch.object(
                    ExperimentBase,
                    "_complete_training_execution",
                    autospec=True,
                    side_effect=complete,
                ),
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                )

        self.assertEqual(events, ["fit", "terminal_save", "test", "complete"])

    def test_checkpoint_disabled_run_never_invokes_terminal_adapter(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                overrides={"RUN_TEST_AFTER_FIT": False},
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
                patch(
                    "model_runtime.runs.experiment.save_terminal_last_checkpoint"
                ) as terminal_save,
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                )

        terminal_save.assert_not_called()

    def test_terminal_save_failure_uses_run_error_path_and_prevents_completion(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                overrides={
                    "CALLBACK_CHECKPOINT_FLAG": True,
                    "RUN_TEST_AFTER_FIT": False,
                },
            ),
        )
        progress = _Progress()

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(root=Path(tmp))
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
                patch(
                    "model_runtime.runs.experiment.save_terminal_last_checkpoint",
                    side_effect=OSError("terminal disk full"),
                ),
                patch.object(
                    FilesystemRunArtifacts,
                    "write_result",
                    autospec=True,
                ) as write_result,
                self.assertRaisesRegex(OSError, "terminal disk full"),
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=artifacts,
                    progress=progress,
                )

        write_result.assert_not_called()
        self.assertEqual(
            [event["type"] for event in progress.events],
            ["dataset_started", "error"],
        )

    def test_training_failure_does_not_invoke_terminal_adapter(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                overrides={
                    "CALLBACK_CHECKPOINT_FLAG": True,
                    "RUN_TEST_AFTER_FIT": False,
                },
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("model_runtime.runs.experiment.Trainer", _FailingTrainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
                patch(
                    "model_runtime.runs.experiment.save_terminal_last_checkpoint"
                ) as terminal_save,
                self.assertRaisesRegex(RuntimeError, "training exploded"),
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                )

        terminal_save.assert_not_called()

    def test_valid_continuation_passes_admitted_snapshot_to_lightning_safely(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                overrides={"NUM_EPOCHS": 2, "RUN_TEST_AFTER_FIT": False},
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=root / "source"),
                )
                state_dict = _Trainer.instances[-1].model.state_dict()
                checkpoint = Path("relative") / "last.ckpt"
                checkpoint_path = root / checkpoint
                checkpoint_path.parent.mkdir()
                torch.save(
                    {
                        "pytorch-lightning_version": "2.5.0",
                        "state_dict": state_dict,
                        "epoch": 0,
                        "global_step": 2,
                        "optimizer_states": [{}],
                    },
                    checkpoint_path,
                )

                _Trainer.instances.clear()
                execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=root / "continued"),
                    continuation=CheckpointContinuation(checkpoint_path),
                )

            fit_kwargs = _Trainer.instances[0].fit_kwargs
            admitted_path = fit_kwargs["ckpt_path"]
            self.assertIsInstance(admitted_path, Path)
            self.assertNotEqual(admitted_path, checkpoint_path)
            self.assertEqual(Path(admitted_path).name, "admitted.ckpt")
            self.assertFalse(Path(admitted_path).exists())
            self.assertEqual(
                fit_kwargs,
                {"ckpt_path": admitted_path, "weights_only": True},
            )

    def test_continuation_rejects_multi_run_plan_before_materialization(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist", "FashionMNIST"),
            ),
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                ModelPackage,
                "build_experiment",
            ) as build_experiment,
            self.assertRaisesRegex(
                InvalidCheckpointContinuation,
                "exactly one Run",
            ),
        ):
            execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                continuation=CheckpointContinuation(Path("missing.ckpt")),
            )

        build_experiment.assert_not_called()
        self.assertEqual(_Trainer.instances, [])

    def test_continuation_rejects_missing_checkpoint_before_materialization(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                ModelPackage,
                "build_experiment",
            ) as build_experiment,
            self.assertRaisesRegex(
                InvalidCheckpointContinuation,
                "readable regular file",
            ),
        ):
            execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                continuation=CheckpointContinuation(Path(tmp) / "missing.ckpt"),
            )

        build_experiment.assert_not_called()
        self.assertEqual(_Trainer.instances, [])

    def test_continuation_rejects_directory_and_unreadable_checkpoint(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "directory.ckpt"
            directory.mkdir()
            unreadable = Path(tmp) / "unreadable.ckpt"
            unreadable.write_bytes(b"checkpoint")
            unreadable.chmod(0)
            try:
                for checkpoint in (directory, unreadable):
                    with (
                        self.subTest(checkpoint=checkpoint.name),
                        self.assertRaisesRegex(
                            InvalidCheckpointContinuation,
                            "readable regular file",
                        ),
                    ):
                        execute_runs(
                            package,
                            plan,
                            artifacts=FilesystemRunArtifacts(root=Path(tmp) / "logs"),
                            continuation=CheckpointContinuation(checkpoint),
                        )
            finally:
                unreadable.chmod(0o600)

        self.assertEqual(_Trainer.instances, [])

    def test_continuation_rejects_malformed_checkpoint_before_materialization(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "malformed.ckpt"
            checkpoint.write_bytes(b"not a torch checkpoint")
            with (
                patch.object(
                    ModelPackage,
                    "build_experiment",
                ) as build_experiment,
                self.assertRaisesRegex(
                    InvalidCheckpointContinuation,
                    "isolated decoder failed|could not be loaded",
                ),
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=Path(tmp) / "logs"),
                    continuation=CheckpointContinuation(checkpoint),
                )

        build_experiment.assert_not_called()
        self.assertEqual(_Trainer.instances, [])

    def test_continuation_rejects_incomplete_lightning_checkpoint(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        base = {
            "pytorch-lightning_version": "2.5.0",
            "state_dict": {"weight": torch.zeros(1)},
            "epoch": 0,
            "global_step": 1,
            "optimizer_states": [{}],
        }
        cases = {
            "mapping payload": [],
            "Lightning version": {**base, "pytorch-lightning_version": ""},
            "nonempty state_dict": {**base, "state_dict": {}},
            "nonnegative epoch": {**base, "epoch": -1},
            "nonnegative global_step": {**base, "global_step": -1},
            "nonempty optimizer_states": {**base, "optimizer_states": []},
        }
        with tempfile.TemporaryDirectory() as tmp:
            for expected, payload in cases.items():
                with self.subTest(expected=expected):
                    checkpoint = Path(tmp) / f"{expected.replace(' ', '-')}.ckpt"
                    torch.save(payload, checkpoint)
                    with (
                        patch.object(
                            ModelPackage,
                            "build_experiment",
                        ) as build_experiment,
                        self.assertRaisesRegex(
                            InvalidCheckpointContinuation,
                            expected,
                        ),
                    ):
                        execute_runs(
                            package,
                            plan,
                            artifacts=FilesystemRunArtifacts(root=Path(tmp) / "logs"),
                            continuation=CheckpointContinuation(checkpoint),
                        )
                    build_experiment.assert_not_called()

        self.assertEqual(_Trainer.instances, [])

    def test_continuation_requires_target_epochs_beyond_completed_checkpoint(
        self,
    ) -> None:
        package = _linears_linear()
        cases = ((1, 0), (1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            for target_epochs, checkpoint_epoch in cases:
                with self.subTest(
                    target_epochs=target_epochs,
                    checkpoint_epoch=checkpoint_epoch,
                ):
                    plan = plan_runs(
                        package,
                        RunRequest(
                            presets=("baseline",),
                            datasets=("Mnist",),
                            overrides={"NUM_EPOCHS": target_epochs},
                        ),
                    )
                    checkpoint = (
                        Path(tmp)
                        / f"target-{target_epochs}-epoch-{checkpoint_epoch}.ckpt"
                    )
                    torch.save(
                        {
                            "pytorch-lightning_version": "2.5.0",
                            "state_dict": {"weight": torch.zeros(1)},
                            "epoch": checkpoint_epoch,
                            "global_step": 1,
                            "optimizer_states": [{}],
                        },
                        checkpoint,
                    )

                    with self.assertRaisesRegex(
                        InvalidCheckpointContinuation,
                        "NUM_EPOCHS.*greater than.*completed epochs",
                    ):
                        execute_runs(
                            package,
                            plan,
                            artifacts=FilesystemRunArtifacts(root=Path(tmp) / "logs"),
                            continuation=CheckpointContinuation(checkpoint),
                        )

        self.assertEqual(_Trainer.instances, [])

    def test_continuation_rejects_mismatched_model_state_keys_before_logging(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                overrides={"NUM_EPOCHS": 2, "RUN_TEST_AFTER_FIT": False},
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=root / "source"),
                )
                state_dict = dict(_Trainer.instances[-1].model.state_dict())
                removed_key = next(iter(state_dict))
                del state_dict[removed_key]
                checkpoint = root / "last.ckpt"
                torch.save(
                    {
                        "pytorch-lightning_version": "2.5.0",
                        "state_dict": state_dict,
                        "epoch": 0,
                        "global_step": 2,
                        "optimizer_states": [{}],
                    },
                    checkpoint,
                )

                _Trainer.instances.clear()
                _Logger.instances.clear()
                with self.assertRaisesRegex(
                    InvalidCheckpointContinuation,
                    f"state keys.*{removed_key}",
                ):
                    execute_runs(
                        package,
                        plan,
                        artifacts=FilesystemRunArtifacts(root=root / "continued"),
                        continuation=CheckpointContinuation(checkpoint),
                    )

            self.assertEqual(_Logger.instances, [])
            self.assertEqual(_Trainer.instances, [])

    def test_continuation_rejects_mismatched_tensor_shapes_before_logging(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                overrides={"NUM_EPOCHS": 2, "RUN_TEST_AFTER_FIT": False},
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=root / "source"),
                )
                state_dict = dict(_Trainer.instances[-1].model.state_dict())
                mismatched_key = next(iter(state_dict))
                value = state_dict[mismatched_key]
                state_dict[mismatched_key] = torch.zeros(
                    (value.numel() + 1,),
                    dtype=value.dtype,
                )
                checkpoint = root / "last.ckpt"
                torch.save(
                    {
                        "pytorch-lightning_version": "2.5.0",
                        "state_dict": state_dict,
                        "epoch": 0,
                        "global_step": 2,
                        "optimizer_states": [{}],
                    },
                    checkpoint,
                )

                _Trainer.instances.clear()
                _Logger.instances.clear()
                with self.assertRaisesRegex(
                    InvalidCheckpointContinuation,
                    f"tensor shape.*{mismatched_key}",
                ):
                    execute_runs(
                        package,
                        plan,
                        artifacts=FilesystemRunArtifacts(root=root / "continued"),
                        continuation=CheckpointContinuation(checkpoint),
                    )

            self.assertEqual(_Logger.instances, [])
            self.assertEqual(_Trainer.instances, [])

    def test_continuation_records_safe_lineage_without_modifying_source(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                overrides={"NUM_EPOCHS": 2, "RUN_TEST_AFTER_FIT": False},
            ),
        )
        progress = _Progress()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=root / "source-run"),
                )
                checkpoint = root / "private" / "last.ckpt"
                checkpoint.parent.mkdir()
                torch.save(
                    {
                        "pytorch-lightning_version": "2.5.0",
                        "state_dict": _Trainer.instances[-1].model.state_dict(),
                        "epoch": 0,
                        "global_step": 17,
                        "optimizer_states": [{}],
                    },
                    checkpoint,
                )
                source_bytes = checkpoint.read_bytes()

                _Trainer.instances.clear()
                results = execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=root / "continued-run"),
                    progress=progress,
                    continuation=CheckpointContinuation(checkpoint),
                )

            expected = {
                "checkpoint": "last.ckpt",
                "epoch": 0,
                "globalStep": 17,
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
            self.assertEqual(results[0].payload["resumedFrom"], expected)
            result_json = json.loads(
                Path(results[0].log_dir, "result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result_json["resumedFrom"], expected)
            self.assertEqual(
                [event["resumedFrom"] for event in progress.events],
                [expected, expected],
            )
            self.assertNotIn(str(checkpoint.parent), json.dumps(result_json))
            self.assertEqual(checkpoint.read_bytes(), source_bytes)

    def test_foreign_plan_rejects_before_framework_side_effects(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        foreign = replace(plan, identity=ModelIdentity("gpt", "linear"))

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(InvalidRunPlan, "does not match"):
                execute_runs(
                    package,
                    foreign,
                    artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                )

        self.assertEqual(_Trainer.instances, [])

    def test_failure_event_follows_started_event_and_preserves_run_identity(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        progress = _Progress()
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("model_runtime.runs.experiment.Trainer", _FailingTrainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
            ):
                with self.assertRaisesRegex(RuntimeError, "training exploded"):
                    execute_runs(
                        package,
                        plan,
                        artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                        progress=progress,
                    )

        self.assertEqual(
            [event["type"] for event in progress.events],
            ["dataset_started", "error"],
        )
        self.assertEqual(progress.events[-1]["status"], "failed")
        self.assertEqual(progress.events[-1]["runId"], "run-0001")
        self.assertEqual(progress.events[-1]["dataset"], "Mnist")
        self.assertEqual(progress.events[-1]["error"], "training exploded")
        self.assertEqual(
            progress.events[-1]["logDir"],
            progress.events[0]["logDir"],
        )

    def test_later_training_failure_exposes_earlier_committed_results(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist", "FashionMNIST", "Cifar10"),
            ),
        )
        long_run_id = "run-" + "x" * 1_024
        plan = replace(
            plan,
            runs=(plan.runs[0], replace(plan.runs[1], id=long_run_id), plan.runs[2]),
        )
        failure = RuntimeError("second Run training failed")

        class FailsSecondTrainer(_Trainer):
            fit_count = 0

            def fit(self, model, datamodule, **kwargs) -> None:
                type(self).fit_count += 1
                if type(self).fit_count == 2:
                    raise failure
                super().fit(model, datamodule, **kwargs)

        progress = _Progress()
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("model_runtime.runs.experiment.Trainer", FailsSecondTrainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
                self.assertRaises(RunPlanExecutionError) as raised,
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                    progress=progress,
                )

        error = raised.exception
        self.assertIs(error.__cause__, failure)
        self.assertEqual(error.phase, "training")
        self.assertEqual(error.affected_run_id, long_run_id)
        self.assertEqual(
            [result.run_id for result in error.completed_results],
            [plan.runs[0].id],
        )
        self.assertEqual(
            error.completed_results[0].payload["executionId"],
            error.execution_id,
        )
        self.assertEqual(FailsSecondTrainer.fit_count, 2)
        self.assertEqual(
            [event["type"] for event in progress.events],
            ["dataset_started", "dataset_completed", "dataset_started", "error"],
        )

    def test_later_result_commit_failure_preserves_only_prior_receipts(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist", "FashionMNIST", "Cifar10"),
            ),
        )
        original_write = FilesystemRunArtifacts.write_result
        failure = OSError("second result commit failed")
        write_count = 0

        def fail_second_write(artifacts, log_dir, result):
            nonlocal write_count
            write_count += 1
            if write_count == 2:
                raise failure
            return original_write(artifacts, log_dir, result)

        progress = _Progress()
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
                patch.object(
                    FilesystemRunArtifacts,
                    "write_result",
                    autospec=True,
                    side_effect=fail_second_write,
                ),
                self.assertRaises(RunPlanExecutionError) as raised,
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                    progress=progress,
                )

            first_receipt = Path(
                raised.exception.completed_results[0].log_dir,
                "result.json",
            ).is_file()

        error = raised.exception
        self.assertIs(error.__cause__, failure)
        self.assertEqual(error.phase, "result_commit")
        self.assertEqual(error.affected_run_id, plan.runs[1].id)
        self.assertEqual(
            [result.run_id for result in error.completed_results],
            [plan.runs[0].id],
        )
        self.assertTrue(first_receipt)
        self.assertEqual(len(_Trainer.instances), 2)
        self.assertEqual(
            [event["type"] for event in progress.events],
            ["dataset_started", "dataset_completed", "dataset_started", "error"],
        )

    def test_post_commit_best_failure_includes_the_completed_affected_run(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        failure = OSError("best results unavailable")
        progress = _Progress()

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(root=Path(tmp))
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
                patch.object(
                    FilesystemRunArtifacts,
                    "update_best_results",
                    side_effect=failure,
                ),
                self.assertRaises(RunPlanExecutionError) as raised,
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=artifacts,
                    progress=progress,
                )

            receipt_existed = Path(
                raised.exception.completed_results[0].log_dir,
                "result.json",
            ).is_file()

        error = raised.exception
        self.assertIs(error.__cause__, failure)
        self.assertEqual(error.phase, "best_results_projection")
        self.assertEqual(error.affected_run_id, plan.runs[0].id)
        self.assertEqual(len(error.completed_results), 1)
        completed = error.completed_results[0]
        self.assertEqual(completed.run_id, plan.runs[0].id)
        self.assertEqual(completed.payload["status"], "completed")
        self.assertEqual(completed.payload["executionId"], error.execution_id)
        self.assertTrue(receipt_existed)
        self.assertEqual(
            [event["type"] for event in progress.events],
            ["dataset_started", "dataset_completed"],
        )

    def test_post_commit_progress_failure_is_a_projection_failure(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        failure = OSError("completion progress unavailable")

        class FailingCompletionProgress(_Progress):
            def write_event(self, event) -> None:
                payload = dict(event)
                if payload["type"] == "dataset_completed":
                    raise failure
                self.events.append(payload)

        progress = FailingCompletionProgress()
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(root=Path(tmp))
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
                self.assertRaises(RunPlanExecutionError) as raised,
            ):
                execute_runs(package, plan, artifacts=artifacts, progress=progress)

            best = artifacts.read_best_results(package.identity)

        error = raised.exception
        self.assertIs(error.__cause__, failure)
        self.assertEqual(error.phase, "progress_projection")
        self.assertEqual(error.affected_run_id, plan.runs[0].id)
        self.assertEqual(
            [result.run_id for result in error.completed_results],
            [plan.runs[0].id],
        )
        self.assertEqual(
            [event["type"] for event in progress.events],
            ["dataset_started"],
        )
        self.assertEqual(len(best["Mnist"]), 1)

    def test_best_results_lock_timeout_keeps_receipt_and_completion(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        progress = _Progress()

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(
                root=Path(tmp),
                best_results_lock_timeout_seconds=0.02,
            )
            summary = artifacts.best_results_path(package.identity)
            lock_path = summary.with_suffix(summary.suffix + ".lock")
            lock_path.parent.mkdir(parents=True)
            held_lock = FileLock(str(lock_path))
            held_lock.acquire()
            try:
                with (
                    patch("model_runtime.runs.experiment.Trainer", _Trainer),
                    patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                    patch("model_runtime.runs.experiment.seed_everything"),
                    self.assertRaises(RunPlanExecutionError) as raised,
                ):
                    execute_runs(
                        package,
                        plan,
                        artifacts=artifacts,
                        progress=progress,
                    )
            finally:
                held_lock.release()

            receipt_exists = Path(
                raised.exception.completed_results[0].log_dir,
                "result.json",
            ).is_file()

        error = raised.exception
        self.assertIsInstance(error.__cause__, TimeoutError)
        self.assertEqual(error.phase, "best_results_projection")
        self.assertTrue(receipt_exists)
        self.assertEqual(
            [event["type"] for event in progress.events],
            ["dataset_started", "dataset_completed"],
        )

    def test_explicit_retry_skips_exact_committed_prefix_and_runs_pending(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist", "FashionMNIST"),
            ),
        )
        failure = RuntimeError("second Run training failed")

        class FailsSecondTrainer(_Trainer):
            fit_count = 0

            def fit(self, model, datamodule, **kwargs) -> None:
                type(self).fit_count += 1
                if type(self).fit_count == 2:
                    raise failure
                super().fit(model, datamodule, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(root=Path(tmp))
            with (
                patch("model_runtime.runs.experiment.Trainer", FailsSecondTrainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
                self.assertRaises(RunPlanExecutionError) as raised,
            ):
                execute_runs(package, plan, artifacts=artifacts)

            partial = raised.exception
            retry_progress = _Progress()
            _Trainer.instances.clear()
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
            ):
                results = execute_runs(
                    package,
                    plan,
                    artifacts=artifacts,
                    progress=retry_progress,
                    retry=RunPlanRetry(
                        execution_id=partial.execution_id,
                        completed_results=partial.completed_results,
                    ),
                )

            best = artifacts.read_best_results(package.identity)

        self.assertEqual(results[0], partial.completed_results[0])
        self.assertEqual(
            [result.run_id for result in results],
            [run.id for run in plan.runs],
        )
        self.assertEqual(results[1].payload["executionId"], partial.execution_id)
        self.assertEqual(len(_Trainer.instances), 1)
        self.assertEqual(
            [event["type"] for event in retry_progress.events],
            ["dataset_started", "dataset_completed"],
        )
        self.assertEqual(retry_progress.events[0]["runId"], plan.runs[1].id)
        self.assertEqual(len(best["Mnist"]), 1)
        self.assertEqual(len(best["FashionMNIST"]), 1)

    def test_retry_builds_only_the_later_pending_preset_experiment(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline", "gating"),
                datasets=("Mnist",),
            ),
        )
        failure = RuntimeError("later preset training failed")

        class FailsSecondTrainer(_Trainer):
            fit_count = 0

            def fit(self, model, datamodule, **kwargs) -> None:
                type(self).fit_count += 1
                if type(self).fit_count == 2:
                    raise failure
                super().fit(model, datamodule, **kwargs)

        original_build_experiment = ModelPackage.build_experiment
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(root=Path(tmp))
            with (
                patch("model_runtime.runs.experiment.Trainer", FailsSecondTrainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
                self.assertRaises(RunPlanExecutionError) as raised,
            ):
                execute_runs(package, plan, artifacts=artifacts)

            partial = raised.exception
            built_presets: list[str] = []

            def tracked_build_experiment(
                selected_package,
                preset,
                *,
                experiment_task,
                run_artifacts,
            ):
                built_presets.append(selected_package.preset_name(preset))
                return original_build_experiment(
                    selected_package,
                    preset,
                    experiment_task=experiment_task,
                    run_artifacts=run_artifacts,
                )

            _Trainer.instances.clear()
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
                patch.object(
                    ModelPackage,
                    "build_experiment",
                    autospec=True,
                    side_effect=tracked_build_experiment,
                ),
            ):
                results = execute_runs(
                    package,
                    plan,
                    artifacts=artifacts,
                    retry=RunPlanRetry(
                        execution_id=partial.execution_id,
                        completed_results=partial.completed_results,
                    ),
                )

        self.assertEqual(built_presets, ["gating"])
        self.assertEqual(
            [result.run_id for result in results],
            [run.id for run in plan.runs],
        )
        self.assertEqual(results[1].payload["executionId"], partial.execution_id)

    def test_retry_rejects_tampered_receipt_before_framework_work(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(root=Path(tmp) / "logs")
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
            ):
                original = execute_runs(package, plan, artifacts=artifacts)[0]

            receipt_path = Path(original.log_dir, "result.json")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["params"] = {"batch_size": 999}
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            _Trainer.instances.clear()

            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                self.assertRaisesRegex(InvalidRunPlan, "retry receipt"),
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=artifacts,
                    retry=RunPlanRetry(
                        execution_id=original.payload["executionId"],
                        completed_results=(original,),
                    ),
                )

        self.assertEqual(_Trainer.instances, [])

    def test_retry_rejects_structural_artifacts_before_framework_work(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        with tempfile.TemporaryDirectory() as tmp:

            class StructuralArtifacts:
                root = Path(tmp)

                def run_name(self, identity, preset_key, dataset, parameters):
                    raise AssertionError("retry must not allocate artifacts")

                def result_metrics_payload(self, metrics):
                    raise AssertionError("retry must not project metrics")

                def write_result(self, log_dir, result):
                    raise AssertionError("retry must not write a receipt")

                def read_best_results(self, identity):
                    raise AssertionError("retry must not read projections")

                def update_best_results(self, identity, experiment_task, result):
                    raise AssertionError("retry must not write projections")

            filesystem_artifacts = FilesystemRunArtifacts(root=Path(tmp))
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
            ):
                completed = execute_runs(
                    package,
                    plan,
                    artifacts=filesystem_artifacts,
                )[0]

            _Trainer.instances.clear()
            structural_artifacts = StructuralArtifacts()
            self.assertIsInstance(structural_artifacts, RunArtifacts)
            with (
                patch.object(ModelPackage, "build_experiment") as build_experiment,
                self.assertRaisesRegex(
                    InvalidRunRequest,
                    "requires FilesystemRunArtifacts",
                ),
            ):
                execute_runs(
                    package,
                    plan,
                    artifacts=structural_artifacts,
                    retry=RunPlanRetry(
                        execution_id=completed.payload["executionId"],
                        completed_results=(completed,),
                    ),
                )

        build_experiment.assert_not_called()
        self.assertEqual(_Trainer.instances, [])

    def test_retry_binds_every_receipt_semantic_to_the_selected_run(self) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = FilesystemRunArtifacts(root=Path(tmp) / "logs")
            with (
                patch("model_runtime.runs.experiment.Trainer", _Trainer),
                patch("model_runtime.runs.experiment.TensorBoardLogger", _Logger),
                patch("model_runtime.runs.experiment.seed_everything"),
            ):
                original = execute_runs(package, plan, artifacts=artifacts)[0]

            receipt_path = Path(original.log_dir, "result.json")
            original_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            cases = (
                ("model", "foreign"),
                ("presetKey", "FOREIGN"),
                ("params", {"batch_size": 999}),
                ("executionId", "foreign-execution"),
            )
            for key, value in cases:
                tampered_payload = {**original_payload, key: value}
                receipt_path.write_text(json.dumps(tampered_payload), encoding="utf-8")
                tampered = replace(original, payload=tampered_payload)
                _Trainer.instances.clear()
                with (
                    self.subTest(key=key),
                    patch("model_runtime.runs.experiment.Trainer", _Trainer),
                    self.assertRaisesRegex(InvalidRunPlan, "retry receipt"),
                ):
                    execute_runs(
                        package,
                        plan,
                        artifacts=artifacts,
                        retry=RunPlanRetry(
                            execution_id=original_payload["executionId"],
                            completed_results=(tampered,),
                        ),
                    )
                self.assertEqual(_Trainer.instances, [])

            receipt_path.write_text(json.dumps(original_payload), encoding="utf-8")
            for completed_results in (
                (replace(original, run_id="foreign-run"),),
                (original, original),
            ):
                with (
                    self.subTest(completed_count=len(completed_results)),
                    self.assertRaisesRegex(InvalidRunPlan, "retry receipt"),
                ):
                    execute_runs(
                        package,
                        plan,
                        artifacts=artifacts,
                        retry=RunPlanRetry(
                            execution_id=original_payload["executionId"],
                            completed_results=completed_results,
                        ),
                    )

            outside = Path(tmp) / "escaped-retry-receipt"
            outside.mkdir()
            outside.joinpath("result.json").write_text(
                json.dumps(original_payload),
                encoding="utf-8",
            )
            escaped = replace(original, log_dir=str(outside))
            try:
                with self.assertRaisesRegex(InvalidRunPlan, "retry receipt"):
                    execute_runs(
                        package,
                        plan,
                        artifacts=artifacts,
                        retry=RunPlanRetry(
                            execution_id=original_payload["executionId"],
                            completed_results=(escaped,),
                        ),
                    )
            finally:
                outside.joinpath("result.json").unlink()
                outside.rmdir()

    def test_invalid_monitor_rejects_before_package_config_materialization(
        self,
    ) -> None:
        package = _linears_linear()
        plan = plan_runs(
            package,
            RunRequest(presets=("baseline",), datasets=("Mnist",)),
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                ModelPackage,
                "build_experiment",
            ) as build_experiment,
            self.assertRaisesRegex(InvalidRunPlan, "Unknown monitor option"),
        ):
            execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                monitors=("missing-monitor",),
            )

        build_experiment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
