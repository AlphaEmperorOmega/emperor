from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from inspect import Parameter, signature
from pathlib import Path
from unittest.mock import patch

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from emperor.monitoring import MonitorOption
from model_runtime.packages import ModelIdentity, ModelPackage
from model_runtime.runs import (
    CheckpointContinuation,
    InvalidCheckpointContinuation,
    InvalidRunPlan,
    PlanningBudget,
    RunParameter,
    RunRequest,
    SubmittedRun,
    accept_run_plan,
    execute_runs,
    plan_runs,
)
from model_runtime.runs.artifacts import FilesystemRunArtifacts
from model_runtime.runs.experiment import ExperimentBase
from models.catalog import model_package


class _Metric:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _Logger:
    instances: list[_Logger] = []

    def __init__(self, save_dir: str, name: str) -> None:
        self.log_dir = str(Path(save_dir) / name / "version_0")
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
        type(self).instances.append(self)

    def fit(self, model, datamodule, **kwargs) -> None:
        self.model = model
        self.fit_datamodule = datamodule
        self.fit_kwargs = kwargs

    def test(self, model, datamodule) -> None:
        self.test_datamodule = datamodule


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
                )
            ],
            [None, 1, (), None, None],
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

    def test_valid_continuation_passes_exact_checkpoint_path_to_lightning(
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

            self.assertEqual(
                _Trainer.instances[0].fit_kwargs,
                {"ckpt_path": checkpoint_path},
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
                    "could not be loaded",
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
