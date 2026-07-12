from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor.model_packages import model_package
from emperor.model_packages.identity import ModelIdentity
from emperor.runs import (
    InvalidRunPlan,
    RunRequest,
    execute_runs,
    plan_runs,
)
from emperor.runs.artifacts import FilesystemRunArtifacts
from lightning.pytorch.callbacks import Callback


class _Metric:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _Logger:
    def __init__(self, save_dir: str, name: str) -> None:
        self.log_dir = str(Path(save_dir) / name / "version_0")


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

    def fit(self, model, datamodule) -> None:
        self.model = model
        self.fit_datamodule = datamodule

    def test(self, model, datamodule) -> None:
        self.test_datamodule = datamodule


class _FailingTrainer(_Trainer):
    def fit(self, model, datamodule) -> None:
        raise RuntimeError("training exploded")


class _Progress(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[dict] = []
        self.events: list[dict] = []

    def set_run_context(
        self,
        dataset,
        log_dir=None,
        preset=None,
        preset_key=None,
        run_id=None,
        run_index=None,
        run_total=None,
        total_epochs=None,
    ) -> None:
        self.contexts.append(
            {
                "dataset": dataset,
                "logDir": log_dir,
                "preset": preset,
                "presetKey": preset_key,
                "runId": run_id,
                "runIndex": run_index,
                "runTotal": run_total,
                "totalEpochs": total_epochs,
            }
        )

    def write_event(self, event) -> None:
        self.events.append(dict(event))


def _linears_linear():
    package = model_package("linears/linear")
    if package is None:
        raise AssertionError("Expected the linears/linear Model Package.")
    return package


class RunsExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        _Trainer.instances.clear()

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
                patch("emperor.experiments.base.Trainer", _Trainer),
                patch("emperor.experiments.base.TensorBoardLogger", _Logger),
                patch("emperor.experiments.base.seed_everything"),
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
            self.assertEqual(progress.contexts[0]["runId"], "run-0001")
            self.assertEqual(progress.contexts[0]["runIndex"], 1)
            self.assertEqual(progress.contexts[0]["runTotal"], 1)
            self.assertEqual(len(_Trainer.instances), 1)

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
                patch("emperor.experiments.base.Trainer", _FailingTrainer),
                patch("emperor.experiments.base.TensorBoardLogger", _Logger),
                patch("emperor.experiments.base.seed_everything"),
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
                package.experiment_type,
                "_materialized_training_runs",
            ) as materialize,
            self.assertRaisesRegex(InvalidRunPlan, "Unknown monitor option"),
        ):
            execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(root=Path(tmp)),
                monitors=("missing-monitor",),
            )

        materialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
