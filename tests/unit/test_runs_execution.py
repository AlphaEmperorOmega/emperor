from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from lightning.pytorch.callbacks import ModelCheckpoint

from model_runtime.packages import ModelIdentity, ModelPackage
from model_runtime.runs import (
    CheckpointContinuation,
    InvalidCheckpointContinuation,
    InvalidRunPlan,
    RunRequest,
    execute_runs,
    plan_runs,
)
from model_runtime.runs.artifacts import FilesystemRunArtifacts
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
