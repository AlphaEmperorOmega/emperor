from __future__ import annotations

import gc
import hashlib
import sys
import tempfile
import unittest
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from emperor.experiments import ExperimentTask
from model_runtime.packages import ModelPackage
from model_runtime.runs import CheckpointAdmissionPolicy, execution
from model_runtime.runs._handoff import TrainingRun, TrainingRunRequest
from model_runtime.runs.checkpoints import (
    CheckpointContinuation,
    CheckpointContinuationLifecycle,
    CheckpointExecution,
)
from model_runtime.runs.errors import InvalidCheckpointContinuation
from models.catalog import model_package


class _TopologyAwareModel:
    def __init__(self) -> None:
        self.loaded_state: dict[str, torch.Tensor] = {"dynamic.weight": torch.zeros(2)}
        self.strict: bool | None = None

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        strict: bool,
    ) -> None:
        self.strict = strict
        self.loaded_state = dict(state_dict)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.loaded_state)


class _Experiment:
    validated_model = _TopologyAwareModel()

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    @staticmethod
    def materialize_training_runs(runs) -> list[TrainingRun]:
        return [
            TrainingRun(
                experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
                preset=run.preset,
                dataset_type=run.dataset_type,
                config=SimpleNamespace(),  # type: ignore[arg-type]
                config_overrides=dict(run.config_overrides),
                num_epochs=3,
                parameters=dict(run.parameters),
                run_id=run.run_id,
                run_index=run.run_index,
                run_total=run.run_total,
            )
            for run in runs
        ]

    @classmethod
    def execute_training(cls, request):
        model_validator = request.model_validator
        assert model_validator is not None
        model_validator(cls.validated_model)
        return {}, "logs/run"


def _failing_experiment(
    failure_stage: str,
    observed_paths: list[Path],
) -> type[_Experiment]:
    class FailingExperiment(_Experiment):
        @classmethod
        def execute_training(cls, request):
            assert request.ckpt_path is not None
            observed_paths.append(request.ckpt_path)
            assert request.model_validator is not None
            request.model_validator(_TopologyAwareModel())
            if failure_stage == "training":
                raise RuntimeError("training failed")
            return {}, "logs/run"

    return FailingExperiment


def _checkpoint_execution(
    directory: str,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    epoch: int = 0,
    target_epochs: int = 3,
) -> tuple[CheckpointContinuationLifecycle, CheckpointExecution]:
    checkpoint_path = Path(directory) / "dynamic.ckpt"
    torch.save(
        {
            "pytorch-lightning_version": "2.5.0",
            "state_dict": state_dict or {"dynamic.weight": torch.ones(2)},
            "epoch": epoch,
            "global_step": 1,
            "optimizer_states": [{}],
        },
        checkpoint_path,
    )
    lifecycle = CheckpointContinuationLifecycle.admit(
        CheckpointContinuation(checkpoint_path),
        SimpleNamespace(runs=(object(),)),
    )
    return lifecycle, lifecycle.bind_training_runs(
        [SimpleNamespace(num_epochs=target_epochs)]
    )


class RunsCheckpointValidationTests(unittest.TestCase):
    def test_checkpoint_file_limit_rejects_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "oversized.ckpt"
            checkpoint.write_bytes(b"x" * 17)
            with (
                patch("model_runtime.runs.checkpoints.torch.load") as torch_load,
                self.assertRaisesRegex(
                    InvalidCheckpointContinuation,
                    "17 bytes.*limit of 16 bytes",
                ),
            ):
                CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(checkpoint),
                    SimpleNamespace(runs=(object(),)),
                    admission_policy=CheckpointAdmissionPolicy(max_file_bytes=16),
                )

            torch_load.assert_not_called()

    def test_execution_uses_and_cleans_an_immutable_admitted_snapshot(self) -> None:
        package = model_package("linears/linear")
        assert package is not None
        semantic_run = SimpleNamespace(
            id="run-0001",
            experiment_task="image-classification",
            preset="baseline",
            dataset="SyntheticDataset",
        )
        plan = SimpleNamespace(runs=(semantic_run,))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.ckpt"
            replacement = root / "replacement.ckpt"
            original_state = torch.ones(2)
            torch.save(
                {
                    "pytorch-lightning_version": "2.5.0",
                    "state_dict": {"dynamic.weight": original_state},
                    "epoch": 0,
                    "global_step": 1,
                    "optimizer_states": [{}],
                },
                source,
            )
            original_bytes = source.read_bytes()
            torch.save(
                {
                    "pytorch-lightning_version": "2.5.0",
                    "state_dict": {"dynamic.weight": torch.full((2,), 9.0)},
                    "epoch": 1,
                    "global_step": 9,
                    "optimizer_states": [{}],
                },
                replacement,
            )
            replacement_bytes = replacement.read_bytes()

            class ReplacingModel(_TopologyAwareModel):
                def load_state_dict(self, state_dict, *, strict):
                    replacement.replace(source)
                    super().load_state_dict(state_dict, strict=strict)

            class SnapshotObservingExperiment(_Experiment):
                observed_path: Path | None = None
                observed_bytes: bytes | None = None
                observed_provenance: dict[str, object] | None = None

                @classmethod
                def execute_training(cls, request):
                    assert request.model_validator is not None
                    request.model_validator(ReplacingModel())
                    cls.observed_path = request.ckpt_path
                    assert cls.observed_path is not None
                    cls.observed_bytes = cls.observed_path.read_bytes()
                    cls.observed_provenance = dict(request.resumed_from)
                    return {}, "logs/run"

            with (
                patch.object(
                    execution,
                    "_validated_materialized_runs",
                    return_value=(
                        ExperimentTask.IMAGE_CLASSIFICATION,
                        ["baseline"],
                        [
                            TrainingRunRequest(
                                run_id="run-0001",
                                run_index=1,
                                run_total=1,
                                preset="baseline",
                                dataset_type=object,
                                parameters={},
                                config_overrides={"num_epochs": 3},
                            )
                        ],
                    ),
                ),
                patch.object(
                    ModelPackage,
                    "build_experiment",
                    return_value=SnapshotObservingExperiment(),
                ),
            ):
                execution.execute_runs(
                    package,
                    plan,
                    artifacts=SimpleNamespace(namespace="runs"),
                    continuation=CheckpointContinuation(source),
                )

            observed_path = SnapshotObservingExperiment.observed_path
            assert observed_path is not None
            self.assertNotEqual(observed_path, source)
            self.assertEqual(
                SnapshotObservingExperiment.observed_bytes,
                original_bytes,
            )
            self.assertEqual(source.read_bytes(), replacement_bytes)
            self.assertFalse(observed_path.exists())
            self.assertEqual(
                SnapshotObservingExperiment.observed_provenance,
                {
                    "checkpoint": "source.ckpt",
                    "epoch": 0,
                    "globalStep": 1,
                    "sha256": hashlib.sha256(original_bytes).hexdigest(),
                },
            )

    def test_execution_cleans_snapshot_after_training_or_projection_failure(
        self,
    ) -> None:
        package = model_package("linears/linear")
        assert package is not None
        semantic_run = SimpleNamespace(
            id="run-0001",
            experiment_task="image-classification",
            preset="baseline",
            dataset="SyntheticDataset",
        )
        plan = SimpleNamespace(runs=(semantic_run,))

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.ckpt"
            torch.save(
                {
                    "pytorch-lightning_version": "2.5.0",
                    "state_dict": {"dynamic.weight": torch.ones(2)},
                    "epoch": 0,
                    "global_step": 1,
                    "optimizer_states": [{}],
                },
                source,
            )
            for stage in ("training", "projection"):
                observed_paths: list[Path] = []
                experiment_type = _failing_experiment(stage, observed_paths)

                projection_failure = (
                    patch.object(
                        execution,
                        "run_result",
                        side_effect=RuntimeError("projection failed"),
                    )
                    if stage == "projection"
                    else nullcontext()
                )
                with (
                    self.subTest(stage=stage),
                    patch.object(
                        execution,
                        "_validated_materialized_runs",
                        return_value=(
                            ExperimentTask.IMAGE_CLASSIFICATION,
                            ["baseline"],
                            [
                                TrainingRunRequest(
                                    run_id="run-0001",
                                    run_index=1,
                                    run_total=1,
                                    preset="baseline",
                                    dataset_type=object,
                                    parameters={},
                                    config_overrides={"num_epochs": 3},
                                )
                            ],
                        ),
                    ),
                    patch.object(
                        ModelPackage,
                        "build_experiment",
                        return_value=experiment_type(),
                    ),
                    projection_failure,
                    self.assertRaisesRegex(RuntimeError, f"{stage} failed"),
                ):
                    execution.execute_runs(
                        package,
                        plan,
                        artifacts=SimpleNamespace(namespace="runs"),
                        continuation=CheckpointContinuation(source),
                    )

                self.assertEqual(len(observed_paths), 1)
                self.assertFalse(observed_paths[0].exists())

    def test_tensor_count_limit_includes_nested_optimizer_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "optimizer-tensors.ckpt"
            torch.save(
                {
                    "pytorch-lightning_version": "2.5.0",
                    "state_dict": {"weight": torch.ones(1)},
                    "epoch": 0,
                    "global_step": 1,
                    "optimizer_states": [
                        {
                            "state": {
                                0: {
                                    "momentum": torch.ones(1),
                                    "variance": torch.ones(1),
                                }
                            }
                        }
                    ],
                },
                checkpoint,
            )

            with self.assertRaisesRegex(
                InvalidCheckpointContinuation,
                "3 tensors.*limit of 2",
            ):
                CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(checkpoint),
                    SimpleNamespace(runs=(object(),)),
                    admission_policy=CheckpointAdmissionPolicy(max_tensor_count=2),
                )

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux containment")
    def test_required_isolated_decode_enforces_worker_wall_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "timeout.ckpt"
            torch.save(
                {
                    "pytorch-lightning_version": "2.5.0",
                    "state_dict": {"weight": torch.ones(1)},
                    "epoch": 0,
                    "global_step": 1,
                    "optimizer_states": [{}],
                },
                checkpoint,
            )

            with self.assertRaisesRegex(
                InvalidCheckpointContinuation,
                "isolated decoder timed out",
            ):
                CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(checkpoint),
                    SimpleNamespace(runs=(object(),)),
                    admission_policy=CheckpointAdmissionPolicy(
                        require_isolated_decode=True,
                        worker_wall_timeout_seconds=0.001,
                    ),
                )

    def test_checkpoint_payload_validation_precedence_is_stable(self) -> None:
        valid = {
            "pytorch-lightning_version": "2.5.0",
            "state_dict": {"weight": torch.zeros(1)},
            "epoch": 0,
            "global_step": 1,
            "optimizer_states": [{}],
        }
        cases = (
            (
                {
                    **valid,
                    "pytorch-lightning_version": "",
                    "state_dict": {},
                },
                "Lightning version",
            ),
            (
                {**valid, "state_dict": {}, "epoch": -1},
                "nonempty state_dict",
            ),
            (
                {
                    **valid,
                    "state_dict": {1: torch.zeros(1)},
                    "epoch": -1,
                },
                "state_dict keys must be strings",
            ),
            (
                {**valid, "epoch": -1, "global_step": -1},
                "nonnegative epoch",
            ),
            (
                {**valid, "global_step": -1, "optimizer_states": []},
                "nonnegative global_step",
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "precedence.ckpt"
            plan = SimpleNamespace(runs=(object(),))
            for payload, message in cases:
                with (
                    self.subTest(message=message),
                    self.assertRaisesRegex(InvalidCheckpointContinuation, message),
                ):
                    torch.save(payload, checkpoint)
                    CheckpointContinuationLifecycle.admit(
                        CheckpointContinuation(checkpoint),
                        plan,
                    )

    def test_absent_lifecycle_does_not_inspect_plan_or_training_run(self) -> None:
        class ExplosivePlan:
            @property
            def runs(self):
                raise AssertionError("absent continuation inspected the Run Plan")

        class ExplosiveTrainingRun:
            @property
            def num_epochs(self):
                raise AssertionError("absent continuation inspected the training Run")

        lifecycle = CheckpointContinuationLifecycle.admit(None, ExplosivePlan())
        execution_options = lifecycle.bind_training_runs([ExplosiveTrainingRun()])

        self.assertIsNone(execution_options.checkpoint_path)
        self.assertIsNone(execution_options.provenance)
        self.assertIsNone(execution_options.strict_model_preloader)

    def test_lifecycle_rejects_run_cardinality_before_checkpoint_file(self) -> None:
        with self.assertRaisesRegex(
            InvalidCheckpointContinuation,
            "exactly one Run",
        ):
            CheckpointContinuationLifecycle.admit(
                CheckpointContinuation(Path("missing.ckpt")),
                SimpleNamespace(runs=(object(), object())),
            )

    def test_validation_loads_checkpoint_state_strictly_into_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            expected = torch.ones(2)
            lifecycle, execution_options = _checkpoint_execution(
                tmp,
                state_dict={"dynamic.weight": expected},
            )
            with lifecycle:
                model = _TopologyAwareModel()

                model_validator = execution_options.strict_model_preloader
                self.assertIsNotNone(model_validator)
                model_validator(model)

        self.assertIs(model.strict, True)
        self.assertEqual(tuple(model.loaded_state), ("dynamic.weight",))
        torch.testing.assert_close(
            model.loaded_state["dynamic.weight"],
            expected,
        )

    def test_strict_preload_is_one_shot_and_releases_parent_payload(self) -> None:
        class NonRetainingModel:
            admitted_tensor: weakref.ReferenceType[torch.Tensor] | None = None

            @staticmethod
            def state_dict() -> dict[str, torch.Tensor]:
                return {"weight": torch.zeros(1)}

            def load_state_dict(self, state_dict, *, strict):
                self.asserted_strict = strict
                self.admitted_tensor = weakref.ref(state_dict["weight"])

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "release.ckpt"
            torch.save(
                {
                    "pytorch-lightning_version": "2.6.5",
                    "state_dict": {"weight": torch.ones(1)},
                    "epoch": 0,
                    "global_step": 1,
                    "optimizer_states": [{}],
                },
                checkpoint,
            )
            with CheckpointContinuationLifecycle.admit(
                CheckpointContinuation(checkpoint),
                SimpleNamespace(runs=(object(),)),
            ) as lifecycle:
                execution_options = lifecycle.bind_training_runs(
                    [SimpleNamespace(num_epochs=2)]
                )
                preloader = execution_options.strict_model_preloader
                assert preloader is not None
                model = NonRetainingModel()
                preloader(model)
                gc.collect()

                self.assertIs(model.asserted_strict, True)
                assert model.admitted_tensor is not None
                self.assertIsNone(model.admitted_tensor())
                with self.assertRaisesRegex(RuntimeError, "already consumed"):
                    preloader(model)

    def test_model_preload_diagnostics_are_bounded_and_snapshot_is_cleaned(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "mismatch.ckpt"
            torch.save(
                {
                    "pytorch-lightning_version": "2.6.5",
                    "state_dict": {
                        f"checkpoint-{index}": torch.ones(1) for index in range(100)
                    },
                    "epoch": 0,
                    "global_step": 1,
                    "optimizer_states": [{}],
                },
                checkpoint,
            )
            snapshot_path: Path | None = None
            with self.assertRaises(InvalidCheckpointContinuation) as raised:
                with CheckpointContinuationLifecycle.admit(
                    CheckpointContinuation(checkpoint),
                    SimpleNamespace(runs=(object(),)),
                ) as lifecycle:
                    execution_options = lifecycle.bind_training_runs(
                        [SimpleNamespace(num_epochs=2)]
                    )
                    snapshot_path = execution_options.checkpoint_path
                    preloader = execution_options.strict_model_preloader
                    assert preloader is not None
                    preloader({f"model-{index}": torch.ones(1) for index in range(100)})

            self.assertLess(len(str(raised.exception)), 1024)
            self.assertIn("missing_count=100", str(raised.exception))
            self.assertIn("unexpected_count=100", str(raised.exception))
            assert snapshot_path is not None
            self.assertFalse(snapshot_path.exists())

    def test_target_epoch_validation_occurs_when_training_runs_are_bound(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "epoch.ckpt"
            torch.save(
                {
                    "pytorch-lightning_version": "2.5.0",
                    "state_dict": {"weight": torch.ones(1)},
                    "epoch": 1,
                    "global_step": 2,
                    "optimizer_states": [{}],
                },
                checkpoint,
            )
            lifecycle = CheckpointContinuationLifecycle.admit(
                CheckpointContinuation(checkpoint),
                SimpleNamespace(runs=(object(),)),
            )

            with lifecycle:
                with self.assertRaisesRegex(
                    InvalidCheckpointContinuation,
                    "Target NUM_EPOCHS.*completed epochs",
                ):
                    lifecycle.bind_training_runs([SimpleNamespace(num_epochs=2)])

    def test_lightning_loop_progress_controls_terminal_epoch_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "terminal.ckpt"
            torch.save(
                {
                    "pytorch-lightning_version": "2.6.5",
                    "state_dict": {"weight": torch.ones(1)},
                    "epoch": 1,
                    "global_step": 2,
                    "optimizer_states": [{}],
                    "loops": {
                        "fit_loop": {
                            "epoch_progress": {
                                "current": {"completed": 1},
                            }
                        }
                    },
                },
                checkpoint,
            )

            with CheckpointContinuationLifecycle.admit(
                CheckpointContinuation(checkpoint),
                SimpleNamespace(runs=(object(),)),
            ) as lifecycle:
                execution_options = lifecycle.bind_training_runs(
                    [SimpleNamespace(num_epochs=2)]
                )

            self.assertEqual(execution_options.provenance["epoch"], 1)

    def test_execution_passes_the_model_to_checkpoint_validation(self) -> None:
        semantic_run = SimpleNamespace(
            id="run-0001",
            experiment_task="image-classification",
            preset="baseline",
            dataset="SyntheticDataset",
        )
        plan = SimpleNamespace(runs=(semantic_run,))
        package = model_package("linears/linear")
        assert package is not None
        artifacts = SimpleNamespace(namespace="runs")
        _Experiment.validated_model = _TopologyAwareModel()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "dynamic.ckpt"
            expected = torch.ones(2)
            torch.save(
                {
                    "pytorch-lightning_version": "2.5.0",
                    "state_dict": {"dynamic.weight": expected},
                    "epoch": 0,
                    "global_step": 1,
                    "optimizer_states": [{}],
                },
                checkpoint,
            )
            with (
                patch.object(
                    execution,
                    "_validated_materialized_runs",
                    return_value=(
                        ExperimentTask.IMAGE_CLASSIFICATION,
                        ["baseline"],
                        [
                            TrainingRunRequest(
                                run_id="run-0001",
                                run_index=1,
                                run_total=1,
                                preset="baseline",
                                dataset_type=object,
                                parameters={},
                                config_overrides={"num_epochs": 3},
                            )
                        ],
                    ),
                ),
                patch.object(
                    ModelPackage,
                    "build_experiment",
                    return_value=_Experiment(),
                ),
            ):
                execution.execute_runs(
                    package,
                    plan,
                    artifacts=artifacts,
                    continuation=CheckpointContinuation(checkpoint),
                )

        self.assertIs(_Experiment.validated_model.strict, True)
        torch.testing.assert_close(
            _Experiment.validated_model.loaded_state["dynamic.weight"],
            expected,
        )


if __name__ == "__main__":
    unittest.main()
