from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from model_runtime.packages import ModelPackage
from model_runtime.runs import execution
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
        self.loaded_state: dict[str, torch.Tensor] = {}
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
                experiment_task=None,
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


def _checkpoint_execution(
    directory: str,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    epoch: int = 0,
    target_epochs: int = 3,
) -> CheckpointExecution:
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
    return lifecycle.bind_training_runs([SimpleNamespace(num_epochs=target_epochs)])


class RunsCheckpointValidationTests(unittest.TestCase):
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
            checkpoint.write_bytes(b"checkpoint")
            plan = SimpleNamespace(runs=(object(),))
            for payload, message in cases:
                with (
                    self.subTest(message=message),
                    patch(
                        "model_runtime.runs.checkpoints.torch.load",
                        return_value=payload,
                    ),
                    self.assertRaisesRegex(InvalidCheckpointContinuation, message),
                ):
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
        self.assertIsNone(execution_options.model_validator)

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
            execution_options = _checkpoint_execution(
                tmp,
                state_dict={"dynamic.weight": expected},
            )
            model = _TopologyAwareModel()

            model_validator = execution_options.model_validator
            self.assertIsNotNone(model_validator)
            model_validator(model)

        self.assertIs(model.strict, True)
        self.assertEqual(tuple(model.loaded_state), ("dynamic.weight",))
        torch.testing.assert_close(
            model.loaded_state["dynamic.weight"],
            expected,
        )

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

            with self.assertRaisesRegex(
                InvalidCheckpointContinuation,
                "Target NUM_EPOCHS.*completed epochs",
            ):
                lifecycle.bind_training_runs([SimpleNamespace(num_epochs=2)])

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
                        "image-classification",
                        ["baseline"],
                        [
                            TrainingRunRequest(
                                run_id="run-0001",
                                run_index=1,
                                run_total=1,
                                preset="baseline",
                                dataset_type=object,
                                parameters={},
                                config_overrides={},
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
