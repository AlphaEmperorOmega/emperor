from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from model_runtime.packages import ModelPackage
from model_runtime.runs import execution
from model_runtime.runs.checkpoints import (
    CheckpointContinuation,
    _LoadedCheckpointContinuation,
    validate_model_state,
)
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
    def materialize_training_runs(_runs) -> list[SimpleNamespace]:
        return [SimpleNamespace(num_epochs=3)]

    @classmethod
    def execute_training_run(cls, _training_run, **kwargs):
        kwargs["model_validator"](cls.validated_model)
        return {}, "logs/run"


def _loaded_continuation() -> _LoadedCheckpointContinuation:
    return _LoadedCheckpointContinuation(
        request=CheckpointContinuation(Path("dynamic.ckpt")),
        state_dict={"dynamic.weight": torch.ones(2)},
        epoch=0,
        global_step=1,
    )


class RunsCheckpointValidationTests(unittest.TestCase):
    def test_validation_loads_checkpoint_state_strictly_into_the_model(self) -> None:
        continuation = _loaded_continuation()
        model = _TopologyAwareModel()

        validate_model_state(continuation, model)

        self.assertIs(model.strict, True)
        self.assertEqual(tuple(model.loaded_state), ("dynamic.weight",))
        torch.testing.assert_close(
            model.loaded_state["dynamic.weight"],
            continuation.state_dict["dynamic.weight"],
        )

    def test_execution_passes_the_model_to_checkpoint_validation(self) -> None:
        continuation = _loaded_continuation()
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

        with (
            patch.object(
                execution,
                "_validated_materialized_runs",
                return_value=("image-classification", ["baseline"], [{}]),
            ),
            patch.object(
                execution,
                "load_checkpoint_continuation",
                return_value=continuation,
            ),
            patch.object(execution, "validate_target_epochs"),
            patch.object(execution, "validate_model_state") as validate_state,
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
                continuation=continuation.request,
            )

        validate_state.assert_called_once_with(
            continuation,
            _Experiment.validated_model,
        )


if __name__ == "__main__":
    unittest.main()
