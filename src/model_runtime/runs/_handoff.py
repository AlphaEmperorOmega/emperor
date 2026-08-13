from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from model_runtime.runs._value_policy import deep_freeze

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback

    from emperor.config import BaseOptions, ModelConfig
    from emperor.experiments import ExperimentTask
    from model_runtime.runs.progress import RunProgress


@dataclass(frozen=True, slots=True)
class TrainingRunRequest:
    """Validated immutable input for Experiment-side Run materialization."""

    run_id: str
    run_index: int
    run_total: int
    preset: Any
    dataset_type: type[Any]
    parameters: Mapping[str, object]
    config_overrides: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            deep_freeze(self.parameters),
        )
        object.__setattr__(
            self,
            "config_overrides",
            deep_freeze(self.config_overrides),
        )


@dataclass
class TrainingRun:
    """Experiment-ready Run produced from one validated handoff request."""

    experiment_task: ExperimentTask | None
    preset: BaseOptions
    dataset_type: type[Any]
    config: ModelConfig
    config_overrides: dict[str, Any]
    num_epochs: int
    parameters: dict[str, object] = field(default_factory=dict[str, object])
    run_id: str | None = None
    run_index: int | None = None
    run_total: int | None = None


@dataclass(frozen=True, slots=True)
class TrainingExecutionRequest:
    """Complete typed request for one Experiment training lifecycle."""

    training_run: TrainingRun
    callbacks: Sequence[Callback]
    progress: RunProgress | None
    progress_step_interval: int
    ckpt_path: Path | None
    model_validator: Callable[[object], None] | None
    resumed_from: Mapping[str, object] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "callbacks", tuple(self.callbacks))
        if self.resumed_from is not None:
            object.__setattr__(
                self,
                "resumed_from",
                deep_freeze(self.resumed_from),
            )


class RunExperiment(Protocol):
    """Runs-owned capability implemented by package-local Experiments."""

    def materialize_training_runs(
        self,
        requests: Sequence[TrainingRunRequest],
    ) -> list[TrainingRun]:
        """Return one Training Run per request in the same order.

        Every result preserves its request's run id, run index, run total,
        preset identity, and Dataset identity.
        """
        ...

    def execute_training(
        self,
        request: TrainingExecutionRequest,
    ) -> tuple[dict[str, Any], str]: ...


def require_run_experiment(value: object, catalog_key: str) -> RunExperiment:
    required_operations = (
        "materialize_training_runs",
        "execute_training",
    )
    if any(not callable(getattr(value, name, None)) for name in required_operations):
        raise TypeError(
            f"Model Package '{catalog_key}' returned an invalid Run Experiment; "
            "expected callable materialize_training_runs and execute_training "
            "operations."
        )
    return cast(RunExperiment, value)


__all__ = [
    "RunExperiment",
    "TrainingExecutionRequest",
    "TrainingRun",
    "TrainingRunRequest",
]
