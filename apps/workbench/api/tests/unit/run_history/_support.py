from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from emperor_workbench.log_experiments import (
    LogExperimentMutationCoordinator,
)
from emperor_workbench.run_history import (
    KnownModelPackageIdentityResolver,
    LogRunDeleteFilters,
    LogRunDeletePlan,
    LogRunDeleteResult,
    RunHistoryService,
)
from emperor_workbench.run_history._scanner import LogRunScanner
from tests.support.model_packages import model_identity_resolver


@dataclass(frozen=True, slots=True)
class ActiveWriter:
    id: str
    status: str
    log_folder: str


class FakeScalarEvent:
    def __init__(self, step: int, value: float, wall_time: float | None = None) -> None:
        self.step = step
        self.value = value
        self.wall_time = float(step if wall_time is None else wall_time)


class FakeTensorBoardAccumulator:
    def Tags(self) -> dict[str, list[str]]:
        return {
            "scalars": ["train/loss"],
            "histograms": ["weights"],
            "images": ["validation/examples/predictions"],
            "tensors": ["validation/examples/predictions/text_summary"],
        }

    def Scalars(self, tag: str) -> list[FakeScalarEvent]:
        if tag != "train/loss":
            raise KeyError(tag)
        return [
            FakeScalarEvent(step=1, value=0.5),
            FakeScalarEvent(step=2, value=0.25),
            FakeScalarEvent(step=3, value=0.125),
        ]


def run_history(
    logs_root: Path,
    *,
    active_writers: list[ActiveWriter] | None = None,
) -> RunHistoryService:
    writers = active_writers or []
    return RunHistoryService(
        logs_root=logs_root,
        mutation_coordinator=LogExperimentMutationCoordinator(),
        active_log_writers=lambda: list(writers),
        model_identity_resolver=model_identity_resolver(),
    )


def delete_plan(
    service: RunHistoryService,
    filters: LogRunDeleteFilters,
) -> LogRunDeletePlan:
    return service.create_delete_plan(
        experiments=list(filters.experiments),
        datasets=list(filters.datasets),
        models=list(filters.models),
        presets=list(filters.presets),
        run_ids=list(filters.run_ids),
    )


def delete_runs(
    service: RunHistoryService,
    filters: LogRunDeleteFilters,
) -> LogRunDeleteResult:
    return service.delete_runs(
        experiments=list(filters.experiments),
        datasets=list(filters.datasets),
        models=list(filters.models),
        presets=list(filters.presets),
        run_ids=list(filters.run_ids),
    )


def log_run_scanner(
    *,
    logs_root: Path | str = "logs",
    cache_ttl_seconds: float = 30.0,
    state_root: Path | None = None,
    identity_resolver: KnownModelPackageIdentityResolver | None = None,
) -> LogRunScanner:
    return LogRunScanner(
        logs_root=logs_root,
        cache_ttl_seconds=cache_ttl_seconds,
        state_root=state_root,
        model_identity_resolver=identity_resolver or model_identity_resolver(),
    )


__all__ = [
    "ActiveWriter",
    "FakeTensorBoardAccumulator",
    "delete_plan",
    "delete_runs",
    "log_run_scanner",
    "run_history",
]
