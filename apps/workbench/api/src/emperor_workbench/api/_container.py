from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor_workbench.config_snapshots import ConfigSnapshotService
from emperor_workbench.inspection import InspectionService
from emperor_workbench.log_experiments import LogExperimentMutationCoordinator
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.run_history import RunHistoryService
from emperor_workbench.run_plans import RunPlanService
from emperor_workbench.settings import WorkbenchApiSettings
from emperor_workbench.training_jobs import TrainingJobService

if TYPE_CHECKING:
    from emperor_workbench.api._blocking import BlockingWorkRuntime
    from emperor_workbench.api._mutations import MutationExecutionRuntime


@dataclass(frozen=True, slots=True)
class WorkbenchContainer:
    """All resources owned by one running Workbench application instance."""

    settings: WorkbenchApiSettings
    project_adapter: ProjectAdapterClient
    config_snapshots: ConfigSnapshotService
    inspection: InspectionService
    run_history: RunHistoryService
    training_jobs: TrainingJobService
    training_run_plans: RunPlanService
    log_experiment_mutations: LogExperimentMutationCoordinator
    blocking_work: BlockingWorkRuntime
    mutation_execution: MutationExecutionRuntime


class WorkbenchContainerSlot:
    """Resource-free handle populated only while application lifespan is active."""

    def __init__(self) -> None:
        self._container: WorkbenchContainer | None = None

    def install(self, container: WorkbenchContainer) -> None:
        if self._container is not None:
            raise RuntimeError("Workbench application lifespan is already active.")
        self._container = container

    def remove(self, container: WorkbenchContainer) -> None:
        if self._container is container:
            self._container = None

    def get(self) -> WorkbenchContainer:
        if self._container is None:
            raise RuntimeError(
                "Workbench application resources are unavailable outside lifespan."
            )
        return self._container


_ACTIVE_CONTAINER: contextvars.ContextVar[WorkbenchContainer | None] = (
    contextvars.ContextVar("workbench_active_container", default=None)
)


@contextmanager
def activate_container(container: WorkbenchContainer) -> Iterator[None]:
    token = _ACTIVE_CONTAINER.set(container)
    try:
        yield
    finally:
        _ACTIVE_CONTAINER.reset(token)


def current_container() -> WorkbenchContainer:
    container = _ACTIVE_CONTAINER.get()
    if container is None:
        raise RuntimeError("No Workbench application container is active.")
    return container


__all__ = ["WorkbenchContainer", "WorkbenchContainerSlot"]
