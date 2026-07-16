from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from emperor_workbench.api._bootstrap import acquire_container
from emperor_workbench.api._container import WorkbenchContainerSlot
from emperor_workbench.log_experiments import LogExperimentMutationCoordinator
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.settings import WorkbenchApiSettings
from emperor_workbench.training_jobs import TrainingJobService

Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


def create_lifespan(
    settings: WorkbenchApiSettings,
    *,
    container_slot: WorkbenchContainerSlot,
    project_adapter: ProjectAdapterClient | None,
    training_jobs: TrainingJobService | None = None,
    log_experiment_mutations: LogExperimentMutationCoordinator | None = None,
) -> Lifespan:
    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        container = acquire_container(
            settings,
            project_adapter=project_adapter,
            training_jobs=training_jobs,
            log_experiment_mutations=log_experiment_mutations,
        )
        slot_installed = False
        state_published = False
        try:
            container_slot.install(container)
            slot_installed = True
            api.state.workbench_container = container
            state_published = True
            yield
        finally:
            try:
                try:
                    await container.mutation_execution.close()
                finally:
                    try:
                        container.blocking_work.close()
                    finally:
                        container.project_adapter.close()
            finally:
                try:
                    if state_published:
                        del api.state.workbench_container
                finally:
                    if slot_installed:
                        container_slot.remove(container)

    return lifespan


__all__ = ["create_lifespan"]
