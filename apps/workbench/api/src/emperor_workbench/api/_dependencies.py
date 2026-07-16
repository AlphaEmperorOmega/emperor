from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from emperor_workbench.api._container import (
    WorkbenchContainer,
    WorkbenchContainerSlot,
)
from emperor_workbench.config_snapshots import ConfigSnapshotService
from emperor_workbench.inspection import InspectionService
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.run_history import RunHistoryService
from emperor_workbench.run_plans import RunPlanService
from emperor_workbench.settings import WorkbenchApiSettings
from emperor_workbench.training_jobs import TrainingJobService


async def get_workbench_container(request: Request) -> WorkbenchContainer:
    slot = cast(
        WorkbenchContainerSlot,
        request.app.state.workbench_container_slot,
    )
    return slot.get()


async def get_workbench_settings(
    services: Annotated[WorkbenchContainer, Depends(get_workbench_container)],
) -> WorkbenchApiSettings:
    return services.settings


async def get_project_adapter_client(
    services: Annotated[WorkbenchContainer, Depends(get_workbench_container)],
) -> ProjectAdapterClient:
    return services.project_adapter


async def get_config_snapshot_service(
    services: Annotated[WorkbenchContainer, Depends(get_workbench_container)],
) -> ConfigSnapshotService:
    return services.config_snapshots


async def get_inspection_service(
    services: Annotated[WorkbenchContainer, Depends(get_workbench_container)],
) -> InspectionService:
    return services.inspection


async def get_run_history_service(
    services: Annotated[WorkbenchContainer, Depends(get_workbench_container)],
) -> RunHistoryService:
    return services.run_history


async def get_training_job_service(
    services: Annotated[WorkbenchContainer, Depends(get_workbench_container)],
) -> TrainingJobService:
    return services.training_jobs


async def get_training_run_plan_service(
    services: Annotated[WorkbenchContainer, Depends(get_workbench_container)],
) -> RunPlanService:
    return services.training_run_plans


__all__ = [
    "get_workbench_container",
    "get_config_snapshot_service",
    "get_inspection_service",
    "get_project_adapter_client",
    "get_run_history_service",
    "get_training_job_service",
    "get_training_run_plan_service",
    "get_workbench_settings",
]
