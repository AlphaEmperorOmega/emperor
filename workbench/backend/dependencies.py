"""FastAPI dependencies for app-scoped Workbench services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, cast

from fastapi import Depends, Request

from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.services.config_snapshots import ConfigSnapshotService
from workbench.backend.services.inspection import InspectionService
from workbench.backend.services.logs import LogRunService
from workbench.backend.services.models import ModelCatalogService
from workbench.backend.services.training import TrainingJobService


@dataclass(frozen=True, slots=True)
class WorkbenchServices:
    settings: WorkbenchApiSettings
    model_catalog: ModelCatalogService
    config_snapshots: ConfigSnapshotService
    inspection: InspectionService
    log_runs: LogRunService
    training_jobs: TrainingJobService


async def get_workbench_services(request: Request) -> WorkbenchServices:
    return cast(WorkbenchServices, request.app.state.workbench_services)


async def get_workbench_settings(
    services: Annotated[WorkbenchServices, Depends(get_workbench_services)],
) -> WorkbenchApiSettings:
    return services.settings


async def get_model_catalog_service(
    services: Annotated[WorkbenchServices, Depends(get_workbench_services)],
) -> ModelCatalogService:
    return services.model_catalog


async def get_config_snapshot_service(
    services: Annotated[WorkbenchServices, Depends(get_workbench_services)],
) -> ConfigSnapshotService:
    return services.config_snapshots


async def get_inspection_service(
    services: Annotated[WorkbenchServices, Depends(get_workbench_services)],
) -> InspectionService:
    return services.inspection


async def get_log_run_service(
    services: Annotated[WorkbenchServices, Depends(get_workbench_services)],
) -> LogRunService:
    return services.log_runs


async def get_training_job_service(
    services: Annotated[WorkbenchServices, Depends(get_workbench_services)],
) -> TrainingJobService:
    return services.training_jobs


__all__ = [
    "WorkbenchServices",
    "get_config_snapshot_service",
    "get_inspection_service",
    "get_log_run_service",
    "get_model_catalog_service",
    "get_training_job_service",
    "get_workbench_services",
    "get_workbench_settings",
]
