"""FastAPI dependencies for app-scoped Viewer services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, cast

from fastapi import Depends, Request

from viewer.backend.core.config import ViewerApiSettings
from viewer.backend.services.config_snapshots import ConfigSnapshotService
from viewer.backend.services.inspection import InspectionService
from viewer.backend.services.logs import LogRunService
from viewer.backend.services.models import ModelCatalogService
from viewer.backend.services.training import TrainingJobService


@dataclass(frozen=True, slots=True)
class ViewerServices:
    settings: ViewerApiSettings
    model_catalog: ModelCatalogService
    config_snapshots: ConfigSnapshotService
    inspection: InspectionService
    log_runs: LogRunService
    training_jobs: TrainingJobService


async def get_viewer_services(request: Request) -> ViewerServices:
    return cast(ViewerServices, request.app.state.viewer_services)


async def get_viewer_settings(
    services: Annotated[ViewerServices, Depends(get_viewer_services)],
) -> ViewerApiSettings:
    return services.settings


async def get_model_catalog_service(
    services: Annotated[ViewerServices, Depends(get_viewer_services)],
) -> ModelCatalogService:
    return services.model_catalog


async def get_config_snapshot_service(
    services: Annotated[ViewerServices, Depends(get_viewer_services)],
) -> ConfigSnapshotService:
    return services.config_snapshots


async def get_inspection_service(
    services: Annotated[ViewerServices, Depends(get_viewer_services)],
) -> InspectionService:
    return services.inspection


async def get_log_run_service(
    services: Annotated[ViewerServices, Depends(get_viewer_services)],
) -> LogRunService:
    return services.log_runs


async def get_training_job_service(
    services: Annotated[ViewerServices, Depends(get_viewer_services)],
) -> TrainingJobService:
    return services.training_jobs


__all__ = [
    "ViewerServices",
    "get_config_snapshot_service",
    "get_inspection_service",
    "get_log_run_service",
    "get_model_catalog_service",
    "get_training_job_service",
    "get_viewer_services",
    "get_viewer_settings",
]
