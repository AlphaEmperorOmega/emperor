"""FastAPI dependencies for app-scoped Viewer services."""

from __future__ import annotations

from typing import cast

from fastapi import Request

from viewer.backend.services.inspection import InspectionService
from viewer.backend.services.logs import LogRunService
from viewer.backend.services.models import ModelCatalogService
from viewer.backend.services.training import TrainingJobService


async def get_model_catalog_service(request: Request) -> ModelCatalogService:
    return cast(ModelCatalogService, request.app.state.model_catalog_service)


async def get_inspection_service(request: Request) -> InspectionService:
    return cast(InspectionService, request.app.state.inspection_service)


async def get_log_run_service(request: Request) -> LogRunService:
    return cast(LogRunService, request.app.state.log_run_service)


async def get_training_job_service(request: Request) -> TrainingJobService:
    return cast(TrainingJobService, request.app.state.training_job_service)
