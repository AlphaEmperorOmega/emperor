"""FastAPI application factory for the Emperor Model Workbench API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from workbench.backend.api.v1.router import (
    PUBLIC_API_PREFIX,
)
from workbench.backend.api.v1.router import (
    router as api_v1_router,
)
from workbench.backend.config_snapshots import FileSystemConfigSnapshotStore
from workbench.backend.core.config import (
    WorkbenchApiSettings,
    get_workbench_api_settings,
)
from workbench.backend.core.errors import ApiError
from workbench.backend.dependencies import WorkbenchServices
from workbench.backend.exceptions import api_error_handler
from workbench.backend.log_runs import LogRunIndex
from workbench.backend.middleware import configure_middleware
from workbench.backend.repositories.config_snapshots import ConfigSnapshotRepository
from workbench.backend.repositories.log_runs import LogRunRepository
from workbench.backend.repositories.training_jobs import TrainingJobRepository
from workbench.backend.services.config_snapshots import ConfigSnapshotService
from workbench.backend.services.inspection import InspectionService
from workbench.backend.services.logs import LogRunService
from workbench.backend.services.models import ModelCatalogService
from workbench.backend.services.training import TrainingJobService
from workbench.backend.training_jobs import TrainingJobManager

__all__ = ["WorkbenchApiSettings", "create_app", "app"]


def create_app(
    settings: WorkbenchApiSettings | None = None,
    training_manager: TrainingJobManager | None = None,
) -> FastAPI:
    api_settings = settings or get_workbench_api_settings()
    jobs = training_manager or TrainingJobManager(
        logs_root=api_settings.logs_root,
        cancellation_mode=api_settings.training_cancellation_mode,
    )
    log_runs = LogRunIndex(logs_root=api_settings.logs_root)
    log_run_repository = LogRunRepository(log_runs)
    snapshot_store = FileSystemConfigSnapshotStore(Path(api_settings.snapshots_root))

    api = FastAPI(title="Emperor Model Workbench API", version="1.0.0")
    configure_middleware(api, api_settings)
    api.add_exception_handler(ApiError, api_error_handler)

    api.state.workbench_services = WorkbenchServices(
        settings=api_settings,
        model_catalog=ModelCatalogService(),
        config_snapshots=ConfigSnapshotService(
            ConfigSnapshotRepository(snapshot_store)
        ),
        inspection=InspectionService(log_run_repository),
        log_runs=LogRunService(log_run_repository),
        training_jobs=TrainingJobService(TrainingJobRepository(jobs)),
    )

    api.include_router(api_v1_router, prefix=PUBLIC_API_PREFIX)
    return api


app = create_app()
