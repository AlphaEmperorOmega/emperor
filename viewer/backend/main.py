"""FastAPI application factory for the Emperor Model Viewer API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from viewer.backend.api.v1.router import router as api_v1_router
from viewer.backend.config_snapshots import FileSystemConfigSnapshotStore
from viewer.backend.core.config import ViewerApiSettings, get_viewer_api_settings
from viewer.backend.exceptions import inspector_error_handler
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.log_runs import LogRunIndex
from viewer.backend.middleware import configure_middleware
from viewer.backend.repositories.config_snapshots import ConfigSnapshotRepository
from viewer.backend.repositories.log_runs import LogRunRepository
from viewer.backend.repositories.training_jobs import TrainingJobRepository
from viewer.backend.services.config_snapshots import ConfigSnapshotService
from viewer.backend.services.inspection import InspectionService
from viewer.backend.services.logs import LogRunService
from viewer.backend.services.models import ModelCatalogService
from viewer.backend.services.training import TrainingJobService
from viewer.backend.training_jobs import TrainingJobManager

__all__ = ["ViewerApiSettings", "create_app", "app"]


def create_app(
    settings: ViewerApiSettings | None = None,
    training_manager: TrainingJobManager | None = None,
) -> FastAPI:
    api_settings = settings or get_viewer_api_settings()
    jobs = training_manager or TrainingJobManager(logs_root=api_settings.logs_root)
    log_runs = LogRunIndex(logs_root=api_settings.logs_root)
    snapshot_store = FileSystemConfigSnapshotStore(Path(api_settings.snapshots_root))

    api = FastAPI(title="Emperor Model Viewer API", version="1.0.0")
    configure_middleware(api, api_settings)
    api.add_exception_handler(InspectorError, inspector_error_handler)

    api.state.settings = api_settings
    api.state.model_catalog_service = ModelCatalogService()
    api.state.inspection_service = InspectionService()
    api.state.log_run_service = LogRunService(LogRunRepository(log_runs))
    api.state.training_job_service = TrainingJobService(TrainingJobRepository(jobs))
    api.state.config_snapshot_service = ConfigSnapshotService(
        ConfigSnapshotRepository(snapshot_store)
    )

    api.include_router(api_v1_router)
    return api


app = create_app()
