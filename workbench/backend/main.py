"""FastAPI application factory for the Emperor Model Workbench API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from workbench.backend.api.mutation_policy import build_http_operation_catalog
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
from workbench.backend.log_experiments import (
    LogExperimentMutationCoordinator,
)
from workbench.backend.middleware import configure_middleware
from workbench.backend.run_history import RunHistoryService
from workbench.backend.services.config_snapshots import ConfigSnapshotService
from workbench.backend.services.inspection import InspectionService
from workbench.backend.training_jobs import TrainingJobService
from workbench.backend.training_jobs.plans import TrainingRunPlanService

__all__ = ["WorkbenchApiSettings", "create_app", "app"]


def create_app(
    settings: WorkbenchApiSettings | None = None,
) -> FastAPI:
    api_settings = settings or get_workbench_api_settings()
    log_experiment_mutations = LogExperimentMutationCoordinator()
    training_jobs = TrainingJobService(
        logs_root=api_settings.logs_root,
        cancellation_mode=api_settings.training_cancellation_mode,
        mutation_coordinator=log_experiment_mutations,
    )
    run_history = RunHistoryService(
        logs_root=api_settings.logs_root,
        mutation_coordinator=log_experiment_mutations,
        active_log_writers=lambda: training_jobs.active_jobs(),
    )
    snapshot_store = FileSystemConfigSnapshotStore(Path(api_settings.snapshots_root))

    api = FastAPI(title="Emperor Model Workbench API", version="1.0.0")
    api.add_exception_handler(ApiError, api_error_handler)

    api.state.workbench_services = WorkbenchServices(
        settings=api_settings,
        config_snapshots=ConfigSnapshotService(snapshot_store),
        inspection=InspectionService(run_history),
        run_history=run_history,
        training_jobs=training_jobs,
        training_run_plans=TrainingRunPlanService(),
        log_experiment_mutations=log_experiment_mutations,
    )

    api.include_router(api_v1_router, prefix=PUBLIC_API_PREFIX)
    operation_catalog = build_http_operation_catalog(
        api.routes,
        declared_routes=api_v1_router.routes,
    )
    configure_middleware(api, api_settings, operation_catalog)
    return api


app = create_app()
