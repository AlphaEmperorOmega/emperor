from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from emperor_workbench.api._container import WorkbenchContainerSlot
from emperor_workbench.api._errors import (
    ApiError,
    api_error_handler,
    domain_failure_handler,
    request_validation_error_handler,
)
from emperor_workbench.api._lifespan import create_lifespan
from emperor_workbench.api._middleware import configure_middleware
from emperor_workbench.api._mutations import build_http_operation_catalog
from emperor_workbench.api.v1 import PUBLIC_API_PREFIX
from emperor_workbench.api.v1 import router as api_v1_router
from emperor_workbench.failures import DomainFailure
from emperor_workbench.log_experiments import LogExperimentMutationCoordinator
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.settings import (
    WorkbenchApiSettings,
    get_workbench_api_settings,
)
from emperor_workbench.training_jobs import TrainingJobService


def create_app(
    settings: WorkbenchApiSettings | None = None,
    *,
    project_adapter: ProjectAdapterClient | None = None,
    training_jobs: TrainingJobService | None = None,
    log_experiment_mutations: LogExperimentMutationCoordinator | None = None,
) -> FastAPI:
    """Build a resource-free FastAPI application configuration."""

    api_settings = settings or get_workbench_api_settings()
    container_slot = WorkbenchContainerSlot()
    api = FastAPI(
        title="Emperor Model Workbench API",
        version="1.0.0",
        strict_content_type=True,
        lifespan=create_lifespan(
            api_settings,
            container_slot=container_slot,
            project_adapter=project_adapter,
            training_jobs=training_jobs,
            log_experiment_mutations=log_experiment_mutations,
        ),
    )
    api.state.workbench_container_slot = container_slot
    api.add_exception_handler(ApiError, api_error_handler)
    api.add_exception_handler(DomainFailure, domain_failure_handler)
    api.add_exception_handler(
        RequestValidationError,
        request_validation_error_handler,
    )
    api.include_router(api_v1_router, prefix=PUBLIC_API_PREFIX)
    operation_catalog = build_http_operation_catalog(
        api.routes,
        declared_routes=api_v1_router.routes,
    )
    configure_middleware(
        api,
        api_settings,
        operation_catalog,
        container_slot=container_slot,
    )
    return api


__all__ = ["create_app"]
