from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from workbench.backend.api.mutation_policy import build_http_operation_catalog
from workbench.backend.api.v1.router import (
    PUBLIC_API_PREFIX,
)
from workbench.backend.api.v1.router import (
    router as api_v1_router,
)
from workbench.backend.config_snapshots import (
    ConfigSnapshotService,
    FileSystemConfigSnapshotStore,
)
from workbench.backend.core.config import (
    WorkbenchApiSettings,
    get_workbench_api_settings,
)
from workbench.backend.core.errors import ApiError
from workbench.backend.dependencies import WorkbenchServices
from workbench.backend.exceptions import (
    api_error_handler,
    domain_failure_handler,
    request_validation_error_handler,
)
from workbench.backend.failures import DomainFailure
from workbench.backend.inspection_worker import (
    InspectionWorkerLimits,
    SubprocessInspectionExecutor,
)
from workbench.backend.log_experiments import (
    LogExperimentMutationCoordinator,
)
from workbench.backend.middleware import configure_middleware
from workbench.backend.run_history import RunHistoryService
from workbench.backend.services.inspection import InspectionService
from workbench.backend.training_jobs import TrainingJobService
from workbench.backend.training_jobs.cgroups import TrainingResourceLimits
from workbench.backend.training_jobs.run_plan_adapter import (
    WorkbenchRunPlanAdapter,
)

__all__ = ["WorkbenchApiSettings", "create_app", "app"]


def create_app(
    settings: WorkbenchApiSettings | None = None,
) -> FastAPI:
    api_settings = settings or get_workbench_api_settings()
    log_experiment_mutations = LogExperimentMutationCoordinator()
    snapshot_store = FileSystemConfigSnapshotStore(
        Path(api_settings.snapshots_root),
        state_root=Path(api_settings.state_root),
    )
    config_snapshots = ConfigSnapshotService(snapshot_store)
    training_jobs: TrainingJobService
    run_history = RunHistoryService(
        logs_root=api_settings.logs_root,
        mutation_coordinator=log_experiment_mutations,
        active_log_writers=lambda: training_jobs.active_jobs(),
        state_root=Path(api_settings.state_root),
        tensorboard_request_work_bytes=(api_settings.tensorboard_request_work_bytes),
        tensorboard_cache_bytes=api_settings.tensorboard_cache_bytes,
    )
    training_jobs = TrainingJobService(
        logs_root=api_settings.logs_root,
        cancellation_mode=api_settings.training_cancellation_mode,
        mutation_coordinator=log_experiment_mutations,
        terminal_log_experiment_invalidator=run_history.invalidate_experiment,
        config_snapshots=config_snapshots,
        max_progress_record_bytes=api_settings.max_progress_record_bytes,
        tensorboard_request_work_bytes=(api_settings.tensorboard_request_work_bytes),
        tensorboard_cache_bytes=api_settings.tensorboard_cache_bytes,
        max_active_training_jobs=api_settings.max_active_training_jobs,
        training_resource_limits=TrainingResourceLimits(
            memory_bytes=api_settings.training_job_memory_limit_bytes,
            cpu_count=api_settings.training_job_cpu_limit,
            process_count=api_settings.training_job_process_limit,
        ),
    )

    api = FastAPI(
        title="Emperor Model Workbench API",
        version="1.0.0",
        strict_content_type=True,
    )
    api.add_exception_handler(ApiError, api_error_handler)
    api.add_exception_handler(DomainFailure, domain_failure_handler)
    api.add_exception_handler(
        RequestValidationError,
        request_validation_error_handler,
    )

    api.state.workbench_services = WorkbenchServices(
        settings=api_settings,
        config_snapshots=config_snapshots,
        inspection=InspectionService(
            run_history,
            executor=SubprocessInspectionExecutor(
                InspectionWorkerLimits(
                    memory_bytes=api_settings.inspection_memory_limit_bytes,
                    cpu_count=api_settings.inspection_cpu_limit,
                    timeout_seconds=api_settings.inspection_timeout_seconds,
                )
            ),
        ),
        run_history=run_history,
        training_jobs=training_jobs,
        training_run_plans=WorkbenchRunPlanAdapter(config_snapshots=config_snapshots),
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
