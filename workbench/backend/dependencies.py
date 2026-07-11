"""FastAPI dependencies for app-scoped Workbench services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, cast

from fastapi import Depends, Request

from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.log_experiments import (
    LogExperimentMutationCoordinator,
)
from workbench.backend.run_history import RunHistoryService
from workbench.backend.services.config_snapshots import ConfigSnapshotService
from workbench.backend.services.inspection import InspectionService
from workbench.backend.training_jobs import TrainingJobService
from workbench.backend.training_jobs.plans import TrainingRunPlanService


@dataclass(frozen=True, slots=True)
class WorkbenchServices:
    settings: WorkbenchApiSettings
    config_snapshots: ConfigSnapshotService
    inspection: InspectionService
    run_history: RunHistoryService
    training_jobs: TrainingJobService
    training_run_plans: TrainingRunPlanService
    log_experiment_mutations: LogExperimentMutationCoordinator


async def get_workbench_services(request: Request) -> WorkbenchServices:
    return cast(WorkbenchServices, request.app.state.workbench_services)


async def get_workbench_settings(
    services: Annotated[WorkbenchServices, Depends(get_workbench_services)],
) -> WorkbenchApiSettings:
    return services.settings


async def get_config_snapshot_service(
    services: Annotated[WorkbenchServices, Depends(get_workbench_services)],
) -> ConfigSnapshotService:
    return services.config_snapshots


async def get_inspection_service(
    services: Annotated[WorkbenchServices, Depends(get_workbench_services)],
) -> InspectionService:
    return services.inspection


async def get_run_history_service(
    services: Annotated[WorkbenchServices, Depends(get_workbench_services)],
) -> RunHistoryService:
    return services.run_history


async def get_training_job_service(
    services: Annotated[WorkbenchServices, Depends(get_workbench_services)],
) -> TrainingJobService:
    return services.training_jobs


async def get_training_run_plan_service(
    services: Annotated[WorkbenchServices, Depends(get_workbench_services)],
) -> TrainingRunPlanService:
    return services.training_run_plans


__all__ = [
    "WorkbenchServices",
    "get_config_snapshot_service",
    "get_inspection_service",
    "get_run_history_service",
    "get_training_job_service",
    "get_training_run_plan_service",
    "get_workbench_services",
    "get_workbench_settings",
]
