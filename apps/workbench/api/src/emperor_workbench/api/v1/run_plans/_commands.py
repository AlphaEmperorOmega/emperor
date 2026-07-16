from __future__ import annotations

from emperor_workbench.api.v1.run_plans._contracts import (
    TrainingRunPlanCreateRequest,
)
from emperor_workbench.api.v1.run_plans._mapping import search_from_request
from emperor_workbench.log_experiments import is_valid_log_experiment_name
from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.run_plans import CreateTrainingRunPlanCommand


def create_run_plan_command(
    request: TrainingRunPlanCreateRequest,
    *,
    project_adapter: ProjectAdapterClient,
) -> CreateTrainingRunPlanCommand:
    return CreateTrainingRunPlanCommand(
        model=ModelPackageCatalog(project_adapter).require_id(
            request.modelType,
            request.model,
        ),
        preset=request.preset,
        presets=request.presets,
        experiment_task=request.experimentTask,
        datasets=request.datasets,
        overrides=request.overrides,
        log_folder=(
            request.logFolder
            if request.logFolder and is_valid_log_experiment_name(request.logFolder)
            else ""
        ),
        monitors=request.monitors,
        search=search_from_request(request.search),
        snapshot_ids=request.snapshotIds,
    )


__all__ = ["create_run_plan_command"]
