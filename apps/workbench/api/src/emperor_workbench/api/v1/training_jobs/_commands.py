from __future__ import annotations

from emperor_workbench.api._mutations import deterministic_mutation_resource_id
from emperor_workbench.api.v1.run_plans._mapping import (
    search_from_request,
    submitted_plan_from_request,
)
from emperor_workbench.api.v1.training_jobs._contracts import TrainingJobCreateRequest
from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.run_plans import (
    ConfigSnapshotRevision,
    MaterializeTrainingRunPlanCommand,
)
from emperor_workbench.training_jobs import CreateTrainingJobCommand


def create_training_job_command(
    request: TrainingJobCreateRequest,
    *,
    project_adapter: ProjectAdapterClient,
) -> CreateTrainingJobCommand:
    return CreateTrainingJobCommand(
        job_id=deterministic_mutation_resource_id("training-job"),
        run_plan=MaterializeTrainingRunPlanCommand(
            model=ModelPackageCatalog(project_adapter).require_id(
                request.modelType,
                request.model,
            ),
            preset=request.preset,
            presets=request.presets,
            experiment_task=request.experimentTask,
            datasets=request.datasets,
            overrides=request.overrides,
            log_folder=request.logFolder,
            monitors=request.monitors,
            search=search_from_request(request.search),
            submitted_plan=(
                submitted_plan_from_request(request.runPlan)
                if request.runPlan is not None
                else None
            ),
            snapshot_ids=request.snapshotIds,
            snapshot_revisions=tuple(
                ConfigSnapshotRevision(
                    id=revision.id,
                    semantic_revision=revision.semanticRevision,
                )
                for revision in request.snapshotRevisions
            ),
        ),
    )


__all__ = ["create_training_job_command"]
