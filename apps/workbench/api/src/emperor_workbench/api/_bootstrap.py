from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from threading import Lock

from emperor_workbench.api._blocking import BlockingWorkRuntime
from emperor_workbench.api._container import WorkbenchContainer
from emperor_workbench.api._mutations import MutationExecutionRuntime
from emperor_workbench.config_snapshots import ConfigSnapshotService
from emperor_workbench.inspection import (
    InspectionService,
    InspectionWorkerLimits,
    SubprocessInspectionExecutor,
)
from emperor_workbench.log_experiments import LogExperimentMutationCoordinator
from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.run_history import (
    KnownModelPackageIdentityResolver,
    RunHistoryService,
)
from emperor_workbench.run_plans import RunPlanService
from emperor_workbench.settings import WorkbenchApiSettings
from emperor_workbench.training_jobs import (
    ActiveTrainingJob,
    TrainingJobService,
    TrainingResourceLimits,
)


def _run_history_model_identity_resolver(
    model_packages: ModelPackageCatalog,
) -> KnownModelPackageIdentityResolver:
    model_ids: Mapping[str, str] | None = None
    lookup_lock = Lock()

    def resolve(model_token: str) -> str | None:
        nonlocal model_ids
        if model_ids is None:
            with lookup_lock:
                if model_ids is None:
                    model_ids = model_packages.identity_lookup()
        return model_ids.get(model_token)

    return resolve


def acquire_container(
    settings: WorkbenchApiSettings,
    *,
    project_adapter: ProjectAdapterClient | None,
    training_jobs: TrainingJobService | None = None,
    log_experiment_mutations: LogExperimentMutationCoordinator | None = None,
) -> WorkbenchContainer:
    """Acquire every app-scoped resource inside the lifespan phase."""

    project_adapter_client = project_adapter or ProjectAdapterClient()
    blocking_work: BlockingWorkRuntime | None = None
    mutation_execution: MutationExecutionRuntime | None = None
    try:
        model_packages = ModelPackageCatalog(project_adapter_client)
        mutation_coordinator = (
            log_experiment_mutations or LogExperimentMutationCoordinator()
        )
        config_snapshots = ConfigSnapshotService.from_filesystem(
            Path(settings.snapshots_root),
            model_packages=model_packages,
            state_root=Path(settings.state_root),
        )
        run_plans = RunPlanService(
            model_packages=model_packages,
            config_snapshots=config_snapshots,
        )
        training_job_service = training_jobs

        def active_log_writers() -> list[ActiveTrainingJob]:
            if training_job_service is None:
                return []
            return training_job_service.active_jobs()

        run_history = RunHistoryService(
            logs_root=settings.logs_root,
            mutation_coordinator=mutation_coordinator,
            active_log_writers=active_log_writers,
            model_identity_resolver=_run_history_model_identity_resolver(
                model_packages
            ),
            state_root=Path(settings.state_root),
            tensorboard_request_work_bytes=settings.tensorboard_request_work_bytes,
            tensorboard_cache_bytes=settings.tensorboard_cache_bytes,
        )
        if training_job_service is None:
            training_job_service = TrainingJobService(
                logs_root=settings.logs_root,
                cancellation_mode=settings.training_cancellation_mode,
                mutation_coordinator=mutation_coordinator,
                terminal_log_experiment_invalidator=run_history.invalidate_experiment,
                run_plans=run_plans,
                max_progress_record_bytes=settings.max_progress_record_bytes,
                tensorboard_request_work_bytes=(
                    settings.tensorboard_request_work_bytes
                ),
                tensorboard_cache_bytes=settings.tensorboard_cache_bytes,
                max_active_training_jobs=settings.max_active_training_jobs,
                training_resource_limits=TrainingResourceLimits(
                    memory_bytes=settings.training_job_memory_limit_bytes,
                    cpu_count=settings.training_job_cpu_limit,
                    process_count=settings.training_job_process_limit,
                ),
            )
        blocking_work = BlockingWorkRuntime()
        mutation_execution = MutationExecutionRuntime(Path(settings.state_root))
        return WorkbenchContainer(
            settings=settings,
            project_adapter=project_adapter_client,
            config_snapshots=config_snapshots,
            inspection=InspectionService(
                SubprocessInspectionExecutor(
                    InspectionWorkerLimits(
                        memory_bytes=settings.inspection_memory_limit_bytes,
                        cpu_count=settings.inspection_cpu_limit,
                        timeout_seconds=settings.inspection_timeout_seconds,
                    )
                ),
                historical_source=run_history,
            ),
            run_history=run_history,
            training_jobs=training_job_service,
            training_run_plans=run_plans,
            log_experiment_mutations=mutation_coordinator,
            blocking_work=blocking_work,
            mutation_execution=mutation_execution,
        )
    except BaseException:
        if mutation_execution is not None:
            mutation_execution.close_executor()
        if blocking_work is not None:
            blocking_work.close()
        project_adapter_client.close()
        raise


__all__ = ["acquire_container"]
