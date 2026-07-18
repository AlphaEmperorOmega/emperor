from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from emperor_workbench.log_experiments import (
    LogExperimentMutationCoordinator,
)
from emperor_workbench.run_plans import RunPlanService
from emperor_workbench.tensorboard import MonitorData, ParameterStatus
from emperor_workbench.training_jobs._containment import (
    CgroupV2Manager,
    ProcessRunner,
    TrainingWorkerLauncher,
)
from emperor_workbench.training_jobs._records import (
    ActiveTrainingJob,
    CreateTrainingJobCommand,
    TrainingCancellationCapability,
    TrainingCancellationMode,
    TrainingJobView,
    TrainingProgressEventsPage,
    TrainingResourceLimits,
)
from emperor_workbench.training_jobs._runtime import (
    _TrainingJobRuntime,
)
from emperor_workbench.training_jobs._store import TrainingJobStore


class TrainingJobService:
    def __init__(
        self,
        *,
        mutation_coordinator: LogExperimentMutationCoordinator,
        root: Path | None = None,
        cwd: Path | None = None,
        logs_root: Path | str = "logs",
        runner: ProcessRunner | None = None,
        job_store: TrainingJobStore | None = None,
        worker_launcher: TrainingWorkerLauncher | None = None,
        cancellation_mode: TrainingCancellationMode | None = None,
        cgroup_manager: CgroupV2Manager | None = None,
        terminal_log_experiment_invalidator: Callable[[str], None] | None = None,
        run_plans: RunPlanService,
        max_progress_record_bytes: int = 1024 * 1024,
        tensorboard_request_work_bytes: int = 64 * 1024 * 1024,
        tensorboard_cache_bytes: int = 128 * 1024 * 1024,
        max_active_training_jobs: int = 2,
        training_resource_limits: TrainingResourceLimits | None = None,
    ) -> None:
        self._runtime = _TrainingJobRuntime(
            root=root,
            cwd=cwd,
            logs_root=logs_root,
            runner=runner,
            job_store=job_store,
            worker_launcher=worker_launcher,
            cancellation_mode=cancellation_mode,
            cgroup_manager=cgroup_manager,
            terminal_log_experiment_invalidator=terminal_log_experiment_invalidator,
            run_plans=run_plans,
            max_progress_record_bytes=max_progress_record_bytes,
            tensorboard_request_work_bytes=tensorboard_request_work_bytes,
            tensorboard_cache_bytes=tensorboard_cache_bytes,
            max_active_training_jobs=max_active_training_jobs,
            training_resource_limits=training_resource_limits,
        )
        self._mutation_coordinator = mutation_coordinator

    def create_job(self, command: CreateTrainingJobCommand) -> TrainingJobView:
        with self._mutation_coordinator.coordinate([command.run_plan.log_folder]):
            return self._runtime.create_job_from_command(command)

    def get_job(self, job_id: str) -> TrainingJobView:
        return self._runtime.get_job_view(job_id)

    def get_job_events(
        self,
        job_id: str,
        *,
        offset: int,
        limit: int,
    ) -> TrainingProgressEventsPage:
        return self._runtime.get_job_events_page(
            job_id,
            offset=offset,
            limit=limit,
        )

    def get_monitor_data(
        self,
        job_id: str,
        *,
        node_path: str,
        dataset: str | None,
        preset: str | None,
    ) -> MonitorData:
        return self._runtime.get_monitor_data(
            job_id,
            node_path=node_path,
            dataset=dataset,
            preset=preset,
        )

    def get_parameter_status(
        self,
        job_id: str,
        *,
        dataset: str | None,
        preset: str | None,
    ) -> ParameterStatus:
        return self._runtime.get_parameter_status(
            job_id,
            dataset=dataset,
            preset=preset,
        )

    def cancel_job(self, job_id: str) -> TrainingJobView:
        return self._runtime.cancel_job_view(job_id)

    def reconcile_job(
        self,
        job_id: str,
        *,
        action: str,
        reason: str,
    ) -> TrainingJobView:
        return self._runtime.reconcile_unknown_job_view(
            job_id,
            action=action,
            reason=reason,
        )

    def cancellation_capability(self) -> TrainingCancellationCapability:
        return self._runtime.cancellation_capability()

    def training_resource_limits_enforced(self) -> bool:
        return self._runtime.training_resource_limits_enforced()

    def active_jobs(self) -> list[ActiveTrainingJob]:
        return self._runtime.active_job_views()
