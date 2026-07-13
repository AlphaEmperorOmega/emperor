from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from workbench.backend.config_snapshots import ConfigSnapshotService
from workbench.backend.log_experiments import (
    LogExperimentMutationCoordinator,
)
from workbench.backend.training_jobs.cgroups import (
    CgroupV2Manager,
    TrainingCancellationCapability,
    TrainingCancellationMode,
    TrainingResourceLimits,
)
from workbench.backend.training_jobs.contracts import (
    ActiveTrainingJob,
    CreateTrainingJobCommand,
    TrainingJobView,
    TrainingProgressEventsPage,
)
from workbench.backend.training_jobs.launcher import (
    ProcessRunner,
    TrainingWorkerLauncher,
)
from workbench.backend.training_jobs.runtime import (
    _TrainingJobRuntime,
)
from workbench.backend.training_jobs.store import TrainingJobStore


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
        config_snapshots: ConfigSnapshotService | None = None,
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
            terminal_log_experiment_invalidator=(terminal_log_experiment_invalidator),
            config_snapshots=config_snapshots,
            max_progress_record_bytes=max_progress_record_bytes,
            tensorboard_request_work_bytes=tensorboard_request_work_bytes,
            tensorboard_cache_bytes=tensorboard_cache_bytes,
            max_active_training_jobs=max_active_training_jobs,
            training_resource_limits=training_resource_limits,
        )
        self._mutation_coordinator = mutation_coordinator

    @classmethod
    def _from_runtime(
        cls,
        runtime: _TrainingJobRuntime,
        *,
        mutation_coordinator: LogExperimentMutationCoordinator,
    ) -> TrainingJobService:
        """Compose the public Interface around an existing private runtime seam."""

        service = cls.__new__(cls)
        service._runtime = runtime
        service._mutation_coordinator = mutation_coordinator
        return service

    def create_job(self, command: CreateTrainingJobCommand) -> TrainingJobView:
        with self._mutation_coordinator.coordinate([command.log_folder]):
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
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
