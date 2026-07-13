from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, RLock
from typing import Any

from workbench.backend.config_snapshots import ConfigSnapshotService
from workbench.backend.failures import FailureKind
from workbench.backend.log_experiments import (
    LogExperimentFailure,
    validate_log_experiment_name,
)
from workbench.backend.model_identity import (
    model_identity_payload_from_id,
    normalize_preset_token,
)
from workbench.backend.mutation_context import deterministic_mutation_resource_id
from workbench.backend.tensorboard.events import TensorBoardEventCache
from workbench.backend.tensorboard.readers import (
    TensorBoardMonitorReader,
    TensorBoardParameterStatusReader,
)
from workbench.backend.training_jobs.cgroups import (
    CgroupV2Manager,
    StrictCancellationUnavailable,
    TrainingCancellationCapability,
    TrainingCancellationMode,
    TrainingResourceLimits,
)
from workbench.backend.training_jobs.contracts import (
    ActiveTrainingJob,
    CreateTrainingJobCommand,
    TrainingJobView,
    TrainingProgressEventsPage,
    TrainingRunPlanView,
)
from workbench.backend.training_jobs.errors import TrainingJobFailure
from workbench.backend.training_jobs.launcher import (
    PersistedCgroupProcessHandle,
    ProcessHandle,
    ProcessRunner,
    TrainingProcessContainment,
    TrainingWorkerLauncher,
    ensure_private_directory,
)
from workbench.backend.training_jobs.lifecycle import (
    terminal_exit_code,
    terminal_status_from_event,
)
from workbench.backend.training_jobs.monitoring import TrainingMonitorLocator
from workbench.backend.training_jobs.progress import (
    TrainingProgressSnapshot,
    TrainingProgressStore,
)
from workbench.backend.training_jobs.projection import (
    TrainingJobLiveProjection,
    TrainingLiveProjectionCache,
)
from workbench.backend.training_jobs.run_plan_adapter import (
    WorkbenchRunPlanAdapter,
    encode_persisted_run_plan,
)
from workbench.backend.training_jobs.snapshot import TrainingJobProjector
from workbench.backend.training_jobs.status import (
    is_active_job_status,
    is_live_process_job_status,
    is_terminal_job_status,
)
from workbench.backend.training_jobs.store import (
    FileSystemTrainingJobStore,
    TrainingJobRecord,
    TrainingJobStore,
)

# Grace period for a cancelled worker to exit after SIGTERM (and again after
# the SIGKILL escalation) before cancellation is reported as failed.
CANCEL_REAP_GRACE_SECONDS = 5.0
LOGGER = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class _TrainingJobRuntime:
    def __init__(
        self,
        *,
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
        self.root = ensure_private_directory(
            root or Path(tempfile.gettempdir()) / "emperor-workbench-training"
        ).resolve()
        self.cwd = cwd or Path.cwd()
        self.logs_root = Path(logs_root).resolve()
        if worker_launcher is not None and any(
            value is not None for value in (runner, cancellation_mode, cgroup_manager)
        ):
            raise ValueError(
                "worker_launcher cannot be combined with runner, "
                "cancellation_mode, or cgroup_manager"
            )
        if worker_launcher is not None:
            self.worker_launcher = worker_launcher
        else:
            self.worker_launcher = TrainingWorkerLauncher(
                cwd=self.cwd,
                runner=runner,
                cancellation_mode=cancellation_mode,
                cgroup_manager=cgroup_manager,
                resource_limits=training_resource_limits,
            )
        tensorboard_cache = TensorBoardEventCache(
            {
                "training_monitor_payload": 128,
                "training_parameter_status_payload": 128,
            },
            max_bytes=tensorboard_cache_bytes,
        )
        self.monitor_reader = TensorBoardMonitorReader(
            max_event_bytes=tensorboard_request_work_bytes,
            event_cache=tensorboard_cache,
            cache_name="training_monitor_payload",
        )
        self.parameter_status_reader = TensorBoardParameterStatusReader(
            max_event_bytes=tensorboard_request_work_bytes,
            event_cache=tensorboard_cache,
            cache_name="training_parameter_status_payload",
        )
        self.run_plan_adapter = WorkbenchRunPlanAdapter(
            config_snapshots=config_snapshots
        )
        self.job_store = job_store or FileSystemTrainingJobStore(self.root)
        self.progress_store = TrainingProgressStore(
            max_record_bytes=max_progress_record_bytes
        )
        self.monitor_locator = TrainingMonitorLocator()
        self.job_projector = TrainingJobProjector()
        self._state_lock = RLock()
        self._launch_admission_lock = Lock()
        self._max_active_training_jobs = max(1, int(max_active_training_jobs))
        self._cancel_reap_grace_seconds = CANCEL_REAP_GRACE_SECONDS
        self._processes: dict[str, ProcessHandle] = {}
        self._released_job_ids: set[str] = set()
        self._terminal_invalidated_job_ids: set[str] = set()
        self._terminal_log_experiment_invalidator = (
            terminal_log_experiment_invalidator or (lambda _experiment: None)
        )
        self._live_projection_cache = TrainingLiveProjectionCache()

    def cancellation_capability(self) -> TrainingCancellationCapability:
        return self.worker_launcher.cancellation_capability()

    def training_resource_limits_enforced(self) -> bool:
        return self.worker_launcher.training_resource_limits_enforced()

    def create_job_from_command(
        self,
        command: CreateTrainingJobCommand,
    ) -> TrainingJobView:
        with self._launch_admission_lock:
            return self._create_job_view(command)

    def _create_job_view(
        self,
        command: CreateTrainingJobCommand,
    ) -> TrainingJobView:
        job_id = deterministic_mutation_resource_id("training-job") or uuid.uuid4().hex
        if self.job_store.get(job_id) is not None:
            return self.get_job_view(job_id)
        active_count = len(self.active_job_views())
        if active_count >= self._max_active_training_jobs:
            raise TrainingJobFailure(
                "Training Job admission is unavailable while "
                f"{self._max_active_training_jobs} active Training Jobs are "
                "already running.",
                kind=FailureKind.UNAVAILABLE,
            )

        try:
            validated_log_folder = validate_log_experiment_name(command.log_folder)
        except LogExperimentFailure as exc:
            raise TrainingJobFailure(exc.detail) from exc
        materialized = self.run_plan_adapter.materialize_training_job(
            command,
            validated_log_folder=validated_log_folder,
        )
        materialized_run_plan = materialized.plan
        selected_monitors = list(materialized.monitors)
        planned_run_count = materialized_run_plan.summary.total_runs

        self._ensure_job_log_folder(validated_log_folder)

        job_root = self._create_job_root(job_id)
        payload = self._build_worker_payload(
            job_id=job_id,
            selected_monitors=selected_monitors,
            planned_run_count=planned_run_count,
            materialized_run_plan=materialized_run_plan,
        )
        try:
            launch = self.worker_launcher.launch(
                job_root=job_root,
                payload=payload,
                logs_root=self.logs_root,
            )
        except StrictCancellationUnavailable as exc:
            raise TrainingJobFailure(str(exc)) from exc
        try:
            job = self._register_job(
                job_id=job_id,
                model=command.model,
                materialized_run_plan=materialized_run_plan,
                selected_monitors=selected_monitors,
                planned_run_count=planned_run_count,
                validated_log_folder=validated_log_folder,
                command=launch.command,
                job_root=job_root,
                process=launch.process,
                containment=launch.containment,
            )
            self._write_event(
                job,
                {
                    "type": "job_started",
                    "status": "running",
                    **model_identity_payload_from_id(command.model),
                    "preset": job.preset,
                    "presets": job.presets,
                    "experimentTask": job.experiment_task,
                    "runTotal": planned_run_count,
                },
            )
            return self.get_job_view(job_id)
        except Exception:
            with self._state_lock:
                self._processes.pop(job_id, None)
            try:
                self._terminate_and_reap_process(job_id, launch.process)
            except Exception:
                pass
            raise

    def _ensure_job_log_folder(self, validated_log_folder: str) -> None:
        # Validate the top-level folder immediately before the worker starts.
        log_folder_path = self.logs_root / validated_log_folder
        if log_folder_path.is_symlink():
            raise TrainingJobFailure(
                f"Refusing to write symlink log experiment: {validated_log_folder}"
            )
        log_folder_path.mkdir(parents=True, exist_ok=True)

    def _create_job_root(self, job_id: str) -> Path:
        job_root = self.root / job_id
        return ensure_private_directory(job_root)

    def _build_worker_payload(
        self,
        *,
        job_id: str,
        selected_monitors: list[str],
        planned_run_count: int,
        materialized_run_plan: TrainingRunPlanView,
    ) -> dict[str, Any]:
        return {
            "id": job_id,
            "plannedRunCount": planned_run_count,
            "runPlan": encode_persisted_run_plan(materialized_run_plan),
            "monitors": selected_monitors,
        }

    def _register_job(
        self,
        *,
        job_id: str,
        model: str,
        materialized_run_plan: TrainingRunPlanView,
        selected_monitors: list[str],
        planned_run_count: int,
        validated_log_folder: str,
        command: list[str],
        job_root: Path,
        process: ProcessHandle,
        containment: TrainingProcessContainment,
    ) -> TrainingJobRecord:
        job = TrainingJobRecord(
            id=job_id,
            model=model,
            preset=materialized_run_plan.preset,
            presets=list(materialized_run_plan.presets),
            experiment_task=materialized_run_plan.experiment_task,
            datasets=list(materialized_run_plan.datasets),
            overrides=dict(materialized_run_plan.overrides),
            search=materialized_run_plan.search,
            planned_run_count=planned_run_count,
            run_plan=materialized_run_plan,
            monitors=selected_monitors,
            log_folder=validated_log_folder,
            command=command,
            root=job_root,
            pid=process.pid,
            cancellation_mode=containment.mode,
            worker_pid=containment.worker_pid,
            process_group_id=containment.process_group_id,
            cgroup_path=containment.cgroup_path,
            windows_job_name=containment.windows_job_name,
        )
        self.job_store.save(job)
        with self._state_lock:
            self._processes[job_id] = process
        return job

    def get_job_view(self, job_id: str) -> TrainingJobView:
        job = self._get_job_record(job_id)
        snapshot = self._progress_updates(job)
        self._refresh(job, snapshot=snapshot)
        payload = self._snapshot(job, snapshot=snapshot)
        self._release_terminal_resources(job)
        return payload

    def cancel_job_view(self, job_id: str) -> TrainingJobView:
        terminal_transition: tuple[TrainingJobRecord, str] | None = None
        with self._state_lock:
            job = self._get_job_record(job_id)
            snapshot = self._progress_updates(job)
            self._refresh(job, snapshot=snapshot)
            if is_terminal_job_status(job.status):
                payload = self._snapshot(job, snapshot=snapshot)
            else:
                previous_status = job.status
                process = self._process_for_job(job)
                if process is None and is_active_job_status(job.status):
                    raise TrainingJobFailure(
                        f"Training job '{job_id}' has no live process handle."
                    )
                reaped_exit_code: int | None = None
                process_exit_code = process.poll() if process is not None else None
                if process is not None and (
                    process_exit_code is None
                    or is_live_process_job_status(job.status)
                    or self._job_has_live_containment(job)
                ):
                    reaped_exit_code = self._terminate_and_reap_process(
                        job_id,
                        process,
                    )
                job.status = "cancelled"
                if reaped_exit_code is not None:
                    job.exit_code = reaped_exit_code
                job.updated_at = _now()
                self.job_store.save(job)
                if (
                    terminal_status_from_event(snapshot.latest_terminal_event or {})
                    != "cancelled"
                ):
                    self._write_event(
                        job,
                        {"type": "cancelled", "status": "cancelled"},
                    )
                snapshot = self._progress_updates(job)
                payload = self._snapshot(job, snapshot=snapshot)
                terminal_transition = (job, previous_status)
        if terminal_transition is not None:
            self._notify_terminal_transition(*terminal_transition)
        self._release_terminal_resources(job)
        return payload

    def reconcile_unknown_job_view(
        self,
        job_id: str,
        *,
        action: str,
        reason: str,
    ) -> TrainingJobView:
        normalized_reason = reason.strip()
        if action != "mark-failed":
            raise TrainingJobFailure("Unknown Training Job reconciliation action.")
        if not normalized_reason or len(normalized_reason) > 500:
            raise TrainingJobFailure(
                "Training Job reconciliation reason must contain 1 to 500 characters."
            )

        with self._state_lock:
            job = self._get_job_record(job_id)
            snapshot = self._progress_summary(job)
            self._refresh(job, snapshot=snapshot)
            if job.status != "unknown":
                raise TrainingJobFailure(
                    f"Training job '{job_id}' is '{job.status}', not unknown.",
                    kind=FailureKind.CONFLICT,
                )
            if self._has_live_reconciliation_evidence(job):
                raise TrainingJobFailure(
                    f"Training job '{job_id}' still has live process evidence.",
                    kind=FailureKind.CONFLICT,
                )
            previous_status = job.status
            self._write_event(
                job,
                {
                    "type": "operator_reconciled_failed",
                    "status": "failed",
                    "action": action,
                    "reason": normalized_reason,
                    "exitCode": None,
                },
            )
            job.status = "failed"
            job.exit_code = None
            job.updated_at = _now()
            self.job_store.save(job)
            snapshot = self._progress_updates(job)
            payload = self._snapshot(job, snapshot=snapshot)

        self._notify_terminal_transition(job, previous_status)
        self._release_terminal_resources(job)
        return payload

    def active_job_views(self) -> list[ActiveTrainingJob]:
        active: list[ActiveTrainingJob] = []
        for job in self.job_store.list():
            snapshot = self._progress_summary(job)
            self._refresh(job, snapshot=snapshot)
            if not is_active_job_status(job.status):
                self._release_terminal_resources(job)
                continue
            active.append(
                ActiveTrainingJob(
                    id=job.id,
                    status=job.status,
                    log_folder=job.log_folder,
                )
            )
        return active

    def get_job_events_page(
        self,
        job_id: str,
        *,
        offset: int = 0,
        limit: int = 500,
    ) -> TrainingProgressEventsPage:
        job = self._get_job_record(job_id)
        snapshot = self._progress_summary(job)
        self._refresh(job, snapshot=snapshot)
        safe_offset = max(0, offset)
        safe_limit = min(5000, max(1, limit))
        page = self.progress_store.read_page(
            job,
            offset=safe_offset,
            limit=safe_limit,
        )
        next_offset = safe_offset + len(page.events)
        payload = TrainingProgressEventsPage(
            job_id=job.id,
            offset=safe_offset,
            limit=safe_limit,
            total_count=page.total_count,
            next_offset=(next_offset if next_offset < page.total_count else None),
            events=page.events,
        )
        self._release_terminal_resources(job)
        return payload

    def get_monitor_data(
        self,
        job_id: str,
        *,
        node_path: str,
        dataset: str | None = None,
        preset: str | None = None,
    ) -> dict[str, Any]:
        job = self._get_job_record(job_id)
        if dataset is not None and dataset not in job.datasets:
            raise TrainingJobFailure(
                f"Unknown dataset '{dataset}' for training job '{job_id}'."
            )
        if preset is not None and not self.monitor_locator.preset_in_job(job, preset):
            raise TrainingJobFailure(
                f"Unknown preset '{preset}' for training job '{job_id}'."
            )
        snapshot = self._progress_summary(job)
        self._refresh(job, snapshot=snapshot)
        reader_log_dir, event_log_dir = self._trusted_monitor_log_dir(
            job,
            events=snapshot.monitor_events,
            dataset=dataset,
            preset=preset,
        )
        try:
            data = self.monitor_reader.read(
                job_id=job.id,
                node_path=node_path,
                dataset=dataset,
                log_dir=reader_log_dir,
            )
            data["logDir"] = event_log_dir
            data["preset"] = normalize_preset_token(preset) if preset else None
            return data
        finally:
            self._release_terminal_resources(job)

    def get_parameter_status(
        self,
        job_id: str,
        *,
        dataset: str | None = None,
        preset: str | None = None,
    ) -> dict[str, Any]:
        job = self._get_job_record(job_id)
        if dataset is not None and dataset not in job.datasets:
            raise TrainingJobFailure(
                f"Unknown dataset '{dataset}' for training job '{job_id}'."
            )
        if preset is not None and not self.monitor_locator.preset_in_job(job, preset):
            raise TrainingJobFailure(
                f"Unknown preset '{preset}' for training job '{job_id}'."
            )
        snapshot = self._progress_summary(job)
        self._refresh(job, snapshot=snapshot)
        reader_log_dir, event_log_dir = self._trusted_monitor_log_dir(
            job,
            events=snapshot.monitor_events,
            dataset=dataset,
            preset=preset,
        )
        try:
            data = self.parameter_status_reader.read(
                source_id=job.id,
                preset=normalize_preset_token(preset) if preset else None,
                dataset=dataset,
                log_dir=reader_log_dir,
            )
            data["logDir"] = event_log_dir
            return data
        finally:
            self._release_terminal_resources(job)

    def _trusted_monitor_log_dir(
        self,
        job: TrainingJobRecord,
        *,
        events: list[dict[str, Any]],
        dataset: str | None,
        preset: str | None,
    ) -> tuple[str | None, str | None]:
        event_log_dir = self.monitor_locator.log_dir_for_monitor_data(
            events=events,
            dataset=dataset,
            preset=preset,
        )
        if event_log_dir is None:
            return None, None

        try:
            log_folder = validate_log_experiment_name(job.log_folder)
            experiment_path = self.logs_root / log_folder
            if experiment_path.is_symlink():
                raise ValueError("symlink Log Experiment")
            resolved_logs_root = self.logs_root.resolve()
            resolved_experiment = experiment_path.resolve()
            resolved_experiment.relative_to(resolved_logs_root)

            raw_candidate = Path(event_log_dir)
            candidates = [raw_candidate.resolve()]
            if not raw_candidate.is_absolute():
                candidates.append((resolved_logs_root / raw_candidate).resolve())
            for resolved_candidate in candidates:
                try:
                    resolved_candidate.relative_to(resolved_experiment)
                except ValueError:
                    continue
                return str(resolved_candidate), event_log_dir
        except (TrainingJobFailure, OSError, ValueError):
            pass
        raise TrainingJobFailure(
            "Training monitor log directory is outside this Training Job's "
            "Log Experiment."
        )

    def _get_job_record(self, job_id: str) -> TrainingJobRecord:
        job = self.job_store.get(job_id)
        if job is None:
            raise TrainingJobFailure(f"Unknown training job '{job_id}'.")
        return job

    def _terminate_and_reap_process(
        self,
        job_id: str,
        process: ProcessHandle,
    ) -> int | None:
        already_exited_code = process.poll()
        if already_exited_code is not None:
            return already_exited_code
        process.terminate()
        try:
            return process.wait(timeout=self._cancel_reap_grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
        try:
            return process.wait(timeout=self._cancel_reap_grace_seconds)
        except subprocess.TimeoutExpired as exc:
            raise TrainingJobFailure(
                f"Training job '{job_id}' process survived terminate and kill."
            ) from exc

    def _process_for_job(
        self,
        job: TrainingJobRecord,
    ) -> ProcessHandle | None:
        with self._state_lock:
            process = self._processes.get(job.id)
        if process is not None:
            return process
        process = self._rehydrate_process_handle(job)
        if process is not None:
            with self._state_lock:
                cached_process = self._processes.get(job.id)
                if cached_process is not None:
                    return cached_process
                self._processes[job.id] = process
        return process

    def _rehydrate_process_handle(
        self,
        job: TrainingJobRecord,
    ) -> ProcessHandle | None:
        if job.cancellation_mode == "strict-cgroup":
            cgroup = self.worker_launcher.recover_job_cgroup(
                job.id,
                persisted_mode=job.cancellation_mode,
            )
            if cgroup is None or not cgroup.has_processes():
                return None
            return PersistedCgroupProcessHandle(
                pid=job.worker_pid or job.pid,
                cgroup=cgroup,
            )
        if job.cancellation_mode == "windows-job-object":
            windows_job = self.worker_launcher.recover_windows_job(job.windows_job_name)
            if windows_job is None or not windows_job.has_processes():
                return None
            from workbench.backend.windows_jobs import (
                PersistedWindowsJobProcessHandle,
            )

            return PersistedWindowsJobProcessHandle(
                pid=job.worker_pid or job.pid,
                job=windows_job,
            )
        return None

    def _job_has_live_containment(self, job: TrainingJobRecord) -> bool:
        with self._state_lock:
            process = self._processes.get(job.id)
        if process is not None and process.poll() is None:
            return True
        if job.cancellation_mode == "strict-cgroup":
            cgroup = self.worker_launcher.recover_job_cgroup(
                job.id,
                persisted_mode=job.cancellation_mode,
            )
            return bool(cgroup and cgroup.has_processes())
        if job.cancellation_mode == "windows-job-object":
            windows_job = self.worker_launcher.recover_windows_job(job.windows_job_name)
            return bool(windows_job and windows_job.has_processes())
        return False

    @staticmethod
    def _process_identity_is_live(pid: int | None, *, group: bool = False) -> bool:
        if pid is None or pid <= 0:
            return False
        try:
            if group:
                if os.name != "posix":
                    return False
                os.killpg(pid, 0)
            else:
                os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _has_live_reconciliation_evidence(self, job: TrainingJobRecord) -> bool:
        process = self._process_for_job(job)
        if process is not None:
            try:
                if process.poll() is None:
                    return True
            except Exception:
                return True
        if self._job_has_live_containment(job):
            return True
        return any(
            (
                self._process_identity_is_live(job.pid),
                self._process_identity_is_live(job.worker_pid),
                self._process_identity_is_live(job.process_group_id, group=True),
            )
        )

    def _release_terminal_resources(self, job: TrainingJobRecord) -> None:
        if not is_terminal_job_status(job.status):
            return
        with self._state_lock:
            if job.id in self._released_job_ids:
                return
            process = self._processes.get(job.id)
            if process is not None:
                if process.poll() is None:
                    return
                try:
                    process.wait(timeout=0)
                except subprocess.TimeoutExpired:
                    return
            cgroup = self.worker_launcher.recover_job_cgroup(
                job.id,
                persisted_mode=job.cancellation_mode,
            )
            if cgroup is not None:
                if cgroup.has_processes():
                    return
                cgroup.cleanup_empty()
            windows_job = self.worker_launcher.recover_windows_job(job.windows_job_name)
            if windows_job is not None:
                if windows_job.has_processes():
                    return
                windows_job.close()
            if self._processes.get(job.id) is process:
                self._processes.pop(job.id, None)
            self._released_job_ids.add(job.id)

    def _notify_terminal_transition(
        self,
        job: TrainingJobRecord,
        previous_status: str,
    ) -> None:
        if is_terminal_job_status(previous_status) or not is_terminal_job_status(
            job.status
        ):
            return
        with self._state_lock:
            if job.id in self._terminal_invalidated_job_ids:
                return
            self._terminal_invalidated_job_ids.add(job.id)
        try:
            self._terminal_log_experiment_invalidator(job.log_folder)
        except Exception:
            LOGGER.exception(
                "Training Job terminal invalidation failed",
                extra={"job_id": job.id, "log_experiment": job.log_folder},
            )

    def _refresh(
        self,
        job: TrainingJobRecord,
        *,
        snapshot: TrainingProgressSnapshot | None = None,
    ) -> None:
        transitioned_from: str | None = None
        with self._state_lock:
            original_state = (job.status, job.exit_code, job.updated_at)
            process = self._process_for_job(job)
            exit_code: int | None = None
            containment_live = False
            exit_code_authoritative = False
            if isinstance(process, PersistedCgroupProcessHandle):
                containment_live = process.has_live_containment()
                if not containment_live:
                    if self._processes.get(job.id) is process:
                        self._processes.pop(job.id, None)
                    process.cgroup.cleanup_empty()
                    process = None
            elif process is not None:
                exit_code = process.poll()
                containment_live = exit_code is None
                exit_code_authoritative = exit_code is not None
            snapshot = snapshot or self._progress_summary(job)
            latest_terminal = snapshot.latest_terminal_event
            terminal_status = (
                terminal_status_from_event(latest_terminal)
                if latest_terminal is not None
                else None
            )
            if containment_live:
                if job.status != "running" or job.exit_code is not None:
                    job.status = "running"
                    job.exit_code = None
                    job.updated_at = _now()
            elif is_terminal_job_status(job.status):
                pass
            elif latest_terminal is not None and terminal_status is not None:
                job.status = terminal_status
                job.exit_code = terminal_exit_code(
                    terminal_status,
                    latest_terminal,
                    job.exit_code,
                )
                job.updated_at = str(latest_terminal.get("timestamp") or _now())
            elif exit_code_authoritative and exit_code is not None:
                job.exit_code = exit_code
                job.status = "completed" if exit_code == 0 else "failed"
                job.updated_at = _now()
            elif job.status != "unknown" or job.exit_code is not None:
                job.status = "unknown"
                job.exit_code = None
                job.updated_at = _now()
            if original_state != (job.status, job.exit_code, job.updated_at):
                self.job_store.save(job)
                if not is_terminal_job_status(
                    original_state[0]
                ) and is_terminal_job_status(job.status):
                    transitioned_from = original_state[0]
        if transitioned_from is not None:
            self._notify_terminal_transition(job, transitioned_from)

    def _progress_updates(
        self,
        job: TrainingJobRecord,
    ) -> TrainingProgressSnapshot:
        # Lock order: Training Job state, projection cache, progress store.
        with self._state_lock:
            return self._live_projection_cache.consume_progress(
                job,
                self.progress_store,
            )

    def _progress_summary(
        self,
        job: TrainingJobRecord,
    ) -> TrainingProgressSnapshot:
        # Lock order: Training Job state, then the private progress-store lock.
        with self._state_lock:
            return self.progress_store.read_summary(job)

    def _write_event(
        self,
        job: TrainingJobRecord,
        event: dict[str, Any],
    ) -> None:
        self.progress_store.append_event(job, event)

    def _snapshot(
        self,
        job: TrainingJobRecord,
        *,
        snapshot: TrainingProgressSnapshot | None = None,
    ) -> TrainingJobView:
        snapshot = snapshot or self._progress_updates(job)
        return self.job_projector.project_snapshot(
            job,
            self._live_projection(job, snapshot),
        )

    def _live_projection(
        self,
        job: TrainingJobRecord,
        snapshot: TrainingProgressSnapshot,
    ) -> TrainingJobLiveProjection:
        # Lock order: Training Job state, then the private projection-cache lock.
        with self._state_lock:
            return self._live_projection_cache.project(
                job,
                snapshot,
            )
