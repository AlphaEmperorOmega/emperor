from __future__ import annotations

import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from models.catalog import model_identity_payload_from_id

from viewer.backend.inspector.discovery import (
    dataset_name,
    resolve_model_monitors,
)
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.job_store import (
    FileSystemTrainingJobStore,
    InMemoryTrainingJobStore,
    TrainingJobRecord,
    TrainingJobStore,
)
from viewer.backend.log_runs import validate_log_experiment_name
from viewer.backend.monitor_data import (
    TensorBoardMonitorReader,
    TensorBoardParameterStatusReader,
)
from viewer.backend.runtime.job_status import (
    is_active_job_status,
    is_live_process_job_status,
    is_terminal_job_status,
)
from viewer.backend.training_cgroups import (
    CgroupV2Manager,
    StrictCancellationUnavailable,
    TrainingCancellationMode,
)
from viewer.backend.training_job_lifecycle import (
    latest_terminal_event,
    terminal_exit_code,
    terminal_status_from_event,
)
from viewer.backend.training_job_projector import (
    TrainingJobLiveProjection,
    TrainingJobProjector,
)
from viewer.backend.training_live_projection import TrainingLiveProjectionCache
from viewer.backend.training_monitor_locator import (
    TrainingMonitorLocator,
    normalize_preset_token,
)
from viewer.backend.training_progress_store import (
    TrainingProgressSnapshot,
    TrainingProgressStore,
)
from viewer.backend.training_run_plans import (
    SelectedTrainingInputs,
    TrainingRunPlanBuilder,
)
from viewer.backend.training_worker_launcher import (
    PersistedCgroupProcessHandle,
    ProcessHandle,
    ProcessRunner,
    SubprocessRunner,
    TrainingProcessContainment,
    TrainingWorkerLauncher,
)

# Grace period for a cancelled worker to exit after SIGTERM (and again after
# the SIGKILL escalation) before cancellation is reported as failed.
CANCEL_REAP_GRACE_SECONDS = 5.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


TrainingJob = TrainingJobRecord

__all__ = [
    "ProcessHandle",
    "ProcessRunner",
    "SubprocessRunner",
    "TrainingJob",
    "TrainingJobManager",
]


class TrainingJobManager:
    def __init__(
        self,
        *,
        root: Path | None = None,
        cwd: Path | None = None,
        logs_root: Path | str = "logs",
        runner: ProcessRunner | None = None,
        monitor_reader: TensorBoardMonitorReader | None = None,
        parameter_status_reader: TensorBoardParameterStatusReader | None = None,
        run_plan_builder: TrainingRunPlanBuilder | None = None,
        job_store: TrainingJobStore | None = None,
        progress_store: TrainingProgressStore | None = None,
        worker_launcher: TrainingWorkerLauncher | None = None,
        job_projector: TrainingJobProjector | None = None,
        monitor_locator: TrainingMonitorLocator | None = None,
        cancellation_mode: TrainingCancellationMode | None = None,
        cgroup_manager: CgroupV2Manager | None = None,
    ) -> None:
        self.root = root or Path("/tmp/emperor-viewer-training")
        self.cwd = cwd or Path.cwd()
        self.logs_root = Path(logs_root)
        self.cgroup_manager = cgroup_manager or CgroupV2Manager()
        self.worker_launcher = worker_launcher or TrainingWorkerLauncher(
            cwd=self.cwd,
            runner=runner,
            cancellation_mode=cancellation_mode,
            cgroup_manager=self.cgroup_manager,
        )
        self.runner = self.worker_launcher.runner
        self.monitor_reader = monitor_reader or TensorBoardMonitorReader()
        self.parameter_status_reader = (
            parameter_status_reader or TensorBoardParameterStatusReader()
        )
        self.run_plan_builder = run_plan_builder or TrainingRunPlanBuilder()
        self.job_store = job_store or FileSystemTrainingJobStore(self.root)
        self.progress_store = progress_store or TrainingProgressStore()
        self.monitor_locator = monitor_locator or TrainingMonitorLocator()
        self.job_projector = job_projector or TrainingJobProjector(
            self.monitor_locator
        )
        self._state_lock = RLock()
        self._processes: dict[str, ProcessHandle] = {}
        self._live_projection_cache = TrainingLiveProjectionCache()

    @property
    def jobs(self) -> dict[str, TrainingJob]:
        if isinstance(
            self.job_store,
            (FileSystemTrainingJobStore, InMemoryTrainingJobStore),
        ):
            return self.job_store.jobs
        return {job.id: job for job in self.job_store.list()}

    def create_run_plan(
        self,
        *,
        model: str,
        preset: str,
        presets: list[str] | None = None,
        experiment_task: str | None = None,
        datasets: list[str],
        overrides: dict[str, Any],
        log_folder: str = "",
        monitors: list[str] | None = None,
        search: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected = self.run_plan_builder.resolve_inputs(
            model=model,
            preset=preset,
            presets=presets,
            experiment_task=experiment_task,
            datasets=datasets,
            overrides=overrides,
            search=search,
        )
        selected_monitors = resolve_model_monitors(selected.parts, monitors)
        return self.run_plan_builder.create(
            model=model,
            selected=selected,
            log_folder=self.run_plan_builder.valid_plan_log_folder(log_folder),
            monitors=[monitor.name for monitor in selected_monitors],
        )

    def create_job(
        self,
        *,
        model: str,
        preset: str,
        presets: list[str] | None = None,
        experiment_task: str | None = None,
        datasets: list[str],
        overrides: dict[str, Any],
        log_folder: str,
        monitors: list[str] | None = None,
        search: dict[str, Any] | None = None,
        run_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        validated_log_folder = validate_log_experiment_name(log_folder)

        selected = self._resolve_job_inputs(
            model=model,
            preset=preset,
            presets=presets,
            experiment_task=experiment_task,
            datasets=datasets,
            overrides=overrides,
            search=search,
        )
        selected_monitors = resolve_model_monitors(selected.parts, monitors)
        materialized_run_plan = self._materialize_job_run_plan(
            model=model,
            selected=selected,
            run_plan=run_plan,
            validated_log_folder=validated_log_folder,
            monitors=[monitor.name for monitor in selected_monitors],
        )
        planned_run_count = materialized_run_plan["summary"]["totalRuns"]

        self._ensure_job_log_folder(validated_log_folder)

        job_id = uuid.uuid4().hex
        job_root = self._create_job_root(job_id)
        payload = self._build_worker_payload(
            job_id=job_id,
            model=model,
            selected=selected,
            selected_monitors=selected_monitors,
            planned_run_count=planned_run_count,
            materialized_run_plan=materialized_run_plan,
            validated_log_folder=validated_log_folder,
        )
        try:
            launch = self.worker_launcher.launch(
                job_root=job_root,
                payload=payload,
            )
        except StrictCancellationUnavailable as exc:
            raise InspectorError(str(exc)) from exc
        try:
            job = self._register_job(
                job_id=job_id,
                model=model,
                payload=payload,
                materialized_run_plan=materialized_run_plan,
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
                    **model_identity_payload_from_id(model),
                    "preset": job.preset,
                    "presets": job.presets,
                    "experimentTask": job.experiment_task,
                    "runTotal": planned_run_count,
                },
            )
            return self.get_job(job_id)
        except Exception:
            with self._state_lock:
                self._processes.pop(job_id, None)
            try:
                self._terminate_and_reap_process(job_id, launch.process)
            except Exception:
                pass
            raise

    def _resolve_job_inputs(
        self,
        *,
        model: str,
        preset: str,
        presets: list[str] | None,
        experiment_task: str | None,
        datasets: list[str],
        overrides: dict[str, Any],
        search: dict[str, Any] | None,
    ) -> SelectedTrainingInputs:
        return self.run_plan_builder.resolve_inputs(
            model=model,
            preset=preset,
            presets=presets,
            experiment_task=experiment_task,
            datasets=datasets,
            overrides=overrides,
            search=search,
        )

    def _materialize_job_run_plan(
        self,
        *,
        model: str,
        selected: SelectedTrainingInputs,
        run_plan: dict[str, Any] | None,
        validated_log_folder: str,
        monitors: list[str],
    ) -> dict[str, Any]:
        if run_plan is not None:
            return self.run_plan_builder.from_submitted(
                model=model,
                selected=selected,
                run_plan=run_plan,
                log_folder=validated_log_folder,
                monitors=monitors,
            )
        return self.run_plan_builder.create(
            model=model,
            selected=selected,
            log_folder=validated_log_folder,
            monitors=monitors,
        )

    def _ensure_job_log_folder(self, validated_log_folder: str) -> None:
        # Validate the top-level folder immediately before the worker starts.
        log_folder_path = self.logs_root / validated_log_folder
        if log_folder_path.is_symlink():
            raise InspectorError(
                f"Refusing to write symlink log experiment: {validated_log_folder}"
            )
        log_folder_path.mkdir(parents=True, exist_ok=True)

    def _create_job_root(self, job_id: str) -> Path:
        job_root = self.root / job_id
        job_root.mkdir(parents=True, exist_ok=True)
        return job_root

    def _build_worker_payload(
        self,
        *,
        job_id: str,
        model: str,
        selected: SelectedTrainingInputs,
        selected_monitors: list[Any],
        planned_run_count: int,
        materialized_run_plan: dict[str, Any],
        validated_log_folder: str,
    ) -> dict[str, Any]:
        return {
            "id": job_id,
            **model_identity_payload_from_id(model),
            "preset": selected.selected_preset_names[0],
            "presets": selected.selected_preset_names,
            "experimentTask": selected.experiment_task_name,
            "datasets": [
                dataset_name(dataset) for dataset in selected.selected_datasets
            ],
            "overrides": dict(materialized_run_plan.get("overrides") or {}),
            "search": (
                selected.parsed_search.to_payload()
                if selected.parsed_search is not None
                else None
            ),
            "plannedRunCount": planned_run_count,
            "runPlan": materialized_run_plan,
            "monitors": [monitor.name for monitor in selected_monitors],
            "logFolder": validated_log_folder,
        }

    def _write_worker_payload(
        self,
        job_root: Path,
        payload: dict[str, Any],
    ) -> Path:
        return self.worker_launcher.write_payload(job_root, payload)

    def _build_worker_command(
        self,
        payload_path: Path,
        progress_path: Path,
    ) -> list[str]:
        return self.worker_launcher.build_command(payload_path, progress_path)

    def _worker_env(self) -> dict[str, str]:
        return self.worker_launcher.worker_env()

    def _register_job(
        self,
        *,
        job_id: str,
        model: str,
        payload: dict[str, Any],
        materialized_run_plan: dict[str, Any],
        validated_log_folder: str,
        command: list[str],
        job_root: Path,
        process: ProcessHandle,
        containment: TrainingProcessContainment,
    ) -> TrainingJob:
        job = TrainingJob(
            id=job_id,
            model=model,
            preset=payload["preset"],
            presets=payload["presets"],
            experiment_task=payload["experimentTask"],
            datasets=payload["datasets"],
            overrides=payload["overrides"],
            search=payload["search"],
            planned_run_count=payload["plannedRunCount"],
            run_plan=materialized_run_plan,
            monitors=payload["monitors"],
            log_folder=validated_log_folder,
            command=command,
            root=job_root,
            pid=process.pid,
            cancellation_mode=containment.mode,
            worker_pid=containment.worker_pid,
            process_group_id=containment.process_group_id,
            cgroup_path=containment.cgroup_path,
        )
        self.job_store.save(job)
        with self._state_lock:
            self._processes[job_id] = process
        return job

    def get_job(self, job_id: str) -> dict[str, Any]:
        job = self._get_job_record(job_id)
        snapshot = self.progress_store.read_snapshot(job)
        self._refresh(job, events=snapshot.events)
        return self._serialize(job, snapshot=snapshot)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = self._get_job_record(job_id)
        process = self._process_for_job(job)
        if process is None and is_active_job_status(job.status):
            raise InspectorError(
                f"Training job '{job_id}' has no live process handle."
            )
        reaped_exit_code: int | None = None
        process_exit_code = process.poll() if process is not None else None
        if process is not None and (
            process_exit_code is None
            or is_live_process_job_status(job.status)
            or self._job_has_live_containment(job)
        ):
            reaped_exit_code = self._terminate_and_reap_process(job_id, process)
        job.status = "cancelled"
        if reaped_exit_code is not None:
            job.exit_code = reaped_exit_code
        job.updated_at = _now()
        self.job_store.save(job)
        self._write_event(job, {"type": "cancelled", "status": "cancelled"})
        snapshot = self.progress_store.read_snapshot(job)
        return self._serialize(job, snapshot=snapshot)

    def active_jobs(self) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        for job in self.job_store.list():
            if (
                is_terminal_job_status(job.status)
                and not self._job_has_live_containment(job)
            ):
                continue
            snapshot = self.progress_store.read_snapshot(job)
            self._refresh(job, events=snapshot.events)
            if not is_active_job_status(job.status):
                continue
            active.append(
                {
                    "id": job.id,
                    "status": job.status,
                    "logFolder": job.log_folder,
                }
            )
        return active

    def get_job_events(
        self,
        job_id: str,
        *,
        offset: int = 0,
        limit: int = 500,
    ) -> dict[str, Any]:
        job = self._get_job_record(job_id)
        snapshot = self.progress_store.read_snapshot(job)
        self._refresh(job, events=snapshot.events)
        safe_offset = max(0, offset)
        safe_limit = min(5000, max(1, limit))
        events = snapshot.events
        page = events[safe_offset : safe_offset + safe_limit]
        next_offset = safe_offset + len(page)
        return {
            "jobId": job.id,
            "offset": safe_offset,
            "limit": safe_limit,
            "totalCount": snapshot.total_count,
            "nextOffset": next_offset if next_offset < snapshot.total_count else None,
            "events": page,
        }

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
            raise InspectorError(
                f"Unknown dataset '{dataset}' for training job '{job_id}'."
            )
        if preset is not None and not self.monitor_locator.preset_in_job(job, preset):
            raise InspectorError(
                f"Unknown preset '{preset}' for training job '{job_id}'."
            )
        self._refresh(job)
        log_dir = self.monitor_locator.log_dir_for_monitor_data(
            events=self._events(job),
            dataset=dataset,
            preset=preset,
        )
        data = self.monitor_reader.read(
            job_id=job.id,
            node_path=node_path,
            dataset=dataset,
            log_dir=log_dir,
        )
        data["preset"] = normalize_preset_token(preset) if preset else None
        return data

    def get_parameter_status(
        self,
        job_id: str,
        *,
        dataset: str | None = None,
        preset: str | None = None,
    ) -> dict[str, Any]:
        job = self._get_job_record(job_id)
        if dataset is not None and dataset not in job.datasets:
            raise InspectorError(
                f"Unknown dataset '{dataset}' for training job '{job_id}'."
            )
        if preset is not None and not self.monitor_locator.preset_in_job(job, preset):
            raise InspectorError(
                f"Unknown preset '{preset}' for training job '{job_id}'."
            )
        self._refresh(job)
        log_dir = self.monitor_locator.log_dir_for_monitor_data(
            events=self._events(job),
            dataset=dataset,
            preset=preset,
        )
        return self.parameter_status_reader.read(
            source_id=job.id,
            preset=normalize_preset_token(preset) if preset else None,
            dataset=dataset,
            log_dir=log_dir,
        )

    def _get_job_record(self, job_id: str) -> TrainingJob:
        job = self.job_store.get(job_id)
        if job is None:
            raise InspectorError(f"Unknown training job '{job_id}'.")
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
            return process.wait(timeout=CANCEL_REAP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
        try:
            return process.wait(timeout=CANCEL_REAP_GRACE_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise InspectorError(
                f"Training job '{job_id}' process survived terminate and kill."
            ) from exc

    def _process_for_job(self, job: TrainingJob) -> ProcessHandle | None:
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

    def _rehydrate_process_handle(self, job: TrainingJob) -> ProcessHandle | None:
        if job.cancellation_mode != "strict-cgroup":
            return None
        cgroup = self.cgroup_manager.from_existing(job.cgroup_path)
        if cgroup is None or not cgroup.has_processes():
            return None
        return PersistedCgroupProcessHandle(
            pid=job.worker_pid or job.pid,
            cgroup=cgroup,
            process_group_id=job.process_group_id,
        )

    def _job_has_live_containment(self, job: TrainingJob) -> bool:
        if job.cancellation_mode != "strict-cgroup":
            return False
        with self._state_lock:
            process = self._processes.get(job.id)
        if process is not None and process.poll() is None:
            return True
        cgroup = self.cgroup_manager.from_existing(job.cgroup_path)
        return bool(cgroup and cgroup.has_processes())

    def _refresh(
        self,
        job: TrainingJob,
        *,
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        original_state = (job.status, job.exit_code, job.updated_at)
        process = self._process_for_job(job)
        exit_code = process.poll() if process is not None else None
        containment_live = process is not None and exit_code is None
        events = events if events is not None else self._events(job)
        latest_terminal = latest_terminal_event(events)
        terminal_status = (
            terminal_status_from_event(latest_terminal)
            if latest_terminal is not None
            else None
        )
        if (
            latest_terminal is not None
            and terminal_status is not None
            and not containment_live
        ):
            job.status = terminal_status
            job.exit_code = terminal_exit_code(
                terminal_status,
                latest_terminal,
                job.exit_code,
            )
            job.updated_at = str(latest_terminal.get("timestamp") or _now())
        elif containment_live and is_terminal_job_status(job.status):
            job.status = "running"
            job.exit_code = None
            job.updated_at = _now()
        elif process is None and is_live_process_job_status(job.status):
            job.status = "unknown"
            job.updated_at = _now()
        elif process is not None and exit_code is None and job.status == "unknown":
            job.status = "running"
            job.updated_at = _now()
        elif (
            process is not None
            and exit_code is not None
            and is_active_job_status(job.status)
        ):
            job.exit_code = exit_code
            job.status = "completed" if exit_code == 0 else "failed"
            job.updated_at = _now()
        if original_state != (job.status, job.exit_code, job.updated_at):
            self.job_store.save(job)

    def _events(self, job: TrainingJob) -> list[dict[str, Any]]:
        return self.progress_store.read_events(job)

    def _write_event(self, job: TrainingJob, event: dict[str, Any]) -> None:
        self.progress_store.append_event(job, event)

    def _normalize_preset_token(self, preset: str | None) -> str | None:
        return normalize_preset_token(preset)

    def _preset_in_job(self, job: TrainingJob, preset: str) -> bool:
        return self.monitor_locator.preset_in_job(job, preset)

    def _event_preset_name(self, event: dict[str, Any]) -> str | None:
        return self.monitor_locator.event_preset_name(event)

    def _event_matches_preset(
        self,
        event: dict[str, Any],
        preset: str | None,
    ) -> bool:
        return self.monitor_locator.event_matches_preset(event, preset)

    def _log_dir_for_monitor_data(
        self,
        job: TrainingJob,
        dataset: str | None,
        preset: str | None,
    ) -> str | None:
        return self.monitor_locator.log_dir_for_monitor_data(
            events=self._events(job),
            dataset=dataset,
            preset=preset,
        )

    def _log_tail(self, job: TrainingJob, line_count: int = 80) -> list[str]:
        return self.job_projector.log_tail(job, line_count=line_count)

    def _run_plan_for_job(
        self,
        job: TrainingJob,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.job_projector.project(
            job,
            events=events,
            summarize=self.run_plan_builder.summarize,
        )["runPlan"]

    def _serialize(
        self,
        job: TrainingJob,
        *,
        snapshot: TrainingProgressSnapshot | None = None,
    ) -> dict[str, Any]:
        snapshot = snapshot or self.progress_store.read_snapshot(job)
        return self.job_projector.project_live(
            job,
            self._live_projection(job, snapshot),
        )

    def _live_projection(
        self,
        job: TrainingJob,
        snapshot: TrainingProgressSnapshot,
    ) -> TrainingJobLiveProjection:
        return self._live_projection_cache.project(
            job,
            snapshot,
            summarize=self.run_plan_builder.summarize,
        )
