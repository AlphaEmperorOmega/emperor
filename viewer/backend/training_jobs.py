from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from viewer.backend.job_store import (
    FileSystemTrainingJobStore,
    InMemoryTrainingJobStore,
    TrainingJobRecord,
    TrainingJobStore,
)
from viewer.backend.inspector.discovery import (
    dataset_name,
    resolve_model_monitors,
)
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.log_runs import validate_log_experiment_name
from viewer.backend.monitor_data import TensorBoardMonitorReader
from viewer.backend.training_run_plans import (
    SelectedTrainingInputs,
    TrainingRunPlanBuilder,
)
from viewer.backend.training_run_progress import project_training_run_progress


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None:
        ...

    def terminate(self) -> None:
        ...


class ProcessRunner(Protocol):
    def start(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
    ) -> ProcessHandle:
        ...


class SubprocessRunner:
    def start(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
    ) -> subprocess.Popen:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_file:
            return subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )


TrainingJob = TrainingJobRecord


class TrainingJobManager:
    def __init__(
        self,
        *,
        root: Path | None = None,
        cwd: Path | None = None,
        logs_root: Path | str = "logs",
        runner: ProcessRunner | None = None,
        monitor_reader: TensorBoardMonitorReader | None = None,
        run_plan_builder: TrainingRunPlanBuilder | None = None,
        job_store: TrainingJobStore | None = None,
    ) -> None:
        self.root = root or Path("/tmp/emperor-viewer-training")
        self.cwd = cwd or Path.cwd()
        self.logs_root = Path(logs_root)
        self.runner = runner or SubprocessRunner()
        self.monitor_reader = monitor_reader or TensorBoardMonitorReader()
        self.run_plan_builder = run_plan_builder or TrainingRunPlanBuilder()
        self.job_store = job_store or FileSystemTrainingJobStore(self.root)
        self._processes: dict[str, ProcessHandle] = {}

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
        datasets: list[str],
        overrides: dict[str, Any],
        log_folder: str = "",
        search: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected = self.run_plan_builder.resolve_inputs(
            model=model,
            preset=preset,
            presets=presets,
            datasets=datasets,
            overrides=overrides,
            search=search,
        )
        return self.run_plan_builder.create(
            model=model,
            selected=selected,
            log_folder=self.run_plan_builder.valid_plan_log_folder(log_folder),
        )

    def create_job(
        self,
        *,
        model: str,
        preset: str,
        presets: list[str] | None = None,
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
        payload_path = self._write_worker_payload(job_root, payload)
        progress_path = job_root / "progress.jsonl"
        command = self._build_worker_command(payload_path, progress_path)
        process = self.runner.start(
            command,
            cwd=self.cwd,
            env=self._worker_env(),
            log_path=job_root / "training.log",
        )
        job = self._register_job(
            job_id=job_id,
            model=model,
            payload=payload,
            materialized_run_plan=materialized_run_plan,
            validated_log_folder=validated_log_folder,
            command=command,
            job_root=job_root,
            process=process,
        )
        self._write_event(
            job,
            {
                "type": "job_started",
                "status": "running",
                "preset": job.preset,
                "presets": job.presets,
                "runTotal": planned_run_count,
            },
        )
        return self.get_job(job_id)

    def _resolve_job_inputs(
        self,
        *,
        model: str,
        preset: str,
        presets: list[str] | None,
        datasets: list[str],
        overrides: dict[str, Any],
        search: dict[str, Any] | None,
    ) -> SelectedTrainingInputs:
        return self.run_plan_builder.resolve_inputs(
            model=model,
            preset=preset,
            presets=presets,
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
    ) -> dict[str, Any]:
        if run_plan is not None:
            return self.run_plan_builder.from_submitted(
                model=model,
                selected=selected,
                run_plan=run_plan,
                log_folder=validated_log_folder,
            )
        return self.run_plan_builder.create(
            model=model,
            selected=selected,
            log_folder=validated_log_folder,
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
            "model": model,
            "preset": selected.selected_preset_names[0],
            "presets": selected.selected_preset_names,
            "datasets": [
                dataset_name(dataset) for dataset in selected.selected_datasets
            ],
            "overrides": selected.effective_overrides,
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
        payload_path = job_root / "payload.json"
        payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload_path

    def _build_worker_command(
        self,
        payload_path: Path,
        progress_path: Path,
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            "viewer.backend.training_worker",
            "--payload",
            str(payload_path),
            "--progress",
            str(progress_path),
        ]

    def _worker_env(self) -> dict[str, str]:
        return {
            **os.environ,
            "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"),
        }

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
    ) -> TrainingJob:
        job = TrainingJob(
            id=job_id,
            model=model,
            preset=payload["preset"],
            presets=payload["presets"],
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
        )
        self._processes[job_id] = process
        self.job_store.save(job)
        return job

    def get_job(self, job_id: str) -> dict[str, Any]:
        job = self._get_job_record(job_id)
        self._refresh(job)
        return self._serialize(job)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = self._get_job_record(job_id)
        process = self._processes.get(job_id)
        if process is None and job.status in {"running", "queued", "unknown"}:
            raise InspectorError(
                f"Training job '{job_id}' has no live process handle."
            )
        if job.status in {"running", "queued"}:
            if process.poll() is None:
                process.terminate()
        job.status = "cancelled"
        job.updated_at = _now()
        self.job_store.save(job)
        self._write_event(job, {"type": "cancelled", "status": "cancelled"})
        return self._serialize(job)

    def active_jobs(self) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        for job in self.job_store.list():
            self._refresh(job)
            if job.status not in {"running", "queued", "unknown"}:
                continue
            active.append(
                {
                    "id": job.id,
                    "status": job.status,
                    "logFolder": job.log_folder,
                }
            )
        return active

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
        if preset is not None and not self._preset_in_job(job, preset):
            raise InspectorError(
                f"Unknown preset '{preset}' for training job '{job_id}'."
            )
        self._refresh(job)
        log_dir = self._log_dir_for_monitor_data(job, dataset, preset)
        data = self.monitor_reader.read(
            job_id=job.id,
            node_path=node_path,
            dataset=dataset,
            log_dir=log_dir,
        )
        data["preset"] = self._normalize_preset_token(preset) if preset else None
        return data

    def _get_job_record(self, job_id: str) -> TrainingJob:
        job = self.job_store.get(job_id)
        if job is None:
            raise InspectorError(f"Unknown training job '{job_id}'.")
        return job

    def _refresh(self, job: TrainingJob) -> None:
        original_state = (job.status, job.exit_code, job.updated_at)
        process = self._processes.get(job.id)
        exit_code = process.poll() if process is not None else None
        events = self._events(job)
        latest_failed = next(
            (event for event in reversed(events) if event.get("status") == "failed"),
            None,
        )
        if latest_failed is not None:
            job.status = "failed"
        elif process is None and job.status in {"running", "queued"}:
            job.status = "unknown"
            job.updated_at = _now()
        elif process is not None and exit_code is None and job.status == "unknown":
            job.status = "running"
            job.updated_at = _now()
        elif process is not None and exit_code is not None and job.status in {
            "running",
            "queued",
            "unknown",
        }:
            job.exit_code = exit_code
            job.status = "completed" if exit_code == 0 else "failed"
            job.updated_at = _now()
        if original_state != (job.status, job.exit_code, job.updated_at):
            self.job_store.save(job)

    def _events(self, job: TrainingJob) -> list[dict[str, Any]]:
        if not job.progress_path.exists():
            return []
        events = []
        for line in job.progress_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def _write_event(self, job: TrainingJob, event: dict[str, Any]) -> None:
        payload = {"timestamp": _now(), "jobId": job.id, **event}
        job.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with job.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def _normalize_preset_token(self, preset: str | None) -> str | None:
        if preset is None:
            return None
        return str(preset).lower().replace("_", "-")

    def _preset_in_job(self, job: TrainingJob, preset: str) -> bool:
        normalized = self._normalize_preset_token(preset)
        return normalized in {self._normalize_preset_token(item) for item in job.presets}

    def _event_preset_name(self, event: dict[str, Any]) -> str | None:
        return self._normalize_preset_token(
            event.get("preset") or event.get("option")
        )

    def _event_matches_preset(
        self,
        event: dict[str, Any],
        preset: str | None,
    ) -> bool:
        if preset is None:
            return True
        return self._event_preset_name(event) == self._normalize_preset_token(preset)

    def _log_dir_for_monitor_data(
        self,
        job: TrainingJob,
        dataset: str | None,
        preset: str | None,
    ) -> str | None:
        for event in reversed(self._events(job)):
            log_dir = event.get("logDir")
            if not log_dir:
                continue
            if dataset is None or event.get("dataset") == dataset:
                if self._event_matches_preset(event, preset):
                    return str(log_dir)
        return None

    def _log_tail(self, job: TrainingJob, line_count: int = 80) -> list[str]:
        if not job.log_path.exists():
            return []
        return job.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
            -line_count:
        ]

    def _run_plan_for_job(
        self,
        job: TrainingJob,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return project_training_run_progress(
            job.run_plan,
            events,
            job.status,
            self.run_plan_builder.summarize,
        )

    def _serialize(self, job: TrainingJob) -> dict[str, Any]:
        events = self._events(job)
        latest_event = events[-1] if events else {}
        metrics_event = next(
            (event for event in reversed(events) if isinstance(event.get("metrics"), dict)),
            {},
        )
        result_events = [
            event for event in events if event.get("type") == "dataset_completed"
        ]
        latest_preset = self._event_preset_name(latest_event)
        run_plan = self._run_plan_for_job(job, events)
        return {
            "id": job.id,
            "status": job.status,
            "model": job.model,
            "preset": job.preset,
            "presets": job.presets,
            "datasets": job.datasets,
            "overrides": job.overrides,
            "search": job.search,
            "plannedRunCount": job.planned_run_count,
            "runPlan": run_plan,
            "monitors": job.monitors,
            "logFolder": job.log_folder,
            "createdAt": job.created_at,
            "updatedAt": job.updated_at,
            "exitCode": job.exit_code,
            "pid": job.pid,
            "currentPreset": latest_preset,
            "currentDataset": latest_event.get("dataset"),
            "epoch": latest_event.get("epoch"),
            "step": latest_event.get("step"),
            "metrics": metrics_event.get("metrics") or {},
            "logDir": latest_event.get("logDir"),
            "events": events,
            "logTail": self._log_tail(job),
            "resultLinks": [
                {
                    "preset": self._event_preset_name(event),
                    "dataset": event.get("dataset"),
                    "logDir": event.get("logDir"),
                }
                for event in result_events
            ],
        }
