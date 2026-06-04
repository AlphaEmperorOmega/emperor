from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from viewer.backend.inspector.discovery import (
    dataset_name,
    resolve_model_monitors,
)
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.log_runs import validate_log_experiment_name
from viewer.backend.monitor_data import TensorBoardMonitorReader
from viewer.backend.training_run_plans import TrainingRunPlanBuilder


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


@dataclass
class TrainingJob:
    id: str
    model: str
    preset: str
    presets: list[str]
    datasets: list[str]
    overrides: dict[str, Any]
    search: dict[str, Any] | None
    planned_run_count: int
    run_plan: dict[str, Any]
    monitors: list[str]
    log_folder: str
    command: list[str]
    root: Path
    process: ProcessHandle
    status: str = "running"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    exit_code: int | None = None

    @property
    def progress_path(self) -> Path:
        return self.root / "progress.jsonl"

    @property
    def log_path(self) -> Path:
        return self.root / "training.log"


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
    ) -> None:
        self.root = root or Path("/tmp/emperor-viewer-training")
        self.cwd = cwd or Path.cwd()
        self.logs_root = Path(logs_root)
        self.runner = runner or SubprocessRunner()
        self.monitor_reader = monitor_reader or TensorBoardMonitorReader()
        self.run_plan_builder = run_plan_builder or TrainingRunPlanBuilder()
        self.jobs: dict[str, TrainingJob] = {}

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

        selected = self.run_plan_builder.resolve_inputs(
            model=model,
            preset=preset,
            presets=presets,
            datasets=datasets,
            overrides=overrides,
            search=search,
        )
        parts = selected.parts
        selected_preset_names = selected.selected_preset_names
        selected_datasets = selected.selected_datasets
        selected_monitors = resolve_model_monitors(parts, monitors)
        if run_plan is not None:
            materialized_run_plan = self.run_plan_builder.from_submitted(
                model=model,
                selected=selected,
                run_plan=run_plan,
                log_folder=validated_log_folder,
            )
        else:
            materialized_run_plan = self.run_plan_builder.create(
                model=model,
                selected=selected,
                log_folder=validated_log_folder,
            )
        planned_run_count = materialized_run_plan["summary"]["totalRuns"]

        # Validate the top-level folder immediately before the worker starts.
        log_folder_path = self.logs_root / validated_log_folder
        if log_folder_path.is_symlink():
            raise InspectorError(
                f"Refusing to write symlink log experiment: {validated_log_folder}"
            )
        log_folder_path.mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex
        job_root = self.root / job_id
        job_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": job_id,
            "model": model,
            "preset": selected_preset_names[0],
            "presets": selected_preset_names,
            "datasets": [dataset_name(dataset) for dataset in selected_datasets],
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
        payload_path = job_root / "payload.json"
        payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        progress_path = job_root / "progress.jsonl"
        command = [
            sys.executable,
            "-m",
            "viewer.backend.training_worker",
            "--payload",
            str(payload_path),
            "--progress",
            str(progress_path),
        ]
        env = {**os.environ, "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib")}
        process = self.runner.start(
            command,
            cwd=self.cwd,
            env=env,
            log_path=job_root / "training.log",
        )
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
            process=process,
        )
        self.jobs[job_id] = job
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

    def get_job(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise InspectorError(f"Unknown training job '{job_id}'.")
        self._refresh(job)
        return self._serialize(job)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise InspectorError(f"Unknown training job '{job_id}'.")
        if job.status in {"running", "queued"} and job.process.poll() is None:
            job.process.terminate()
        job.status = "cancelled"
        job.updated_at = _now()
        self._write_event(job, {"type": "cancelled", "status": "cancelled"})
        return self._serialize(job)

    def active_jobs(self) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        for job in self.jobs.values():
            self._refresh(job)
            if job.status not in {"running", "queued"}:
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
        job = self.jobs.get(job_id)
        if job is None:
            raise InspectorError(f"Unknown training job '{job_id}'.")
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

    def _refresh(self, job: TrainingJob) -> None:
        exit_code = job.process.poll()
        events = self._events(job)
        latest_failed = next(
            (event for event in reversed(events) if event.get("status") == "failed"),
            None,
        )
        if latest_failed is not None:
            job.status = "failed"
        if exit_code is not None and job.status in {"running", "queued"}:
            job.exit_code = exit_code
            job.status = "completed" if exit_code == 0 else "failed"
            job.updated_at = _now()

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

    def _run_for_event(
        self,
        *,
        event: dict[str, Any],
        runs: list[dict[str, Any]],
        run_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        run_id = event.get("runId")
        if isinstance(run_id, str) and run_id in run_by_id:
            return run_by_id[run_id]

        run_index = event.get("runIndex")
        if isinstance(run_index, int):
            if 1 <= run_index <= len(runs):
                return runs[run_index - 1]
            if 0 <= run_index < len(runs):
                return runs[run_index]

        dataset = event.get("dataset")
        preset = self._event_preset_name(event)
        if dataset is None:
            return None
        candidates = [
            run
            for run in runs
            if run.get("dataset") == dataset
            and (
                preset is None
                or self._normalize_preset_token(str(run.get("preset")))
                == self._normalize_preset_token(preset)
            )
        ]
        return next(
            (
                run
                for run in candidates
                if run.get("status") not in {"Completed", "Failed", "Cancelled"}
            ),
            candidates[0] if candidates else None,
        )

    def _event_epoch(self, event: dict[str, Any], total_epochs: int) -> int:
        raw_epoch = event.get("epoch")
        if not isinstance(raw_epoch, int):
            return 0
        return min(total_epochs, max(0, raw_epoch + 1))

    def _run_plan_for_job(self, job: TrainingJob, events: list[dict[str, Any]]) -> dict[str, Any]:
        plan = copy.deepcopy(job.run_plan)
        runs = plan.get("runs") or []
        latest_failed_event = next(
            (event for event in reversed(events) if event.get("status") == "failed"),
            {},
        )
        run_by_id = {
            str(run.get("id")): run
            for run in runs
            if run.get("id") is not None
        }
        for event in events:
            row = self._run_for_event(
                event=event,
                runs=runs,
                run_by_id=run_by_id,
            )
            if row is None:
                continue

            event_type = event.get("type")
            total_epochs = int(row.get("totalEpochs") or 0)
            if event.get("logDir"):
                row["logDir"] = event.get("logDir")
            if isinstance(event.get("metrics"), dict):
                row["metrics"] = event["metrics"]

            if event_type == "dataset_started":
                row["status"] = "Running"
                row["currentEpoch"] = max(0, int(row.get("currentEpoch") or 0))
            elif event_type in {
                "epoch_started",
                "step",
                "validation",
                "fit_completed",
                "test_completed",
            }:
                row["status"] = "Running"
                row["currentEpoch"] = max(
                    int(row.get("currentEpoch") or 0),
                    self._event_epoch(event, total_epochs),
                )
            elif event_type == "dataset_completed":
                row["status"] = "Completed"
                row["currentEpoch"] = total_epochs
            elif event_type == "error":
                row["status"] = "Failed"
                row["currentEpoch"] = max(
                    int(row.get("currentEpoch") or 0),
                    self._event_epoch(event, total_epochs),
                )
                row["error"] = str(event.get("error") or "Training failed")
                if event.get("traceback"):
                    row["errorTraceback"] = str(event.get("traceback"))

        if job.status == "cancelled":
            for row in runs:
                if row.get("status") == "Running":
                    row["status"] = "Cancelled"
                elif row.get("status") == "Pending":
                    row["status"] = "Skipped"
        elif job.status == "failed":
            failed_seen = any(row.get("status") == "Failed" for row in runs)
            for row in runs:
                if row.get("status") == "Running":
                    row["status"] = "Failed"
                    failed_seen = True
                elif row.get("status") == "Pending":
                    if not failed_seen:
                        row["status"] = "Failed"
                        row["error"] = "Training failed"
                        if latest_failed_event.get("traceback"):
                            row["errorTraceback"] = str(
                                latest_failed_event.get("traceback")
                            )
                        failed_seen = True
                    else:
                        row["status"] = "Skipped"
        elif job.status == "completed":
            for row in runs:
                if row.get("status") == "Running":
                    row["status"] = "Completed"
                    row["currentEpoch"] = int(row.get("totalEpochs") or 0)

        plan["summary"] = self.run_plan_builder.summarize(runs)
        return plan

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
            "pid": job.process.pid,
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
