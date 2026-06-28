"""Training job record storage interfaces and local adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from models.catalog import model_id_from_payload, model_identity_payload_from_id

from viewer.backend.storage.local_files import (
    read_json_object,
    require_safe_name,
    resolve_root,
    resolve_under_root,
    safe_child_path,
    write_json_atomic,
)

METADATA_FILENAME = "metadata.json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class TrainingJobRecord:
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
    pid: int
    cancellation_mode: str = "process-group"
    worker_pid: int | None = None
    process_group_id: int | None = None
    cgroup_path: str | None = None
    status: str = "running"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    exit_code: int | None = None

    @property
    def payload_path(self) -> Path:
        return self.root / "payload.json"

    @property
    def progress_path(self) -> Path:
        return self.root / "progress.jsonl"

    @property
    def log_path(self) -> Path:
        return self.root / "training.log"


class TrainingJobStore(Protocol):
    def save(self, job: TrainingJobRecord) -> None: ...

    def get(self, job_id: str) -> TrainingJobRecord | None: ...

    def list(self) -> list[TrainingJobRecord]: ...


class InMemoryTrainingJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, TrainingJobRecord] = {}
        self._lock = RLock()

    @property
    def jobs(self) -> dict[str, TrainingJobRecord]:
        return self._jobs

    def save(self, job: TrainingJobRecord) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> TrainingJobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[TrainingJobRecord]:
        with self._lock:
            return list(self._jobs.values())


class FileSystemTrainingJobStore:
    def __init__(self, root: Path) -> None:
        self.root = resolve_root(Path(root))
        self._jobs: dict[str, TrainingJobRecord] = {}
        self._lock = RLock()

    @property
    def jobs(self) -> dict[str, TrainingJobRecord]:
        self.list()
        return self._jobs

    def save(self, job: TrainingJobRecord) -> None:
        if self._safe_job_id(job.id) is None:
            raise ValueError("training job id contains unsafe path characters")
        metadata_path = self._metadata_path(job.root)
        write_json_atomic(metadata_path, _record_to_metadata(job))
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> TrainingJobRecord | None:
        safe_job_id = self._safe_job_id(job_id)
        if safe_job_id is None:
            return None
        with self._lock:
            cached_job = self._jobs.get(job_id)
        if cached_job is not None:
            return cached_job
        metadata_path = safe_child_path(
            self.root,
            f"{safe_job_id}/{METADATA_FILENAME}",
        )
        job = self._read_metadata(metadata_path)
        if job is None or job.id != job_id:
            return None
        with self._lock:
            self._jobs[job.id] = job
        return job

    def list(self) -> list[TrainingJobRecord]:
        if self.root.exists():
            for metadata_path in sorted(self.root.glob(f"*/{METADATA_FILENAME}")):
                job = self._read_metadata(metadata_path)
                if job is None or self._safe_job_id(job.id) is None:
                    continue
                with self._lock:
                    if job.id in self._jobs:
                        continue
                    self._jobs[job.id] = job
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.id)

    def _metadata_path(self, job_root: Path) -> Path:
        return resolve_under_root(self.root, Path(job_root) / METADATA_FILENAME)

    def _safe_job_id(self, job_id: str) -> str | None:
        try:
            return require_safe_name(job_id, "training job id")
        except ValueError:
            return None

    def _read_metadata(self, metadata_path: Path) -> TrainingJobRecord | None:
        payload = read_json_object(metadata_path)
        if payload is None:
            return None
        try:
            return _record_from_metadata(payload, metadata_path)
        except (KeyError, TypeError, ValueError):
            return None


def _record_to_metadata(job: TrainingJobRecord) -> dict[str, Any]:
    return {
        "id": job.id,
        **model_identity_payload_from_id(job.model),
        "preset": job.preset,
        "presets": job.presets,
        "datasets": job.datasets,
        "overrides": job.overrides,
        "search": job.search,
        "planned_run_count": job.planned_run_count,
        "run_plan": job.run_plan,
        "monitors": job.monitors,
        "log_folder": job.log_folder,
        "command": job.command,
        "root": str(job.root),
        "payload_path": str(job.payload_path),
        "progress_path": str(job.progress_path),
        "log_path": str(job.log_path),
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "status": job.status,
        "pid": job.pid,
        "cancellation_mode": job.cancellation_mode,
        "worker_pid": job.worker_pid,
        "process_group_id": job.process_group_id,
        "cgroup_path": job.cgroup_path,
        "exit_code": job.exit_code,
    }


def _record_from_metadata(
    payload: dict[str, Any],
    metadata_path: Path,
) -> TrainingJobRecord:
    root = _metadata_root(payload, metadata_path)
    model_id = model_id_from_payload(payload)
    if model_id is None:
        raise ValueError("Training job metadata has invalid model identity.")
    return TrainingJobRecord(
        id=str(payload["id"]),
        model=model_id,
        preset=str(payload["preset"]),
        presets=[str(item) for item in payload["presets"]],
        datasets=[str(item) for item in payload["datasets"]],
        overrides=dict(payload["overrides"]),
        search=(dict(payload["search"]) if payload.get("search") is not None else None),
        planned_run_count=int(payload["planned_run_count"]),
        run_plan=dict(payload["run_plan"]),
        monitors=[str(item) for item in payload["monitors"]],
        log_folder=str(payload["log_folder"]),
        command=[str(item) for item in payload["command"]],
        root=root,
        pid=int(payload["pid"]),
        cancellation_mode=str(payload.get("cancellation_mode") or "process-group"),
        worker_pid=(
            int(payload["worker_pid"])
            if payload.get("worker_pid") is not None
            else int(payload["pid"])
        ),
        process_group_id=(
            int(payload["process_group_id"])
            if payload.get("process_group_id") is not None
            else None
        ),
        cgroup_path=(
            str(payload["cgroup_path"])
            if payload.get("cgroup_path") is not None
            else None
        ),
        status=str(payload["status"]),
        created_at=str(payload["created_at"]),
        updated_at=str(payload["updated_at"]),
        exit_code=(
            int(payload["exit_code"]) if payload.get("exit_code") is not None else None
        ),
    )


def _metadata_root(payload: dict[str, Any], metadata_path: Path) -> Path:
    root = Path(str(payload["root"]))
    metadata_root = metadata_path.parent
    try:
        if root.resolve() != metadata_root.resolve():
            return metadata_root
    except OSError:
        return metadata_root
    return root
