from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, cast

from emperor_workbench.filesystem import (
    read_json_object,
    require_safe_name,
    resolve_root,
    resolve_under_root,
    write_json_atomic,
)
from emperor_workbench.run_plans import RunPlanPersistenceCodec
from emperor_workbench.training_jobs._records import (
    ResolvedTrainingCancellationMode,
    TrainingJobRecord,
    TrainingJobStatus,
)
from emperor_workbench.training_jobs._store import (
    PRIVATE_FILE_MODE,
    ensure_private_directory,
)

METADATA_FILENAME = "metadata.json"
METADATA_FIELDS = frozenset(
    {
        "id",
        "preset",
        "presets",
        "experiment_task",
        "datasets",
        "overrides",
        "search",
        "planned_run_count",
        "run_plan",
        "monitors",
        "log_folder",
        "command",
        "root",
        "payload_path",
        "progress_path",
        "log_path",
        "created_at",
        "updated_at",
        "status",
        "pid",
        "cancellation_mode",
        "worker_pid",
        "process_group_id",
        "cgroup_path",
        "windows_job_name",
        "exit_code",
    }
)


class FileSystemTrainingJobStore:
    def __init__(self, root: Path) -> None:
        self.root = resolve_root(ensure_private_directory(Path(root)))
        self._jobs: dict[str, TrainingJobRecord] = {}
        self._lock = RLock()

    def save(self, job: TrainingJobRecord) -> None:
        safe_job_id = self._safe_job_id(job.id)
        if safe_job_id is None:
            raise ValueError("training job id contains unsafe path characters")
        with self._lock:
            metadata_path = self._metadata_path_for_save(
                job_root=job.root,
                safe_job_id=safe_job_id,
            )
            write_json_atomic(metadata_path, _record_to_metadata(job))
            metadata_path.chmod(PRIVATE_FILE_MODE)
            self._jobs[job.id] = job

    def get(self, job_id: str) -> TrainingJobRecord | None:
        safe_job_id = self._safe_job_id(job_id)
        if safe_job_id is None:
            return None
        with self._lock:
            return self._validated_record(safe_job_id)

    def list(self) -> list[TrainingJobRecord]:
        with self._lock:
            job_ids = set(self._jobs)
            if self.root.exists():
                try:
                    children = list(self.root.iterdir())
                except OSError:
                    children = ()
                for child in children:
                    safe_job_id = self._safe_job_id(child.name)
                    if safe_job_id is not None:
                        job_ids.add(safe_job_id)
            return [
                job
                for job_id in sorted(job_ids)
                if (job := self._validated_record(job_id)) is not None
            ]

    def _validated_record(self, safe_job_id: str) -> TrainingJobRecord | None:
        """Return one cached/disk record only while its path authority is valid.

        The caller holds ``_lock`` so in-process saves cannot interleave the
        canonical-path and metadata validation with cache selection.
        """
        try:
            job_root = self._canonical_job_root(
                safe_job_id,
                require_directory=True,
            )
            metadata_path = self._metadata_path_for_read(job_root)
        except ValueError:
            self._jobs.pop(safe_job_id, None)
            return None
        observed_job = self._read_metadata(metadata_path)
        if observed_job is None or observed_job.id != safe_job_id:
            self._jobs.pop(safe_job_id, None)
            return None

        cached_job = self._jobs.get(safe_job_id)
        if cached_job is not None:
            if cached_job.id != safe_job_id:
                self._jobs.pop(safe_job_id, None)
                return None
            cached_job.root = job_root
            return cached_job
        observed_job.root = job_root
        self._jobs[safe_job_id] = observed_job
        return observed_job

    def _canonical_job_root(
        self,
        safe_job_id: str,
        *,
        require_directory: bool,
    ) -> Path:
        candidate = self.root / safe_job_id
        if candidate.is_symlink():
            raise ValueError("training job directory must not be a symlink")
        canonical = resolve_under_root(self.root, candidate)
        expected = self.root / safe_job_id
        if canonical != expected:
            raise ValueError("training job directory is not canonical")
        if require_directory and not candidate.is_dir():
            raise ValueError("training job directory does not exist")
        return canonical

    def _metadata_path_for_save(
        self,
        *,
        job_root: Path,
        safe_job_id: str,
    ) -> Path:
        canonical_root = self._canonical_job_root(
            safe_job_id,
            require_directory=False,
        )
        provided_root = Path(job_root).resolve()
        if provided_root != canonical_root:
            raise ValueError("training job root does not match its job id")
        if Path(job_root).is_symlink():
            raise ValueError("training job directory must not be a symlink")
        metadata_path = canonical_root / METADATA_FILENAME
        if metadata_path.is_symlink():
            raise ValueError("training job metadata must not be a symlink")
        return resolve_under_root(self.root, metadata_path)

    def _metadata_path_for_read(self, job_root: Path) -> Path:
        metadata_path = job_root / METADATA_FILENAME
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError("training job metadata is not a regular file")
        canonical = resolve_under_root(self.root, metadata_path)
        if canonical.parent != job_root:
            raise ValueError("training job metadata is not canonical")
        return canonical

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
    run_plan = RunPlanPersistenceCodec.encode(job.run_plan)
    return {
        "id": job.id,
        "preset": job.preset,
        "presets": job.presets,
        "experiment_task": job.experiment_task,
        "datasets": job.datasets,
        "overrides": job.overrides,
        "search": (
            RunPlanPersistenceCodec.encode_search(job.search)
            if job.search is not None
            else None
        ),
        "planned_run_count": job.planned_run_count,
        "run_plan": run_plan,
        "monitors": job.monitors,
        "log_folder": job.log_folder,
        # Observation only. Recovery never imports or executes this command.
        "command": job.observed_command,
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
        "windows_job_name": job.windows_job_name,
        "exit_code": job.exit_code,
    }


def _record_from_metadata(
    payload: dict[str, Any],
    metadata_path: Path,
) -> TrainingJobRecord:
    if set(payload) != METADATA_FIELDS:
        raise ValueError("Training job metadata fields are not canonical.")
    root = _metadata_root(payload, metadata_path)
    run_plan = RunPlanPersistenceCodec.decode(payload["run_plan"])
    observed_command = payload["command"]
    if not isinstance(observed_command, list) or not all(
        isinstance(item, str) for item in observed_command
    ):
        raise ValueError("Training job command observation must be a string list.")
    return TrainingJobRecord(
        id=str(payload["id"]),
        model=run_plan.model,
        preset=str(payload["preset"]),
        presets=[str(item) for item in payload["presets"]],
        experiment_task=str(payload["experiment_task"]),
        datasets=[str(item) for item in payload["datasets"]],
        overrides=dict(payload["overrides"]),
        search=RunPlanPersistenceCodec.decode_search(
            dict(payload["search"]) if payload.get("search") is not None else None
        ),
        planned_run_count=int(payload["planned_run_count"]),
        run_plan=run_plan,
        monitors=[str(item) for item in payload["monitors"]],
        log_folder=str(payload["log_folder"]),
        observed_command=list(observed_command),
        root=root,
        pid=int(payload["pid"]),
        cancellation_mode=cast(
            ResolvedTrainingCancellationMode,
            str(payload["cancellation_mode"]),
        ),
        worker_pid=(
            int(payload["worker_pid"]) if payload["worker_pid"] is not None else None
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
        windows_job_name=(
            str(payload["windows_job_name"])
            if payload.get("windows_job_name") is not None
            else None
        ),
        status=cast(TrainingJobStatus, str(payload["status"])),
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
            raise ValueError("Training job metadata root is not canonical.")
    except OSError as exc:
        raise ValueError("Training job metadata root cannot be resolved.") from exc
    return metadata_root


__all__ = ["FileSystemTrainingJobStore", "METADATA_FIELDS"]
