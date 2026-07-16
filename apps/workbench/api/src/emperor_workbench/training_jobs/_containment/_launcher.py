from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any

from emperor_workbench.filesystem import apply_owner_only_permissions
from emperor_workbench.training_jobs._containment._cgroup_v2 import (
    CgroupProcessHandle,
    CgroupV2Job,
    CgroupV2Manager,
    StrictCancellationUnavailable,
)
from emperor_workbench.training_jobs._containment._process_group import (
    ProcessGroupHandle,
)
from emperor_workbench.training_jobs._containment._protocols import (
    ProcessHandle,
    ProcessRunner,
    TrainingProcessContainment,
    TrainingWorkerLaunch,
)
from emperor_workbench.training_jobs._records import (
    ResolvedTrainingCancellationMode,
    TrainingCancellationCapability,
    TrainingCancellationMode,
    TrainingResourceLimits,
)
from emperor_workbench.training_jobs._store import (
    ensure_private_directory,
    ensure_private_file,
    open_private_binary,
)

TRAINING_LOGS_ROOT_ENV = "WORKBENCH_TRAINING_LOGS_ROOT"

_WORKER_ENVIRONMENT_NAMES = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "HOME",
        "LANG",
        "LD_LIBRARY_PATH",
        "LOCALAPPDATA",
        "MPLCONFIGDIR",
        "PATH",
        "PATHEXT",
        "EMPEROR_PROJECT_ADAPTER_COMMAND",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PYTHONHOME",
        "PYTHONUNBUFFERED",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMPDIR",
        "TMP",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "WINDIR",
    }
)
_WORKER_ENVIRONMENT_PREFIXES = (
    "CUDA_",
    "KMP_",
    "LC_",
    "MKL_",
    "NVIDIA_",
    "OMP_",
    "TORCH_",
)


class TrainingCancellationAdapter:
    """Configured containment selection, observation, and cgroup recovery."""

    def __init__(
        self,
        *,
        mode: TrainingCancellationMode,
        cgroup_manager: CgroupV2Manager | None = None,
        resource_limits: TrainingResourceLimits | None = None,
    ) -> None:
        self._cgroup_manager = cgroup_manager
        self._resource_limits = resource_limits or TrainingResourceLimits()
        self._manager_lock = Lock()
        self.mode = self._resolve_mode(mode)

    def _resolve_mode(
        self,
        mode: TrainingCancellationMode,
    ) -> ResolvedTrainingCancellationMode:
        if mode != "auto":
            return mode
        if os.name == "nt":
            return "windows-job-object"
        if sys.platform.startswith("linux"):
            try:
                if self._manager().is_available():
                    return "strict-cgroup"
            except (OSError, StrictCancellationUnavailable):
                pass
        return "process-group"

    def capability(self) -> TrainingCancellationCapability:
        if self.mode == "process-group":
            return "process-group" if os.name == "posix" else "unsupported"
        if self.mode == "windows-job-object":
            return "windows-job-object" if os.name == "nt" else "unsupported"
        try:
            manager = self._manager()
            return "strict-cgroup" if manager.is_available() else "unsupported"
        except (OSError, StrictCancellationUnavailable):
            return "unsupported"

    def create_job_cgroup(self, job_id: str) -> CgroupV2Job:
        if self.mode != "strict-cgroup":
            raise RuntimeError("Per-job cgroups are only valid in strict-cgroup mode.")
        return self._manager().create_job_cgroup(job_id)

    def create_windows_job(self, job_id: str):
        if self.mode != "windows-job-object":
            raise RuntimeError(
                "Windows Job Objects are only valid in windows-job-object mode."
            )
        from emperor_workbench.training_jobs._containment._windows_job import (
            WindowsJob,
            WindowsJobLimits,
            training_job_object_name,
        )

        return WindowsJob.create(
            name=training_job_object_name(job_id),
            limits=WindowsJobLimits(
                memory_bytes=self._resource_limits.memory_bytes,
                cpu_count=self._resource_limits.cpu_count,
                process_count=self._resource_limits.process_count,
            ),
        )

    def recover_windows_job(self, name: str | None):
        if self.mode != "windows-job-object" or not name:
            return None
        from emperor_workbench.training_jobs._containment._windows_job import WindowsJob

        return WindowsJob.open(name)

    @property
    def resource_limits_enforced(self) -> bool:
        return self.capability() in {"strict-cgroup", "windows-job-object"}

    def recover_job_cgroup(
        self,
        job_id: str,
        *,
        persisted_mode: ResolvedTrainingCancellationMode,
    ) -> CgroupV2Job | None:
        if persisted_mode != "strict-cgroup":
            return None
        try:
            return self._manager().from_job_id(job_id)
        except (OSError, StrictCancellationUnavailable):
            return None

    def _manager(self) -> CgroupV2Manager:
        manager = self._cgroup_manager
        if manager is not None:
            return manager
        with self._manager_lock:
            manager = self._cgroup_manager
            if manager is None:
                manager = CgroupV2Manager(resource_limits=self._resource_limits)
                self._cgroup_manager = manager
        return manager


class SubprocessRunner:
    def start(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
    ) -> ProcessHandle:
        ensure_private_directory(log_path.parent)
        with open_private_binary(
            log_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        ) as log_file:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
        return ProcessGroupHandle(
            process,
            process_group_id=process.pid if os.name == "posix" else None,
        )

    def start_windows_job(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
        job,
    ) -> ProcessHandle:
        ensure_private_directory(log_path.parent)
        with open_private_binary(
            log_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        ) as log_file:
            return job.start_suspended(
                command,
                cwd=cwd,
                env=env,
                stdout=log_file,
                stderr=log_file,
            )


class TrainingWorkerLauncher:
    def __init__(
        self,
        *,
        cwd: Path,
        runner: ProcessRunner | None = None,
        cancellation_mode: TrainingCancellationMode | None = None,
        cgroup_manager: CgroupV2Manager | None = None,
        resource_limits: TrainingResourceLimits | None = None,
        cgroup_join_timeout: float = 2.0,
    ) -> None:
        self.cwd = cwd
        self.runner = runner or SubprocessRunner()
        selected_mode: TrainingCancellationMode = cancellation_mode or (
            "process-group" if runner is not None else "auto"
        )
        self.cancellation = TrainingCancellationAdapter(
            mode=selected_mode,
            cgroup_manager=cgroup_manager,
            resource_limits=resource_limits,
        )
        self.cgroup_join_timeout = cgroup_join_timeout

    @property
    def cancellation_mode(self) -> ResolvedTrainingCancellationMode:
        return self.cancellation.mode

    def cancellation_capability(self) -> TrainingCancellationCapability:
        return self.cancellation.capability()

    def training_resource_limits_enforced(self) -> bool:
        return self.cancellation.resource_limits_enforced

    def recover_job_cgroup(
        self,
        job_id: str,
        *,
        persisted_mode: ResolvedTrainingCancellationMode,
    ) -> CgroupV2Job | None:
        return self.cancellation.recover_job_cgroup(
            job_id,
            persisted_mode=persisted_mode,
        )

    def recover_windows_job(self, name: str | None):
        return self.cancellation.recover_windows_job(name)

    def launch(
        self,
        *,
        job_root: Path,
        payload: dict[str, Any],
        logs_root: Path | str = "logs",
    ) -> TrainingWorkerLaunch:
        ensure_private_directory(job_root)
        payload_path = self.write_payload(job_root, payload)
        progress_path = ensure_private_file(job_root / "progress.jsonl")
        log_path = ensure_private_file(job_root / "training.log")
        command = self.build_command(payload_path, progress_path)
        cgroup: CgroupV2Job | None = None
        windows_job = None
        ready_path: Path | None = None
        launched_command = command
        if self.cancellation_mode == "strict-cgroup":
            cgroup = self.cancellation.create_job_cgroup(str(payload["id"]))
            ready_path = job_root / "cgroup.ready"
            launched_command = self._wrap_strict_cgroup_command(
                command,
                cgroup=cgroup,
                ready_path=ready_path,
            )
        elif self.cancellation_mode == "windows-job-object":
            windows_job = self.cancellation.create_windows_job(str(payload["id"]))
        try:
            worker_env = self.worker_env()
            worker_env[TRAINING_LOGS_ROOT_ENV] = str(Path(logs_root).resolve())
            if windows_job is not None:
                start_windows_job = getattr(self.runner, "start_windows_job", None)
                if not callable(start_windows_job):
                    raise RuntimeError(
                        "The configured process runner cannot assign a suspended "
                        "worker to a Windows Job Object."
                    )
                process = start_windows_job(
                    launched_command,
                    cwd=self.cwd,
                    env=worker_env,
                    log_path=log_path,
                    job=windows_job,
                )
            else:
                process = self.runner.start(
                    launched_command,
                    cwd=self.cwd,
                    env=worker_env,
                    log_path=log_path,
                )
        except Exception:
            if cgroup is not None:
                cgroup.cleanup_empty()
            if windows_job is not None:
                windows_job.close()
            raise
        if cgroup is not None and ready_path is not None:
            process = CgroupProcessHandle(
                process=process,
                cgroup=cgroup,
                process_group_id=getattr(process, "process_group_id", None),
            )
            self._wait_for_cgroup_join(
                process=process,
                ready_path=ready_path,
                cgroup=cgroup,
            )
        return TrainingWorkerLaunch(
            command=command,
            process=process,
            containment=TrainingProcessContainment(
                mode=self.cancellation_mode,
                worker_pid=process.pid,
                process_group_id=getattr(process, "process_group_id", None),
                cgroup_path=cgroup.cgroup_path if cgroup is not None else None,
                windows_job_name=(
                    windows_job.name if windows_job is not None else None
                ),
            ),
        )

    def write_payload(self, job_root: Path, payload: dict[str, Any]) -> Path:
        payload_path = job_root / "payload.json"
        ensure_private_directory(payload_path.parent)
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        with open_private_binary(
            payload_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        ) as payload_file:
            payload_file.write(encoded)
        apply_owner_only_permissions(payload_path)
        return payload_path

    def build_command(
        self,
        payload_path: Path,
        progress_path: Path,
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            "emperor_workbench.training_jobs.worker",
            "--payload",
            str(payload_path),
            "--progress",
            str(progress_path),
        ]

    def worker_env(self) -> dict[str, str]:
        environment = {
            name: value
            for name, value in os.environ.items()
            if name in _WORKER_ENVIRONMENT_NAMES
            or name.startswith(_WORKER_ENVIRONMENT_PREFIXES)
        }
        environment.setdefault(
            "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib")
        )
        return environment

    def _wrap_strict_cgroup_command(
        self,
        command: list[str],
        *,
        cgroup: CgroupV2Job,
        ready_path: Path,
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            "emperor_workbench.training_jobs.cgroup_worker",
            "--cgroup",
            cgroup.cgroup_path,
            "--ready",
            str(ready_path),
            "--",
            *command,
        ]

    def _wait_for_cgroup_join(
        self,
        *,
        process: ProcessHandle,
        ready_path: Path,
        cgroup: CgroupV2Job,
    ) -> None:
        deadline = time.monotonic() + self.cgroup_join_timeout
        while time.monotonic() < deadline:
            if ready_path.is_file():
                return
            exit_code = process.poll()
            if exit_code is not None:
                cgroup.cleanup_empty()
                raise StrictCancellationUnavailable(
                    "Strict training cancellation failed before the worker "
                    "joined its cgroup. See the training log for details."
                )
            time.sleep(0.02)
        process.kill()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        cgroup.cleanup_empty()
        raise StrictCancellationUnavailable(
            "Strict training cancellation timed out while waiting for the "
            "worker to join its cgroup."
        )


__all__ = [
    "SubprocessRunner",
    "TrainingCancellationAdapter",
    "TrainingWorkerLauncher",
]
