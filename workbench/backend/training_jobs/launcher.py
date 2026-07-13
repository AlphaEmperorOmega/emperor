from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from workbench.backend.storage.local_files import (
    apply_owner_only_permissions,
    reject_link_like,
)
from workbench.backend.training_jobs.cgroups import (
    CgroupV2Job,
    CgroupV2Manager,
    ResolvedTrainingCancellationMode,
    StrictCancellationUnavailable,
    TrainingCancellationCapability,
    TrainingCancellationMode,
    TrainingResourceLimits,
)

TRAINING_LOGS_ROOT_ENV = "WORKBENCH_TRAINING_LOGS_ROOT"
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600

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


def ensure_private_directory(path: Path) -> Path:
    """Create or tighten one private directory without accepting symlinks."""

    candidate = Path(path)
    reject_link_like(candidate, "private directory")
    candidate.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIRECTORY_MODE)
    reject_link_like(candidate, "private directory")
    if not candidate.is_dir():
        raise ValueError(f"Private directory is not canonical: {candidate}")
    apply_owner_only_permissions(candidate)
    return candidate


def _open_private_binary(path: Path, flags: int):
    if path.exists():
        reject_link_like(path, "private file")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(
        path,
        flags | no_follow | close_on_exec,
        PRIVATE_FILE_MODE,
    )
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
        return os.fdopen(descriptor, "ab" if flags & os.O_APPEND else "wb")
    except Exception:
        os.close(descriptor)
        raise


def ensure_private_file(path: Path) -> Path:
    """Create or tighten a regular private file without following symlinks."""

    ensure_private_directory(path.parent)
    with _open_private_binary(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND):
        pass
    apply_owner_only_permissions(path)
    return path


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class ProcessRunner(Protocol):
    def start(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
    ) -> ProcessHandle: ...


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
        from workbench.backend.windows_jobs import (
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
        from workbench.backend.windows_jobs import WindowsJob

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


@dataclass(frozen=True, slots=True)
class TrainingProcessContainment:
    mode: ResolvedTrainingCancellationMode
    worker_pid: int
    process_group_id: int | None = None
    cgroup_path: str | None = None
    windows_job_name: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessGroupHandle:
    process: subprocess.Popen
    process_group_id: int | None = None

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        exit_code = self.process.poll()
        if exit_code is not None and self._process_group_exists():
            return None
        return exit_code

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        exit_code = self.process.wait(timeout=timeout)
        self._wait_for_process_group(deadline=deadline, timeout=timeout)
        return exit_code

    def terminate(self) -> None:
        self._signal(signal.SIGTERM, self.process.terminate)

    def kill(self) -> None:
        self._signal(getattr(signal, "SIGKILL", signal.SIGTERM), self.process.kill)

    def _signal(self, signum: int, fallback: Callable[[], None]) -> None:
        if self.process_group_id is None:
            fallback()
            return
        try:
            os.killpg(self.process_group_id, signum)
        except ProcessLookupError:
            return
        except OSError:
            fallback()

    def _wait_for_process_group(
        self,
        *,
        deadline: float | None,
        timeout: float | None,
    ) -> None:
        if self.process_group_id is None:
            return
        while self._process_group_exists():
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(
                        cmd=self.process.args,
                        timeout=timeout or 0.0,
                    )
                time.sleep(min(0.05, remaining))
            else:
                time.sleep(0.05)

    def _process_group_exists(self) -> bool:
        if self.process_group_id is None:
            return False
        try:
            os.killpg(self.process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


@dataclass(slots=True)
class CgroupProcessHandle:
    process: ProcessHandle
    cgroup: CgroupV2Job
    process_group_id: int | None = None

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        exit_code = self.process.poll()
        if exit_code is not None and self.cgroup.has_processes():
            return None
        return exit_code

    def terminate(self) -> None:
        self.process.terminate()
        self.cgroup.terminate()

    def kill(self) -> None:
        self.process.kill()
        self.cgroup.kill()

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        exit_code = self.process.wait(timeout=timeout)
        cgroup_timeout = (
            None if deadline is None else max(0.0, deadline - time.monotonic())
        )
        try:
            self.cgroup.wait_empty(timeout=cgroup_timeout)
        except TimeoutError as exc:
            raise subprocess.TimeoutExpired(
                cmd=["training-cgroup", self.cgroup.cgroup_path],
                timeout=timeout or 0.0,
            ) from exc
        self.cgroup.cleanup_empty()
        return exit_code


@dataclass(slots=True)
class PersistedCgroupProcessHandle:
    pid: int
    cgroup: CgroupV2Job
    _last_signal: int | None = None

    def has_live_containment(self) -> bool:
        return self.cgroup.has_processes()

    def poll(self) -> int | None:
        if self.has_live_containment():
            return None
        if self._last_signal is not None:
            return -self._last_signal
        return 0

    def terminate(self) -> None:
        self._last_signal = signal.SIGTERM
        self.cgroup.terminate()

    def kill(self) -> None:
        signum = getattr(signal, "SIGKILL", signal.SIGTERM)
        self._last_signal = signum
        self.cgroup.kill()

    def wait(self, timeout: float | None = None) -> int:
        try:
            self.cgroup.wait_empty(timeout=timeout)
        except TimeoutError as exc:
            raise subprocess.TimeoutExpired(
                cmd=["training-cgroup", self.cgroup.cgroup_path],
                timeout=timeout or 0.0,
            ) from exc
        self.cgroup.cleanup_empty()
        return self.poll() or 0


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
        with _open_private_binary(
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
        with _open_private_binary(
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


@dataclass(frozen=True, slots=True)
class TrainingWorkerLaunch:
    command: list[str]
    process: ProcessHandle
    containment: TrainingProcessContainment


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
        with _open_private_binary(
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
            "workbench.backend.training_worker",
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
            "workbench.backend.cgroup_worker_wrapper",
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
    "CgroupProcessHandle",
    "PersistedCgroupProcessHandle",
    "ProcessHandle",
    "ProcessGroupHandle",
    "ProcessRunner",
    "SubprocessRunner",
    "TrainingCancellationAdapter",
    "TrainingProcessContainment",
    "TrainingWorkerLaunch",
    "TrainingWorkerLauncher",
    "ensure_private_directory",
    "ensure_private_file",
]
