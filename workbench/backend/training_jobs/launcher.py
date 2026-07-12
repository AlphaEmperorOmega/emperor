"""Training Job worker payload, launch, and process-containment Adapters."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from workbench.backend.training_jobs.cgroups import (
    CgroupV2Job,
    CgroupV2Manager,
    StrictCancellationUnavailable,
    TrainingCancellationCapability,
    TrainingCancellationMode,
)

TRAINING_LOGS_ROOT_ENV = "WORKBENCH_TRAINING_LOGS_ROOT"


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None:
        ...

    def terminate(self) -> None:
        ...

    def wait(self, timeout: float | None = None) -> int:
        ...

    def kill(self) -> None:
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


class TrainingCancellationAdapter:
    """Configured containment selection, observation, and cgroup recovery."""

    def __init__(
        self,
        *,
        mode: TrainingCancellationMode,
        cgroup_manager: CgroupV2Manager | None = None,
    ) -> None:
        self.mode = mode
        self._cgroup_manager = cgroup_manager
        self._manager_lock = Lock()

    def capability(self) -> TrainingCancellationCapability:
        if self.mode == "process-group":
            return "process-group" if os.name == "posix" else "unsupported"
        try:
            manager = self._manager()
            return "strict-cgroup" if manager.is_available() else "unsupported"
        except (OSError, StrictCancellationUnavailable):
            return "unsupported"

    def create_job_cgroup(self, job_id: str) -> CgroupV2Job:
        if self.mode != "strict-cgroup":
            raise RuntimeError(
                "Per-job cgroups are only valid in strict-cgroup mode."
            )
        return self._manager().create_job_cgroup(job_id)

    def recover_job_cgroup(
        self,
        job_id: str,
        *,
        persisted_mode: TrainingCancellationMode,
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
                manager = CgroupV2Manager()
                self._cgroup_manager = manager
        return manager


@dataclass(frozen=True, slots=True)
class TrainingProcessContainment:
    mode: TrainingCancellationMode
    worker_pid: int
    process_group_id: int | None = None
    cgroup_path: str | None = None


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
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_file:
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
        cgroup_join_timeout: float = 2.0,
    ) -> None:
        self.cwd = cwd
        self.runner = runner or SubprocessRunner()
        selected_mode: TrainingCancellationMode = (
            cancellation_mode
            or ("process-group" if runner is not None else "strict-cgroup")
        )
        self.cancellation = TrainingCancellationAdapter(
            mode=selected_mode,
            cgroup_manager=cgroup_manager,
        )
        self.cgroup_join_timeout = cgroup_join_timeout

    @property
    def cancellation_mode(self) -> TrainingCancellationMode:
        return self.cancellation.mode

    def cancellation_capability(self) -> TrainingCancellationCapability:
        return self.cancellation.capability()

    def recover_job_cgroup(
        self,
        job_id: str,
        *,
        persisted_mode: TrainingCancellationMode,
    ) -> CgroupV2Job | None:
        return self.cancellation.recover_job_cgroup(
            job_id,
            persisted_mode=persisted_mode,
        )

    def launch(
        self,
        *,
        job_root: Path,
        payload: dict[str, Any],
        logs_root: Path | str = "logs",
    ) -> TrainingWorkerLaunch:
        payload_path = self.write_payload(job_root, payload)
        progress_path = job_root / "progress.jsonl"
        command = self.build_command(payload_path, progress_path)
        cgroup: CgroupV2Job | None = None
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
        try:
            worker_env = self.worker_env()
            worker_env[TRAINING_LOGS_ROOT_ENV] = str(Path(logs_root).resolve())
            process = self.runner.start(
                launched_command,
                cwd=self.cwd,
                env=worker_env,
                log_path=job_root / "training.log",
            )
        except Exception:
            if cgroup is not None:
                cgroup.cleanup_empty()
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
            ),
        )

    def write_payload(self, job_root: Path, payload: dict[str, Any]) -> Path:
        payload_path = job_root / "payload.json"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
        return {
            **os.environ,
            "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"),
        }

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
]
