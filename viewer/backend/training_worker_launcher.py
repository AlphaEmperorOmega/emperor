"""Training worker payload, command, and process launch helpers."""

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
from typing import Any, Protocol

from viewer.backend.training_cgroups import (
    CgroupV2Job,
    CgroupV2Manager,
    StrictCancellationUnavailable,
    TrainingCancellationMode,
)


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
    process_group_id: int | None = None
    _last_signal: int | None = None

    def poll(self) -> int | None:
        if self.cgroup.has_processes():
            return None
        if self._last_signal is not None:
            return -self._last_signal
        return 0

    def terminate(self) -> None:
        self._last_signal = signal.SIGTERM
        self._signal_process_group(signal.SIGTERM)
        self.cgroup.terminate()

    def kill(self) -> None:
        signum = getattr(signal, "SIGKILL", signal.SIGTERM)
        self._last_signal = signum
        self._signal_process_group(signum)
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

    def _signal_process_group(self, signum: int) -> None:
        if self.process_group_id is None:
            return
        try:
            os.killpg(self.process_group_id, signum)
        except ProcessLookupError:
            return
        except OSError:
            return


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
        self.cancellation_mode: TrainingCancellationMode = (
            cancellation_mode
            or ("process-group" if runner is not None else "strict-cgroup")
        )
        self.cgroup_manager = cgroup_manager or CgroupV2Manager()
        self.cgroup_join_timeout = cgroup_join_timeout

    def launch(
        self,
        *,
        job_root: Path,
        payload: dict[str, Any],
    ) -> TrainingWorkerLaunch:
        payload_path = self.write_payload(job_root, payload)
        progress_path = job_root / "progress.jsonl"
        command = self.build_command(payload_path, progress_path)
        cgroup: CgroupV2Job | None = None
        ready_path: Path | None = None
        launched_command = command
        if self.cancellation_mode == "strict-cgroup":
            cgroup = self.cgroup_manager.create_job_cgroup(str(payload["id"]))
            ready_path = job_root / "cgroup.ready"
            launched_command = self._wrap_strict_cgroup_command(
                command,
                cgroup=cgroup,
                ready_path=ready_path,
            )
        try:
            process = self.runner.start(
                launched_command,
                cwd=self.cwd,
                env=self.worker_env(),
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
            "viewer.backend.training_worker",
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
            "viewer.backend.cgroup_worker_wrapper",
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
    "TrainingProcessContainment",
    "TrainingWorkerLaunch",
    "TrainingWorkerLauncher",
]
