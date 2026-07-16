from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from emperor_workbench.filesystem import require_safe_name
from emperor_workbench.training_jobs._containment._protocols import ProcessHandle
from emperor_workbench.training_jobs._records import TrainingResourceLimits

CGROUP_V2_MOUNT = Path("/sys/fs/cgroup")
CGROUP_NAMESPACE = "emperor-workbench-training"


class StrictCancellationUnavailable(RuntimeError):
    """Raised when strict cgroup containment cannot be created."""


def _is_linux() -> bool:
    return os.name == "posix" and sys.platform.startswith("linux")


def _current_cgroup_relative_path() -> str:
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StrictCancellationUnavailable(
            "Strict training cancellation requires /proc/self/cgroup."
        ) from exc
    for line in lines:
        if line.startswith("0::"):
            return line.removeprefix("0::").strip() or "/"
    raise StrictCancellationUnavailable(
        "Strict training cancellation requires a unified cgroup v2 hierarchy."
    )


def current_cgroup_path() -> Path:
    relative = _current_cgroup_relative_path()
    return CGROUP_V2_MOUNT / relative.lstrip("/")


@dataclass(frozen=True, slots=True)
class CgroupV2Job:
    path: Path

    @property
    def cgroup_path(self) -> str:
        return str(self.path)

    def read_pids(self) -> set[int]:
        pids: set[int] = set()
        for procs_path in self._process_files():
            try:
                text = procs_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            except OSError:
                continue
            for raw_pid in text.split():
                try:
                    pids.add(int(raw_pid))
                except ValueError:
                    continue
        return pids

    def has_processes(self) -> bool:
        return bool(self.read_pids())

    def signal(self, signum: int) -> None:
        for pid in sorted(self.read_pids(), reverse=True):
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue

    def terminate(self) -> None:
        self.signal(signal.SIGTERM)

    def kill(self) -> None:
        kill_path = self.path / "cgroup.kill"
        try:
            kill_path.write_text("1", encoding="utf-8")
            return
        except FileNotFoundError:
            pass
        except OSError:
            pass
        self.signal(getattr(signal, "SIGKILL", signal.SIGTERM))

    def wait_empty(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.has_processes():
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for cgroup to empty: {self.path}"
                    )
                time.sleep(min(0.05, remaining))
            else:
                time.sleep(0.05)

    def cleanup_empty(self) -> None:
        if self.has_processes():
            return
        try:
            self.path.rmdir()
        except OSError:
            return

    def _process_files(self) -> list[Path]:
        if not self.path.exists():
            return []
        process_files: list[Path] = []
        pending = [self.path]
        while pending:
            current = pending.pop()
            process_files.append(current / "cgroup.procs")
            try:
                children = list(current.iterdir())
            except OSError:
                continue
            pending.extend(child for child in children if child.is_dir())
        return process_files


class CgroupV2Manager:
    def __init__(
        self,
        *,
        base_path: Path | None = None,
        namespace: str = CGROUP_NAMESPACE,
        resource_limits: TrainingResourceLimits | None = None,
    ) -> None:
        self._base_path = base_path
        self.namespace = require_safe_name(namespace, "cgroup namespace")
        self.resource_limits = resource_limits or TrainingResourceLimits()

    @property
    def base_path(self) -> Path:
        if self._base_path is None:
            self._base_path = current_cgroup_path() if _is_linux() else CGROUP_V2_MOUNT
        return self._base_path

    def is_available(self) -> bool:
        if not _is_linux():
            return False
        if not (CGROUP_V2_MOUNT / "cgroup.controllers").exists():
            return False
        try:
            base_path = self.base_path
            if not base_path.is_dir() or not os.access(base_path, os.W_OK | os.X_OK):
                return False
            self._probe_resource_controls()
            return True
        except (OSError, StrictCancellationUnavailable):
            return False

    def _probe_resource_controls(self) -> None:
        namespace_path = self.base_path / self.namespace
        if namespace_path.is_symlink():
            raise StrictCancellationUnavailable(
                "Strict training cancellation refuses a symlink cgroup namespace."
            )
        created_namespace = False
        try:
            namespace_path.mkdir()
            created_namespace = True
        except FileExistsError:
            if not namespace_path.is_dir():
                raise
        probe_path = namespace_path / f".probe-{os.getpid()}-{time.time_ns()}"
        try:
            probe_path.mkdir()
            self._write_resource_limits(probe_path)
        finally:
            # Regular-filesystem tests create these files; real cgroup control
            # files are virtual and reject unlinking before the cgroup is removed.
            for filename in self._resource_limit_settings():
                try:
                    (probe_path / filename).unlink()
                except OSError:
                    pass
            try:
                probe_path.rmdir()
            except OSError:
                pass
            if created_namespace:
                try:
                    namespace_path.rmdir()
                except OSError:
                    pass

    def require_available(self) -> None:
        if not _is_linux():
            raise StrictCancellationUnavailable(
                "Strict training cancellation requires Linux cgroup v2."
            )
        if not (CGROUP_V2_MOUNT / "cgroup.controllers").exists():
            raise StrictCancellationUnavailable(
                "Strict training cancellation requires a cgroup v2 mount at "
                f"{CGROUP_V2_MOUNT}."
            )
        if not self.base_path.is_dir():
            raise StrictCancellationUnavailable(
                "Strict training cancellation requires the current cgroup path "
                f"to exist: {self.base_path}."
            )
        if not os.access(self.base_path, os.W_OK | os.X_OK):
            raise StrictCancellationUnavailable(
                "Strict training cancellation requires a writable/delegated "
                f"cgroup under {self.base_path}."
            )

    def create_job_cgroup(self, job_id: str) -> CgroupV2Job:
        self.require_available()
        namespace_path = self.base_path / self.namespace
        if namespace_path.is_symlink():
            raise StrictCancellationUnavailable(
                "Strict training cancellation refuses a symlink cgroup namespace."
            )
        job_path = self._job_path(job_id)
        try:
            namespace_path.mkdir(exist_ok=True)
            job_path.mkdir(exist_ok=False)
            self._write_resource_limits(job_path)
        except OSError as exc:
            try:
                job_path.rmdir()
            except OSError:
                pass
            raise StrictCancellationUnavailable(
                "Strict Training Job containment could not create and limit a "
                f"per-job cgroup under {self.base_path}: {exc}"
            ) from exc
        return CgroupV2Job(job_path)

    def _write_resource_limits(self, job_path: Path) -> None:
        for filename, value in self._resource_limit_settings().items():
            (job_path / filename).write_text(value, encoding="ascii")

    def _resource_limit_settings(self) -> dict[str, str]:
        period_us = 100_000
        return {
            "memory.max": str(self.resource_limits.memory_bytes),
            "cpu.max": f"{self.resource_limits.cpu_count * period_us} {period_us}",
            "pids.max": str(self.resource_limits.process_count),
        }

    def from_job_id(self, job_id: str) -> CgroupV2Job | None:
        """Recover only the canonical cgroup owned by this manager and job."""
        try:
            job_path = self._job_path(job_id)
        except ValueError:
            return None
        namespace_path = self.base_path / self.namespace
        if namespace_path.is_symlink() or job_path.is_symlink():
            return None
        if not namespace_path.is_dir() or not job_path.is_dir():
            return None
        try:
            resolved_namespace = namespace_path.resolve()
            resolved_job_path = job_path.resolve()
        except OSError:
            return None
        if resolved_job_path.parent != resolved_namespace:
            return None
        return CgroupV2Job(resolved_job_path)

    def _job_path(self, job_id: str) -> Path:
        safe_job_id = require_safe_name(job_id, "training job id")
        return self.base_path / self.namespace / f"job-{safe_job_id}"


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


__all__ = [
    "CgroupProcessHandle",
    "CgroupV2Job",
    "CgroupV2Manager",
    "PersistedCgroupProcessHandle",
    "StrictCancellationUnavailable",
]
