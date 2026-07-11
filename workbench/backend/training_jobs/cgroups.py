"""Linux cgroup-v2 Adapter for strict Training Job containment."""

from __future__ import annotations

import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from workbench.backend.storage.local_files import require_safe_name

TrainingCancellationCapability = Literal[
    "strict-cgroup",
    "process-group",
    "unsupported",
]
TrainingCancellationMode = Literal["strict-cgroup", "process-group"]

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


def requested_cancellation_capability(
    mode: TrainingCancellationMode,
) -> TrainingCancellationCapability:
    if mode == "process-group":
        return "process-group" if os.name == "posix" else "unsupported"
    return "strict-cgroup" if CgroupV2Manager().is_available() else "unsupported"


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
    ) -> None:
        self.base_path = base_path or (
            current_cgroup_path() if _is_linux() else CGROUP_V2_MOUNT
        )
        self.namespace = require_safe_name(namespace, "cgroup namespace")

    def is_available(self) -> bool:
        if not _is_linux():
            return False
        if not (CGROUP_V2_MOUNT / "cgroup.controllers").exists():
            return False
        return (
            self.base_path.is_dir()
            and os.access(self.base_path, os.W_OK | os.X_OK)
        )

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
        except OSError as exc:
            raise StrictCancellationUnavailable(
                "Strict training cancellation could not create a per-job "
                f"cgroup under {self.base_path}: {exc}"
            ) from exc
        return CgroupV2Job(job_path)

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


__all__ = [
    "CgroupV2Job",
    "CgroupV2Manager",
    "StrictCancellationUnavailable",
    "TrainingCancellationCapability",
    "TrainingCancellationMode",
    "requested_cancellation_capability",
]
