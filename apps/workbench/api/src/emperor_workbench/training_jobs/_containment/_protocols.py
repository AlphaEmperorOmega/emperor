from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from emperor_workbench.training_jobs._records import (
    ResolvedTrainingCancellationMode,
)


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


@dataclass(frozen=True, slots=True)
class TrainingProcessContainment:
    mode: ResolvedTrainingCancellationMode
    worker_pid: int
    process_group_id: int | None = None
    cgroup_path: str | None = None
    windows_job_name: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingWorkerLaunch:
    command: list[str]
    process: ProcessHandle
    containment: TrainingProcessContainment


__all__ = [
    "ProcessHandle",
    "ProcessRunner",
    "TrainingProcessContainment",
    "TrainingWorkerLaunch",
]
