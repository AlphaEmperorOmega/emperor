"""Training worker payload, command, and process launch helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


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


@dataclass(frozen=True, slots=True)
class TrainingWorkerLaunch:
    command: list[str]
    process: ProcessHandle


class TrainingWorkerLauncher:
    def __init__(
        self,
        *,
        cwd: Path,
        runner: ProcessRunner | None = None,
    ) -> None:
        self.cwd = cwd
        self.runner = runner or SubprocessRunner()

    def launch(
        self,
        *,
        job_root: Path,
        payload: dict[str, Any],
    ) -> TrainingWorkerLaunch:
        payload_path = self.write_payload(job_root, payload)
        progress_path = job_root / "progress.jsonl"
        command = self.build_command(payload_path, progress_path)
        process = self.runner.start(
            command,
            cwd=self.cwd,
            env=self.worker_env(),
            log_path=job_root / "training.log",
        )
        return TrainingWorkerLaunch(command=command, process=process)

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


__all__ = [
    "ProcessHandle",
    "ProcessRunner",
    "SubprocessRunner",
    "TrainingWorkerLaunch",
    "TrainingWorkerLauncher",
]
