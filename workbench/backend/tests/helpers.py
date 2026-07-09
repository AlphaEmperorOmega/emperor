from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from torch.utils.tensorboard import SummaryWriter

from workbench.backend.log_runs import LogRunDeleteFilters
from workbench.backend.training_jobs import TrainingJobManager


class FakeProcess:
    pid = 1234

    def __init__(
        self,
        exit_code: int | None = None,
        *,
        ignores_terminate: bool = False,
        ignores_kill: bool = False,
    ) -> None:
        self.exit_code = exit_code
        self.terminated = False
        self.killed = False
        self.ignores_terminate = ignores_terminate
        self.ignores_kill = ignores_kill

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True
        if not self.ignores_terminate:
            self.exit_code = -15

    def kill(self) -> None:
        self.killed = True
        if not self.ignores_kill:
            self.exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.exit_code is None:
            raise subprocess.TimeoutExpired(
                cmd=["fake-training-worker"],
                timeout=timeout or 0.0,
            )
        return self.exit_code


class FakeRunner:
    def __init__(self, process: FakeProcess | None = None) -> None:
        self.process = process or FakeProcess()
        self.commands: list[list[str]] = []
        self.log_paths: list[Path] = []

    def start(self, command, *, cwd, env, log_path):
        self.commands.append(command)
        self.log_paths.append(Path(log_path))
        log_path.write_text("fake training log\n", encoding="utf-8")
        return self.process


def write_tensorboard_run(
    logs_root: Path,
    relative_parts: list[str],
    *,
    scalars: dict[str, list[tuple[int, float]]] | None = None,
    metrics: dict[str, object] | None = None,
    hparams: bool = True,
    checkpoint: bool = True,
) -> Path:
    run_dir = logs_root.joinpath(*relative_parts)
    writer = SummaryWriter(log_dir=str(run_dir))
    for tag, points in (scalars or {"train/loss": [(1, 0.5)]}).items():
        for step, value in points:
            writer.add_scalar(tag, value, step)
    writer.flush()
    writer.close()

    if hparams:
        (run_dir / "hparams.yaml").write_text("batch_size: 4\n", encoding="utf-8")
    if checkpoint:
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        (checkpoint_dir / "epoch=0-step=1.ckpt").write_text(
            "checkpoint", encoding="utf-8"
        )
    if metrics is not None:
        (run_dir / "result.json").write_text(
            json.dumps({"metrics": metrics}),
            encoding="utf-8",
        )
    return run_dir


def delete_filters_for_runs(
    runs,
    *,
    experiments: list[str] | None = None,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    presets: list[str] | None = None,
    run_ids: list[str] | None = None,
) -> LogRunDeleteFilters:
    return LogRunDeleteFilters(
        experiments=(
            experiments
            if experiments is not None
            else sorted({run.experiment for run in runs})
        ),
        datasets=(
            datasets if datasets is not None else sorted({run.dataset for run in runs})
        ),
        models=models if models is not None else sorted({run.model for run in runs}),
        presets=(
            presets if presets is not None else sorted({run.preset for run in runs})
        ),
        runIds=run_ids if run_ids is not None else sorted({run.id for run in runs}),
    )


def create_progress_test_job(
    root: Path,
) -> tuple[TrainingJobManager, dict[str, object], Path]:
    manager = TrainingJobManager(
        root=root / "jobs",
        logs_root=root / "logs",
        runner=FakeRunner(),
    )
    payload = manager.create_job(
        model="linears/linear",
        preset="baseline",
        datasets=["Mnist"],
        overrides={},
        log_folder="progress_jsonl",
        monitors=[],
    )
    job = manager.jobs[str(payload["id"])]
    return manager, payload, job.progress_path
