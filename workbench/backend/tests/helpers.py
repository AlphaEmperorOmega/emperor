from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from torch.utils.tensorboard import SummaryWriter

from workbench.backend.api.v1.training_mapping import (
    active_training_job_to_payload,
    training_events_page_to_payload,
    training_job_to_payload,
)
from workbench.backend.log_experiments import (
    LogExperimentMutationCoordinator,
)
from workbench.backend.run_history import RunHistoryService
from workbench.backend.run_history.records import LogRunDeleteFilters
from workbench.backend.services.inspection import InspectionService
from workbench.backend.training_jobs.contracts import (
    CreateTrainingJobCommand,
    TrainingSearch,
)
from workbench.backend.training_jobs.run_plan_adapter import (
    submitted_run_plan_from_payload,
    training_search_from_payload,
)
from workbench.backend.training_jobs.runtime import _TrainingJobRuntime
from workbench.backend.training_jobs.service import TrainingJobService


class TrainingJobServiceHarness:
    """Real public Training Job composition plus explicit private-seam access."""

    def __init__(self, **runtime_options: Any) -> None:
        mutation_coordinator = runtime_options.pop(
            "mutation_coordinator",
            LogExperimentMutationCoordinator(),
        )
        self.service = TrainingJobService(
            mutation_coordinator=mutation_coordinator,
            **runtime_options,
        )

    def __getattr__(self, name: str) -> Any:
        """Temporarily preserve private-runtime access for unmigrated tests."""

        return getattr(self.runtime, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "service" or "service" not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        runtime = self.runtime
        if hasattr(runtime, name):
            setattr(runtime, name, value)
            return
        object.__setattr__(self, name, value)

    @property
    def runtime(self) -> _TrainingJobRuntime:
        return self.service._runtime

    @property
    def jobs(self):
        return {job.id: job for job in self.runtime.job_store.list()}

    @property
    def runner(self):
        return self.runtime.worker_launcher.runner

    def create_run_plan(self, **kwargs: Any) -> dict[str, Any]:
        return self.runtime.run_plan_adapter.create_for_request(
            model=kwargs["model"],
            preset=kwargs["preset"],
            presets=kwargs.get("presets"),
            experiment_task=kwargs.get("experiment_task"),
            datasets=kwargs["datasets"],
            overrides=kwargs["overrides"],
            log_folder=kwargs.get("log_folder", ""),
            monitors=kwargs.get("monitors"),
            search=kwargs.get("search"),
        )

    def create_job_payload(self, **kwargs: Any) -> dict[str, Any]:
        search = kwargs.get("search")
        if search is not None and not isinstance(search, TrainingSearch):
            search = training_search_from_payload(search)
        run_plan = kwargs.get("run_plan")
        if run_plan is not None and not hasattr(run_plan, "runs"):
            run_plan = submitted_run_plan_from_payload(run_plan)
        return training_job_to_payload(
            self.service.create_job(
                CreateTrainingJobCommand(
                    model=kwargs["model"],
                    preset=kwargs["preset"],
                    presets=kwargs.get("presets"),
                    experiment_task=kwargs.get("experiment_task"),
                    datasets=kwargs["datasets"],
                    overrides=kwargs["overrides"],
                    log_folder=kwargs["log_folder"],
                    monitors=list(kwargs.get("monitors") or []),
                    search=search,
                    run_plan=run_plan,
                )
            )
        )

    def get_job_payload(self, job_id: str) -> dict[str, Any]:
        return training_job_to_payload(self.service.get_job(job_id))

    def cancel_job_payload(self, job_id: str) -> dict[str, Any]:
        return training_job_to_payload(self.service.cancel_job(job_id))

    def active_job_payloads(self) -> list[dict[str, Any]]:
        return [
            active_training_job_to_payload(job)
            for job in self.service.active_jobs()
        ]

    def get_job_events_payload(
        self,
        job_id: str,
        *,
        offset: int = 0,
        limit: int = 500,
    ) -> dict[str, Any]:
        return training_events_page_to_payload(
            self.service.get_job_events(
                job_id,
                offset=offset,
                limit=limit,
            )
        )

    def get_monitor_data(
        self,
        job_id: str,
        *,
        node_path: str,
        dataset: str | None = None,
        preset: str | None = None,
    ) -> dict[str, Any]:
        return self.service.get_monitor_data(
            job_id,
            node_path=node_path,
            dataset=dataset,
            preset=preset,
        )

    def get_parameter_status(
        self,
        job_id: str,
        *,
        dataset: str | None = None,
        preset: str | None = None,
    ) -> dict[str, Any]:
        return self.service.get_parameter_status(
            job_id,
            dataset=dataset,
            preset=preset,
        )

    create_job = create_job_payload
    get_job = get_job_payload
    cancel_job = cancel_job_payload
    active_jobs = active_job_payloads
    get_job_events = get_job_events_payload


def attach_training_service(app, harness: TrainingJobServiceHarness):
    """Install a service-backed test runtime behind the app capability."""
    services = app.state.workbench_services
    training_jobs = TrainingJobService._from_runtime(
        harness.runtime,
        mutation_coordinator=services.log_experiment_mutations,
    )
    harness.service = training_jobs
    run_history = RunHistoryService(
        logs_root=services.settings.logs_root,
        mutation_coordinator=services.log_experiment_mutations,
        active_log_writers=lambda: training_jobs.active_jobs(),
    )
    app.state.workbench_services = replace(
        services,
        inspection=InspectionService(run_history),
        run_history=run_history,
        training_jobs=training_jobs,
    )
    return app


def create_app_with_training_service(settings, harness: TrainingJobServiceHarness):
    """Create an app whose Training Jobs capability uses a service test rig."""
    from workbench.backend.main import create_app

    return attach_training_service(create_app(settings), harness)


# Transitional aliases keep the untouched tests green while their callers move
# to the public-service vocabulary in a following commit.
TrainingJobRuntimeHarness = TrainingJobServiceHarness
attach_training_runtime = attach_training_service
create_app_with_training_runtime = create_app_with_training_service


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
) -> tuple[TrainingJobServiceHarness, dict[str, object], Path]:
    manager = TrainingJobServiceHarness(
        root=root / "jobs",
        logs_root=root / "logs",
        runner=FakeRunner(),
    )
    payload = manager.create_job_payload(
        model="linears/linear",
        preset="baseline",
        datasets=["Mnist"],
        overrides={},
        log_folder="progress_jsonl",
        monitors=[],
    )
    job = manager.jobs[str(payload["id"])]
    return manager, payload, job.progress_path
