from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from torch.utils.tensorboard import SummaryWriter

from emperor_workbench.log_experiments import (
    LogExperimentMutationCoordinator,
)
from emperor_workbench.model_packages import (
    ModelPackageCatalog,
    ModelPackageIdentity,
)
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.run_history import LogRunDeleteFilters
from emperor_workbench.run_plans import (
    ConfigSnapshotRevision,
    CreateTrainingRunPlanCommand,
    MaterializeTrainingRunPlanCommand,
    RunPlanService,
    SubmittedTrainingRun,
    SubmittedTrainingRunPlan,
    TrainingRunPlanView,
    TrainingSearch,
)
from emperor_workbench.training_jobs import (
    ActiveTrainingJob,
    CreateTrainingJobCommand,
    TrainingJobService,
    TrainingJobView,
    TrainingProgressEventsPage,
)
from tests.support.model_packages import project_adapter_client

_OPTIONAL_EVENT_METADATA_KEYS = {
    "eventBytes",
    "returnedItemCount",
    "skippedEventFiles",
    "sourceItemCount",
    "truncated",
    "truncationReason",
}


def _model_identity(model_id: str) -> dict[str, str]:
    identity = ModelPackageIdentity.from_id(model_id)
    return {"modelType": identity.model_type, "model": identity.model}


def _search_to_payload(search: TrainingSearch) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": search.mode,
        "values": search.values,
    }
    if search.random_samples is not None:
        payload["randomSamples"] = search.random_samples
    return payload


def run_plan_payload(plan: TrainingRunPlanView) -> dict[str, Any]:
    return {
        **_model_identity(plan.model),
        "preset": plan.preset,
        "presets": plan.presets,
        "experimentTask": plan.experiment_task,
        "datasets": plan.datasets,
        "overrides": plan.overrides,
        "search": _search_to_payload(plan.search) if plan.search else None,
        "logFolder": plan.log_folder,
        "isRandomSearch": plan.is_random_search,
        "runs": [
            {
                "id": run.id,
                "index": run.index,
                "status": run.status,
                "preset": run.preset,
                **(
                    {"snapshotId": run.snapshot_id}
                    if run.snapshot_id_present or run.snapshot_id is not None
                    else {}
                ),
                **(
                    {"snapshotName": run.snapshot_name}
                    if run.snapshot_name_present or run.snapshot_name is not None
                    else {}
                ),
                "dataset": run.dataset,
                "experimentTask": run.experiment_task,
                "changes": [
                    {
                        "key": change.key,
                        "label": change.label,
                        "value": change.value,
                        "source": change.source,
                    }
                    for change in run.changes
                ],
                "overrides": run.overrides,
                "command": run.command,
                "commandArgv": run.command_argv,
                "commands": {
                    "posix": run.commands.posix,
                    "powershell": run.commands.powershell,
                },
                "totalEpochs": run.total_epochs,
                "currentEpoch": run.current_epoch,
                "metrics": run.metrics,
                "logDir": run.log_dir,
                "error": run.error,
                "errorTraceback": run.error_traceback,
            }
            for run in plan.runs
        ],
        "summary": {
            "totalRuns": plan.summary.total_runs,
            "completedRuns": plan.summary.completed_runs,
            "runningRuns": plan.summary.running_runs,
            "pendingRuns": plan.summary.pending_runs,
            "failedRuns": plan.summary.failed_runs,
            "cancelledRuns": plan.summary.cancelled_runs,
            "skippedRuns": plan.summary.skipped_runs,
            "totalEpochs": plan.summary.total_epochs,
            "completedEpochs": plan.summary.completed_epochs,
            "remainingEpochs": plan.summary.remaining_epochs,
        },
        "snapshotRevisions": [
            {
                "id": revision.id,
                "semanticRevision": revision.semantic_revision,
            }
            for revision in plan.snapshot_revisions
        ],
    }


def training_job_payload(job: TrainingJobView) -> dict[str, Any]:
    return {
        "id": job.id,
        "status": job.status,
        **_model_identity(job.model),
        "preset": job.preset,
        "presets": job.presets,
        "experimentTask": job.experiment_task,
        "datasets": job.datasets,
        "overrides": job.overrides,
        "search": _search_to_payload(job.search) if job.search else None,
        "plannedRunCount": job.planned_run_count,
        "runPlan": run_plan_payload(job.run_plan) if job.run_plan else None,
        "monitors": job.monitors,
        "logFolder": job.log_folder,
        "createdAt": job.created_at,
        "updatedAt": job.updated_at,
        "exitCode": job.exit_code,
        "pid": job.pid,
        "cancellationMode": job.cancellation_mode,
        "currentPreset": job.current_preset,
        "currentDataset": job.current_dataset,
        "epoch": job.epoch,
        "step": job.step,
        "metrics": job.metrics,
        "logDir": job.log_dir,
        "events": job.events,
        "eventCount": job.event_count,
        "eventCounts": job.event_counts,
        "eventsTruncated": job.events_truncated,
        "clusterGrowth": job.cluster_growth,
        "logTail": job.log_tail,
        "logTailTruncated": job.log_tail_truncated,
        "resultLinks": [
            {
                "preset": link.preset,
                "dataset": link.dataset,
                "logDir": link.log_dir,
            }
            for link in job.result_links
        ],
    }


def _active_training_job_to_payload(job: ActiveTrainingJob) -> dict[str, str]:
    return {
        "id": job.id,
        "status": job.status,
        "logFolder": job.log_folder,
    }


def _training_events_page_to_payload(
    page: TrainingProgressEventsPage,
) -> dict[str, Any]:
    return {
        "jobId": page.job_id,
        "offset": page.offset,
        "limit": page.limit,
        "totalCount": page.total_count,
        "nextOffset": page.next_offset,
        "events": page.events,
    }


def _scalar_point_to_payload(point: Any) -> dict[str, Any]:
    return {
        "step": point.step,
        "wallTime": point.wall_time,
        "value": point.value,
    }


def _monitor_data_to_payload(data: Any) -> dict[str, Any]:
    return {
        "jobId": data.job_id,
        "nodePath": data.node_path,
        "preset": data.preset,
        "dataset": data.dataset,
        "logDir": data.log_dir,
        "eventBytes": data.event_bytes,
        "skippedEventFiles": data.skipped_event_files,
        "truncated": data.truncated,
        "truncationReason": data.truncation_reason,
        "sourceItemCount": data.source_item_count,
        "returnedItemCount": data.returned_item_count,
        "scalarSeries": [
            {
                "tag": series.tag,
                "label": series.label,
                "points": [_scalar_point_to_payload(point) for point in series.points],
                "sourceItemCount": series.source_item_count,
                "returnedItemCount": series.returned_item_count,
                "truncated": series.truncated,
                "truncationReason": series.truncation_reason,
            }
            for series in data.scalar_series
        ],
        "histograms": [
            {
                "tag": histogram.tag,
                "step": histogram.step,
                "wallTime": histogram.wall_time,
                "buckets": [
                    {
                        "left": bucket.left,
                        "right": bucket.right,
                        "count": bucket.count,
                    }
                    for bucket in histogram.buckets
                ],
                "sourceItemCount": histogram.source_item_count,
                "returnedItemCount": histogram.returned_item_count,
                "truncated": histogram.truncated,
                "truncationReason": histogram.truncation_reason,
            }
            for histogram in data.histograms
        ],
        "images": [
            {
                "tag": image.tag,
                "step": image.step,
                "wallTime": image.wall_time,
                "mimeType": image.mime_type,
                "dataUrl": image.data_url,
                "eventBytes": image.event_bytes,
                "sourceItemCount": image.source_item_count,
                "returnedItemCount": image.returned_item_count,
                "truncated": image.truncated,
                "truncationReason": image.truncation_reason,
            }
            for image in data.images
        ],
    }


def _parameter_channel_to_payload(channel: Any) -> dict[str, Any]:
    return {
        "status": channel.status,
        "metric": channel.metric,
        "lastStep": channel.last_step,
        "observedPoints": channel.observed_points,
    }


def _parameter_status_to_payload(status: Any) -> dict[str, Any]:
    return {
        "sourceId": status.source_id,
        "preset": status.preset,
        "dataset": status.dataset,
        "logDir": status.log_dir,
        "eventBytes": status.event_bytes,
        "skippedEventFiles": status.skipped_event_files,
        "truncated": status.truncated,
        "truncationReason": status.truncation_reason,
        "sourceItemCount": status.source_item_count,
        "returnedItemCount": status.returned_item_count,
        "nodes": [
            {
                "nodePath": node.node_path,
                "weights": _parameter_channel_to_payload(node.weights),
                "bias": _parameter_channel_to_payload(node.bias),
            }
            for node in status.nodes
        ],
    }


def _without_absent_event_metadata(value: Any) -> Any:
    if isinstance(value, list):
        return [_without_absent_event_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _without_absent_event_metadata(item)
        for key, item in value.items()
        if key not in _OPTIONAL_EVENT_METADATA_KEYS or item is not None
    }


class TrainingJobServiceHarness:
    """Real public Training Job composition plus worker-wire test helpers."""

    def __init__(self, **runtime_options: Any) -> None:
        self.mutation_coordinator = runtime_options.pop(
            "mutation_coordinator",
            LogExperimentMutationCoordinator(),
        )
        project_adapter = runtime_options.pop("project_adapter", None)
        if project_adapter is None:
            project_adapter = project_adapter_client()
        config_snapshots = runtime_options.pop("config_snapshots", None)
        self.run_plans = runtime_options.pop(
            "run_plans",
            RunPlanService(
                model_packages=ModelPackageCatalog(project_adapter),
                config_snapshots=config_snapshots,
            ),
        )
        self.root = Path(
            runtime_options.get("root")
            or Path(tempfile.gettempdir()) / "emperor-workbench-training"
        ).resolve()
        self.runner = runtime_options.get("runner")
        self.service = TrainingJobService(
            mutation_coordinator=self.mutation_coordinator,
            run_plans=self.run_plans,
            **runtime_options,
        )

    def create_run_plan(self, **kwargs: Any) -> TrainingRunPlanView:
        search = kwargs.get("search")
        if search is not None and not isinstance(search, TrainingSearch):
            search = TrainingSearch(
                mode=search["mode"],
                values={
                    key: list(values)
                    for key, values in search.get("values", {}).items()
                },
                random_samples=search.get("randomSamples"),
            )
        return self.run_plans.preview(
            CreateTrainingRunPlanCommand(
                model=kwargs["model"],
                preset=kwargs["preset"],
                presets=kwargs.get("presets"),
                experiment_task=kwargs.get("experiment_task"),
                datasets=kwargs["datasets"],
                overrides=kwargs["overrides"],
                log_folder=kwargs.get("log_folder", ""),
                monitors=list(kwargs.get("monitors") or []),
                search=search,
                snapshot_ids=list(kwargs.get("snapshot_ids") or []),
            )
        )

    def progress_path(self, job_id: str) -> Path:
        return self.root / job_id / "progress.jsonl"

    def append_progress_event(
        self,
        job_id: str,
        event: dict[str, Any],
    ) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "jobId": job_id,
            **event,
        }
        progress_path = self.progress_path(job_id)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8") as progress_file:
            progress_file.write(json.dumps(payload, default=str))
            progress_file.write("\n")

    def create_job_payload(self, **kwargs: Any) -> dict[str, Any]:
        search = kwargs.get("search")
        if search is not None and not isinstance(search, TrainingSearch):
            search = TrainingSearch(
                mode=search["mode"],
                values={
                    key: list(values)
                    for key, values in search.get("values", {}).items()
                },
                random_samples=search.get("randomSamples"),
            )
        run_plan = kwargs.get("run_plan")
        if isinstance(run_plan, TrainingRunPlanView):
            run_plan = SubmittedTrainingRunPlan(
                runs=[
                    SubmittedTrainingRun(
                        id=run.id,
                        preset=run.preset,
                        dataset=run.dataset,
                        overrides=dict(run.overrides),
                        snapshot_id=run.snapshot_id,
                        snapshot_name=run.snapshot_name,
                    )
                    for run in run_plan.runs
                ],
                snapshot_revisions=run_plan.snapshot_revisions,
            )
        elif run_plan is not None and not isinstance(
            run_plan,
            SubmittedTrainingRunPlan,
        ):
            run_plan = SubmittedTrainingRunPlan(
                runs=[
                    SubmittedTrainingRun(
                        id=row["id"],
                        preset=row["preset"],
                        dataset=row["dataset"],
                        overrides=dict(row.get("overrides") or {}),
                        snapshot_id=row.get("snapshotId"),
                        snapshot_name=row.get("snapshotName"),
                    )
                    for row in run_plan.get("runs", [])
                ],
                snapshot_revisions=tuple(
                    ConfigSnapshotRevision(
                        id=revision["id"],
                        semantic_revision=revision["semanticRevision"],
                    )
                    for revision in run_plan.get("snapshotRevisions", [])
                ),
            )
        return training_job_payload(
            self.service.create_job(
                CreateTrainingJobCommand(
                    run_plan=MaterializeTrainingRunPlanCommand(
                        model=kwargs["model"],
                        preset=kwargs["preset"],
                        presets=kwargs.get("presets"),
                        experiment_task=kwargs.get("experiment_task"),
                        datasets=kwargs["datasets"],
                        overrides=kwargs["overrides"],
                        log_folder=kwargs["log_folder"],
                        monitors=list(kwargs.get("monitors") or []),
                        search=search,
                        submitted_plan=run_plan,
                        snapshot_ids=list(kwargs.get("snapshot_ids") or []),
                        snapshot_revisions=tuple(
                            ConfigSnapshotRevision(
                                id=str(revision["id"]),
                                semantic_revision=str(revision["semanticRevision"]),
                            )
                            for revision in kwargs.get("snapshot_revisions") or []
                        ),
                    ),
                )
            )
        )

    def get_job_payload(self, job_id: str) -> dict[str, Any]:
        return training_job_payload(self.service.get_job(job_id))

    def cancel_job_payload(self, job_id: str) -> dict[str, Any]:
        return training_job_payload(self.service.cancel_job(job_id))

    def reconcile_job_payload(
        self,
        job_id: str,
        *,
        action: str,
        reason: str,
    ) -> dict[str, Any]:
        return training_job_payload(
            self.service.reconcile_job(
                job_id,
                action=action,
                reason=reason,
            )
        )

    def active_job_payloads(self) -> list[dict[str, Any]]:
        return [
            _active_training_job_to_payload(job) for job in self.service.active_jobs()
        ]

    def get_job_events_payload(
        self,
        job_id: str,
        *,
        offset: int = 0,
        limit: int = 500,
    ) -> dict[str, Any]:
        return _training_events_page_to_payload(
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
        return _without_absent_event_metadata(
            _monitor_data_to_payload(
                self.service.get_monitor_data(
                    job_id,
                    node_path=node_path,
                    dataset=dataset,
                    preset=preset,
                )
            )
        )

    def get_parameter_status(
        self,
        job_id: str,
        *,
        dataset: str | None = None,
        preset: str | None = None,
    ) -> dict[str, Any]:
        return _without_absent_event_metadata(
            _parameter_status_to_payload(
                self.service.get_parameter_status(
                    job_id,
                    dataset=dataset,
                    preset=preset,
                )
            )
        )


def create_app_with_training_service(
    settings,
    harness: TrainingJobServiceHarness,
    *,
    project_adapter: ProjectAdapterClient | None = None,
):
    """Create an app whose Training Jobs capability uses a service test rig."""
    from emperor_workbench.api import create_app

    return create_app(
        settings,
        project_adapter=project_adapter,
        training_jobs=harness.service,
        log_experiment_mutations=harness.mutation_coordinator,
    )


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
            tuple(experiments)
            if experiments is not None
            else tuple(sorted({run.experiment for run in runs}))
        ),
        datasets=(
            tuple(datasets)
            if datasets is not None
            else tuple(sorted({run.dataset for run in runs}))
        ),
        models=(
            tuple(models)
            if models is not None
            else tuple(sorted({run.model for run in runs}))
        ),
        presets=(
            tuple(presets)
            if presets is not None
            else tuple(sorted({run.preset for run in runs}))
        ),
        run_ids=(
            tuple(run_ids)
            if run_ids is not None
            else tuple(sorted({run.id for run in runs}))
        ),
    )
