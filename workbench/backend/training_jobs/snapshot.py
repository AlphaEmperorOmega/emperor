"""Project reduced state into typed Training Job snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from workbench.backend.training_jobs.contracts import (
    TrainingJobView,
    TrainingResultLinkView,
)
from workbench.backend.training_jobs.monitoring import TrainingMonitorLocator
from workbench.backend.training_jobs.projection import TrainingJobLiveProjection
from workbench.backend.training_jobs.run_plan_adapter import (
    training_run_plan_from_payload,
    training_search_from_payload,
)
from workbench.backend.training_jobs.store import TrainingJobRecord

TRAINING_JOB_LOG_TAIL_CHUNK_BYTES = 8192


class TrainingJobProjector:
    def __init__(self) -> None:
        self._monitor_locator = TrainingMonitorLocator()

    def project_snapshot(
        self,
        job: TrainingJobRecord,
        projection: TrainingJobLiveProjection,
    ) -> TrainingJobView:
        latest_event = projection.latest_event
        metrics_event = projection.metrics_event
        latest_preset = self._monitor_locator.event_preset_name(latest_event)
        return TrainingJobView(
            id=job.id,
            status=job.status,
            model=job.model,
            preset=job.preset,
            presets=list(job.presets),
            experiment_task=job.experiment_task,
            datasets=list(job.datasets),
            overrides=dict(job.overrides),
            search=training_search_from_payload(job.search),
            planned_run_count=job.planned_run_count,
            run_plan=training_run_plan_from_payload(projection.run_plan),
            monitors=list(job.monitors),
            log_folder=job.log_folder,
            created_at=job.created_at,
            updated_at=job.updated_at,
            exit_code=job.exit_code,
            pid=job.pid,
            cancellation_mode=job.cancellation_mode,
            current_preset=latest_preset,
            current_dataset=_optional_str(latest_event.get("dataset")),
            epoch=_optional_int(latest_event.get("epoch")),
            step=_optional_int(latest_event.get("step")),
            metrics=dict(metrics_event.get("metrics") or {}),
            log_dir=_optional_str(latest_event.get("logDir")),
            events=list(projection.events_tail),
            event_count=projection.event_count,
            event_counts=dict(projection.event_counts),
            events_truncated=projection.events_truncated,
            cluster_growth=list(projection.cluster_growth),
            log_tail=self.log_tail(job),
            result_links=[
                TrainingResultLinkView(
                    preset=self._monitor_locator.event_preset_name(event),
                    dataset=_optional_str(event.get("dataset")),
                    log_dir=_optional_str(event.get("logDir")),
                )
                for event in projection.result_events
            ],
        )

    def log_tail(
        self,
        job: TrainingJobRecord,
        line_count: int = 80,
    ) -> list[str]:
        if not job.log_path.exists():
            return []
        return _tail_lines(job.log_path, line_count)


def _tail_lines(path: Path, line_count: int) -> list[str]:
    if line_count <= 0:
        return []

    chunks: list[bytes] = []
    newline_count = 0
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        while position > 0 and newline_count <= line_count:
            read_size = min(TRAINING_JOB_LOG_TAIL_CHUNK_BYTES, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")

    data = b"".join(reversed(chunks))
    return data.decode("utf-8", errors="replace").splitlines()[-line_count:]


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


__all__ = ["TrainingJobProjector"]
