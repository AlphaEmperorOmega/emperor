"""Project reduced state into typed Training Job snapshots."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workbench.backend.training_jobs.contracts import (
    TrainingJobView,
    TrainingResultLinkView,
)
from workbench.backend.training_jobs.monitoring import TrainingMonitorLocator
from workbench.backend.training_jobs.projection import TrainingJobLiveProjection
from workbench.backend.training_jobs.store import TrainingJobRecord

TRAINING_JOB_LOG_TAIL_CHUNK_BYTES = 8192
TRAINING_JOB_LOG_TAIL_LINE_LIMIT = 80
TRAINING_JOB_LOG_TAIL_MAX_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class TrainingLogTail:
    lines: tuple[str, ...]
    truncated: bool


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
        log_tail = self.log_tail_snapshot(job)
        return TrainingJobView(
            id=job.id,
            status=job.status,
            model=job.model,
            preset=job.preset,
            presets=list(job.presets),
            experiment_task=job.experiment_task,
            datasets=list(job.datasets),
            overrides=dict(job.overrides),
            search=job.search,
            planned_run_count=job.planned_run_count,
            run_plan=projection.run_plan,
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
            log_tail=list(log_tail.lines),
            result_links=[
                TrainingResultLinkView(
                    preset=self._monitor_locator.event_preset_name(event),
                    dataset=_optional_str(event.get("dataset")),
                    log_dir=_optional_str(event.get("logDir")),
                )
                for event in projection.result_events
            ],
            log_tail_truncated=log_tail.truncated,
        )

    def log_tail(
        self,
        job: TrainingJobRecord,
        line_count: int = 80,
    ) -> list[str]:
        return list(self.log_tail_snapshot(job, line_count=line_count).lines)

    def log_tail_snapshot(
        self,
        job: TrainingJobRecord,
        *,
        line_count: int = TRAINING_JOB_LOG_TAIL_LINE_LIMIT,
        max_bytes: int = TRAINING_JOB_LOG_TAIL_MAX_BYTES,
    ) -> TrainingLogTail:
        if not job.log_path.exists():
            return TrainingLogTail((), False)
        return _tail_lines(job.log_path, line_count, max_bytes)


def _utf8_suffix(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    suffix = encoded[-max_bytes:]
    while suffix and suffix[0] & 0xC0 == 0x80:
        suffix = suffix[1:]
    return suffix.decode("utf-8")


def _bounded_decoded_lines(
    lines: list[str],
    max_bytes: int,
) -> tuple[tuple[str, ...], bool]:
    retained: deque[str] = deque()
    remaining = max_bytes
    truncated = False
    for line in reversed(lines):
        separator_bytes = 1 if retained else 0
        if separator_bytes > remaining:
            truncated = True
            break
        line_budget = remaining - separator_bytes
        encoded_size = len(line.encode("utf-8"))
        if encoded_size <= line_budget:
            retained.appendleft(line)
            remaining -= separator_bytes + encoded_size
            continue
        if line_budget > 0:
            retained.appendleft(_utf8_suffix(line, line_budget))
        truncated = True
        break
    return tuple(retained), truncated


def _tail_lines(
    path: Path,
    line_count: int,
    max_bytes: int,
) -> TrainingLogTail:
    if line_count <= 0 or max_bytes <= 0:
        return TrainingLogTail((), path.stat().st_size > 0)

    with path.open("rb") as handle:
        size = int(os.fstat(handle.fileno()).st_size)
        read_size = min(size, max_bytes + 4)
        position = size - read_size
        handle.seek(position)
        data = handle.read(read_size)

    decoded_lines = data.decode("utf-8", errors="replace").splitlines()
    selected_lines = decoded_lines[-line_count:]
    bounded_lines, byte_truncated = _bounded_decoded_lines(
        selected_lines,
        max_bytes,
    )
    return TrainingLogTail(
        bounded_lines,
        position > 0 or len(decoded_lines) > line_count or byte_truncated,
    )


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


__all__ = [
    "TRAINING_JOB_LOG_TAIL_LINE_LIMIT",
    "TRAINING_JOB_LOG_TAIL_MAX_BYTES",
    "TrainingJobProjector",
    "TrainingLogTail",
]
