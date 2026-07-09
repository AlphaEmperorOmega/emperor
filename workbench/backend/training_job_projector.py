"""Project stored training jobs and progress events into API payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models.catalog import model_identity_payload_from_id

from workbench.backend.job_store import TrainingJobRecord
from workbench.backend.training_monitor_locator import TrainingMonitorLocator
from workbench.backend.training_run_progress import project_training_run_progress

TRAINING_JOB_EVENT_TAIL_LIMIT = 100
TRAINING_JOB_LOG_TAIL_CHUNK_BYTES = 8192


@dataclass(frozen=True)
class TrainingJobLiveProjection:
    run_plan: dict[str, Any]
    latest_event: dict[str, Any] = field(default_factory=dict)
    metrics_event: dict[str, Any] = field(default_factory=dict)
    result_events: list[dict[str, Any]] = field(default_factory=list)
    events_tail: list[dict[str, Any]] = field(default_factory=list)
    event_count: int = 0
    event_counts: dict[str, int] = field(default_factory=dict)
    events_truncated: bool = False
    cluster_growth: list[dict[str, Any]] = field(default_factory=list)


class TrainingJobProjector:
    def __init__(self, monitor_locator: TrainingMonitorLocator | None = None) -> None:
        self._monitor_locator = monitor_locator or TrainingMonitorLocator()

    def project(
        self,
        job: TrainingJobRecord,
        *,
        events: list[dict[str, Any]],
        summarize,
    ) -> dict[str, Any]:
        latest_event = events[-1] if events else {}
        metrics_event = next(
            (
                event
                for event in reversed(events)
                if isinstance(event.get("metrics"), dict)
            ),
            {},
        )
        result_events = [
            event for event in events if event.get("type") == "dataset_completed"
        ]
        latest_preset = self._monitor_locator.event_preset_name(latest_event)
        run_plan = project_training_run_progress(
            job.run_plan,
            events,
            job.status,
            summarize,
        )
        return {
            "id": job.id,
            "status": job.status,
            **model_identity_payload_from_id(job.model),
            "preset": job.preset,
            "presets": job.presets,
            "experimentTask": job.experiment_task,
            "datasets": job.datasets,
            "overrides": job.overrides,
            "search": job.search,
            "plannedRunCount": job.planned_run_count,
            "runPlan": run_plan,
            "monitors": job.monitors,
            "logFolder": job.log_folder,
            "createdAt": job.created_at,
            "updatedAt": job.updated_at,
            "exitCode": job.exit_code,
            "pid": job.pid,
            "cancellationMode": job.cancellation_mode,
            "currentPreset": latest_preset,
            "currentDataset": latest_event.get("dataset"),
            "epoch": latest_event.get("epoch"),
            "step": latest_event.get("step"),
            "metrics": metrics_event.get("metrics") or {},
            "logDir": latest_event.get("logDir"),
            "events": events,
            "logTail": self.log_tail(job),
            "resultLinks": [
                {
                    "preset": self._monitor_locator.event_preset_name(event),
                    "dataset": event.get("dataset"),
                    "logDir": event.get("logDir"),
                }
                for event in result_events
            ],
            "eventCount": len(events),
            "eventCounts": _event_counts(events),
            "eventsTruncated": False,
            "clusterGrowth": _cluster_growth_from_events(events),
        }

    def project_live(
        self,
        job: TrainingJobRecord,
        projection: TrainingJobLiveProjection,
    ) -> dict[str, Any]:
        latest_event = projection.latest_event
        metrics_event = projection.metrics_event
        latest_preset = self._monitor_locator.event_preset_name(latest_event)
        return {
            "id": job.id,
            "status": job.status,
            **model_identity_payload_from_id(job.model),
            "preset": job.preset,
            "presets": job.presets,
            "experimentTask": job.experiment_task,
            "datasets": job.datasets,
            "overrides": job.overrides,
            "search": job.search,
            "plannedRunCount": job.planned_run_count,
            "runPlan": projection.run_plan,
            "monitors": job.monitors,
            "logFolder": job.log_folder,
            "createdAt": job.created_at,
            "updatedAt": job.updated_at,
            "exitCode": job.exit_code,
            "pid": job.pid,
            "cancellationMode": job.cancellation_mode,
            "currentPreset": latest_preset,
            "currentDataset": latest_event.get("dataset"),
            "epoch": latest_event.get("epoch"),
            "step": latest_event.get("step"),
            "metrics": metrics_event.get("metrics") or {},
            "logDir": latest_event.get("logDir"),
            "events": projection.events_tail,
            "eventCount": projection.event_count,
            "eventCounts": projection.event_counts,
            "eventsTruncated": projection.events_truncated,
            "clusterGrowth": projection.cluster_growth,
            "logTail": self.log_tail(job),
            "resultLinks": [
                {
                    "preset": self._monitor_locator.event_preset_name(event),
                    "dataset": event.get("dataset"),
                    "logDir": event.get("logDir"),
                }
                for event in projection.result_events
            ],
        }

    def log_tail(
        self,
        job: TrainingJobRecord,
        line_count: int = 80,
    ) -> list[str]:
        if not job.log_path.exists():
            return []
        return _tail_lines(job.log_path, line_count)


def _tail_lines(path, line_count: int) -> list[str]:
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


def _event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("type") or "unknown")
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def _capacity_total(value: Any) -> int:
    if not isinstance(value, list):
        return 0
    total = 1
    for axis in value:
        if not isinstance(axis, int | float):
            return 0
        total *= int(axis)
    return total


def _coord(value: Any) -> list[int] | None:
    if (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, int | float) for item in value)
    ):
        return [int(value[0]), int(value[1]), int(value[2])]
    return None


def _cluster_growth_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_node: dict[str, dict[str, Any]] = {}
    for event in events:
        node = event.get("node")
        if not isinstance(node, str) or not node:
            continue
        summary = by_node.setdefault(
            node,
            {
                "node": node,
                "count": 0,
                "capacityTotal": 0,
                "additionCount": 0,
                "additions": [],
            },
        )
        event_type = event.get("type")
        if event_type == "cluster_initialized":
            if isinstance(event.get("count"), int):
                summary["count"] = event["count"]
            capacity_total = _capacity_total(event.get("capacity"))
            if capacity_total:
                summary["capacityTotal"] = capacity_total
        elif event_type in {"neuron_added", "neurons_added"}:
            if isinstance(event.get("count"), int):
                summary["count"] = event["count"]
            capacity_total = _capacity_total(event.get("capacity"))
            if capacity_total:
                summary["capacityTotal"] = capacity_total
            if event_type == "neurons_added":
                coordinates = event.get("coordinates")
                if not isinstance(coordinates, list):
                    coordinates = []
                coordinate_count = event.get("coordinateCount")
                if not isinstance(coordinate_count, int):
                    coordinate_count = len(coordinates)
                summary["additionCount"] += max(0, coordinate_count)
                for coordinate_value in coordinates[-50:]:
                    coord = _coord(coordinate_value)
                    if coord is None:
                        continue
                    summary["additions"].append(
                        {
                            "coord": coord,
                            "step": event.get("step")
                            if isinstance(event.get("step"), int)
                            else None,
                            "epoch": event.get("epoch")
                            if isinstance(event.get("epoch"), int)
                            else None,
                        }
                    )
                summary["additions"] = summary["additions"][-50:]
                continue
            coord = _coord(event.get("coord"))
            if coord is not None:
                summary["additionCount"] += 1
                summary["additions"].append(
                    {
                        "coord": coord,
                        "step": event.get("step")
                        if isinstance(event.get("step"), int)
                        else None,
                        "epoch": event.get("epoch")
                        if isinstance(event.get("epoch"), int)
                        else None,
                    }
                )
                summary["additions"] = summary["additions"][-50:]
    return list(by_node.values())


__all__ = [
    "TRAINING_JOB_EVENT_TAIL_LIMIT",
    "TrainingJobLiveProjection",
    "TrainingJobProjector",
]
