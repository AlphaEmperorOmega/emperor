"""Authoritative event reduction for live and replayed Training Jobs."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from workbench.backend.training_jobs.progress import TrainingProgressSnapshot
from workbench.backend.training_jobs.run_progress import (
    SummaryCallback,
    apply_training_run_progress_event,
    finalize_training_run_progress,
    run_lookup_by_id,
)
from workbench.backend.training_jobs.store import TrainingJobRecord

TRAINING_JOB_EVENT_TAIL_LIMIT = 100


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


@dataclass
class _ClusterGrowthState:
    node: str
    count: int = 0
    capacity_total: int = 0
    addition_count: int = 0
    additions: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "count": self.count,
            "capacityTotal": self.capacity_total,
            "additionCount": self.addition_count,
            "additions": list(self.additions),
        }


@dataclass
class _TrainingEventReducer:
    run_plan_base: dict[str, Any]
    run_by_id: dict[str, dict[str, Any]]
    event_count: int = 0
    event_counts: dict[str, int] = field(default_factory=dict)
    events_tail: list[dict[str, Any]] = field(default_factory=list)
    latest_event: dict[str, Any] = field(default_factory=dict)
    metrics_event: dict[str, Any] = field(default_factory=dict)
    latest_failed_event: dict[str, Any] = field(default_factory=dict)
    result_events: list[dict[str, Any]] = field(default_factory=list)
    cluster_growth: dict[str, _ClusterGrowthState] = field(default_factory=dict)

    @classmethod
    def from_job(cls, job: TrainingJobRecord) -> _TrainingEventReducer:
        run_plan_base = copy.deepcopy(job.run_plan)
        return cls(
            run_plan_base=run_plan_base,
            run_by_id=run_lookup_by_id(run_plan_base.get("runs") or []),
        )

    def apply(self, event: dict[str, Any]) -> None:
        self.event_count += 1
        event_type = str(event.get("type") or "unknown")
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
        self.events_tail.append(event)
        self.events_tail = self.events_tail[-TRAINING_JOB_EVENT_TAIL_LIMIT:]
        self.latest_event = event
        if isinstance(event.get("metrics"), dict):
            self.metrics_event = event
        if event.get("status") == "failed":
            self.latest_failed_event = event
        if event_type == "dataset_completed":
            self.result_events.append(event)
        apply_training_run_progress_event(
            runs=self.run_plan_base.get("runs") or [],
            run_by_id=self.run_by_id,
            event=event,
        )
        self._apply_cluster_growth(event)

    def snapshot(
        self,
        *,
        job_status: str,
        summarize: SummaryCallback,
    ) -> TrainingJobLiveProjection:
        run_plan = finalize_training_run_progress(
            self.run_plan_base,
            job_status=job_status,
            summarize=summarize,
            latest_failed_event=self.latest_failed_event,
        )
        return TrainingJobLiveProjection(
            run_plan=run_plan,
            latest_event=dict(self.latest_event),
            metrics_event=dict(self.metrics_event),
            result_events=list(self.result_events),
            events_tail=list(self.events_tail),
            event_count=self.event_count,
            event_counts=dict(self.event_counts),
            events_truncated=self.event_count > len(self.events_tail),
            cluster_growth=[
                entry.to_payload()
                for entry in self.cluster_growth.values()
            ],
        )

    def _apply_cluster_growth(self, event: dict[str, Any]) -> None:
        node = event.get("node")
        if not isinstance(node, str) or not node:
            return

        event_type = event.get("type")
        if event_type not in {"cluster_initialized", "neuron_added", "neurons_added"}:
            return

        summary = self.cluster_growth.setdefault(
            node,
            _ClusterGrowthState(node=node),
        )
        if isinstance(event.get("count"), int):
            summary.count = event["count"]
        capacity_total = _capacity_total(event.get("capacity"))
        if capacity_total:
            summary.capacity_total = capacity_total

        if event_type == "cluster_initialized":
            return
        if event_type == "neurons_added":
            coordinates = event.get("coordinates")
            if not isinstance(coordinates, list):
                coordinates = []
            coordinate_count = event.get("coordinateCount")
            if not isinstance(coordinate_count, int):
                coordinate_count = len(coordinates)
            summary.addition_count += max(0, coordinate_count)
            for coordinate_value in coordinates[-50:]:
                coord = _coord(coordinate_value)
                if coord is None:
                    continue
                summary.additions.append(
                    {
                        "coord": coord,
                        "step": _optional_int(event.get("step")),
                        "epoch": _optional_int(event.get("epoch")),
                    }
                )
            summary.additions = summary.additions[-50:]
            return

        coord = _coord(event.get("coord"))
        if coord is None:
            return
        summary.addition_count += 1
        summary.additions.append(
            {
                "coord": coord,
                "step": _optional_int(event.get("step")),
                "epoch": _optional_int(event.get("epoch")),
            }
        )
        summary.additions = summary.additions[-50:]


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


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


class TrainingLiveProjectionCache:
    """Thread-safe process-local cache around the shared event reducer."""

    def __init__(self) -> None:
        self._cache: dict[str, _TrainingEventReducer] = {}
        self._lock = RLock()

    def evict(self, job_id: str) -> None:
        """Drop a terminal Training Job's process-local projection state."""
        with self._lock:
            self._cache.pop(job_id, None)

    def project(
        self,
        job: TrainingJobRecord,
        snapshot: TrainingProgressSnapshot,
        *,
        summarize: SummaryCallback,
    ) -> TrainingJobLiveProjection:
        with self._lock:
            reducer = self._cache.get(job.id)
            if (
                reducer is None
                or snapshot.reset
                or snapshot.total_count < reducer.event_count
                or reducer.run_plan_base.get("runs") is None
            ):
                reducer = _TrainingEventReducer.from_job(job)
                self._cache[job.id] = reducer
                events_to_apply = snapshot.events
            else:
                events_to_apply = snapshot.events[reducer.event_count :]

            for event in events_to_apply:
                reducer.apply(event)
            return reducer.snapshot(
                job_status=job.status,
                summarize=summarize,
            )

    def replay(
        self,
        job: TrainingJobRecord,
        snapshot: TrainingProgressSnapshot,
        *,
        summarize: SummaryCallback,
    ) -> TrainingJobLiveProjection:
        """Project a released terminal job without retaining cache state."""
        reducer = _TrainingEventReducer.from_job(job)
        for event in snapshot.events:
            reducer.apply(event)
        return reducer.snapshot(
            job_status=job.status,
            summarize=summarize,
        )


__all__ = [
    "TRAINING_JOB_EVENT_TAIL_LIMIT",
    "TrainingJobLiveProjection",
    "TrainingLiveProjectionCache",
]
