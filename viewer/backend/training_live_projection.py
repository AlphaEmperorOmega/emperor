"""Incremental live Training Run Progress projection for Training Jobs."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from viewer.backend.job_store import TrainingJobRecord
from viewer.backend.training_job_projector import (
    TRAINING_JOB_EVENT_TAIL_LIMIT,
    TrainingJobLiveProjection,
)
from viewer.backend.training_progress_store import TrainingProgressSnapshot
from viewer.backend.training_run_progress import (
    SummaryCallback,
    apply_training_run_progress_event,
    finalize_training_run_progress,
    run_lookup_by_id,
)


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
class _LiveProjectionCache:
    event_count: int
    event_counts: dict[str, int]
    events_tail: list[dict[str, Any]]
    run_plan_base: dict[str, Any]
    run_by_id: dict[str, dict[str, Any]]
    latest_event: dict[str, Any] = field(default_factory=dict)
    metrics_event: dict[str, Any] = field(default_factory=dict)
    latest_failed_event: dict[str, Any] = field(default_factory=dict)
    result_events: list[dict[str, Any]] = field(default_factory=list)
    cluster_growth: dict[str, _ClusterGrowthState] = field(default_factory=dict)


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


class TrainingLiveProjectionCache:
    """Thread-safe incremental projection cache for one manager process."""

    def __init__(self) -> None:
        self._cache: dict[str, _LiveProjectionCache] = {}
        self._lock = RLock()

    def project(
        self,
        job: TrainingJobRecord,
        snapshot: TrainingProgressSnapshot,
        *,
        summarize: SummaryCallback,
    ) -> TrainingJobLiveProjection:
        with self._lock:
            cache = self._cache.get(job.id)
            if (
                cache is None
                or snapshot.reset
                or snapshot.total_count < cache.event_count
                or cache.run_plan_base.get("runs") is None
            ):
                cache = self._new_cache(job)
                self._cache[job.id] = cache
                events_to_apply = snapshot.events
            else:
                events_to_apply = snapshot.events[cache.event_count :]

            for event in events_to_apply:
                self._apply_event(cache, event)
            cache.event_count = snapshot.total_count

            run_plan = finalize_training_run_progress(
                cache.run_plan_base,
                job_status=job.status,
                summarize=summarize,
                latest_failed_event=cache.latest_failed_event,
            )
            return TrainingJobLiveProjection(
                run_plan=run_plan,
                latest_event=dict(cache.latest_event),
                metrics_event=dict(cache.metrics_event),
                result_events=list(cache.result_events),
                events_tail=list(cache.events_tail),
                event_count=cache.event_count,
                event_counts=dict(cache.event_counts),
                events_truncated=cache.event_count > len(cache.events_tail),
                cluster_growth=[
                    entry.to_payload()
                    for entry in cache.cluster_growth.values()
                ],
            )

    def _new_cache(self, job: TrainingJobRecord) -> _LiveProjectionCache:
        run_plan_base = copy.deepcopy(job.run_plan)
        runs = run_plan_base.get("runs") or []
        return _LiveProjectionCache(
            event_count=0,
            event_counts={},
            events_tail=[],
            run_plan_base=run_plan_base,
            run_by_id=run_lookup_by_id(runs),
        )

    def _apply_event(
        self,
        cache: _LiveProjectionCache,
        event: dict[str, Any],
    ) -> None:
        event_type = str(event.get("type") or "unknown")
        cache.event_counts[event_type] = cache.event_counts.get(event_type, 0) + 1
        cache.events_tail.append(event)
        cache.events_tail = cache.events_tail[-TRAINING_JOB_EVENT_TAIL_LIMIT:]
        cache.latest_event = event
        if isinstance(event.get("metrics"), dict):
            cache.metrics_event = event
        if event.get("status") == "failed":
            cache.latest_failed_event = event
        if event_type == "dataset_completed":
            cache.result_events.append(event)
        apply_training_run_progress_event(
            runs=cache.run_plan_base.get("runs") or [],
            run_by_id=cache.run_by_id,
            event=event,
        )
        self._apply_cluster_growth_event(cache, event)

    def _apply_cluster_growth_event(
        self,
        cache: _LiveProjectionCache,
        event: dict[str, Any],
    ) -> None:
        node = event.get("node")
        if not isinstance(node, str) or not node:
            return

        event_type = event.get("type")
        if event_type not in {"cluster_initialized", "neuron_added", "neurons_added"}:
            return

        summary = cache.cluster_growth.setdefault(
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
                        "step": (
                            event.get("step")
                            if isinstance(event.get("step"), int)
                            else None
                        ),
                        "epoch": (
                            event.get("epoch")
                            if isinstance(event.get("epoch"), int)
                            else None
                        ),
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
                "step": (
                    event.get("step") if isinstance(event.get("step"), int) else None
                ),
                "epoch": (
                    event.get("epoch") if isinstance(event.get("epoch"), int) else None
                ),
            }
        )
        summary.additions = summary.additions[-50:]


__all__ = ["TrainingLiveProjectionCache"]
