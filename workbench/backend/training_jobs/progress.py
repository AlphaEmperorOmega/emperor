"""Bounded Training Job progress persistence and cursor-based access."""

from __future__ import annotations

import json
import uuid
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from workbench.backend.training_jobs.lifecycle import terminal_status_from_event
from workbench.backend.training_jobs.store import TrainingJobRecord

TRAINING_PROGRESS_TAIL_LIMIT = 100
TRAINING_PROGRESS_MONITOR_EVENT_LIMIT = 2_000
TRAINING_PROGRESS_EVENT_TYPE_LIMIT = 256
TRAINING_PROGRESS_CACHE_JOB_LIMIT = 128
_FILE_IDENTITY_SAMPLE_BYTES = 256


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class TrainingProgressCursor:
    """One reader's position in one generation of an authoritative JSONL file."""

    generation: str
    offset: int
    total_count: int


@dataclass(frozen=True)
class TrainingProgressSnapshot:
    """Bounded aggregate state plus the delta for one explicit reader cursor."""

    events: list[dict[str, Any]]
    new_events: list[dict[str, Any]]
    total_count: int
    reset: bool = False
    cursor: TrainingProgressCursor | None = None
    event_counts: dict[str, int] = field(default_factory=dict)
    latest_terminal_event: dict[str, Any] | None = None
    monitor_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TrainingProgressPage:
    events: list[dict[str, Any]]
    total_count: int


@dataclass(frozen=True, slots=True)
class TrainingProgressCacheStats:
    retained_event_count: int
    total_count: int
    aggregate_key_count: int = 0


@dataclass
class _ProgressCacheEntry:
    generation: str = field(default_factory=lambda: uuid.uuid4().hex)
    offset: int = 0
    prefix: bytes = b""
    boundary: bytes = b""
    total_count: int = 0
    event_counts: dict[str, int] = field(default_factory=dict)
    tail: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=TRAINING_PROGRESS_TAIL_LIMIT)
    )
    latest_terminal_event: dict[str, Any] | None = None
    monitor_events: OrderedDict[
        tuple[str | None, str | None],
        dict[str, Any],
    ] = field(default_factory=OrderedDict)


class TrainingProgressStore:
    """Deep JSONL seam for bounded summaries, cursor deltas, and raw pages."""

    def __init__(
        self,
        *,
        max_cached_jobs: int = TRAINING_PROGRESS_CACHE_JOB_LIMIT,
    ) -> None:
        self._cache: OrderedDict[Path, _ProgressCacheEntry] = OrderedDict()
        self._max_cached_jobs = max(1, max_cached_jobs)
        self._lock = RLock()

    @property
    def cached_job_count(self) -> int:
        with self._lock:
            return len(self._cache)

    def read_snapshot(
        self,
        job: TrainingJobRecord,
        *,
        cursor: TrainingProgressCursor | None = None,
    ) -> TrainingProgressSnapshot:
        """Read bounded aggregate state and this cursor's unconsumed delta."""

        with self._lock:
            refreshed = self._refresh_entry(job, collect_events=True)
            if refreshed is None:
                return self._empty_snapshot(reset=True)
            entry, entry_reset, old_offset, old_generation, appended = refreshed
            reset = entry_reset or cursor is None
            if cursor is None:
                delta = (
                    appended
                    if entry_reset and old_offset == 0
                    else self._read_events_range(
                        job.progress_path,
                        start=0,
                        end=entry.offset,
                    )
                )
            elif (
                cursor.generation != entry.generation
                or cursor.offset > entry.offset
            ):
                reset = True
                delta = self._read_events_range(
                    job.progress_path,
                    start=0,
                    end=entry.offset,
                )
            elif (
                not entry_reset
                and cursor.generation == old_generation
                and cursor.offset == old_offset
            ):
                delta = appended
            elif cursor.offset == entry.offset:
                delta = []
            else:
                delta = self._read_events_range(
                    job.progress_path,
                    start=cursor.offset,
                    end=entry.offset,
                )
            return self._snapshot(entry, new_events=delta, reset=reset)

    def read_summary(self, job: TrainingJobRecord) -> TrainingProgressSnapshot:
        """Read only bounded tail and aggregate state; consume no reader cursor."""

        with self._lock:
            refreshed = self._refresh_entry(job, collect_events=False)
            if refreshed is None:
                return self._empty_snapshot(reset=True)
            entry, reset, _old_offset, _old_generation, _appended = refreshed
            return self._snapshot(entry, new_events=[], reset=reset)

    def read_page(
        self,
        job: TrainingJobRecord,
        *,
        offset: int,
        limit: int,
    ) -> TrainingProgressPage:
        """Stream one logical event page without materializing full history."""

        safe_offset = max(0, offset)
        safe_limit = max(1, limit)
        with self._lock:
            refreshed = self._refresh_entry(job, collect_events=False)
            if refreshed is None:
                return TrainingProgressPage(events=[], total_count=0)
            entry = refreshed[0]
            events = self._read_page_events(
                job.progress_path,
                offset=safe_offset,
                limit=safe_limit,
                end=entry.offset,
            )
            return TrainingProgressPage(
                events=events,
                total_count=entry.total_count,
            )

    def cache_stats(self, job: TrainingJobRecord) -> TrainingProgressCacheStats:
        """Expose bounded-retention evidence without leaking cache internals."""

        with self._lock:
            entry = self._cache.get(job.progress_path)
            if entry is None:
                return TrainingProgressCacheStats(0, 0)
            return TrainingProgressCacheStats(
                retained_event_count=len(entry.tail) + len(entry.monitor_events),
                total_count=entry.total_count,
                aggregate_key_count=(
                    len(entry.event_counts) + len(entry.monitor_events)
                ),
            )

    def read_snapshot_uncached(
        self,
        job: TrainingJobRecord,
    ) -> TrainingProgressSnapshot:
        """Compatibility read that retains no state in this store."""

        temporary = TrainingProgressStore(max_cached_jobs=1)
        return temporary.read_snapshot(job)

    def _refresh_entry(
        self,
        job: TrainingJobRecord,
        *,
        collect_events: bool,
    ) -> tuple[_ProgressCacheEntry, bool, int, str | None, list[dict[str, Any]]] | None:
        path = job.progress_path
        if not path.exists():
            self._cache.pop(path, None)
            return None

        stat = path.stat()
        existing = self._cache.get(path)
        old_offset = existing.offset if existing is not None else 0
        old_generation = existing.generation if existing is not None else None
        with path.open("rb") as handle:
            current_prefix = handle.read(_FILE_IDENTITY_SAMPLE_BYTES)
            reset = existing is None or stat.st_size < old_offset
            if existing is not None and not reset:
                if (
                    existing.prefix
                    and current_prefix[: len(existing.prefix)] != existing.prefix
                ):
                    reset = True
                elif existing.offset:
                    boundary_start = max(
                        0,
                        existing.offset - _FILE_IDENTITY_SAMPLE_BYTES,
                    )
                    handle.seek(boundary_start)
                    current_boundary = handle.read(
                        existing.offset - boundary_start
                    )
                    if current_boundary != existing.boundary:
                        reset = True

            if reset:
                entry = _ProgressCacheEntry(prefix=current_prefix)
                self._cache[path] = entry
                old_offset = 0
            else:
                assert existing is not None
                entry = existing
                entry.prefix = current_prefix

            appended: list[dict[str, Any]] = []
            handle.seek(entry.offset)
            while True:
                line_start = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                if not raw_line.endswith(b"\n"):
                    handle.seek(line_start)
                    break
                event = self._decode_event(raw_line.decode("utf-8"))
                if event is None:
                    continue
                self._record_event(entry, event)
                if collect_events:
                    appended.append(event)
            entry.offset = handle.tell()
            boundary_start = max(
                0,
                entry.offset - _FILE_IDENTITY_SAMPLE_BYTES,
            )
            handle.seek(boundary_start)
            entry.boundary = handle.read(entry.offset - boundary_start)

        self._touch(path)
        return entry, reset, old_offset, old_generation, appended

    def _record_event(
        self,
        entry: _ProgressCacheEntry,
        event: dict[str, Any],
    ) -> None:
        entry.total_count += 1
        event_type = str(event.get("type") or "unknown")
        if (
            event_type not in entry.event_counts
            and len(entry.event_counts) >= TRAINING_PROGRESS_EVENT_TYPE_LIMIT
        ):
            event_type = "other"
        entry.event_counts[event_type] = entry.event_counts.get(event_type, 0) + 1
        entry.tail.append(event)
        if terminal_status_from_event(event) is not None:
            entry.latest_terminal_event = event
        if event.get("logDir"):
            preset = event.get("preset")
            dataset = event.get("dataset")
            key = (
                preset if isinstance(preset, str) else None,
                dataset if isinstance(dataset, str) else None,
            )
            entry.monitor_events[key] = event
            entry.monitor_events.move_to_end(key)
            while (
                len(entry.monitor_events)
                > TRAINING_PROGRESS_MONITOR_EVENT_LIMIT
            ):
                entry.monitor_events.popitem(last=False)

    def _snapshot(
        self,
        entry: _ProgressCacheEntry,
        *,
        new_events: list[dict[str, Any]],
        reset: bool,
    ) -> TrainingProgressSnapshot:
        return TrainingProgressSnapshot(
            events=list(entry.tail),
            new_events=new_events,
            total_count=entry.total_count,
            reset=reset,
            cursor=TrainingProgressCursor(
                generation=entry.generation,
                offset=entry.offset,
                total_count=entry.total_count,
            ),
            event_counts=dict(entry.event_counts),
            latest_terminal_event=(
                dict(entry.latest_terminal_event)
                if entry.latest_terminal_event is not None
                else None
            ),
            monitor_events=list(entry.monitor_events.values()),
        )

    def _empty_snapshot(self, *, reset: bool) -> TrainingProgressSnapshot:
        return TrainingProgressSnapshot(
            events=[],
            new_events=[],
            total_count=0,
            reset=reset,
        )

    def _read_events_range(
        self,
        path: Path,
        *,
        start: int,
        end: int,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        with path.open("rb") as handle:
            handle.seek(start)
            while handle.tell() < end:
                raw_line = handle.readline()
                if not raw_line or not raw_line.endswith(b"\n"):
                    break
                event = self._decode_event(raw_line.decode("utf-8"))
                if event is not None:
                    events.append(event)
        return events

    def _read_page_events(
        self,
        path: Path,
        *,
        offset: int,
        limit: int,
        end: int,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        logical_index = 0
        with path.open("rb") as handle:
            while handle.tell() < end and len(events) < limit:
                raw_line = handle.readline()
                if not raw_line or not raw_line.endswith(b"\n"):
                    break
                event = self._decode_event(raw_line.decode("utf-8"))
                if event is None:
                    continue
                if logical_index >= offset:
                    events.append(event)
                logical_index += 1
        return events

    def _decode_event(self, line: str) -> dict[str, Any] | None:
        if not line.strip():
            return None
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, Mapping):
            return None
        return dict(value)

    def _touch(self, path: Path) -> None:
        self._cache.move_to_end(path)
        while len(self._cache) > self._max_cached_jobs:
            self._cache.popitem(last=False)

    def evict(self, job: TrainingJobRecord) -> None:
        """Drop one job's bounded aggregate and cursor state."""

        with self._lock:
            self._cache.pop(job.progress_path, None)

    def append_event(
        self,
        job: TrainingJobRecord,
        event: dict[str, Any],
    ) -> None:
        payload = {"timestamp": _now(), "jobId": job.id, **event}
        job.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, job.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")


__all__ = [
    "TRAINING_PROGRESS_TAIL_LIMIT",
    "TrainingProgressCacheStats",
    "TrainingProgressCursor",
    "TrainingProgressPage",
    "TrainingProgressSnapshot",
    "TrainingProgressStore",
]
