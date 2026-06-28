"""Training progress event persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from viewer.backend.job_store import TrainingJobRecord


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class TrainingProgressSnapshot:
    events: list[dict[str, Any]]
    new_events: list[dict[str, Any]]
    total_count: int
    reset: bool = False


@dataclass
class _ProgressCacheEntry:
    offset: int = 0
    prefix: bytes = b""
    events: list[dict[str, Any]] = field(default_factory=list)


class TrainingProgressStore:
    def __init__(self) -> None:
        self._cache: dict[Path, _ProgressCacheEntry] = {}
        self._lock = RLock()

    def read_snapshot(self, job: TrainingJobRecord) -> TrainingProgressSnapshot:
        with self._lock:
            return self._read_snapshot(job)

    def _read_snapshot(self, job: TrainingJobRecord) -> TrainingProgressSnapshot:
        path = job.progress_path
        if not path.exists():
            self._cache.pop(path, None)
            return TrainingProgressSnapshot(
                events=[],
                new_events=[],
                total_count=0,
                reset=True,
            )

        stat = path.stat()
        entry = self._cache.get(path)
        reset = entry is None
        with path.open("rb") as handle:
            current_prefix = handle.read(256)
            if (
                entry is None
                or stat.st_size < entry.offset
                or (
                    entry.prefix and current_prefix[: len(entry.prefix)] != entry.prefix
                )
            ):
                entry = _ProgressCacheEntry(prefix=current_prefix)
                self._cache[path] = entry
                reset = True
            else:
                entry.prefix = current_prefix

            new_events: list[dict[str, Any]] = []
            handle.seek(entry.offset)
            while True:
                line_start = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                if not raw_line.endswith(b"\n"):
                    handle.seek(line_start)
                    break
                line = raw_line.decode("utf-8")
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry.events.append(event)
                new_events.append(event)
            entry.offset = handle.tell()

        return TrainingProgressSnapshot(
            events=list(entry.events),
            new_events=new_events,
            total_count=len(entry.events),
            reset=reset,
        )

    def read_events(self, job: TrainingJobRecord) -> list[dict[str, Any]]:
        return self.read_snapshot(job).events

    def append_event(
        self,
        job: TrainingJobRecord,
        event: dict[str, Any],
    ) -> None:
        payload = {"timestamp": _now(), "jobId": job.id, **event}
        job.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with job.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")


__all__ = ["TrainingProgressSnapshot", "TrainingProgressStore"]
