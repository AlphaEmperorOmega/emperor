"""Training progress event persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from viewer.backend.job_store import TrainingJobRecord


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TrainingProgressStore:
    def read_events(self, job: TrainingJobRecord) -> list[dict[str, Any]]:
        if not job.progress_path.exists():
            return []
        events = []
        for line in job.progress_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def append_event(
        self,
        job: TrainingJobRecord,
        event: dict[str, Any],
    ) -> None:
        payload = {"timestamp": _now(), "jobId": job.id, **event}
        job.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with job.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")


__all__ = ["TrainingProgressStore"]
