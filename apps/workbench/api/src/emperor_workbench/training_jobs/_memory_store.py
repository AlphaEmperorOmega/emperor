from __future__ import annotations

from threading import RLock

from emperor_workbench.training_jobs._records import TrainingJobRecord


class InMemoryTrainingJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, TrainingJobRecord] = {}
        self._lock = RLock()

    def save(self, job: TrainingJobRecord) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> TrainingJobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[TrainingJobRecord]:
        with self._lock:
            return list(self._jobs.values())


__all__ = ["InMemoryTrainingJobStore"]
