"""Training-job API use cases."""

from __future__ import annotations

from typing import Any

from viewer.backend.repositories.training_jobs import TrainingJobRepository
from viewer.backend.training_contracts import (
    ActiveTrainingJob,
    CreateTrainingJobCommand,
    CreateTrainingRunPlanCommand,
    TrainingJobView,
    TrainingRunPlanView,
)


class TrainingJobService:
    def __init__(self, repository: TrainingJobRepository) -> None:
        self._repository = repository

    def create_job(self, command: CreateTrainingJobCommand) -> TrainingJobView:
        return self._repository.create_job(command)

    def create_run_plan(
        self,
        command: CreateTrainingRunPlanCommand,
    ) -> TrainingRunPlanView:
        return self._repository.create_run_plan(command)

    def get_job(self, job_id: str) -> TrainingJobView:
        return self._repository.get_job(job_id)

    def get_job_events(
        self,
        job_id: str,
        *,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        return self._repository.get_job_events(
            job_id,
            offset=offset,
            limit=limit,
        )

    def get_monitor_data(
        self,
        job_id: str,
        *,
        node_path: str,
        dataset: str | None,
        preset: str | None,
    ) -> dict[str, Any]:
        return self._repository.get_monitor_data(
            job_id,
            node_path=node_path,
            dataset=dataset,
            preset=preset,
        )

    def get_parameter_status(
        self,
        job_id: str,
        *,
        dataset: str | None,
        preset: str | None,
    ) -> dict[str, Any]:
        return self._repository.get_parameter_status(
            job_id,
            dataset=dataset,
            preset=preset,
        )

    def cancel_job(self, job_id: str) -> TrainingJobView:
        return self._repository.cancel_job(job_id)

    def active_jobs(self) -> list[ActiveTrainingJob]:
        return self._repository.active_jobs()
