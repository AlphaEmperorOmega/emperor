"""In-memory training-job data-access adapter.

This repository is intentionally thin: it is an extension point between
services and the concrete local ``TrainingJobManager`` job-manager object.
"""

from __future__ import annotations

from typing import Any

from viewer.backend.training_contracts import (
    ActiveTrainingJob,
    CreateTrainingJobCommand,
    CreateTrainingRunPlanCommand,
    TrainingJobView,
    TrainingRunPlanView,
)
from viewer.backend.training_jobs import TrainingJobManager


class TrainingJobRepository:
    def __init__(self, manager: TrainingJobManager) -> None:
        self._manager = manager

    def create_job(self, command: CreateTrainingJobCommand) -> TrainingJobView:
        kwargs: dict[str, Any] = {
            "model": command.model,
            "preset": command.preset,
            "presets": command.presets,
            "datasets": command.datasets,
            "overrides": command.overrides,
            "log_folder": command.log_folder,
            "monitors": command.monitors,
            "search": (
                command.search.to_api_payload()
                if command.search is not None
                else None
            ),
            "run_plan": (
                command.run_plan.to_api_payload()
                if command.run_plan is not None
                else None
            ),
        }
        if command.experiment_task is not None:
            kwargs["experiment_task"] = command.experiment_task
        return TrainingJobView.from_payload(
            self._manager.create_job(**kwargs)
        )

    def create_run_plan(
        self,
        command: CreateTrainingRunPlanCommand,
    ) -> TrainingRunPlanView:
        kwargs: dict[str, Any] = {
            "model": command.model,
            "preset": command.preset,
            "presets": command.presets,
            "datasets": command.datasets,
            "overrides": command.overrides,
            "log_folder": command.log_folder,
            "monitors": command.monitors,
            "search": (
                command.search.to_api_payload()
                if command.search is not None
                else None
            ),
        }
        if command.experiment_task is not None:
            kwargs["experiment_task"] = command.experiment_task
        return TrainingRunPlanView.from_payload(
            self._manager.create_run_plan(**kwargs)
        )

    def get_job(self, job_id: str) -> TrainingJobView:
        return TrainingJobView.from_payload(self._manager.get_job(job_id))

    def get_job_events(
        self,
        job_id: str,
        *,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        return self._manager.get_job_events(
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
        return self._manager.get_monitor_data(
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
        return self._manager.get_parameter_status(
            job_id,
            dataset=dataset,
            preset=preset,
        )

    def cancel_job(self, job_id: str) -> TrainingJobView:
        return TrainingJobView.from_payload(self._manager.cancel_job(job_id))

    def active_jobs(self) -> list[ActiveTrainingJob]:
        return [
            ActiveTrainingJob.from_payload(job)
            for job in self._manager.active_jobs()
        ]
