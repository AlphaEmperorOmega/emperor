"""In-memory training-job data-access adapter.

This repository is intentionally thin: it is an extension point between
services and the concrete local ``TrainingJobManager`` job-manager object.
"""

from __future__ import annotations

from typing import Any

from viewer.backend.training_jobs import TrainingJobManager


class TrainingJobRepository:
    def __init__(self, manager: TrainingJobManager) -> None:
        self._manager = manager

    def create_job(
        self,
        *,
        model: str,
        preset: str,
        presets: list[str] | None,
        datasets: list[str],
        overrides: dict[str, Any],
        log_folder: str,
        monitors: list[str],
        search: dict[str, Any] | None,
        run_plan: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self._manager.create_job(
            model=model,
            preset=preset,
            presets=presets,
            datasets=datasets,
            overrides=overrides,
            log_folder=log_folder,
            monitors=monitors,
            search=search,
            run_plan=run_plan,
        )

    def create_run_plan(
        self,
        *,
        model: str,
        preset: str,
        presets: list[str] | None,
        datasets: list[str],
        overrides: dict[str, Any],
        log_folder: str,
        search: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self._manager.create_run_plan(
            model=model,
            preset=preset,
            presets=presets,
            datasets=datasets,
            overrides=overrides,
            log_folder=log_folder,
            search=search,
        )

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._manager.get_job(job_id)

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

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return self._manager.cancel_job(job_id)

    def active_jobs(self) -> list[dict[str, Any]]:
        return self._manager.active_jobs()
