"""Log-run API use cases."""

from __future__ import annotations

from typing import Any

from viewer.backend.log_runs import LogRunDeleteFilters
from viewer.backend.repositories.log_runs import LogRunRepository


class LogRunService:
    def __init__(self, repository: LogRunRepository) -> None:
        self._repository = repository

    def list_runs(self, *, limit: int, offset: int) -> dict[str, Any]:
        runs = [run.to_response() for run in self._repository.list_runs()]
        page = runs[offset : offset + limit]
        return {
            "runs": page,
            "total": len(runs),
            "limit": limit,
            "offset": offset,
            "hasMore": offset + limit < len(runs),
        }

    def list_experiments(self, *, limit: int, offset: int) -> dict[str, Any]:
        experiments = [
            experiment.to_response()
            for experiment in self._repository.list_experiments()
        ]
        page = experiments[offset : offset + limit]
        return {
            "experiments": page,
            "total": len(experiments),
            "limit": limit,
            "offset": offset,
            "hasMore": offset + limit < len(experiments),
        }

    def delete_experiment(self, experiment: str) -> dict[str, Any]:
        return self._repository.delete_experiment(experiment).to_response()

    def create_delete_plan(
        self,
        *,
        experiments: list[str],
        datasets: list[str],
        models: list[str],
        presets: list[str],
        run_ids: list[str],
        active_jobs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        filters = LogRunDeleteFilters(
            experiments=experiments,
            datasets=datasets,
            models=models,
            presets=presets,
            runIds=run_ids,
        )
        return self._repository.create_delete_plan(
            filters,
            active_jobs=active_jobs,
        ).to_response()

    def delete_runs(
        self,
        *,
        experiments: list[str],
        datasets: list[str],
        models: list[str],
        presets: list[str],
        run_ids: list[str],
        active_jobs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        filters = LogRunDeleteFilters(
            experiments=experiments,
            datasets=datasets,
            models=models,
            presets=presets,
            runIds=run_ids,
        )
        return self._repository.delete_runs(
            filters,
            active_jobs=active_jobs,
        ).to_response()

    def tags_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return self._repository.tags_for_runs(run_ids)

    def scalars_for_runs(
        self,
        *,
        run_ids: list[str],
        tags: list[str],
    ) -> list[dict[str, Any]]:
        return self._repository.scalars_for_runs(run_ids=run_ids, tags=tags)

    def monitor_data_for_run(self, run_id: str, node_path: str) -> dict[str, Any]:
        return self._repository.monitor_data_for_run(run_id, node_path=node_path)
