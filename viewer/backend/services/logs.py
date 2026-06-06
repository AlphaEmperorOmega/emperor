"""Log-run API use cases."""

from __future__ import annotations

from typing import Any

from viewer.backend.log_runs import LogRunDeleteFilters
from viewer.backend.repositories.log_runs import LogRunRepository


def _paginate(
    items: list[dict[str, Any]],
    *,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    return {
        "items": items[offset : offset + limit],
        "total": len(items),
        "limit": limit,
        "offset": offset,
        "hasMore": offset + limit < len(items),
    }


def _delete_filters_from_fields(
    *,
    experiments: list[str],
    datasets: list[str],
    models: list[str],
    presets: list[str],
    run_ids: list[str],
) -> LogRunDeleteFilters:
    return LogRunDeleteFilters(
        experiments=experiments,
        datasets=datasets,
        models=models,
        presets=presets,
        runIds=run_ids,
    )


class LogRunService:
    def __init__(self, repository: LogRunRepository) -> None:
        self._repository = repository

    def list_runs(self, *, limit: int, offset: int) -> dict[str, Any]:
        runs = [run.to_response() for run in self._repository.list_runs()]
        page = _paginate(runs, limit=limit, offset=offset)
        return {
            "runs": page["items"],
            "total": page["total"],
            "limit": page["limit"],
            "offset": page["offset"],
            "hasMore": page["hasMore"],
        }

    def list_experiments(self, *, limit: int, offset: int) -> dict[str, Any]:
        experiments = [
            experiment.to_response()
            for experiment in self._repository.list_experiments()
        ]
        page = _paginate(experiments, limit=limit, offset=offset)
        return {
            "experiments": page["items"],
            "total": page["total"],
            "limit": page["limit"],
            "offset": page["offset"],
            "hasMore": page["hasMore"],
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
        filters = _delete_filters_from_fields(
            experiments=experiments,
            datasets=datasets,
            models=models,
            presets=presets,
            run_ids=run_ids,
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
        filters = _delete_filters_from_fields(
            experiments=experiments,
            datasets=datasets,
            models=models,
            presets=presets,
            run_ids=run_ids,
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
