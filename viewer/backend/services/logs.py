"""Log-run API use cases."""

from __future__ import annotations

from typing import Any

from viewer.backend.inspector.errors import InspectorError
from viewer.backend.log_runs import LogRunDeleteFilters
from viewer.backend.repositories.log_runs import LogRunRepository

ACTIVE_LOG_EXPERIMENT_DELETE_MESSAGE = (
    "A training job is still writing to this log folder."
)


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


def _paginated_response(
    items: list[dict[str, Any]],
    *,
    collection_key: str,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    page = _paginate(items, limit=limit, offset=offset)
    return {
        collection_key: page["items"],
        "total": page["total"],
        "limit": page["limit"],
        "offset": page["offset"],
        "hasMore": page["hasMore"],
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


def _has_active_job_for_log_folder(
    active_jobs: list[dict[str, Any]],
    log_folder: str,
) -> bool:
    return any(
        str(job.get("logFolder") or "") == log_folder
        for job in active_jobs
    )


class LogRunService:
    def __init__(self, repository: LogRunRepository) -> None:
        self._repository = repository

    def list_runs(self, *, limit: int, offset: int) -> dict[str, Any]:
        runs = [run.to_response() for run in self._repository.list_runs()]
        return _paginated_response(
            runs,
            collection_key="runs",
            limit=limit,
            offset=offset,
        )

    def list_experiments(self, *, limit: int, offset: int) -> dict[str, Any]:
        experiments = [
            experiment.to_response()
            for experiment in self._repository.list_experiments()
        ]
        return _paginated_response(
            experiments,
            collection_key="experiments",
            limit=limit,
            offset=offset,
        )

    def delete_experiment(
        self,
        experiment: str,
        *,
        active_jobs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if _has_active_job_for_log_folder(active_jobs, experiment):
            raise InspectorError(ACTIVE_LOG_EXPERIMENT_DELETE_MESSAGE)
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

    def parameter_status_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return self._repository.parameter_status_for_runs(run_ids)
