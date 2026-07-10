"""Log-run API use cases."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Literal

from models.catalog import model_identity_payload_from_id

from workbench.backend.inspector.errors import InspectorError
from workbench.backend.log_runs import LogRunDeleteFilters
from workbench.backend.repositories.log_runs import LogRunRepository

ACTIVE_LOG_EXPERIMENT_DELETE_MESSAGE = (
    "A training job is still writing to this log folder."
)
LOG_RUN_LISTING_CACHE_MAX_ENTRIES = 32
RunListingCacheKey = tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    bool | None,
]
CachedRunListing = tuple[list[Any], dict[str, Any]]


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


def _selected_values(values: list[str] | None) -> set[str]:
    return set(values or [])


def _matches_model_filter(run_model: str, selected_models: set[str]) -> bool:
    if not selected_models:
        return True
    if run_model in selected_models:
        return True
    run_model_leaf = run_model.rsplit("/", 1)[-1]
    return any(model.rsplit("/", 1)[-1] == run_model_leaf for model in selected_models)


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
    return any(str(job.get("logFolder") or "") == log_folder for job in active_jobs)


def _cached_layer_monitor_data(repository: LogRunRepository, run: Any) -> bool | None:
    cache_lookup = getattr(repository, "cached_layer_monitor_data_for_run", None)
    if not callable(cache_lookup):
        return None
    return cache_lookup(run)


def _value_facets(values: list[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(Counter(values).items())
    ]


def _run_facets(runs: list[Any]) -> dict[str, Any]:
    runs_by_experiment: dict[str, list[Any]] = defaultdict(list)
    for run in runs:
        runs_by_experiment[run.experiment].append(run)

    experiments = []
    for experiment, experiment_runs in sorted(runs_by_experiment.items()):
        model_counts: Counter[tuple[str, str]] = Counter()
        for run in experiment_runs:
            try:
                identity = model_identity_payload_from_id(run.model)
            except ValueError:
                identity = {
                    "modelType": str(getattr(run, "modelType", "models")),
                    "model": str(run.model),
                }
            model_counts[(identity["modelType"], identity["model"])] += 1
        experiments.append(
            {
                "experiment": experiment,
                "runCount": len(experiment_runs),
                "datasets": _value_facets([run.dataset for run in experiment_runs]),
                "models": [
                    {"modelType": model_type, "model": model, "count": count}
                    for (model_type, model), count in sorted(model_counts.items())
                ],
                "presets": _value_facets([run.preset for run in experiment_runs]),
            }
        )
    return {"experiments": experiments}


class LogRunService:
    def __init__(self, repository: LogRunRepository) -> None:
        self._repository = repository
        self._run_catalog_signature: tuple[int, ...] | None = None
        self._run_listing_cache: dict[RunListingCacheKey, CachedRunListing] = {}

    def _clear_run_listing_cache(self) -> None:
        self._run_catalog_signature = None
        self._run_listing_cache.clear()

    def _filtered_run_listing(
        self,
        *,
        experiment_set: set[str],
        model_set: set[str],
        preset_set: set[str],
        dataset_set: set[str],
        has_event_files: bool | None,
    ) -> tuple[list[Any], dict[str, Any]]:
        catalog = self._repository.list_runs()
        # LogRunScanner returns the same immutable run objects while its catalog
        # fingerprint is unchanged. The signature lets later pages reuse one
        # filter/facet pass without hiding filesystem changes.
        catalog_signature = tuple(id(run) for run in catalog)
        if catalog_signature != self._run_catalog_signature:
            self._run_catalog_signature = catalog_signature
            self._run_listing_cache.clear()

        cache_key = (
            tuple(sorted(experiment_set)),
            tuple(sorted(model_set)),
            tuple(sorted(preset_set)),
            tuple(sorted(dataset_set)),
            has_event_files,
        )
        cached = self._run_listing_cache.get(cache_key)
        if cached is not None:
            return cached

        filtered_runs = []
        for run in catalog:
            if experiment_set and run.experiment not in experiment_set:
                continue
            if not _matches_model_filter(run.model, model_set):
                continue
            if preset_set and run.preset not in preset_set:
                continue
            if dataset_set and run.dataset not in dataset_set:
                continue
            if (
                has_event_files is not None
                and (run.eventFileCount > 0) != has_event_files
            ):
                continue
            filtered_runs.append(run)

        listing = (filtered_runs, _run_facets(filtered_runs))
        if len(self._run_listing_cache) >= LOG_RUN_LISTING_CACHE_MAX_ENTRIES:
            oldest_key = next(iter(self._run_listing_cache))
            self._run_listing_cache.pop(oldest_key)
        self._run_listing_cache[cache_key] = listing
        return listing

    def list_runs(
        self,
        *,
        limit: int,
        offset: int,
        experiment: list[str] | None = None,
        model: list[str] | None = None,
        preset: list[str] | None = None,
        dataset: list[str] | None = None,
        has_event_files: bool | None = None,
        projection: Literal["full", "summary"] = "full",
    ) -> dict[str, Any]:
        experiment_set = _selected_values(experiment)
        model_set = _selected_values(model)
        preset_set = _selected_values(preset)
        dataset_set = _selected_values(dataset)
        filtered_runs, facets = self._filtered_run_listing(
            experiment_set=experiment_set,
            model_set=model_set,
            preset_set=preset_set,
            dataset_set=dataset_set,
            has_event_files=has_event_files,
        )

        page_runs = filtered_runs[offset : offset + limit]
        runs = []
        for run in page_runs:
            response = run.to_response()
            if projection == "summary":
                response["metrics"] = {}
                response["hasLayerMonitorData"] = None
            else:
                response["hasLayerMonitorData"] = _cached_layer_monitor_data(
                    self._repository,
                    run,
                )
            runs.append(response)
        return {
            "runs": runs,
            "total": len(filtered_runs),
            "limit": limit,
            "offset": offset,
            "hasMore": offset + limit < len(filtered_runs),
            "facets": facets,
        }

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
        result = self._repository.delete_experiment(experiment).to_response()
        self._clear_run_listing_cache()
        return result

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
        result = self._repository.delete_runs(
            filters,
            active_jobs=active_jobs,
        ).to_response()
        self._clear_run_listing_cache()
        return result

    def import_archive(
        self,
        *,
        archive: bytes,
        filename: str,
        max_upload_size: int | None,
        max_extracted_size: int | None,
    ) -> dict[str, object]:
        result = self._repository.import_archive(
            archive=archive,
            filename=filename,
            max_upload_size=max_upload_size,
            max_extracted_size=max_extracted_size,
        )
        self._clear_run_listing_cache()
        return result

    def tags_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return self._repository.tags_for_runs(run_ids)

    def scalars_for_runs(
        self,
        *,
        run_ids: list[str],
        tags: list[str],
        max_points: int,
        sampling: str,
    ) -> list[dict[str, Any]]:
        return self._repository.scalars_for_runs(
            run_ids=run_ids,
            tags=tags,
            max_points=max_points,
            sampling=sampling,
        )

    def media_for_runs(
        self,
        *,
        run_ids: list[str],
        image_tags: list[str],
        text_tags: list[str],
    ) -> dict[str, Any]:
        return self._repository.media_for_runs(
            run_ids=run_ids,
            image_tags=image_tags,
            text_tags=text_tags,
        )

    def monitor_data_for_run(self, run_id: str, node_path: str) -> dict[str, Any]:
        return self._repository.monitor_data_for_run(run_id, node_path=node_path)

    def parameter_status_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return self._repository.parameter_status_for_runs(run_ids)

    def checkpoints_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return self._repository.checkpoints_for_runs(run_ids)

    def artifacts_for_run(self, run_id: str) -> dict[str, Any]:
        return self._repository.artifacts_for_run(run_id)
