"""Deep Run History read and mutation Interface."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, BinaryIO, Literal

from emperor.model_packages import model_identity_payload_from_id

from workbench.backend.core.limits import (
    DEFAULT_MAX_LOG_ARCHIVE_MEMBER_COUNT,
    DEFAULT_MAX_LOG_ARCHIVE_PATH_BYTES,
)
from workbench.backend.inspector.errors import InspectorError
from workbench.backend.log_experiments import (
    LogExperimentMutationCoordinator,
)
from workbench.backend.run_history.archive import (
    import_log_archive,
    inspect_log_archive_experiments,
)
from workbench.backend.run_history.contracts import (
    ActiveLogWriterSource,
    HistoricalCheckpointCandidate,
    HistoricalInspectionContext,
)
from workbench.backend.run_history.deletion import (
    LogRunDeletionExecutor,
    LogRunDeletionPlanner,
)
from workbench.backend.run_history.query import LogRunQueryService
from workbench.backend.run_history.records import LogRun, LogRunDeleteFilters
from workbench.backend.run_history.scanner import LogRunScanner
from workbench.backend.tensorboard.readers import (
    DEFAULT_SCALAR_POINT_LIMIT,
    TensorBoardMonitorReader,
    TensorBoardParameterStatusReader,
)

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
CachedRunListing = tuple[list[LogRun], dict[str, Any]]


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
    if not selected_models or run_model in selected_models:
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


def _value_facets(values: list[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(Counter(values).items())
    ]


def _run_facets(runs: list[LogRun]) -> dict[str, Any]:
    runs_by_experiment: dict[str, list[LogRun]] = defaultdict(list)
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


class RunHistoryService:
    """Own one process-local Run History runtime and cache graph."""

    def __init__(
        self,
        *,
        logs_root: Path | str,
        mutation_coordinator: LogExperimentMutationCoordinator,
        active_log_writers: ActiveLogWriterSource,
        scalar_point_limit: int = DEFAULT_SCALAR_POINT_LIMIT,
    ) -> None:
        self._logs_root = Path(logs_root)
        self._mutation_coordinator = mutation_coordinator
        self._active_log_writers = active_log_writers
        self._scanner = LogRunScanner(logs_root=self._logs_root)
        monitor_reader = TensorBoardMonitorReader(
            scalar_point_limit=scalar_point_limit,
        )
        parameter_status_reader = TensorBoardParameterStatusReader()
        self._query = LogRunQueryService(
            scanner=self._scanner,
            scalar_point_limit=scalar_point_limit,
            monitor_reader=monitor_reader,
            parameter_status_reader=parameter_status_reader,
        )
        self._deletion_planner = LogRunDeletionPlanner(scanner=self._scanner)
        self._deletion_executor = LogRunDeletionExecutor(scanner=self._scanner)
        self._run_catalog_signature: tuple[int, ...] | None = None
        self._run_listing_cache: dict[RunListingCacheKey, CachedRunListing] = {}
        self._run_listing_generation = 0
        self._run_listing_lock = RLock()

    def _clear_run_listing_cache(self) -> None:
        with self._run_listing_lock:
            self._run_listing_generation += 1
            self._run_catalog_signature = None
            self._run_listing_cache.clear()

    def _invalidate_all(self) -> None:
        self._scanner.clear_cache()
        self._query.clear_cache()
        self._clear_run_listing_cache()

    def _invalidate_runs(self, run_paths: list[Path]) -> None:
        self._scanner.clear_cache()
        self._query.clear_run_caches(run_paths)
        self._clear_run_listing_cache()

    def _filtered_run_listing(
        self,
        *,
        experiment_set: set[str],
        model_set: set[str],
        preset_set: set[str],
        dataset_set: set[str],
        has_event_files: bool | None,
    ) -> tuple[list[LogRun], dict[str, Any]]:
        with self._run_listing_lock:
            generation = self._run_listing_generation
        catalog = self._scanner.list_runs()
        catalog_signature = tuple(id(run) for run in catalog)
        cache_key = (
            tuple(sorted(experiment_set)),
            tuple(sorted(model_set)),
            tuple(sorted(preset_set)),
            tuple(sorted(dataset_set)),
            has_event_files,
        )
        with self._run_listing_lock:
            if generation != self._run_listing_generation:
                cached = None
            else:
                if catalog_signature != self._run_catalog_signature:
                    self._run_catalog_signature = catalog_signature
                    self._run_listing_cache.clear()
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
        with self._run_listing_lock:
            if (
                generation == self._run_listing_generation
                and self._run_catalog_signature == catalog_signature
            ):
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
        filtered_runs, facets = self._filtered_run_listing(
            experiment_set=_selected_values(experiment),
            model_set=_selected_values(model),
            preset_set=_selected_values(preset),
            dataset_set=_selected_values(dataset),
            has_event_files=has_event_files,
        )
        runs = []
        for run in filtered_runs[offset : offset + limit]:
            response = run.to_response()
            if projection == "summary":
                response["metrics"] = {}
                response["hasLayerMonitorData"] = None
            else:
                response["hasLayerMonitorData"] = (
                    self._query.cached_layer_monitor_data_for_run(run)
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
            for experiment in self._scanner.list_experiments()
        ]
        return _paginated_response(
            experiments,
            collection_key="experiments",
            limit=limit,
            offset=offset,
        )

    def delete_experiment(self, experiment: str) -> dict[str, Any]:
        with self._mutation_coordinator.coordinate([experiment]):
            if any(
                writer.log_folder == experiment
                for writer in self._active_log_writers()
            ):
                raise InspectorError(ACTIVE_LOG_EXPERIMENT_DELETE_MESSAGE)
            try:
                return self._deletion_executor.delete_experiment(
                    experiment
                ).to_response()
            finally:
                self._invalidate_all()

    def create_delete_plan(
        self,
        *,
        experiments: list[str],
        datasets: list[str],
        models: list[str],
        presets: list[str],
        run_ids: list[str],
    ) -> dict[str, Any]:
        filters = _delete_filters_from_fields(
            experiments=experiments,
            datasets=datasets,
            models=models,
            presets=presets,
            run_ids=run_ids,
        )
        return self._deletion_planner.create_delete_plan(
            filters,
            active_writers=self._active_log_writers(),
        ).to_response()

    def delete_runs(
        self,
        *,
        experiments: list[str],
        datasets: list[str],
        models: list[str],
        presets: list[str],
        run_ids: list[str],
    ) -> dict[str, Any]:
        filters = _delete_filters_from_fields(
            experiments=experiments,
            datasets=datasets,
            models=models,
            presets=presets,
            run_ids=run_ids,
        )
        with self._mutation_coordinator.coordinate(experiments):
            plan = self._deletion_planner.create_delete_plan(
                filters,
                active_writers=self._active_log_writers(),
            )
            affected_run_paths = [candidate.path for candidate in plan.candidates]
            try:
                return self._deletion_executor.delete_runs(plan).to_response()
            finally:
                self._invalidate_runs(affected_run_paths)

    def import_archive(
        self,
        *,
        archive: BinaryIO,
        filename: str,
        max_upload_size: int | None,
        max_extracted_size: int | None,
        max_member_count: int = DEFAULT_MAX_LOG_ARCHIVE_MEMBER_COUNT,
        max_path_bytes: int = DEFAULT_MAX_LOG_ARCHIVE_PATH_BYTES,
    ) -> dict[str, object]:
        experiments = inspect_log_archive_experiments(
            archive=archive,
            filename=filename,
            max_upload_size=max_upload_size,
            max_extracted_size=max_extracted_size,
            max_member_count=max_member_count,
            max_path_bytes=max_path_bytes,
        )
        with self._mutation_coordinator.coordinate(experiments):
            active_experiments = {
                writer.log_folder for writer in self._active_log_writers()
            }
            if active_experiments.intersection(experiments):
                raise InspectorError(ACTIVE_LOG_EXPERIMENT_DELETE_MESSAGE)
            try:
                return import_log_archive(
                    archive=archive,
                    filename=filename,
                    logs_root=self._logs_root,
                    max_upload_size=max_upload_size,
                    max_extracted_size=max_extracted_size,
                    max_member_count=max_member_count,
                    max_path_bytes=max_path_bytes,
                )
            finally:
                self._invalidate_all()

    def tags_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return self._query.tags_for_runs(run_ids)

    def scalars_for_runs(
        self,
        *,
        run_ids: list[str],
        tags: list[str],
        max_points: int,
        sampling: str,
    ) -> list[dict[str, Any]]:
        return self._query.scalars_for_runs(
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
        return self._query.media_for_runs(
            run_ids=run_ids,
            image_tags=image_tags,
            text_tags=text_tags,
        )

    def monitor_data_for_run(self, run_id: str, node_path: str) -> dict[str, Any]:
        return self._query.monitor_data_for_run(run_id, node_path=node_path)

    def parameter_status_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return self._query.parameter_status_for_runs(run_ids)

    def checkpoints_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return self._query.checkpoints_for_runs(run_ids)

    def artifacts_for_run(self, run_id: str) -> dict[str, Any]:
        return self._query.artifacts_for_run(run_id)

    def inspection_context(self, run_id: str) -> HistoricalInspectionContext:
        run = self._scanner.resolve_runs([run_id])[0]
        params: Mapping[str, Any] = MappingProxyType(
            self._query.saved_params_for_run(run)
        )
        root = self._scanner.resolved_root()
        checkpoint_candidates: list[HistoricalCheckpointCandidate] = []
        for candidate in self._query.checkpoint_paths_for_resolved_run(run):
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
                stat = resolved.stat()
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                checkpoint_candidates.append(
                    HistoricalCheckpointCandidate(
                        path=resolved,
                        size_bytes=int(stat.st_size),
                        modified_at_ns=int(stat.st_mtime_ns),
                    )
                )
        return HistoricalInspectionContext(
            run_id=run.id,
            model=run.model,
            preset=run.preset,
            dataset=run.dataset,
            params=params,
            checkpoint_candidates=tuple(checkpoint_candidates),
        )


__all__ = ["RunHistoryService"]
