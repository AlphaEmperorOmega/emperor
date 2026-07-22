from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, BinaryIO, Literal

from emperor_workbench.log_experiments import (
    LogExperimentMutationCoordinator,
)
from emperor_workbench.run_history._archive import (
    import_log_archive,
    inspect_log_archive_experiments,
)
from emperor_workbench.run_history._contracts import (
    ActiveLogWriterSource,
    HistoricalCheckpointCandidate,
    HistoricalInspectionContext,
    KnownModelPackageIdentityResolver,
)
from emperor_workbench.run_history._deletion import (
    LogRunDeletionExecutor,
    LogRunDeletionPlanner,
)
from emperor_workbench.run_history._errors import RunHistoryFailure
from emperor_workbench.run_history._query import (
    LOG_EVENT_CACHE_MAX_ENTRIES,
    LOG_SCALAR_ACCUMULATOR_CACHE_MAX_ENTRIES,
    LogRunQueryService,
)
from emperor_workbench.run_history._records import (
    LogArchiveImportResult,
    LogCheckpoint,
    LogExperimentDeleteResult,
    LogExperimentPage,
    LogMedia,
    LogRun,
    LogRunArtifacts,
    LogRunDeleteFilters,
    LogRunDeletePlan,
    LogRunDeleteResult,
    LogRunExperimentFacets,
    LogRunFacets,
    LogRunFacetValue,
    LogRunModelFacet,
    LogRunPage,
    LogRunTags,
    LogScalarSeries,
)
from emperor_workbench.run_history._scanner import LogRunScanner
from emperor_workbench.tensorboard import (
    DEFAULT_SCALAR_POINT_LIMIT,
    MonitorData,
    ParameterStatus,
    TensorBoardEventCache,
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
    str | None,
    bool | None,
]
CachedRunListing = tuple[list[LogRun], LogRunFacets]


def _selected_values(values: list[str] | None) -> set[str]:
    return set(values or [])


def _matches_model_filter(run_model: str, selected_models: set[str]) -> bool:
    return not selected_models or run_model in selected_models


def _delete_filters_from_fields(
    *,
    experiments: list[str],
    datasets: list[str],
    models: list[str],
    presets: list[str],
    run_ids: list[str],
) -> LogRunDeleteFilters:
    return LogRunDeleteFilters(
        run_ids=tuple(run_ids),
        experiments=tuple(experiments),
        datasets=tuple(datasets),
        models=tuple(models),
        presets=tuple(presets),
    )


def _value_facets(values: list[str]) -> tuple[LogRunFacetValue, ...]:
    return tuple(
        LogRunFacetValue(value=value, count=count)
        for value, count in sorted(Counter(values).items())
    )


def _run_facets(runs: list[LogRun]) -> LogRunFacets:
    runs_by_experiment: dict[str, list[LogRun]] = defaultdict(list)
    for run in runs:
        runs_by_experiment[run.experiment].append(run)

    experiments: list[LogRunExperimentFacets] = []
    for experiment, experiment_runs in sorted(runs_by_experiment.items()):
        model_counts = Counter(run.model for run in experiment_runs)
        experiments.append(
            LogRunExperimentFacets(
                experiment=experiment,
                run_count=len(experiment_runs),
                datasets=_value_facets([run.dataset for run in experiment_runs]),
                models=tuple(
                    LogRunModelFacet(model=model, count=count)
                    for model, count in sorted(model_counts.items())
                ),
                presets=_value_facets([run.preset for run in experiment_runs]),
            )
        )
    return LogRunFacets(experiments=tuple(experiments))


class RunHistoryService:
    """Own one process-local Run History runtime and cache graph."""

    def __init__(
        self,
        *,
        logs_root: Path | str,
        mutation_coordinator: LogExperimentMutationCoordinator,
        active_log_writers: ActiveLogWriterSource,
        model_identity_resolver: KnownModelPackageIdentityResolver,
        state_root: Path | None = None,
        scalar_point_limit: int = DEFAULT_SCALAR_POINT_LIMIT,
        tensorboard_request_work_bytes: int = 64 * 1024 * 1024,
        tensorboard_cache_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        self._logs_root = Path(logs_root)
        self._mutation_coordinator = mutation_coordinator
        self._active_log_writers = active_log_writers
        self._scanner = LogRunScanner(
            logs_root=self._logs_root,
            state_root=state_root,
            model_identity_resolver=model_identity_resolver,
        )
        event_cache = TensorBoardEventCache(
            {
                "tags": LOG_EVENT_CACHE_MAX_ENTRIES,
                "scalars": LOG_EVENT_CACHE_MAX_ENTRIES,
                "scalar_accumulators": LOG_SCALAR_ACCUMULATOR_CACHE_MAX_ENTRIES,
                "monitor_payload": LOG_EVENT_CACHE_MAX_ENTRIES,
                "parameter_status_payload": LOG_EVENT_CACHE_MAX_ENTRIES,
            },
            max_bytes=tensorboard_cache_bytes,
        )
        monitor_reader = TensorBoardMonitorReader(
            scalar_point_limit=scalar_point_limit,
            max_event_bytes=tensorboard_request_work_bytes,
            event_cache=event_cache,
        )
        parameter_status_reader = TensorBoardParameterStatusReader(
            max_event_bytes=tensorboard_request_work_bytes,
            event_cache=event_cache,
        )
        self._query = LogRunQueryService(
            scanner=self._scanner,
            scalar_point_limit=scalar_point_limit,
            monitor_reader=monitor_reader,
            parameter_status_reader=parameter_status_reader,
            event_cache=event_cache,
            max_request_event_bytes=tensorboard_request_work_bytes,
            max_tag_event_bytes=tensorboard_request_work_bytes,
            max_tag_batch_event_bytes=tensorboard_request_work_bytes,
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

    def invalidate_experiment(self, experiment: str) -> None:
        """Inward seam for a terminal Training Job's Log Experiment."""

        run_paths = self._scanner.invalidate_experiment(experiment)
        self._query.clear_run_caches(run_paths)
        self._clear_run_listing_cache()

    def _filtered_run_listing(
        self,
        *,
        experiment_set: set[str],
        model_set: set[str],
        preset_set: set[str],
        dataset_set: set[str],
        experiment_task: str | None,
        has_event_files: bool | None,
    ) -> tuple[list[LogRun], LogRunFacets]:
        with self._run_listing_lock:
            generation = self._run_listing_generation
        catalog = self._scanner.list_runs(result_projection="none")
        catalog_signature = tuple(id(run) for run in catalog)
        cache_key = (
            tuple(sorted(experiment_set)),
            tuple(sorted(model_set)),
            tuple(sorted(preset_set)),
            tuple(sorted(dataset_set)),
            experiment_task,
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

        filtered_runs: list[LogRun] = []
        for run in catalog:
            if experiment_set and run.experiment not in experiment_set:
                continue
            if not _matches_model_filter(run.model, model_set):
                continue
            if preset_set and run.preset not in preset_set:
                continue
            if dataset_set and run.dataset not in dataset_set:
                continue
            if experiment_task is not None:
                projected = self._scanner.project_run(run, include_metrics=False)
                if projected.experiment_task != experiment_task:
                    continue
            if (
                has_event_files is not None
                and (run.event_file_count > 0) != has_event_files
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
        experiment_task: str | None = None,
        has_event_files: bool | None = None,
        projection: Literal["full", "summary"] = "full",
    ) -> LogRunPage:
        filtered_runs, facets = self._filtered_run_listing(
            experiment_set=_selected_values(experiment),
            model_set=_selected_values(model),
            preset_set=_selected_values(preset),
            dataset_set=_selected_values(dataset),
            experiment_task=experiment_task,
            has_event_files=has_event_files,
        )
        runs: list[LogRun] = []
        for run in filtered_runs[offset : offset + limit]:
            projected = self._scanner.project_run(
                run,
                include_metrics=projection == "full",
            )
            if projection == "summary":
                projected = replace(
                    projected,
                    metrics={},
                    has_layer_monitor_data=None,
                )
            else:
                projected = replace(
                    projected,
                    has_layer_monitor_data=(
                        self._query.cached_layer_monitor_data_for_run(run)
                    ),
                )
            runs.append(projected)
        return LogRunPage(
            runs=tuple(runs),
            total=len(filtered_runs),
            limit=limit,
            offset=offset,
            has_more=offset + limit < len(filtered_runs),
            facets=facets,
        )

    def list_experiments(self, *, limit: int, offset: int) -> LogExperimentPage:
        experiments = self._scanner.list_experiments()
        return LogExperimentPage(
            experiments=tuple(experiments[offset : offset + limit]),
            total=len(experiments),
            limit=limit,
            offset=offset,
            has_more=offset + limit < len(experiments),
        )

    def delete_experiment(self, experiment: str) -> LogExperimentDeleteResult:
        with self._mutation_coordinator.coordinate([experiment]):
            self._scanner.reconcile_catalog()
            if any(
                writer.log_folder == experiment for writer in self._active_log_writers()
            ):
                raise RunHistoryFailure(ACTIVE_LOG_EXPERIMENT_DELETE_MESSAGE)
            try:
                return self._deletion_executor.delete_experiment(experiment)
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
    ) -> LogRunDeletePlan:
        self._scanner.reconcile_catalog()
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
        )

    def delete_runs(
        self,
        *,
        experiments: list[str],
        datasets: list[str],
        models: list[str],
        presets: list[str],
        run_ids: list[str],
    ) -> LogRunDeleteResult:
        filters = _delete_filters_from_fields(
            experiments=experiments,
            datasets=datasets,
            models=models,
            presets=presets,
            run_ids=run_ids,
        )
        with self._mutation_coordinator.coordinate(experiments):
            self._scanner.reconcile_catalog()
            plan = self._deletion_planner.create_delete_plan(
                filters,
                active_writers=self._active_log_writers(),
            )
            affected_run_paths = [candidate.path for candidate in plan.candidates]
            try:
                return self._deletion_executor.delete_runs(plan)
            finally:
                self._invalidate_runs(affected_run_paths)

    def create_preset_delete_plan(
        self,
        *,
        experiment: str,
        preset: str,
    ) -> LogRunDeletePlan:
        with self._mutation_coordinator.coordinate([experiment]):
            self._scanner.reconcile_catalog()
            return self._deletion_planner.create_preset_delete_plan(
                experiment=experiment,
                preset=preset,
                active_writers=self._active_log_writers(),
            )

    def delete_preset(
        self,
        *,
        experiment: str,
        preset: str,
    ) -> LogRunDeleteResult:
        with self._mutation_coordinator.coordinate([experiment]):
            self._scanner.reconcile_catalog()
            plan = self._deletion_planner.create_preset_delete_plan(
                experiment=experiment,
                preset=preset,
                active_writers=self._active_log_writers(),
            )
            affected_run_paths = [candidate.path for candidate in plan.candidates]
            try:
                return self._deletion_executor.delete_runs(plan)
            finally:
                self._invalidate_runs(affected_run_paths)

    def import_archive(
        self,
        *,
        archive: BinaryIO,
        filename: str,
        max_upload_size: int | None,
        max_extracted_size: int | None,
        max_member_count: int,
        max_path_bytes: int,
    ) -> LogArchiveImportResult:
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
                raise RunHistoryFailure(ACTIVE_LOG_EXPERIMENT_DELETE_MESSAGE)
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

    def tags_for_runs(self, run_ids: list[str]) -> list[LogRunTags]:
        return self._query.tags_for_runs(run_ids)

    def scalars_for_runs(
        self,
        *,
        run_ids: list[str],
        tags: list[str],
        max_points: int,
        sampling: str,
    ) -> list[LogScalarSeries]:
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
    ) -> LogMedia:
        return self._query.media_for_runs(
            run_ids=run_ids,
            image_tags=image_tags,
            text_tags=text_tags,
        )

    def monitor_data_for_run(self, run_id: str, node_path: str) -> MonitorData:
        return self._query.monitor_data_for_run(run_id, node_path=node_path)

    def parameter_status_for_runs(
        self,
        run_ids: list[str],
    ) -> list[ParameterStatus]:
        return self._query.parameter_status_for_runs(run_ids)

    def checkpoints_for_runs(self, run_ids: list[str]) -> list[LogCheckpoint]:
        return self._query.checkpoints_for_runs(run_ids)

    def artifacts_for_run(self, run_id: str) -> LogRunArtifacts:
        return self._query.artifacts_for_run(run_id)

    def inspection_context(self, run_id: str) -> HistoricalInspectionContext:
        run = self._scanner.resolve_runs([run_id])[0]
        params: Mapping[str, Any] = MappingProxyType(
            self._query.saved_params_for_run(run)
        )
        root = self._scanner.resolved_root()
        try:
            run_root = run.path.resolve(strict=True)
            run_root.relative_to(root)
        except (OSError, ValueError):
            run_root = run.path
        checkpoint_candidates: list[HistoricalCheckpointCandidate] = []
        for candidate in self._query.checkpoint_paths_for_resolved_run(run):
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(run_root)
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
