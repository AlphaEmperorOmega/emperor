from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, BinaryIO, Literal

from workbench.backend.core.limits import (
    DEFAULT_MAX_LOG_ARCHIVE_MEMBER_COUNT,
    DEFAULT_MAX_LOG_ARCHIVE_PATH_BYTES,
)
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
from workbench.backend.run_history.errors import RunHistoryFailure
from workbench.backend.run_history.query import (
    LOG_EVENT_CACHE_MAX_ENTRIES,
    LOG_SCALAR_ACCUMULATOR_CACHE_MAX_ENTRIES,
    LogRunQueryService,
)
from workbench.backend.run_history.records import (
    LogArchiveImportResult,
    LogCheckpoint,
    LogExperimentDeleteResult,
    LogExperimentPage,
    LogHistogram,
    LogHistogramBucket,
    LogImageSummary,
    LogMedia,
    LogMonitorData,
    LogMonitorImage,
    LogMonitorScalarSeries,
    LogParameterChannelStatus,
    LogParameterNodeStatus,
    LogParameterStatus,
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
    LogScalarPoint,
    LogScalarSeries,
    LogTextSummary,
)
from workbench.backend.run_history.scanner import LogRunScanner
from workbench.backend.tensorboard.events import TensorBoardEventCache
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
    str | None,
    bool | None,
]
CachedRunListing = tuple[list[LogRun], LogRunFacets]


def _selected_values(values: list[str] | None) -> set[str]:
    return set(values or [])


def _matches_model_filter(run_model: str, selected_models: set[str]) -> bool:
    if not selected_models or run_model in selected_models:
        return True
    run_model_leaf = run_model.rsplit("/", 1)[-1]
    return any(
        "/" not in model and model == run_model_leaf for model in selected_models
    )


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


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None


def _optional_bool(value: object) -> bool | None:
    return bool(value) if isinstance(value, bool) else None


def _optional_str(value: object) -> str | None:
    return str(value) if isinstance(value, str) else None


def _scalar_point(value: Mapping[str, Any]) -> LogScalarPoint:
    return LogScalarPoint(
        step=int(value["step"]),
        wall_time=float(value["wallTime"]),
        value=float(value["value"]),
    )


def _image_summary(value: Mapping[str, Any]) -> LogImageSummary:
    return LogImageSummary(
        run_id=str(value["runId"]),
        tag=str(value["tag"]),
        step=int(value["step"]),
        wall_time=float(value["wallTime"]),
        mime_type=str(value["mimeType"]),
        data_url=str(value["dataUrl"]),
        event_bytes=_optional_int(value.get("eventBytes")),
        source_item_count=_optional_int(value.get("sourceItemCount")),
        returned_item_count=_optional_int(value.get("returnedItemCount")),
        truncated=_optional_bool(value.get("truncated")),
        truncation_reason=_optional_str(value.get("truncationReason")),
    )


def _text_summary(value: Mapping[str, Any]) -> LogTextSummary:
    return LogTextSummary(
        run_id=str(value["runId"]),
        tag=str(value["tag"]),
        step=int(value["step"]),
        wall_time=float(value["wallTime"]),
        text=str(value["text"]),
        event_bytes=_optional_int(value.get("eventBytes")),
        source_item_count=_optional_int(value.get("sourceItemCount")),
        returned_item_count=_optional_int(value.get("returnedItemCount")),
        truncated=_optional_bool(value.get("truncated")),
        truncation_reason=_optional_str(value.get("truncationReason")),
    )


def _monitor_data(value: Mapping[str, Any]) -> LogMonitorData:
    scalar_series = tuple(
        LogMonitorScalarSeries(
            tag=str(series["tag"]),
            label=str(series["label"]),
            points=tuple(_scalar_point(point) for point in series["points"]),
            source_item_count=_optional_int(series.get("sourceItemCount")),
            returned_item_count=_optional_int(series.get("returnedItemCount")),
            truncated=_optional_bool(series.get("truncated")),
            truncation_reason=_optional_str(series.get("truncationReason")),
        )
        for series in value["scalarSeries"]
    )
    histograms = tuple(
        LogHistogram(
            tag=str(histogram["tag"]),
            step=int(histogram["step"]),
            wall_time=float(histogram["wallTime"]),
            buckets=tuple(
                LogHistogramBucket(
                    left=float(bucket["left"]),
                    right=float(bucket["right"]),
                    count=float(bucket["count"]),
                )
                for bucket in histogram["buckets"]
            ),
            source_item_count=_optional_int(histogram.get("sourceItemCount")),
            returned_item_count=_optional_int(histogram.get("returnedItemCount")),
            truncated=_optional_bool(histogram.get("truncated")),
            truncation_reason=_optional_str(histogram.get("truncationReason")),
        )
        for histogram in value["histograms"]
    )
    images = tuple(
        LogMonitorImage(
            tag=str(image["tag"]),
            step=int(image["step"]),
            wall_time=float(image["wallTime"]),
            mime_type=str(image["mimeType"]),
            data_url=str(image["dataUrl"]),
            event_bytes=_optional_int(image.get("eventBytes")),
            source_item_count=_optional_int(image.get("sourceItemCount")),
            returned_item_count=_optional_int(image.get("returnedItemCount")),
            truncated=_optional_bool(image.get("truncated")),
            truncation_reason=_optional_str(image.get("truncationReason")),
        )
        for image in value["images"]
    )
    return LogMonitorData(
        job_id=str(value["jobId"]),
        node_path=str(value["nodePath"]),
        preset=_optional_str(value.get("preset")),
        dataset=_optional_str(value.get("dataset")),
        log_dir=_optional_str(value.get("logDir")),
        scalar_series=scalar_series,
        histograms=histograms,
        images=images,
        event_bytes=_optional_int(value.get("eventBytes")),
        skipped_event_files=_optional_int(value.get("skippedEventFiles")),
        truncated=_optional_bool(value.get("truncated")),
        truncation_reason=_optional_str(value.get("truncationReason")),
        source_item_count=_optional_int(value.get("sourceItemCount")),
        returned_item_count=_optional_int(value.get("returnedItemCount")),
    )


def _parameter_channel(value: Mapping[str, Any]) -> LogParameterChannelStatus:
    return LogParameterChannelStatus(
        status=str(value["status"]),
        metric=_optional_str(value.get("metric")),
        last_step=_optional_int(value.get("lastStep")),
        observed_points=int(value["observedPoints"]),
    )


def _parameter_status(value: Mapping[str, Any]) -> LogParameterStatus:
    return LogParameterStatus(
        source_id=str(value["sourceId"]),
        preset=_optional_str(value.get("preset")),
        dataset=_optional_str(value.get("dataset")),
        log_dir=_optional_str(value.get("logDir")),
        nodes=tuple(
            LogParameterNodeStatus(
                node_path=str(node["nodePath"]),
                weights=_parameter_channel(node["weights"]),
                bias=_parameter_channel(node["bias"]),
            )
            for node in value["nodes"]
        ),
        event_bytes=_optional_int(value.get("eventBytes")),
        skipped_event_files=_optional_int(value.get("skippedEventFiles")),
        truncated=_optional_bool(value.get("truncated")),
        truncation_reason=_optional_str(value.get("truncationReason")),
        source_item_count=_optional_int(value.get("sourceItemCount")),
        returned_item_count=_optional_int(value.get("returnedItemCount")),
    )


class RunHistoryService:
    """Own one process-local Run History runtime and cache graph."""

    def __init__(
        self,
        *,
        logs_root: Path | str,
        mutation_coordinator: LogExperimentMutationCoordinator,
        active_log_writers: ActiveLogWriterSource,
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
        max_member_count: int = DEFAULT_MAX_LOG_ARCHIVE_MEMBER_COUNT,
        max_path_bytes: int = DEFAULT_MAX_LOG_ARCHIVE_PATH_BYTES,
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
        return [
            LogRunTags(
                run_id=str(value["runId"]),
                has_layer_monitor_data=_optional_bool(
                    value.get("hasLayerMonitorData")
                ),
                scalar_tags=tuple(str(item) for item in value["scalarTags"]),
                histogram_tags=tuple(str(item) for item in value["histogramTags"]),
                image_tags=tuple(str(item) for item in value["imageTags"]),
                text_tags=tuple(str(item) for item in value["textTags"]),
                event_bytes=_optional_int(value.get("eventBytes")),
                skipped_event_files=_optional_int(value.get("skippedEventFiles")),
                truncated=_optional_bool(value.get("truncated")),
                truncation_reason=_optional_str(value.get("truncationReason")),
                source_item_count=_optional_int(value.get("sourceItemCount")),
                returned_item_count=_optional_int(value.get("returnedItemCount")),
            )
            for value in self._query.tags_for_runs(run_ids)
        ]

    def scalars_for_runs(
        self,
        *,
        run_ids: list[str],
        tags: list[str],
        max_points: int,
        sampling: str,
    ) -> list[LogScalarSeries]:
        values = self._query.scalars_for_runs(
            run_ids=run_ids,
            tags=tags,
            max_points=max_points,
            sampling=sampling,
        )
        return [
            LogScalarSeries(
                run_id=str(value["runId"]),
                tag=str(value["tag"]),
                points=tuple(_scalar_point(point) for point in value["points"]),
                source_point_count=_optional_int(value.get("sourcePointCount")),
                truncated=_optional_bool(value.get("truncated")),
            )
            for value in values
        ]

    def media_for_runs(
        self,
        *,
        run_ids: list[str],
        image_tags: list[str],
        text_tags: list[str],
    ) -> LogMedia:
        value = self._query.media_for_runs(
            run_ids=run_ids,
            image_tags=image_tags,
            text_tags=text_tags,
        )
        return LogMedia(
            images=tuple(_image_summary(item) for item in value["images"]),
            texts=tuple(_text_summary(item) for item in value["texts"]),
            event_bytes=_optional_int(value.get("eventBytes")),
            skipped_event_files=_optional_int(value.get("skippedEventFiles")),
            source_item_count=_optional_int(value.get("sourceItemCount")),
            returned_item_count=_optional_int(value.get("returnedItemCount")),
            truncated=_optional_bool(value.get("truncated")),
            truncation_reason=_optional_str(value.get("truncationReason")),
        )

    def monitor_data_for_run(self, run_id: str, node_path: str) -> LogMonitorData:
        return _monitor_data(
            self._query.monitor_data_for_run(run_id, node_path=node_path)
        )

    def parameter_status_for_runs(
        self,
        run_ids: list[str],
    ) -> list[LogParameterStatus]:
        return [
            _parameter_status(value)
            for value in self._query.parameter_status_for_runs(run_ids)
        ]

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
