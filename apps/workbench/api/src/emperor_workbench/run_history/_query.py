from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import emperor_workbench.tensorboard as tensorboard
from emperor_workbench.failures import FailureKind
from emperor_workbench.run_history._artifacts import (
    ObservedRunArtifact,
    RunArtifactObservation,
    _file_id,
    _parse_checkpoint_epoch,
    _parse_checkpoint_step,
    _run_relative_file_label,
)
from emperor_workbench.run_history._errors import RunHistoryFailure
from emperor_workbench.run_history._records import (
    LogCheckpoint,
    LogImageSummary,
    LogMedia,
    LogRun,
    LogRunArtifact,
    LogRunArtifacts,
    LogRunTags,
    LogScalarPoint,
    LogScalarSeries,
    LogTextSummary,
)
from emperor_workbench.run_history._scanner import LogRunScanner
from emperor_workbench.tensorboard import (
    DEFAULT_SCALAR_POINT_LIMIT,
    EventFileIndex,
    ImageSummary,
    MonitorData,
    ParameterStatus,
    ScalarTail,
    TagCatalog,
    TensorBoardEventCache,
    TensorBoardMonitorReader,
    TensorBoardParameterStatusReader,
    TextSummary,
)

LOG_EVENT_CACHE_MAX_ENTRIES = 256
LOG_TAG_READ_MAX_EVENT_BYTES = 64 * 1024 * 1024
LOG_TAG_BATCH_READ_MAX_EVENT_BYTES = 64 * 1024 * 1024
LOG_SCALAR_ACCUMULATOR_CACHE_MAX_ENTRIES = 32
PARAMETER_MONITOR_CHANNELS = {"weights", "bias"}
LAYER_MONITOR_METRICS = {
    "relative_delta_norm",
    "delta_norm",
    "l2_norm",
    "mean",
    "var",
    "grad_mean",
    "grad_var",
    "grad_norm",
    "update_ratio",
    "spectral_norm",
    "condition_number",
    "effective_rank",
    "dead_input_fraction",
    "dead_output_fraction",
}
SummaryT = TypeVar("SummaryT", ImageSummary, TextSummary)


def _is_layer_monitor_tag(tag: str) -> bool:
    parts = tag.rsplit("/", 2)
    if len(parts) != 3:
        return False
    node_path, channel, metric = parts
    return (
        bool(node_path)
        and channel in PARAMETER_MONITOR_CHANNELS
        and metric in LAYER_MONITOR_METRICS
    )


def _tags_have_layer_monitor_data(tags: TagCatalog) -> bool:
    return any(
        _is_layer_monitor_tag(tag)
        for tag_group in (
            tags.scalar_tags,
            tags.histogram_tags,
            tags.image_tags,
        )
        for tag in tag_group
    )


class LogRunQueryService:
    def __init__(
        self,
        *,
        scanner: LogRunScanner,
        scalar_point_limit: int = DEFAULT_SCALAR_POINT_LIMIT,
        max_tag_event_bytes: int = LOG_TAG_READ_MAX_EVENT_BYTES,
        max_tag_batch_event_bytes: int = LOG_TAG_BATCH_READ_MAX_EVENT_BYTES,
        monitor_reader: TensorBoardMonitorReader | None = None,
        parameter_status_reader: TensorBoardParameterStatusReader | None = None,
        event_cache: TensorBoardEventCache | None = None,
        max_request_event_bytes: int | None = None,
    ) -> None:
        self.scanner = scanner
        self.scalar_point_limit = scalar_point_limit
        self.max_tag_event_bytes = max(0, int(max_tag_event_bytes))
        self.max_tag_batch_event_bytes = max(0, int(max_tag_batch_event_bytes))
        self.max_request_event_bytes = max(
            1,
            int(
                max_tag_batch_event_bytes
                if max_request_event_bytes is None
                else max_request_event_bytes
            ),
        )
        self.monitor_reader = monitor_reader or TensorBoardMonitorReader(
            scalar_point_limit=scalar_point_limit,
        )
        self.parameter_status_reader = (
            parameter_status_reader or TensorBoardParameterStatusReader()
        )
        self._event_cache = event_cache or TensorBoardEventCache(
            {
                "tags": LOG_EVENT_CACHE_MAX_ENTRIES,
                "scalars": LOG_EVENT_CACHE_MAX_ENTRIES,
                "scalar_accumulators": LOG_SCALAR_ACCUMULATOR_CACHE_MAX_ENTRIES,
            }
        )

    def _cache_token(self) -> int:
        return self._event_cache.token()

    def _cache_get(
        self,
        cache_name: str,
        key: tuple[Any, ...],
    ) -> Any | None:
        return self._event_cache.get(cache_name, key)

    def _cache_set(
        self,
        cache_name: str,
        key: tuple[Any, ...],
        value: Any,
        *,
        generation: int,
    ) -> None:
        self._event_cache.publish(
            cache_name,
            key,
            value,
            generation=generation,
        )

    def clear_cache(self) -> None:
        self._event_cache.clear()
        self.monitor_reader.clear_cache()
        self.parameter_status_reader.clear_cache()

    def clear_run_caches(self, run_paths: list[Path]) -> None:
        roots = {path.as_posix() for path in run_paths}
        if not roots:
            return
        self._event_cache.clear_roots(roots)
        self.monitor_reader.clear_roots(roots)
        self.parameter_status_reader.clear_roots(roots)

    def _tags_cache_key(
        self,
        event_files: EventFileIndex,
    ) -> tuple[Any, ...]:
        return event_files.cache_key()

    def _cached_tags_if_current(
        self,
        run_dir: Path,
        *,
        event_files: EventFileIndex | None = None,
    ) -> TagCatalog | None:
        event_files = event_files or tensorboard.event_file_index(run_dir)
        cached = self._cache_get("tags", self._tags_cache_key(event_files))
        return cached if isinstance(cached, TagCatalog) else None

    def _scalar_cache_key(
        self,
        event_files: EventFileIndex,
        *,
        tag: str,
        max_points: int,
        sampling: str,
    ) -> tuple[Any, ...]:
        return event_files.cache_key(
            tag,
            max_points,
            sampling,
        )

    def tags_for_runs(self, run_ids: list[str]) -> list[LogRunTags]:
        runs = self.scanner.resolve_runs(run_ids)
        results: list[LogRunTags] = []
        uncached_event_bytes = 0
        for run in runs:
            cache_generation = self._cache_token()
            event_files = self.scanner.artifact_observation(run).event_files
            tags = self._cached_tags_if_current(
                run.path,
                event_files=event_files,
            )
            if tags is None:
                exceeds_per_run_tag_budget = event_files.exceeds(
                    self.max_tag_event_bytes
                )
                would_exceed_batch_budget = (
                    self.max_tag_batch_event_bytes > 0
                    and uncached_event_bytes + event_files.total_size
                    > self.max_tag_batch_event_bytes
                )
                if exceeds_per_run_tag_budget:
                    tags = self.read_tags(
                        run.path,
                        event_files=event_files,
                        cache_generation=cache_generation,
                    )
                elif would_exceed_batch_budget:
                    raise RunHistoryFailure(
                        "TensorBoard tag event files exceed the shared "
                        f"{self.max_tag_batch_event_bytes} byte read budget.",
                        kind=FailureKind.TOO_LARGE,
                    )
                else:
                    uncached_event_bytes += event_files.total_size
                    tags = self.read_tags(
                        run.path,
                        event_files=event_files,
                        cache_generation=cache_generation,
                    )
            results.append(
                LogRunTags(
                    run_id=run.id,
                    has_layer_monitor_data=_tags_have_layer_monitor_data(tags),
                    scalar_tags=tags.scalar_tags,
                    histogram_tags=tags.histogram_tags,
                    image_tags=tags.image_tags,
                    text_tags=tags.text_tags,
                    event_bytes=tags.event_bytes,
                    skipped_event_files=tags.skipped_event_files,
                    truncated=tags.truncated,
                    truncation_reason=tags.truncation_reason,
                    source_item_count=tags.source_item_count,
                    returned_item_count=tags.returned_item_count,
                )
            )
        return results

    def cached_layer_monitor_data_for_run(self, run: LogRun) -> bool | None:
        event_files = self.scanner.artifact_observation(run).event_files
        tags = self._cached_tags_if_current(
            run.path,
            event_files=event_files,
        )
        if tags is None:
            return None
        return _tags_have_layer_monitor_data(tags)

    def scalars_for_runs(
        self,
        *,
        run_ids: list[str],
        tags: list[str],
        max_points: int | None = None,
        sampling: str = "tail",
    ) -> list[LogScalarSeries]:
        runs = self.scanner.resolve_runs(run_ids)
        requested_tags = list(dict.fromkeys(tags))
        if not requested_tags:
            return []

        series: list[LogScalarSeries] = []
        event_read_bytes = 0
        for run in runs:
            cache_generation = self._cache_token()
            event_files = self.scanner.artifact_observation(run).event_files
            event_read_bytes += event_files.total_size
            if event_read_bytes > self.max_request_event_bytes:
                raise RunHistoryFailure(
                    "TensorBoard scalar event files exceed the shared "
                    f"{self.max_request_event_bytes} byte read budget.",
                    kind=FailureKind.TOO_LARGE,
                )
            cached_tags = self._cached_tags_if_current(
                run.path,
                event_files=event_files,
            )
            run_tags = set(
                cached_tags.scalar_tags
                if cached_tags is not None
                else self.read_tags(
                    run.path,
                    event_files=event_files,
                    cache_generation=cache_generation,
                ).scalar_tags,
            )
            available_tags = [tag for tag in requested_tags if tag in run_tags]
            scalar_series_by_tag = self.read_scalar_series_batch(
                run.path,
                available_tags,
                max_points=max_points,
                sampling=sampling,
                event_files=event_files,
                cache_generation=cache_generation,
            )
            for tag in requested_tags:
                scalar_series = scalar_series_by_tag.get(tag)
                if scalar_series is None:
                    continue
                if scalar_series.points:
                    series.append(
                        LogScalarSeries(
                            run_id=run.id,
                            tag=tag,
                            points=tuple(
                                LogScalarPoint(
                                    step=point.step,
                                    wall_time=point.wall_time,
                                    value=point.value,
                                )
                                for point in scalar_series.points
                            ),
                            source_point_count=scalar_series.source_point_count,
                            truncated=scalar_series.truncated,
                        )
                    )
        return series

    def media_for_runs(
        self,
        *,
        run_ids: list[str],
        image_tags: list[str],
        text_tags: list[str],
    ) -> LogMedia:
        runs = self.scanner.resolve_runs(run_ids)
        requested_image_tags = list(dict.fromkeys(image_tags))
        requested_text_tags = list(dict.fromkeys(text_tags))
        images: list[LogImageSummary] = []
        texts: list[LogTextSummary] = []
        source_item_count = len(runs) * (
            len(requested_image_tags) + len(requested_text_tags)
        )
        skipped_event_files = 0
        event_bytes = 0
        skipped_reasons: list[str] = []
        request_event_bytes = 0

        for run in runs:
            cache_generation = self._cache_token()
            event_files = self.scanner.artifact_observation(run).event_files
            request_event_bytes += event_files.total_size
            if request_event_bytes > self.max_request_event_bytes:
                raise RunHistoryFailure(
                    "TensorBoard media event files exceed the shared "
                    f"{self.max_request_event_bytes} byte read budget.",
                    kind=FailureKind.TOO_LARGE,
                )
            run_tags = self.read_tags(
                run.path,
                event_files=event_files,
                cache_generation=cache_generation,
            )
            if run_tags.truncated:
                skipped_event_files += run_tags.skipped_event_files or 0
                event_bytes += run_tags.event_bytes or 0
                if run_tags.truncation_reason:
                    skipped_reasons.append(run_tags.truncation_reason)
            image_tag_set = set(run_tags.image_tags)
            text_tag_set = set(run_tags.text_tags)
            for tag in requested_image_tags:
                if tag not in image_tag_set:
                    continue
                summary = self.read_image_summary(
                    run.path,
                    tag,
                    event_files=event_files,
                )
                if summary is not None:
                    images.append(
                        LogImageSummary(
                            run_id=run.id,
                            tag=summary.tag,
                            step=summary.step,
                            wall_time=summary.wall_time,
                            mime_type=summary.mime_type,
                            data_url=summary.data_url,
                            event_bytes=summary.event_bytes,
                            source_item_count=summary.source_item_count,
                            returned_item_count=summary.returned_item_count,
                            truncated=summary.truncated,
                            truncation_reason=summary.truncation_reason,
                        )
                    )
            for tag in requested_text_tags:
                if tag not in text_tag_set:
                    continue
                summary = self.read_text_summary(
                    run.path,
                    tag,
                    event_files=event_files,
                )
                if summary is not None:
                    texts.append(
                        LogTextSummary(
                            run_id=run.id,
                            tag=summary.tag,
                            step=summary.step,
                            wall_time=summary.wall_time,
                            text=summary.text,
                            event_bytes=summary.event_bytes,
                            source_item_count=summary.source_item_count,
                            returned_item_count=summary.returned_item_count,
                            truncated=summary.truncated,
                            truncation_reason=summary.truncation_reason,
                        )
                    )

        returned_item_count = len(images) + len(texts)
        truncated_items = [item for item in [*images, *texts] if item.truncated]
        truncated = skipped_event_files > 0 or bool(truncated_items)
        reason = None
        if skipped_reasons:
            reason = skipped_reasons[0]
        elif truncated_items:
            reason = truncated_items[0].truncation_reason or "media truncated"

        return LogMedia(
            source_item_count=source_item_count,
            returned_item_count=returned_item_count,
            truncated=truncated,
            truncation_reason=reason,
            event_bytes=event_bytes or None,
            skipped_event_files=skipped_event_files or None,
            images=tuple(images),
            texts=tuple(texts),
        )

    def monitor_data_for_run(self, run_id: str, node_path: str) -> MonitorData:
        run = self.scanner.resolve_runs([run_id])[0]
        event_files = self.scanner.artifact_observation(run).event_files
        return self.monitor_reader.read(
            job_id=run.id,
            node_path=node_path,
            dataset=run.dataset,
            log_dir=str(run.path),
            event_files=event_files,
        )

    def parameter_status_for_runs(self, run_ids: list[str]) -> list[ParameterStatus]:
        runs = self.scanner.resolve_runs(run_ids)
        return [
            self.parameter_status_reader.read(
                source_id=run.id,
                preset=run.preset,
                dataset=run.dataset,
                log_dir=str(run.path),
                event_files=self.scanner.artifact_observation(run).event_files,
            )
            for run in runs
        ]

    def checkpoints_for_runs(self, run_ids: list[str]) -> list[LogCheckpoint]:
        runs = self.scanner.resolve_runs(run_ids)
        return [checkpoint for run in runs for checkpoint in self.read_checkpoints(run)]

    def checkpoint_paths_for_resolved_run(self, run: LogRun) -> list[Path]:
        return [
            artifact.path
            for artifact in self.scanner.artifact_observation(run).checkpoints
        ]

    def saved_params_for_run(
        self,
        run: LogRun,
        *,
        artifacts: RunArtifactObservation | None = None,
    ) -> dict[str, Any]:
        artifacts = artifacts or self.scanner.artifact_observation(run)
        return {**artifacts.hparams_values(), **artifacts.params()}

    def artifacts_for_run(self, run_id: str) -> LogRunArtifacts:
        run = self.scanner.resolve_runs([run_id])[0]
        observation = self.scanner.artifact_observation(run)
        metrics = observation.metrics()
        checkpoints = self.read_checkpoints(run, artifacts=observation)
        artifacts: list[LogRunArtifact] = []

        artifacts.extend(
            self.artifact_metadata(
                run,
                artifact,
                kind="event_file",
                label=_run_relative_file_label(run.path, artifact.path),
            )
            for artifact in observation.event_artifacts
        )
        for artifact, kind in (
            (observation.hparams, "hparams"),
            (observation.result, "result"),
        ):
            if artifact is not None:
                artifacts.append(
                    self.artifact_metadata(
                        run,
                        artifact,
                        kind=kind,
                        label=artifact.path.name,
                    )
                )
        artifacts.extend(
            self.artifact_metadata(
                run,
                artifact,
                kind="checkpoint",
                label=_run_relative_file_label(run.path, artifact.path),
            )
            for artifact in observation.checkpoints
        )

        return LogRunArtifacts(
            run_id=run.id,
            params=self.saved_params_for_run(run, artifacts=observation),
            metrics=metrics,
            artifacts=tuple(artifacts),
            checkpoints=tuple(checkpoints),
            truncation_reasons=observation.truncation_reasons,
        )

    def read_checkpoints(
        self,
        run: LogRun,
        *,
        artifacts: RunArtifactObservation | None = None,
    ) -> list[LogCheckpoint]:
        artifacts = artifacts or self.scanner.artifact_observation(run)
        checkpoints = [
            self.checkpoint_metadata(run, artifact)
            for artifact in artifacts.checkpoints
        ]
        return sorted(
            checkpoints,
            key=lambda checkpoint: (
                checkpoint.step is None,
                checkpoint.step if checkpoint.step is not None else -1,
                checkpoint.epoch is None,
                checkpoint.epoch if checkpoint.epoch is not None else -1,
                checkpoint.filename,
                checkpoint.relative_path,
            ),
        )

    def checkpoint_metadata(
        self,
        run: LogRun,
        artifact: ObservedRunArtifact,
    ) -> LogCheckpoint:
        return LogCheckpoint(
            id=_file_id(run.id, artifact.relative_path),
            run_id=run.id,
            filename=artifact.path.name,
            relative_path=artifact.relative_path,
            epoch=_parse_checkpoint_epoch(artifact.path.name),
            step=_parse_checkpoint_step(artifact.path.name),
            size_bytes=artifact.size,
            modified_at=artifact.modified_at,
        )

    def artifact_metadata(
        self,
        run: LogRun,
        artifact: ObservedRunArtifact,
        *,
        kind: str,
        label: str,
    ) -> LogRunArtifact:
        return LogRunArtifact(
            id=_file_id(run.id, artifact.relative_path),
            kind=kind,
            label=label,
            relative_path=artifact.relative_path,
            size_bytes=artifact.size,
            modified_at=artifact.modified_at,
        )

    def read_tags(
        self,
        run_dir: Path,
        *,
        event_files: EventFileIndex | None = None,
        cache_generation: int | None = None,
    ) -> TagCatalog:
        generation = (
            self._cache_token() if cache_generation is None else cache_generation
        )
        event_files = event_files or tensorboard.event_file_index(run_dir)
        cache_key = self._tags_cache_key(event_files)
        cached = self._cache_get("tags", cache_key)
        if isinstance(cached, TagCatalog):
            return cached

        tags = {"scalars": set(), "histograms": set(), "images": set(), "texts": set()}
        if event_files.exceeds(self.max_tag_event_bytes):
            result = TagCatalog(
                scalar_tags=(),
                histogram_tags=(),
                image_tags=(),
                text_tags=(),
                event_bytes=event_files.total_size,
                skipped_event_files=len(event_files.fingerprint),
                truncated=True,
                truncation_reason=(
                    "event files skipped: "
                    f"{event_files.total_size} bytes exceeds "
                    f"{self.max_tag_event_bytes} byte tag-read cap"
                ),
                source_item_count=len(event_files.fingerprint),
                returned_item_count=0,
            )
            self._cache_set(
                "tags",
                cache_key,
                result,
                generation=generation,
            )
            return result
        for event_dir in event_files.dirs:
            accumulator = event_files.load_accumulator(
                event_dir,
                size_guidance=tensorboard.TENSORBOARD_TAG_SIZE_GUIDANCE,
            )
            if accumulator is None:
                continue
            try:
                accumulator_tags = accumulator.Tags()
            except Exception:
                continue
            tags["scalars"].update(accumulator_tags.get("scalars", []))
            tags["histograms"].update(accumulator_tags.get("histograms", []))
            tags["images"].update(accumulator_tags.get("images", []))
            tags["texts"].update(
                tag
                for tag in accumulator_tags.get("tensors", [])
                if tag.endswith("/text_summary")
            )
        returned_item_count = sum(len(value) for value in tags.values())
        result = TagCatalog(
            scalar_tags=tuple(sorted(tags["scalars"])),
            histogram_tags=tuple(sorted(tags["histograms"])),
            image_tags=tuple(sorted(tags["images"])),
            text_tags=tuple(sorted(tags["texts"])),
            truncated=False,
            source_item_count=returned_item_count,
            returned_item_count=returned_item_count,
        )
        self._cache_set(
            "tags",
            cache_key,
            result,
            generation=generation,
        )
        return result

    def read_scalar_series_batch(
        self,
        run_dir: Path,
        tags: list[str],
        *,
        max_points: int | None = None,
        sampling: str = "tail",
        event_files: EventFileIndex | None = None,
        cache_generation: int | None = None,
    ) -> dict[str, ScalarTail]:
        generation = (
            self._cache_token() if cache_generation is None else cache_generation
        )
        event_files = event_files or tensorboard.event_file_index(run_dir)
        point_limit = max_points if max_points is not None else self.scalar_point_limit
        requested_tags = list(dict.fromkeys(tags))
        results: dict[str, ScalarTail] = {}
        uncached_tags: list[str] = []
        for tag in requested_tags:
            cache_key = self._scalar_cache_key(
                event_files,
                tag=tag,
                max_points=point_limit,
                sampling=sampling,
            )
            cached = self._cache_get("scalars", cache_key)
            if isinstance(cached, ScalarTail):
                results[tag] = cached
            else:
                uncached_tags.append(tag)

        if uncached_tags:
            try:
                streamed = tensorboard.exact_scalar_tails(
                    event_files,
                    uncached_tags,
                    max_points=point_limit,
                    byte_budget=self.max_request_event_bytes,
                )
            except ValueError as exc:
                raise RunHistoryFailure(str(exc)) from exc

            for tag in uncached_tags:
                result = streamed[tag]
                self._cache_set(
                    "scalars",
                    self._scalar_cache_key(
                        event_files,
                        tag=tag,
                        max_points=point_limit,
                        sampling=sampling,
                    ),
                    result,
                    generation=generation,
                )
                results[tag] = result

        return {tag: results[tag] for tag in requested_tags if tag in results}

    def read_scalar_series(
        self,
        run_dir: Path,
        tag: str,
        *,
        max_points: int | None = None,
        sampling: str = "tail",
    ) -> ScalarTail:
        return self.read_scalar_series_batch(
            run_dir,
            [tag],
            max_points=max_points,
            sampling=sampling,
        )[tag]

    def read_image_summary(
        self,
        run_dir: Path,
        tag: str,
        *,
        event_files: EventFileIndex | None = None,
    ) -> ImageSummary | None:
        return self._read_latest_summary(
            run_dir,
            tag,
            tensorboard.image_summary,
            event_files=event_files,
        )

    def read_text_summary(
        self,
        run_dir: Path,
        tag: str,
        *,
        event_files: EventFileIndex | None = None,
    ) -> TextSummary | None:
        return self._read_latest_summary(
            run_dir,
            tag,
            tensorboard.text_summary,
            event_files=event_files,
        )

    def _read_latest_summary(
        self,
        run_dir: Path,
        tag: str,
        summary_reader: Callable[[Any, str], SummaryT | None],
        *,
        event_files: EventFileIndex | None = None,
    ) -> SummaryT | None:
        event_files = event_files or tensorboard.event_file_index(run_dir)
        summaries: list[SummaryT] = []
        for event_dir in event_files.dirs:
            accumulator = event_files.load_accumulator(event_dir)
            if accumulator is None:
                continue
            try:
                summary = summary_reader(accumulator, tag)
            except Exception:
                continue
            if summary is not None:
                summaries.append(summary)
        if not summaries:
            return None
        summaries.sort(key=lambda item: (item.step, item.wall_time))
        return summaries[-1]


__all__ = [
    "LOG_EVENT_CACHE_MAX_ENTRIES",
    "LOG_SCALAR_ACCUMULATOR_CACHE_MAX_ENTRIES",
    "LOG_TAG_BATCH_READ_MAX_EVENT_BYTES",
    "LOG_TAG_READ_MAX_EVENT_BYTES",
    "LogRunQueryService",
]
