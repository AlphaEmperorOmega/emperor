"""Log Run TensorBoard query and cache service."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import Any

from workbench.backend.run_history.artifacts import (
    EventFingerprint,
    _event_file_fingerprint,
    _file_id,
    _file_modified_at,
    _parse_checkpoint_epoch,
    _parse_checkpoint_step,
    _read_hparams_flat,
    _read_result_metrics,
    _read_result_params,
    _relative_file_path,
    _run_relative_file_label,
)
from workbench.backend.run_history.records import (
    LOG_RESPONSE_ITEM_LIMIT,
    LogCheckpoint,
    LogRun,
    LogRunArtifact,
    LogRunArtifacts,
)
from workbench.backend.run_history.scanner import LogRunScanner
from workbench.backend.tensorboard.events import (
    TENSORBOARD_TAG_SIZE_GUIDANCE,
    EventFileIndex,
    event_dirs,
    event_file_fingerprint,
    event_file_index,
    event_file_total_size,
    image_summary,
    load_event_accumulator,
    scalar_points,
    text_summary,
)
from workbench.backend.tensorboard.readers import (
    DEFAULT_SCALAR_POINT_LIMIT,
    TensorBoardMonitorReader,
    TensorBoardParameterStatusReader,
)

LOG_EVENT_CACHE_MAX_ENTRIES = 256
LOG_TAG_READ_MAX_EVENT_BYTES = 96 * 1024 * 1024
LOG_TAG_BATCH_READ_MAX_EVENT_BYTES = 64 * 1024 * 1024
LOG_SCALAR_ACCUMULATOR_CACHE_MAX_ENTRIES = 32
LOG_TAG_KEYS = ("scalars", "histograms", "images", "texts")
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


def _tags_have_layer_monitor_data(tags: dict[str, Any]) -> bool:
    return any(
        _is_layer_monitor_tag(tag)
        for key in ("scalars", "histograms", "images")
        for tag in tags.get(key, [])
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
    ) -> None:
        self.scanner = scanner
        self.scalar_point_limit = scalar_point_limit
        self.max_tag_event_bytes = max(0, int(max_tag_event_bytes))
        self.max_tag_batch_event_bytes = max(0, int(max_tag_batch_event_bytes))
        self.monitor_reader = monitor_reader or TensorBoardMonitorReader(
            scalar_point_limit=scalar_point_limit,
        )
        self.parameter_status_reader = (
            parameter_status_reader or TensorBoardParameterStatusReader()
        )
        self._tags_cache: OrderedDict[
            tuple[str, EventFingerprint],
            dict[str, Any],
        ] = OrderedDict()
        self._scalar_cache: OrderedDict[
            tuple[str, EventFingerprint, str, int, str],
            dict[str, Any],
        ] = OrderedDict()
        self._scalar_accumulator_cache: OrderedDict[
            tuple[str, EventFingerprint],
            Any,
        ] = OrderedDict()
        self._cache_generation = 0
        self._cache_lock = RLock()

    def _cache_token(self) -> int:
        with self._cache_lock:
            return self._cache_generation

    def _cache_get(
        self,
        cache: OrderedDict[Any, Any],
        key: Any,
    ) -> Any | None:
        with self._cache_lock:
            if key not in cache:
                return None
            cache.move_to_end(key)
            return cache[key]

    def _cache_set(
        self,
        cache: OrderedDict[Any, Any],
        key: Any,
        value: Any,
        *,
        generation: int,
        max_entries: int = LOG_EVENT_CACHE_MAX_ENTRIES,
    ) -> None:
        with self._cache_lock:
            if generation != self._cache_generation:
                return
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > max_entries:
                cache.popitem(last=False)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache_generation += 1
            self._tags_cache.clear()
            self._scalar_cache.clear()
            self._scalar_accumulator_cache.clear()
        self.monitor_reader.clear_cache()
        self.parameter_status_reader.clear_cache()

    def clear_run_caches(self, run_paths: list[Path]) -> None:
        roots = {path.as_posix() for path in run_paths}
        if not roots:
            return
        with self._cache_lock:
            self._cache_generation += 1
            for cache in (self._tags_cache, self._scalar_cache):
                for key in list(cache):
                    if key and key[0] in roots:
                        cache.pop(key, None)
            for key in list(self._scalar_accumulator_cache):
                cached_root = key[0]
                if any(
                    cached_root == root or cached_root.startswith(f"{root}/")
                    for root in roots
                ):
                    self._scalar_accumulator_cache.pop(key, None)
        self.monitor_reader.clear_roots(roots)
        self.parameter_status_reader.clear_roots(roots)

    def _tags_cache_key(
        self,
        run_dir: Path,
    ) -> tuple[str, EventFingerprint]:
        return (run_dir.as_posix(), _event_file_fingerprint(run_dir))

    def _copy_tags_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        copied = dict(payload)
        for key in LOG_TAG_KEYS:
            value = copied.get(key)
            copied[key] = list(value) if isinstance(value, list) else []
        return copied

    def _cached_tags_if_current(
        self,
        run_dir: Path,
    ) -> dict[str, Any] | None:
        cached = self._cache_get(self._tags_cache, self._tags_cache_key(run_dir))
        if cached is None:
            return None
        return self._copy_tags_payload(cached)

    def _batch_budget_skip_tags(
        self,
        index: EventFileIndex,
    ) -> dict[str, Any]:
        return {
            key: []
            for key in LOG_TAG_KEYS
        } | {
            "eventBytes": index.total_size,
            "skippedEventFiles": len(index.fingerprint),
            "truncated": True,
            "truncationReason": (
                "event files skipped: uncached batch tag-read cap would "
                f"exceed {self.max_tag_batch_event_bytes} bytes"
            ),
            "sourceItemCount": len(index.fingerprint),
            "returnedItemCount": 0,
        }

    def _scalar_cache_key(
        self,
        run_dir: Path,
        *,
        tag: str,
        max_points: int,
        sampling: str,
    ) -> tuple[str, EventFingerprint, str, int, str]:
        return (
            run_dir.as_posix(),
            _event_file_fingerprint(run_dir),
            tag,
            max_points,
            sampling,
        )

    def _copy_scalar_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "points": [dict(point) for point in payload["points"]],
            "sourcePointCount": payload["sourcePointCount"],
            "truncated": payload["truncated"],
        }

    def _scalar_payload_from_points(
        self,
        points: list[dict[str, Any]],
        *,
        point_limit: int,
        sampling: str,
    ) -> dict[str, Any]:
        points.sort(key=lambda point: (point["step"], point["wallTime"]))
        source_point_count = len(points)
        if sampling == "tail":
            sampled_points = points[-point_limit:]
        else:
            sampled_points = points[-point_limit:]
        return {
            "points": [dict(point) for point in sampled_points],
            "sourcePointCount": source_point_count,
            "truncated": source_point_count > len(sampled_points),
        }

    def _load_scalar_accumulator(
        self,
        event_dir: Path,
        *,
        generation: int,
    ) -> Any | None:
        cache_key = (event_dir.as_posix(), event_file_fingerprint(event_dir))
        cached = self._cache_get(self._scalar_accumulator_cache, cache_key)
        if cached is not None:
            return cached
        accumulator = load_event_accumulator(event_dir)
        if accumulator is not None:
            self._cache_set(
                self._scalar_accumulator_cache,
                cache_key,
                accumulator,
                generation=generation,
                max_entries=LOG_SCALAR_ACCUMULATOR_CACHE_MAX_ENTRIES,
            )
        return accumulator

    def tags_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        runs = self.scanner.resolve_runs(run_ids)
        results: list[dict[str, Any]] = []
        uncached_event_bytes = 0
        for run in runs:
            tags = self._cached_tags_if_current(run.path)
            if tags is None:
                index = event_file_index(run.path)
                exceeds_per_run_tag_budget = (
                    self.max_tag_event_bytes > 0
                    and index.total_size > self.max_tag_event_bytes
                )
                would_exceed_batch_budget = (
                    self.max_tag_batch_event_bytes > 0
                    and uncached_event_bytes + index.total_size
                    > self.max_tag_batch_event_bytes
                )
                if exceeds_per_run_tag_budget:
                    tags = self.read_tags(run.path)
                elif would_exceed_batch_budget:
                    tags = self._batch_budget_skip_tags(index)
                else:
                    uncached_event_bytes += index.total_size
                    tags = self.read_tags(run.path)
            results.append(
                {
                    "runId": run.id,
                    "hasLayerMonitorData": _tags_have_layer_monitor_data(tags),
                    "scalarTags": tags["scalars"],
                    "histogramTags": tags["histograms"],
                    "imageTags": tags["images"],
                    "textTags": tags["texts"],
                    "eventBytes": tags.get("eventBytes"),
                    "skippedEventFiles": tags.get("skippedEventFiles"),
                    "truncated": tags.get("truncated"),
                    "truncationReason": tags.get("truncationReason"),
                    "sourceItemCount": tags.get("sourceItemCount"),
                    "returnedItemCount": tags.get("returnedItemCount"),
                }
            )
        return results

    def cached_layer_monitor_data_for_run(self, run: LogRun) -> bool | None:
        tags = self._cached_tags_if_current(run.path)
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
    ) -> list[dict[str, Any]]:
        runs = self.scanner.resolve_runs(run_ids)
        requested_tags = list(dict.fromkeys(tags))
        if not requested_tags:
            return []

        series: list[dict[str, Any]] = []
        for run in runs:
            cached_tags = self._cached_tags_if_current(run.path)
            run_tags = set(
                cached_tags["scalars"]
                if cached_tags is not None
                else self.read_tags(run.path)["scalars"],
            )
            available_tags = [tag for tag in requested_tags if tag in run_tags]
            scalar_series_by_tag = self.read_scalar_series_batch(
                run.path,
                available_tags,
                max_points=max_points,
                sampling=sampling,
            )
            for tag in requested_tags:
                scalar_series = scalar_series_by_tag.get(tag)
                if scalar_series is None:
                    continue
                if scalar_series["points"]:
                    series.append(
                        {
                            "runId": run.id,
                            "tag": tag,
                            **scalar_series,
                        }
                    )
        return series

    def media_for_runs(
        self,
        *,
        run_ids: list[str],
        image_tags: list[str],
        text_tags: list[str],
    ) -> dict[str, Any]:
        runs = self.scanner.resolve_runs(run_ids)
        requested_image_tags = list(dict.fromkeys(image_tags))
        requested_text_tags = list(dict.fromkeys(text_tags))
        images: list[dict[str, Any]] = []
        texts: list[dict[str, Any]] = []
        source_item_count = len(runs) * (
            len(requested_image_tags) + len(requested_text_tags)
        )
        skipped_event_files = 0
        event_bytes = 0
        skipped_reasons: list[str] = []

        for run in runs:
            run_tags = self.read_tags(run.path)
            if run_tags.get("truncated"):
                skipped_event_files += int(run_tags.get("skippedEventFiles") or 0)
                event_bytes += int(run_tags.get("eventBytes") or 0)
                reason = run_tags.get("truncationReason")
                if isinstance(reason, str) and reason:
                    skipped_reasons.append(reason)
            image_tag_set = set(run_tags["images"])
            text_tag_set = set(run_tags["texts"])
            for tag in requested_image_tags:
                if tag not in image_tag_set:
                    continue
                summary = self.read_image_summary(run.path, tag)
                if summary is not None:
                    images.append({"runId": run.id, **summary})
            for tag in requested_text_tags:
                if tag not in text_tag_set:
                    continue
                summary = self.read_text_summary(run.path, tag)
                if summary is not None:
                    texts.append({"runId": run.id, **summary})

        returned_item_count = len(images) + len(texts)
        truncated_items = [
            item
            for item in [*images, *texts]
            if bool(item.get("truncated"))
        ]
        truncated = skipped_event_files > 0 or bool(truncated_items)
        reason = None
        if skipped_reasons:
            reason = skipped_reasons[0]
        elif truncated_items:
            reason = str(
                truncated_items[0].get("truncationReason") or "media truncated"
            )

        return {
            "sourceItemCount": source_item_count,
            "returnedItemCount": returned_item_count,
            "truncated": truncated,
            "truncationReason": reason,
            "eventBytes": event_bytes or None,
            "skippedEventFiles": skipped_event_files or None,
            "images": images,
            "texts": texts,
        }

    def monitor_data_for_run(self, run_id: str, node_path: str) -> dict[str, Any]:
        run = self.scanner.resolve_runs([run_id])[0]
        return self.monitor_reader.read(
            job_id=run.id,
            node_path=node_path,
            dataset=run.dataset,
            log_dir=str(run.path),
        )

    def parameter_status_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        runs = self.scanner.resolve_runs(run_ids)
        return [
            self.parameter_status_reader.read(
                source_id=run.id,
                preset=run.preset,
                dataset=run.dataset,
                log_dir=str(run.path),
            )
            for run in runs
        ]

    def checkpoints_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        runs = self.scanner.resolve_runs(run_ids)
        return [
            checkpoint.to_response()
            for run in runs
            for checkpoint in self.read_checkpoints(run)
        ]

    def checkpoint_paths_for_resolved_run(self, run: LogRun) -> list[Path]:
        root = self.scanner.resolved_root()
        return [
            root / checkpoint.relativePath
            for checkpoint in self.read_checkpoints(run)
        ]

    def saved_params_for_run(self, run: LogRun) -> dict[str, Any]:
        result_path = self.scanner.artifact_path(run, "result.json")
        hparams_path = self.scanner.artifact_path(run, "hparams.yaml")
        result_params = _read_result_params(result_path) if result_path else {}
        hparams = _read_hparams_flat(hparams_path) if hparams_path else {}
        return {**hparams, **result_params}

    def artifacts_for_run(self, run_id: str) -> dict[str, Any]:
        run = self.scanner.resolve_runs([run_id])[0]
        result_path = self.scanner.artifact_path(run, "result.json")
        metrics = _read_result_metrics(result_path) if result_path else {}
        checkpoints = self.read_checkpoints(run)
        artifacts: list[LogRunArtifact] = []

        artifacts.extend(
            self.artifact_metadata(
                run,
                path,
                kind="event_file",
                label=_run_relative_file_label(run.path, path),
            )
            for path in self.scanner.artifact_files(run, "events.out.tfevents.*")
        )
        for filename, kind in (
            ("hparams.yaml", "hparams"),
            ("result.json", "result"),
        ):
            path = self.scanner.artifact_path(run, filename)
            if path is not None:
                artifacts.append(
                    self.artifact_metadata(
                        run,
                        path,
                        kind=kind,
                        label=filename,
                    )
                )
        artifacts.extend(
            self.artifact_metadata(
                run,
                self.scanner.resolved_root() / checkpoint.relativePath,
                kind="checkpoint",
                label=_run_relative_file_label(
                    run.path,
                    self.scanner.resolved_root() / checkpoint.relativePath,
                ),
            )
            for checkpoint in checkpoints
        )

        source_item_count = len(artifacts) + len(checkpoints)
        returned_artifacts = artifacts[:LOG_RESPONSE_ITEM_LIMIT]
        remaining_budget = max(0, LOG_RESPONSE_ITEM_LIMIT - len(returned_artifacts))
        returned_checkpoints = checkpoints[:remaining_budget]
        returned_item_count = len(returned_artifacts) + len(returned_checkpoints)
        truncated = source_item_count > returned_item_count
        response = LogRunArtifacts(
            runId=run.id,
            params=self.saved_params_for_run(run),
            metrics=metrics,
            artifacts=returned_artifacts,
            checkpoints=returned_checkpoints,
        ).to_response()
        response.update(
            {
                "sourceItemCount": source_item_count,
                "returnedItemCount": returned_item_count,
                "truncated": truncated,
                "truncationReason": (
                    f"artifact metadata capped at {LOG_RESPONSE_ITEM_LIMIT} rows"
                    if truncated
                    else None
                ),
            }
        )
        return response

    def read_checkpoints(self, run: LogRun) -> list[LogCheckpoint]:
        checkpoints = [
            self.checkpoint_metadata(run, path)
            for path in self.scanner.artifact_files(run, "*.ckpt")
        ]
        return sorted(
            checkpoints,
            key=lambda checkpoint: (
                checkpoint.step is None,
                checkpoint.step if checkpoint.step is not None else -1,
                checkpoint.epoch is None,
                checkpoint.epoch if checkpoint.epoch is not None else -1,
                checkpoint.filename,
                checkpoint.relativePath,
            ),
        )

    def checkpoint_metadata(self, run: LogRun, path: Path) -> LogCheckpoint:
        root = self.scanner.resolved_root()
        relative_path = _relative_file_path(root, path)
        return LogCheckpoint(
            id=_file_id(run.id, relative_path),
            runId=run.id,
            filename=path.name,
            relativePath=relative_path,
            epoch=_parse_checkpoint_epoch(path.name),
            step=_parse_checkpoint_step(path.name),
            sizeBytes=path.stat().st_size,
            modifiedAt=_file_modified_at(path),
        )

    def artifact_metadata(
        self,
        run: LogRun,
        path: Path,
        *,
        kind: str,
        label: str,
    ) -> LogRunArtifact:
        root = self.scanner.resolved_root()
        relative_path = _relative_file_path(root, path)
        return LogRunArtifact(
            id=_file_id(run.id, relative_path),
            kind=kind,
            label=label,
            relativePath=relative_path,
            sizeBytes=path.stat().st_size,
            modifiedAt=_file_modified_at(path),
        )

    def read_tags(self, run_dir: Path) -> dict[str, Any]:
        generation = self._cache_token()
        cache_key = self._tags_cache_key(run_dir)
        cached = self._cache_get(self._tags_cache, cache_key)
        if cached is not None:
            return self._copy_tags_payload(cached)

        tags = {"scalars": set(), "histograms": set(), "images": set(), "texts": set()}
        if (
            self.max_tag_event_bytes > 0
            and event_file_total_size(run_dir) > self.max_tag_event_bytes
        ):
            index = event_file_index(run_dir)
            result = {
                key: sorted(value)
                for key, value in tags.items()
            } | {
                "eventBytes": index.total_size,
                "skippedEventFiles": len(index.fingerprint),
                "truncated": True,
                "truncationReason": (
                    "event files skipped: "
                    f"{index.total_size} bytes exceeds "
                    f"{self.max_tag_event_bytes} byte tag-read cap"
                ),
                "sourceItemCount": len(index.fingerprint),
                "returnedItemCount": 0,
            }
            self._cache_set(
                self._tags_cache,
                cache_key,
                result,
                generation=generation,
            )
            return self._copy_tags_payload(result)
        for event_dir in event_dirs(run_dir):
            accumulator = load_event_accumulator(
                event_dir,
                size_guidance=TENSORBOARD_TAG_SIZE_GUIDANCE,
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
        result = {
            key: sorted(value)
            for key, value in tags.items()
        } | {
            "truncated": False,
            "sourceItemCount": returned_item_count,
            "returnedItemCount": returned_item_count,
        }
        self._cache_set(
            self._tags_cache,
            cache_key,
            result,
            generation=generation,
        )
        return self._copy_tags_payload(result)

    def read_scalar_series_batch(
        self,
        run_dir: Path,
        tags: list[str],
        *,
        max_points: int | None = None,
        sampling: str = "tail",
    ) -> dict[str, dict[str, Any]]:
        generation = self._cache_token()
        point_limit = max_points if max_points is not None else self.scalar_point_limit
        requested_tags = list(dict.fromkeys(tags))
        results: dict[str, dict[str, Any]] = {}
        uncached_tags: list[str] = []
        for tag in requested_tags:
            cache_key = self._scalar_cache_key(
                run_dir,
                tag=tag,
                max_points=point_limit,
                sampling=sampling,
            )
            cached = self._cache_get(self._scalar_cache, cache_key)
            if cached is None:
                uncached_tags.append(tag)
            else:
                results[tag] = self._copy_scalar_payload(cached)

        if uncached_tags:
            points_by_tag: dict[str, list[dict[str, Any]]] = {
                tag: [] for tag in uncached_tags
            }
            for event_dir in event_dirs(run_dir):
                accumulator = self._load_scalar_accumulator(
                    event_dir,
                    generation=generation,
                )
                if accumulator is None:
                    continue
                for tag in uncached_tags:
                    try:
                        points_by_tag[tag].extend(scalar_points(accumulator, tag, None))
                    except Exception:
                        continue

            for tag in uncached_tags:
                result = self._scalar_payload_from_points(
                    points_by_tag[tag],
                    point_limit=point_limit,
                    sampling=sampling,
                )
                self._cache_set(
                    self._scalar_cache,
                    self._scalar_cache_key(
                        run_dir,
                        tag=tag,
                        max_points=point_limit,
                        sampling=sampling,
                    ),
                    result,
                    generation=generation,
                )
                results[tag] = self._copy_scalar_payload(result)

        return {tag: results[tag] for tag in requested_tags if tag in results}

    def read_scalar_series(
        self,
        run_dir: Path,
        tag: str,
        *,
        max_points: int | None = None,
        sampling: str = "tail",
    ) -> dict[str, Any]:
        return self.read_scalar_series_batch(
            run_dir,
            [tag],
            max_points=max_points,
            sampling=sampling,
        )[tag]

    def read_image_summary(self, run_dir: Path, tag: str) -> dict[str, Any] | None:
        return self._read_latest_summary(run_dir, tag, image_summary)

    def read_text_summary(self, run_dir: Path, tag: str) -> dict[str, Any] | None:
        return self._read_latest_summary(run_dir, tag, text_summary)

    def _read_latest_summary(
        self,
        run_dir: Path,
        tag: str,
        summary_reader: Callable[[Any, str], dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        summaries: list[dict[str, Any]] = []
        for event_dir in event_dirs(run_dir):
            accumulator = load_event_accumulator(event_dir)
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
        summaries.sort(key=lambda item: (item["step"], item["wallTime"]))
        return summaries[-1]


__all__ = [
    "LOG_EVENT_CACHE_MAX_ENTRIES",
    "LOG_SCALAR_ACCUMULATOR_CACHE_MAX_ENTRIES",
    "LOG_TAG_BATCH_READ_MAX_EVENT_BYTES",
    "LOG_TAG_KEYS",
    "LOG_TAG_READ_MAX_EVENT_BYTES",
    "LogRunQueryService",
]
