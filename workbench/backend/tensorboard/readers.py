from __future__ import annotations

import base64
import copy
import re
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing import event_accumulator

from workbench.backend.tensorboard import events as tensorboard_events
from workbench.backend.tensorboard.events import (
    DEFAULT_TENSORBOARD_SIZE_GUIDANCE,
    MAX_TENSORBOARD_IMAGE_SUMMARY_BYTES,
    EventFileIndex,
    TensorBoardEventCache,
    finite_float,
    scalar_points,
)

DEFAULT_SCALAR_POINT_LIMIT = 500
DEFAULT_PARAMETER_STATUS_SCALAR_POINT_LIMIT = 20
DEFAULT_BUCKET_LIMIT = 128
DEFAULT_MONITOR_EVENT_READ_MAX_BYTES = 96 * 1024 * 1024
MONITOR_EVENT_CACHE_MAX_ENTRIES = 256
RELATIVE_DELTA_EPSILON = 1e-12
ABSOLUTE_DELTA_EPSILON = 1e-9
PARAMETER_CHANNELS = ("weights", "bias")
DELTA_METRICS = ("relative_delta_norm", "delta_norm")
VALUE_FALLBACK_METRICS = ("l2_norm", "mean", "var")


def monitor_path_aliases(node_path: str | None) -> list[str]:
    if not node_path:
        return []
    aliases = {node_path}
    aliases.add(re.sub(r"(^|\.)layers\.(\d+)(?=\.|$)", r"\1\2", node_path))
    match = re.match(r"^main_model\.(\d+)(.*)$", node_path)
    if match:
        aliases.add(f"main_model.layers.{match.group(1)}{match.group(2)}")
    return list(aliases)


def parse_parameter_monitor_tag(tag: str) -> tuple[str, str, str] | None:
    parts = tag.rsplit("/", 2)
    if len(parts) != 3:
        return None
    node_path, channel, metric = parts
    if not node_path or channel not in PARAMETER_CHANNELS:
        return None
    if metric not in DELTA_METRICS and metric not in VALUE_FALLBACK_METRICS:
        return None
    return node_path, channel, metric


def is_parameter_monitor_tag(tag: str) -> bool:
    return parse_parameter_monitor_tag(tag) is not None


def empty_monitor_data(
    *,
    job_id: str,
    node_path: str,
    dataset: str | None,
    log_dir: str | None,
) -> dict[str, Any]:
    return {
        "jobId": job_id,
        "nodePath": node_path,
        "dataset": dataset,
        "logDir": log_dir,
        "scalarSeries": [],
        "histograms": [],
        "images": [],
    }


def empty_parameter_status(
    *,
    source_id: str,
    preset: str | None,
    dataset: str | None,
    log_dir: str | None,
) -> dict[str, Any]:
    return {
        "sourceId": source_id,
        "preset": preset,
        "dataset": dataset,
        "logDir": log_dir,
        "nodes": [],
    }


def skipped_event_metadata(
    index: EventFileIndex,
    *,
    limit: int,
) -> dict[str, Any]:
    return {
        "eventBytes": index.total_size,
        "skippedEventFiles": len(index.fingerprint),
        "truncated": True,
        "truncationReason": (
            "event files skipped: "
            f"{index.total_size} bytes exceeds {limit} byte read cap"
        ),
        "sourceItemCount": len(index.fingerprint),
        "returnedItemCount": 0,
    }


class _TensorBoardPayloadCache:
    def __init__(
        self,
        *,
        event_cache: TensorBoardEventCache | None = None,
        cache_name: str = "payload",
    ) -> None:
        self._cache = event_cache or TensorBoardEventCache(
            {cache_name: MONITOR_EVENT_CACHE_MAX_ENTRIES}
        )
        self._cache_name = cache_name

    def token(self) -> int:
        return self._cache.token()

    def get(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        cached = self._cache.get(self._cache_name, key)
        return copy.deepcopy(cached) if cached is not None else None

    def publish(
        self,
        key: tuple[Any, ...],
        value: dict[str, Any],
        *,
        generation: int,
    ) -> None:
        self._cache.publish(
            self._cache_name,
            key,
            copy.deepcopy(value),
            generation=generation,
        )

    def clear_roots(self, roots: set[str]) -> None:
        self._cache.clear_roots(roots)

    def clear(self) -> None:
        self._cache.clear()


class TensorBoardMonitorReader:
    def __init__(
        self,
        *,
        scalar_point_limit: int = DEFAULT_SCALAR_POINT_LIMIT,
        bucket_limit: int = DEFAULT_BUCKET_LIMIT,
        max_event_bytes: int = DEFAULT_MONITOR_EVENT_READ_MAX_BYTES,
        event_cache: TensorBoardEventCache | None = None,
        cache_name: str = "monitor_payload",
    ) -> None:
        self.scalar_point_limit = scalar_point_limit
        self.bucket_limit = bucket_limit
        self.max_event_bytes = max(0, int(max_event_bytes))
        self._payload_cache = _TensorBoardPayloadCache(
            event_cache=event_cache,
            cache_name=cache_name,
        )

    def clear_roots(self, roots: set[str]) -> None:
        self._payload_cache.clear_roots(roots)

    def clear_cache(self) -> None:
        self._payload_cache.clear()

    def read(
        self,
        *,
        job_id: str,
        node_path: str,
        dataset: str | None,
        log_dir: str | None,
        event_files: EventFileIndex | None = None,
    ) -> dict[str, Any]:
        generation = self._payload_cache.token()
        response = empty_monitor_data(
            job_id=job_id,
            node_path=node_path,
            dataset=dataset,
            log_dir=log_dir,
        )
        if not log_dir:
            return response
        root = Path(log_dir)
        if not root.exists():
            return response
        index = event_files or tensorboard_events.event_file_index(root)
        if index.exceeds(self.max_event_bytes):
            response.update(skipped_event_metadata(index, limit=self.max_event_bytes))
            return response

        cache_key = index.cache_key(
            job_id,
            node_path,
            dataset,
        )
        cached = self._payload_cache.get(cache_key)
        if cached is not None:
            return cached

        prefixes = [f"{alias}/" for alias in monitor_path_aliases(node_path)]
        for run_dir in index.dirs:
            accumulator = index.load_accumulator(run_dir)
            if accumulator is None:
                continue
            try:
                tags = accumulator.Tags()
            except Exception:
                continue
            response["scalarSeries"].extend(
                self._read_scalars(accumulator, tags.get("scalars", []), prefixes)
            )
            response["histograms"].extend(
                self._read_histograms(accumulator, tags.get("histograms", []), prefixes)
            )
            response["images"].extend(
                self._read_images(accumulator, tags.get("images", []), prefixes)
            )

        response["scalarSeries"].sort(key=lambda series: series["tag"])
        response["histograms"].sort(key=lambda item: item["tag"])
        response["images"].sort(key=lambda item: item["tag"])
        self._payload_cache.publish(cache_key, response, generation=generation)
        return response

    def _matching_tags(
        self,
        tags: list[str],
        prefixes: list[str],
    ) -> list[tuple[str, str]]:
        matches: dict[str, str] = {}
        for tag in sorted(tags):
            for prefix in prefixes:
                if tag.startswith(prefix):
                    matches.setdefault(tag, prefix)
                    break
        return [(tag, matches[tag]) for tag in sorted(matches)]

    def _label(self, tag: str, prefix: str) -> str:
        return tag[len(prefix) :]

    def _read_scalars(
        self, accumulator, tags: list[str], prefixes: list[str]
    ) -> list[dict[str, Any]]:
        series = []
        for tag, prefix in self._matching_tags(tags, prefixes):
            try:
                points = scalar_points(accumulator, tag, self.scalar_point_limit)
            except Exception:
                continue
            series.append(
                {"tag": tag, "label": self._label(tag, prefix), "points": points}
            )
        return series

    def _read_histograms(
        self,
        accumulator,
        tags: list[str],
        prefixes: list[str],
    ) -> list[dict[str, Any]]:
        histograms = []
        for tag, _prefix in self._matching_tags(tags, prefixes):
            try:
                events = accumulator.Histograms(tag)
            except Exception:
                continue
            if not events:
                continue
            event = events[-1]
            histograms.append(
                {
                    "tag": tag,
                    "step": int(event.step),
                    "wallTime": finite_float(event.wall_time),
                    "buckets": self._histogram_buckets(event.histogram_value),
                }
            )
        return histograms

    def _histogram_buckets(self, histogram_value) -> list[dict[str, float]]:
        buckets = []
        left = finite_float(getattr(histogram_value, "min", 0.0))
        for count, right in zip(
            histogram_value.bucket,
            histogram_value.bucket_limit,
            strict=True,
        ):
            bucket = {
                "left": left,
                "right": finite_float(right),
                "count": finite_float(count),
            }
            if bucket["count"] > 0:
                buckets.append(bucket)
            left = bucket["right"]
            if len(buckets) >= self.bucket_limit:
                break
        return buckets

    def _read_images(
        self, accumulator, tags: list[str], prefixes: list[str]
    ) -> list[dict[str, Any]]:
        images = []
        for tag, _prefix in self._matching_tags(tags, prefixes):
            try:
                events = accumulator.Images(tag)
            except Exception:
                continue
            if not events:
                continue
            event = events[-1]
            encoded = event.encoded_image_string
            if isinstance(encoded, str):
                encoded = encoded.encode("latin1")
            raw_bytes = len(encoded)
            if raw_bytes > MAX_TENSORBOARD_IMAGE_SUMMARY_BYTES:
                images.append(
                    {
                        "tag": tag,
                        "step": int(event.step),
                        "wallTime": finite_float(event.wall_time),
                        "mimeType": "image/png",
                        "dataUrl": "",
                        "eventBytes": raw_bytes,
                        "sourceItemCount": 1,
                        "returnedItemCount": 0,
                        "truncated": True,
                        "truncationReason": (
                            "image payload omitted: "
                            f"{raw_bytes} bytes exceeds "
                            f"{MAX_TENSORBOARD_IMAGE_SUMMARY_BYTES} byte cap"
                        ),
                    }
                )
                continue
            data = base64.b64encode(encoded).decode("ascii")
            images.append(
                {
                    "tag": tag,
                    "step": int(event.step),
                    "wallTime": finite_float(event.wall_time),
                    "mimeType": "image/png",
                    "dataUrl": f"data:image/png;base64,{data}",
                    "eventBytes": raw_bytes,
                    "sourceItemCount": 1,
                    "returnedItemCount": 1,
                    "truncated": False,
                }
            )
        return images


class TensorBoardParameterStatusReader:
    def __init__(
        self,
        *,
        scalar_point_limit: int = DEFAULT_PARAMETER_STATUS_SCALAR_POINT_LIMIT,
        max_event_bytes: int = DEFAULT_MONITOR_EVENT_READ_MAX_BYTES,
        event_cache: TensorBoardEventCache | None = None,
        cache_name: str = "parameter_status_payload",
    ) -> None:
        self.scalar_point_limit = max(1, int(scalar_point_limit))
        self.max_event_bytes = max(0, int(max_event_bytes))
        self._size_guidance = {
            **DEFAULT_TENSORBOARD_SIZE_GUIDANCE,
            event_accumulator.SCALARS: self.scalar_point_limit,
        }
        self._payload_cache = _TensorBoardPayloadCache(
            event_cache=event_cache,
            cache_name=cache_name,
        )

    def clear_roots(self, roots: set[str]) -> None:
        self._payload_cache.clear_roots(roots)

    def clear_cache(self) -> None:
        self._payload_cache.clear()

    def read(
        self,
        *,
        source_id: str,
        preset: str | None,
        dataset: str | None,
        log_dir: str | None,
        event_files: EventFileIndex | None = None,
    ) -> dict[str, Any]:
        generation = self._payload_cache.token()
        response = empty_parameter_status(
            source_id=source_id,
            preset=preset,
            dataset=dataset,
            log_dir=log_dir,
        )
        if not log_dir:
            return response
        root = Path(log_dir)
        if not root.exists():
            return response
        index = event_files or tensorboard_events.event_file_index(root)
        if index.exceeds(self.max_event_bytes):
            response.update(skipped_event_metadata(index, limit=self.max_event_bytes))
            return response

        cache_key = index.cache_key(
            source_id,
            preset,
            dataset,
        )
        cached = self._payload_cache.get(cache_key)
        if cached is not None:
            return cached

        scalars_by_node = self._read_parameter_scalars(index)
        response["nodes"] = [
            {
                "nodePath": node_path,
                "weights": self._channel_status(
                    node_path,
                    "weights",
                    channel_data.get("weights", self._empty_channel_data()),
                ),
                "bias": self._channel_status(
                    node_path,
                    "bias",
                    channel_data.get("bias", self._empty_channel_data()),
                ),
            }
            for node_path, channel_data in sorted(scalars_by_node.items())
        ]
        self._payload_cache.publish(cache_key, response, generation=generation)
        return response

    def _read_parameter_scalars(
        self,
        index: EventFileIndex,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        scalars_by_node: dict[str, dict[str, dict[str, Any]]] = {}
        for run_dir in index.dirs:
            accumulator = index.load_accumulator(
                run_dir,
                size_guidance=self._size_guidance,
            )
            if accumulator is None:
                continue
            try:
                tags = accumulator.Tags().get("scalars", [])
            except Exception:
                continue
            for tag in tags:
                parsed = self._parameter_tag(tag)
                if parsed is None:
                    continue
                node_path, channel, metric = parsed
                channel_data = scalars_by_node.setdefault(node_path, {}).setdefault(
                    channel, self._empty_channel_data()
                )
                channel_data["seen"].add(metric)
                try:
                    points = scalar_points(
                        accumulator,
                        tag,
                        self.scalar_point_limit,
                    )
                except Exception:
                    continue
                if points:
                    channel_data["points"].setdefault(metric, []).extend(points)

        for channel_data_by_node in scalars_by_node.values():
            for channel_data in channel_data_by_node.values():
                for metric_points in channel_data["points"].values():
                    metric_points.sort(
                        key=lambda point: (point["step"], point["wallTime"])
                    )
                    del metric_points[: -self.scalar_point_limit]
        return scalars_by_node

    def _parameter_tag(self, tag: str) -> tuple[str, str, str] | None:
        return parse_parameter_monitor_tag(tag)

    def _empty_channel_data(self) -> dict[str, Any]:
        return {"seen": set(), "points": {}}

    def _empty_channel_status(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "metric": None,
            "lastStep": None,
            "observedPoints": 0,
        }

    def _channel_status(
        self,
        node_path: str,
        channel: str,
        channel_data: dict[str, Any],
    ) -> dict[str, Any]:
        seen_metrics = channel_data["seen"]
        points_by_metric = channel_data["points"]
        if not seen_metrics:
            return self._empty_channel_status("missing")

        delta_status = self._delta_channel_status(
            node_path,
            channel,
            points_by_metric,
        )
        if delta_status is not None:
            return delta_status

        fallback_seen = [
            metric for metric in VALUE_FALLBACK_METRICS if metric in seen_metrics
        ]
        if fallback_seen:
            return self._fallback_channel_status(
                node_path,
                channel,
                fallback_seen,
                points_by_metric,
            )
        return self._empty_channel_status("unknown")

    def _delta_channel_status(
        self,
        node_path: str,
        channel: str,
        points_by_metric: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        delta_metrics = [
            metric for metric in DELTA_METRICS if points_by_metric.get(metric)
        ]
        if not delta_metrics:
            return None

        evidence_metric = next(
            (
                metric
                for metric in delta_metrics
                if self._has_update_evidence(
                    points_by_metric[metric],
                    RELATIVE_DELTA_EPSILON
                    if metric == "relative_delta_norm"
                    else ABSOLUTE_DELTA_EPSILON,
                )
            ),
            None,
        )
        two_sample_metric = next(
            (metric for metric in delta_metrics if len(points_by_metric[metric]) >= 2),
            None,
        )
        metric = evidence_metric or two_sample_metric or delta_metrics[0]
        points = points_by_metric[metric]
        status = (
            "updated"
            if evidence_metric
            else "unchanged"
            if two_sample_metric
            else "unknown"
        )
        return {
            "status": status,
            "metric": f"{node_path}/{channel}/{metric}",
            "lastStep": int(points[-1]["step"]),
            "observedPoints": len(points),
        }

    def _fallback_channel_status(
        self,
        node_path: str,
        channel: str,
        fallback_seen: list[str],
        points_by_metric: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        observed_metrics = [
            metric for metric in fallback_seen if points_by_metric.get(metric)
        ]
        if not observed_metrics:
            return self._empty_channel_status("unknown")

        evidence_metric = next(
            (
                metric
                for metric in observed_metrics
                if self._has_value_change(points_by_metric[metric])
            ),
            None,
        )
        two_sample_metric = next(
            (
                metric
                for metric in observed_metrics
                if len(points_by_metric[metric]) >= 2
            ),
            None,
        )
        metric = evidence_metric or two_sample_metric or observed_metrics[0]
        points = points_by_metric[metric]
        status = (
            "updated"
            if evidence_metric
            else "unchanged"
            if two_sample_metric
            else "unknown"
        )
        return {
            "status": status,
            "metric": f"{node_path}/{channel}/{metric}",
            "lastStep": int(points[-1]["step"]),
            "observedPoints": len(points),
        }

    def _has_update_evidence(
        self,
        points: list[dict[str, Any]],
        epsilon: float,
    ) -> bool:
        return any(abs(point["value"]) > epsilon for point in points)

    def _has_value_change(self, points: list[dict[str, Any]]) -> bool:
        if len(points) < 2:
            return False
        previous = points[0]["value"]
        for point in points[1:]:
            current = point["value"]
            if abs(current - previous) > ABSOLUTE_DELTA_EPSILON:
                return True
            previous = current
        return False
