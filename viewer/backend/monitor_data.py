from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from viewer.backend.tensorboard_reader import (
    event_dirs,
    finite_float,
    load_event_accumulator,
)


DEFAULT_SCALAR_POINT_LIMIT = 500
DEFAULT_BUCKET_LIMIT = 128


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


class TensorBoardMonitorReader:
    def __init__(
        self,
        *,
        scalar_point_limit: int = DEFAULT_SCALAR_POINT_LIMIT,
        bucket_limit: int = DEFAULT_BUCKET_LIMIT,
    ) -> None:
        self.scalar_point_limit = scalar_point_limit
        self.bucket_limit = bucket_limit

    def read(
        self,
        *,
        job_id: str,
        node_path: str,
        dataset: str | None,
        log_dir: str | None,
    ) -> dict[str, Any]:
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

        prefix = f"{node_path}/"
        for run_dir in event_dirs(root):
            accumulator = load_event_accumulator(run_dir)
            if accumulator is None:
                continue
            tags = accumulator.Tags()
            response["scalarSeries"].extend(
                self._read_scalars(accumulator, tags.get("scalars", []), prefix)
            )
            response["histograms"].extend(
                self._read_histograms(accumulator, tags.get("histograms", []), prefix)
            )
            response["images"].extend(
                self._read_images(accumulator, tags.get("images", []), prefix)
            )

        response["scalarSeries"].sort(key=lambda series: series["tag"])
        response["histograms"].sort(key=lambda item: item["tag"])
        response["images"].sort(key=lambda item: item["tag"])
        return response

    def _matching_tags(self, tags: list[str], prefix: str) -> list[str]:
        return sorted(tag for tag in tags if tag.startswith(prefix))

    def _label(self, tag: str, prefix: str) -> str:
        return tag[len(prefix) :]

    def _read_scalars(self, accumulator, tags: list[str], prefix: str) -> list[dict[str, Any]]:
        series = []
        for tag in self._matching_tags(tags, prefix):
            try:
                events = accumulator.Scalars(tag)
            except Exception:
                continue
            points = [
                {
                    "step": int(event.step),
                    "wallTime": finite_float(event.wall_time),
                    "value": finite_float(event.value),
                }
                for event in events[-self.scalar_point_limit :]
            ]
            series.append({"tag": tag, "label": self._label(tag, prefix), "points": points})
        return series

    def _read_histograms(
        self,
        accumulator,
        tags: list[str],
        prefix: str,
    ) -> list[dict[str, Any]]:
        histograms = []
        for tag in self._matching_tags(tags, prefix):
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
        for count, right in zip(histogram_value.bucket, histogram_value.bucket_limit):
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

    def _read_images(self, accumulator, tags: list[str], prefix: str) -> list[dict[str, Any]]:
        images = []
        for tag in self._matching_tags(tags, prefix):
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
            data = base64.b64encode(encoded).decode("ascii")
            images.append(
                {
                    "tag": tag,
                    "step": int(event.step),
                    "wallTime": finite_float(event.wall_time),
                    "mimeType": "image/png",
                    "dataUrl": f"data:image/png;base64,{data}",
                }
            )
        return images
