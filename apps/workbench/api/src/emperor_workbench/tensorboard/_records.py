from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ParameterActivityStatus = Literal["updated", "unchanged", "missing", "unknown"]


@dataclass(frozen=True, slots=True)
class ScalarPoint:
    step: int
    wall_time: float
    value: float


@dataclass(frozen=True, slots=True)
class ScalarTail:
    points: tuple[ScalarPoint, ...]
    source_point_count: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class TagCatalog:
    scalar_tags: tuple[str, ...]
    histogram_tags: tuple[str, ...]
    image_tags: tuple[str, ...]
    text_tags: tuple[str, ...]
    event_bytes: int | None = None
    skipped_event_files: int | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None
    source_item_count: int | None = None
    returned_item_count: int | None = None


@dataclass(frozen=True, slots=True)
class ScalarSeries:
    tag: str
    label: str
    points: tuple[ScalarPoint, ...]
    source_item_count: int | None = None
    returned_item_count: int | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class HistogramBucket:
    left: float
    right: float
    count: float


@dataclass(frozen=True, slots=True)
class Histogram:
    tag: str
    step: int
    wall_time: float
    buckets: tuple[HistogramBucket, ...]
    source_item_count: int | None = None
    returned_item_count: int | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ImageSummary:
    tag: str
    step: int
    wall_time: float
    mime_type: str
    data_url: str
    event_bytes: int | None = None
    source_item_count: int | None = None
    returned_item_count: int | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TextSummary:
    tag: str
    step: int
    wall_time: float
    text: str
    event_bytes: int | None = None
    source_item_count: int | None = None
    returned_item_count: int | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class MonitorData:
    job_id: str
    node_path: str
    preset: str | None
    dataset: str | None
    log_dir: str | None
    scalar_series: tuple[ScalarSeries, ...]
    histograms: tuple[Histogram, ...]
    images: tuple[ImageSummary, ...]
    event_bytes: int | None = None
    skipped_event_files: int | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None
    source_item_count: int | None = None
    returned_item_count: int | None = None


@dataclass(frozen=True, slots=True)
class ParameterChannelStatus:
    status: ParameterActivityStatus
    metric: str | None
    last_step: int | None
    observed_points: int


@dataclass(frozen=True, slots=True)
class ParameterNodeStatus:
    node_path: str
    weights: ParameterChannelStatus
    bias: ParameterChannelStatus


@dataclass(frozen=True, slots=True)
class ParameterStatus:
    source_id: str
    preset: str | None
    dataset: str | None
    log_dir: str | None
    nodes: tuple[ParameterNodeStatus, ...]
    event_bytes: int | None = None
    skipped_event_files: int | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None
    source_item_count: int | None = None
    returned_item_count: int | None = None


__all__ = [
    "Histogram",
    "HistogramBucket",
    "ImageSummary",
    "MonitorData",
    "ParameterActivityStatus",
    "ParameterChannelStatus",
    "ParameterNodeStatus",
    "ParameterStatus",
    "ScalarPoint",
    "ScalarSeries",
    "ScalarTail",
    "TagCatalog",
    "TextSummary",
]
