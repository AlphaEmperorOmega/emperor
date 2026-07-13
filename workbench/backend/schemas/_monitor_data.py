from __future__ import annotations

from typing import Literal

from pydantic import Field

from workbench.backend.schemas._base import ApiResponseModel, BoundedIdentifier
from workbench.backend.schemas._limits import MAX_LOG_REQUEST_RUN_IDS


def _is_none(value: object) -> bool:
    return value is None


class ScalarPointResponse(ApiResponseModel):
    step: int
    wallTime: float
    value: float


class ScalarSeriesResponse(ApiResponseModel):
    tag: str
    label: str
    points: list[ScalarPointResponse]
    sourceItemCount: int | None = Field(default=None, exclude_if=_is_none)
    returnedItemCount: int | None = Field(default=None, exclude_if=_is_none)
    truncated: bool | None = Field(default=None, exclude_if=_is_none)
    truncationReason: str | None = Field(default=None, exclude_if=_is_none)


class HistogramBucketResponse(ApiResponseModel):
    left: float
    right: float
    count: float


class HistogramResponse(ApiResponseModel):
    tag: str
    step: int
    wallTime: float
    buckets: list[HistogramBucketResponse]
    sourceItemCount: int | None = Field(default=None, exclude_if=_is_none)
    returnedItemCount: int | None = Field(default=None, exclude_if=_is_none)
    truncated: bool | None = Field(default=None, exclude_if=_is_none)
    truncationReason: str | None = Field(default=None, exclude_if=_is_none)


class ImageResponse(ApiResponseModel):
    tag: str
    step: int
    wallTime: float
    mimeType: str
    dataUrl: str
    eventBytes: int | None = Field(default=None, exclude_if=_is_none)
    sourceItemCount: int | None = Field(default=None, exclude_if=_is_none)
    returnedItemCount: int | None = Field(default=None, exclude_if=_is_none)
    truncated: bool | None = Field(default=None, exclude_if=_is_none)
    truncationReason: str | None = Field(default=None, exclude_if=_is_none)


class MonitorDataResponse(ApiResponseModel):
    jobId: str
    nodePath: str
    preset: str | None = None
    dataset: str | None = None
    logDir: str | None = None
    eventBytes: int | None = Field(default=None, exclude_if=_is_none)
    skippedEventFiles: int | None = Field(default=None, exclude_if=_is_none)
    truncated: bool | None = Field(default=None, exclude_if=_is_none)
    truncationReason: str | None = Field(default=None, exclude_if=_is_none)
    sourceItemCount: int | None = Field(default=None, exclude_if=_is_none)
    returnedItemCount: int | None = Field(default=None, exclude_if=_is_none)
    scalarSeries: list[ScalarSeriesResponse]
    histograms: list[HistogramResponse]
    images: list[ImageResponse]


ParameterActivityStatus = Literal["updated", "unchanged", "missing", "unknown"]


class ParameterChannelStatusResponse(ApiResponseModel):
    status: ParameterActivityStatus
    metric: str | None = None
    lastStep: int | None = None
    observedPoints: int


class ParameterNodeStatusResponse(ApiResponseModel):
    nodePath: str
    weights: ParameterChannelStatusResponse
    bias: ParameterChannelStatusResponse


class ParameterStatusResponse(ApiResponseModel):
    sourceId: str
    preset: str | None = None
    dataset: str | None = None
    logDir: str | None = None
    eventBytes: int | None = Field(default=None, exclude_if=_is_none)
    skippedEventFiles: int | None = Field(default=None, exclude_if=_is_none)
    truncated: bool | None = Field(default=None, exclude_if=_is_none)
    truncationReason: str | None = Field(default=None, exclude_if=_is_none)
    sourceItemCount: int | None = Field(default=None, exclude_if=_is_none)
    returnedItemCount: int | None = Field(default=None, exclude_if=_is_none)
    nodes: list[ParameterNodeStatusResponse]


class LogParameterStatusRequest(ApiResponseModel):
    runIds: list[BoundedIdentifier] = Field(
        default_factory=list,
        max_length=MAX_LOG_REQUEST_RUN_IDS,
    )


class LogParameterStatusResponse(ApiResponseModel):
    runs: list[ParameterStatusResponse]
