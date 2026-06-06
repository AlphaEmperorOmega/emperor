"""TensorBoard monitor-data schemas."""

from __future__ import annotations

from viewer.backend.schemas._base import ApiResponseModel


class ScalarPointResponse(ApiResponseModel):
    step: int
    wallTime: float
    value: float


class ScalarSeriesResponse(ApiResponseModel):
    tag: str
    label: str
    points: list[ScalarPointResponse]


class HistogramBucketResponse(ApiResponseModel):
    left: float
    right: float
    count: float


class HistogramResponse(ApiResponseModel):
    tag: str
    step: int
    wallTime: float
    buckets: list[HistogramBucketResponse]


class ImageResponse(ApiResponseModel):
    tag: str
    step: int
    wallTime: float
    mimeType: str
    dataUrl: str


class MonitorDataResponse(ApiResponseModel):
    jobId: str
    nodePath: str
    preset: str | None = None
    dataset: str | None = None
    logDir: str | None = None
    scalarSeries: list[ScalarSeriesResponse]
    histograms: list[HistogramResponse]
    images: list[ImageResponse]
