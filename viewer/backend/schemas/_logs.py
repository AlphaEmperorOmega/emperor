"""Log run and historical metric schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from viewer.backend.schemas._base import ApiResponseModel, JsonObject
from viewer.backend.schemas._monitor_data import ScalarPointResponse

MAX_LOG_REQUEST_RUN_IDS = 50
MAX_LOG_SCALAR_TAGS = 50
MAX_LOG_MEDIA_TAGS = 20
DEFAULT_LOG_SCALAR_MAX_POINTS = 500
MAX_LOG_SCALAR_MAX_POINTS = 2000
LogScalarSampling = Literal["tail"]


def _is_none(value: object) -> bool:
    return value is None


class LogRunResponse(ApiResponseModel):
    id: str
    group: str | None = None
    experiment: str
    modelType: str
    model: str
    preset: str
    dataset: str
    runName: str
    timestamp: str | None = None
    version: str
    relativePath: str
    hasResult: bool
    eventFileCount: int
    checkpointCount: int
    hasHparams: bool
    metrics: JsonObject


class LogRunsResponse(ApiResponseModel):
    total: int = 0
    limit: int = 0
    offset: int = 0
    hasMore: bool = False
    runs: list[LogRunResponse]


class LogCheckpointsRequest(ApiResponseModel):
    runIds: list[str] = Field(
        default_factory=list,
        max_length=MAX_LOG_REQUEST_RUN_IDS,
    )


class LogCheckpointResponse(ApiResponseModel):
    id: str
    runId: str
    filename: str
    relativePath: str
    epoch: int | None = None
    step: int | None = None
    sizeBytes: int
    modifiedAt: str


class LogCheckpointsResponse(ApiResponseModel):
    sourceItemCount: int | None = Field(default=None, exclude_if=_is_none)
    returnedItemCount: int | None = Field(default=None, exclude_if=_is_none)
    truncated: bool | None = Field(default=None, exclude_if=_is_none)
    truncationReason: str | None = Field(default=None, exclude_if=_is_none)
    checkpoints: list[LogCheckpointResponse]


class LogRunArtifactResponse(ApiResponseModel):
    id: str
    kind: str
    label: str
    relativePath: str
    sizeBytes: int
    modifiedAt: str


class LogRunArtifactsResponse(ApiResponseModel):
    runId: str
    params: JsonObject
    metrics: JsonObject
    sourceItemCount: int | None = Field(default=None, exclude_if=_is_none)
    returnedItemCount: int | None = Field(default=None, exclude_if=_is_none)
    truncated: bool | None = Field(default=None, exclude_if=_is_none)
    truncationReason: str | None = Field(default=None, exclude_if=_is_none)
    artifacts: list[LogRunArtifactResponse]
    checkpoints: list[LogCheckpointResponse]


class LogExperimentResponse(ApiResponseModel):
    experiment: str
    runCount: int
    relativePath: str


class LogExperimentsResponse(ApiResponseModel):
    total: int = 0
    limit: int = 0
    offset: int = 0
    hasMore: bool = False
    experiments: list[LogExperimentResponse]


class LogExperimentDeleteResponse(ApiResponseModel):
    experiment: str
    deletedRunIds: list[str]
    deletedRunCount: int
    deletedRelativePath: str


class LogRunModelFilterRequest(ApiResponseModel):
    modelType: str
    model: str


class LogRunDeleteFiltersRequest(ApiResponseModel):
    experiments: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    models: list[LogRunModelFilterRequest] = Field(default_factory=list)
    presets: list[str] = Field(default_factory=list)
    runIds: list[str] = Field(default_factory=list)


class LogRunDeleteCandidateResponse(ApiResponseModel):
    id: str
    experiment: str
    modelType: str
    model: str
    preset: str
    dataset: str
    runName: str
    version: str
    relativePath: str


class LogRunDeleteAffectedValuesResponse(ApiResponseModel):
    experiments: list[str]
    datasets: list[str]
    models: list[LogRunModelFilterRequest]
    presets: list[str]
    runIds: list[str]


class LogRunDeleteCountsResponse(ApiResponseModel):
    runs: int
    experiments: int
    datasets: int
    models: int
    presets: int


class LogRunDeleteBlockerResponse(ApiResponseModel):
    id: str
    logFolder: str
    status: str


class LogRunDeletePlanResponse(ApiResponseModel):
    candidateCount: int
    sourceItemCount: int | None = Field(default=None, exclude_if=_is_none)
    returnedItemCount: int | None = Field(default=None, exclude_if=_is_none)
    truncated: bool | None = Field(default=None, exclude_if=_is_none)
    truncationReason: str | None = Field(default=None, exclude_if=_is_none)
    counts: LogRunDeleteCountsResponse
    affected: LogRunDeleteAffectedValuesResponse
    candidates: list[LogRunDeleteCandidateResponse]
    blockedByActiveJobs: list[LogRunDeleteBlockerResponse]
    canDelete: bool


class LogRunDeleteResponse(LogRunDeletePlanResponse):
    deletedRunIds: list[str]
    deletedRunCount: int
    deletedRelativePaths: list[str]


class LogTagsRequest(ApiResponseModel):
    runIds: list[str] = Field(
        default_factory=list,
        max_length=MAX_LOG_REQUEST_RUN_IDS,
    )


class LogRunTagsResponse(ApiResponseModel):
    runId: str
    eventBytes: int | None = Field(default=None, exclude_if=_is_none)
    skippedEventFiles: int | None = Field(default=None, exclude_if=_is_none)
    truncated: bool | None = Field(default=None, exclude_if=_is_none)
    truncationReason: str | None = Field(default=None, exclude_if=_is_none)
    sourceItemCount: int | None = Field(default=None, exclude_if=_is_none)
    returnedItemCount: int | None = Field(default=None, exclude_if=_is_none)
    scalarTags: list[str]
    histogramTags: list[str]
    imageTags: list[str]
    textTags: list[str] = Field(default_factory=list)


class LogTagsResponse(ApiResponseModel):
    runs: list[LogRunTagsResponse]


class LogScalarsRequest(ApiResponseModel):
    runIds: list[str] = Field(
        default_factory=list,
        max_length=MAX_LOG_REQUEST_RUN_IDS,
    )
    tags: list[str] = Field(default_factory=list, max_length=MAX_LOG_SCALAR_TAGS)
    maxPoints: int = Field(
        default=DEFAULT_LOG_SCALAR_MAX_POINTS,
        ge=1,
        le=MAX_LOG_SCALAR_MAX_POINTS,
    )
    sampling: LogScalarSampling = "tail"


class LogScalarSeriesResponse(ApiResponseModel):
    runId: str
    tag: str
    points: list[ScalarPointResponse]
    sourcePointCount: int | None = Field(default=None, exclude_if=_is_none)
    truncated: bool | None = Field(default=None, exclude_if=_is_none)


class LogScalarsResponse(ApiResponseModel):
    series: list[LogScalarSeriesResponse]


class LogMediaRequest(ApiResponseModel):
    runIds: list[str] = Field(
        default_factory=list,
        max_length=MAX_LOG_REQUEST_RUN_IDS,
    )
    imageTags: list[str] = Field(default_factory=list, max_length=MAX_LOG_MEDIA_TAGS)
    textTags: list[str] = Field(default_factory=list, max_length=MAX_LOG_MEDIA_TAGS)


class LogImageSummaryResponse(ApiResponseModel):
    runId: str
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


class LogTextSummaryResponse(ApiResponseModel):
    runId: str
    tag: str
    step: int
    wallTime: float
    text: str
    eventBytes: int | None = Field(default=None, exclude_if=_is_none)
    sourceItemCount: int | None = Field(default=None, exclude_if=_is_none)
    returnedItemCount: int | None = Field(default=None, exclude_if=_is_none)
    truncated: bool | None = Field(default=None, exclude_if=_is_none)
    truncationReason: str | None = Field(default=None, exclude_if=_is_none)


class LogMediaResponse(ApiResponseModel):
    eventBytes: int | None = Field(default=None, exclude_if=_is_none)
    skippedEventFiles: int | None = Field(default=None, exclude_if=_is_none)
    sourceItemCount: int | None = Field(default=None, exclude_if=_is_none)
    returnedItemCount: int | None = Field(default=None, exclude_if=_is_none)
    truncated: bool | None = Field(default=None, exclude_if=_is_none)
    truncationReason: str | None = Field(default=None, exclude_if=_is_none)
    images: list[LogImageSummaryResponse]
    texts: list[LogTextSummaryResponse]
