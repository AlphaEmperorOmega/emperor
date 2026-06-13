"""Log run and historical metric schemas."""

from __future__ import annotations

from pydantic import Field

from viewer.backend.schemas._base import ApiResponseModel, JsonObject
from viewer.backend.schemas._monitor_data import ScalarPointResponse


class LogRunResponse(ApiResponseModel):
    id: str
    group: str | None = None
    experiment: str
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
    runIds: list[str] = Field(default_factory=list)


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


class LogRunDeleteFiltersRequest(ApiResponseModel):
    experiments: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    presets: list[str] = Field(default_factory=list)
    runIds: list[str] = Field(default_factory=list)


class LogRunDeleteCandidateResponse(ApiResponseModel):
    id: str
    experiment: str
    model: str
    preset: str
    dataset: str
    runName: str
    version: str
    relativePath: str


class LogRunDeleteAffectedValuesResponse(ApiResponseModel):
    experiments: list[str]
    datasets: list[str]
    models: list[str]
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
    runIds: list[str] = Field(default_factory=list)


class LogRunTagsResponse(ApiResponseModel):
    runId: str
    scalarTags: list[str]
    histogramTags: list[str]
    imageTags: list[str]


class LogTagsResponse(ApiResponseModel):
    runs: list[LogRunTagsResponse]


class LogScalarsRequest(ApiResponseModel):
    runIds: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class LogScalarSeriesResponse(ApiResponseModel):
    runId: str
    tag: str
    points: list[ScalarPointResponse]


class LogScalarsResponse(ApiResponseModel):
    series: list[LogScalarSeriesResponse]
