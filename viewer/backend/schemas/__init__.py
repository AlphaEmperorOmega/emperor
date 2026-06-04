"""Pydantic request/response schemas that define the viewer HTTP contract.

These shapes are validated on the frontend by the Zod schemas in
``viewer/frontend/src/lib/api.ts``; keep the two in sync.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ConfigValue = bool | int | float | str | None


class ApiResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiResponseModel):
    status: str


class ModelsResponse(ApiResponseModel):
    models: list[str]


class PresetResponse(ApiResponseModel):
    name: str
    label: str
    description: str


class PresetsResponse(ApiResponseModel):
    model: str
    presets: list[PresetResponse]


class DatasetResponse(ApiResponseModel):
    name: str
    label: str
    inputDim: int
    outputDim: int


class DatasetsResponse(ApiResponseModel):
    model: str
    datasets: list[DatasetResponse]


class MonitorOptionResponse(ApiResponseModel):
    name: str
    label: str
    description: str
    kinds: list[str]
    defaultEnabled: bool = False


class MonitorsResponse(ApiResponseModel):
    model: str
    monitors: list[MonitorOptionResponse]


class ConfigFieldResponse(ApiResponseModel):
    key: str
    configKey: str
    flag: str
    label: str
    section: str
    type: str
    default: ConfigValue
    nullable: bool
    choices: list[ConfigValue]
    locked: bool = False
    lockedValue: ConfigValue = None
    lockedReason: str = ""


class ConfigSchemaResponse(ApiResponseModel):
    model: str
    fields: list[ConfigFieldResponse]


class SearchAxisResponse(ApiResponseModel):
    key: str
    configKey: str
    searchKey: str
    label: str
    section: str
    type: str
    values: list[ConfigValue]
    locked: bool = False
    lockedValue: ConfigValue = None
    lockedReason: str = ""


class SearchSpaceResponse(ApiResponseModel):
    model: str
    preset: str | None = None
    axes: list[SearchAxisResponse]


class GraphConfigFieldResponse(ApiResponseModel):
    key: str
    value: Any


class GraphConfigResponse(ApiResponseModel):
    typeName: str
    fields: list[GraphConfigFieldResponse]


class GraphNodeResponse(ApiResponseModel):
    id: str
    label: str
    typeName: str
    path: str
    graphRole: Literal["architecture", "internal", "runtime"]
    parameterCount: int
    details: dict[str, Any]
    config: GraphConfigResponse | None


class GraphEdgeResponse(ApiResponseModel):
    id: str
    source: str
    target: str


class InspectRequest(ApiResponseModel):
    model: str
    preset: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    dataset: str | None = None


class InspectResponse(ApiResponseModel):
    model: str
    preset: str
    parameterCount: int
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]


class TrainingJobCreateRequest(ApiResponseModel):
    model: str
    preset: str
    presets: list[str] | None = None
    datasets: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    logFolder: str
    monitors: list[str] = Field(default_factory=list)
    search: dict[str, Any] | None = None
    runPlan: "SubmittedTrainingRunPlanRequest | None" = None


class TrainingRunPlanCreateRequest(ApiResponseModel):
    model: str
    preset: str
    presets: list[str] | None = None
    datasets: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    logFolder: str = ""
    search: dict[str, Any] | None = None


class TrainingSearchResponse(ApiResponseModel):
    mode: Literal["grid", "random"]
    values: dict[str, list[ConfigValue]]
    randomSamples: int | None = None


class TrainingSearchRequest(ApiResponseModel):
    mode: Literal["grid", "random"]
    values: dict[str, list[ConfigValue]]
    randomSamples: int | None = None


class TrainingRunChangeResponse(ApiResponseModel):
    key: str
    label: str
    value: ConfigValue
    source: Literal["override", "search"]


class SubmittedTrainingRunChangeRequest(ApiResponseModel):
    key: str
    label: str
    value: ConfigValue
    source: Literal["override", "search"]


class TrainingRunResponse(ApiResponseModel):
    id: str
    index: int
    status: Literal[
        "Pending",
        "Running",
        "Completed",
        "Failed",
        "Cancelled",
        "Skipped",
    ] = "Pending"
    preset: str
    snapshotId: str | None = None
    snapshotName: str | None = None
    dataset: str
    changes: list[TrainingRunChangeResponse] = Field(default_factory=list)
    overrides: dict[str, ConfigValue] = Field(default_factory=dict)
    command: str
    totalEpochs: int
    currentEpoch: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    logDir: str | None = None
    error: str | None = None
    errorTraceback: str | None = None


class SubmittedTrainingRunRequest(ApiResponseModel):
    id: str
    index: int
    status: Literal[
        "Pending",
        "Running",
        "Completed",
        "Failed",
        "Cancelled",
        "Skipped",
    ] = "Pending"
    preset: str
    snapshotId: str | None = None
    snapshotName: str | None = None
    dataset: str
    changes: list[SubmittedTrainingRunChangeRequest] = Field(default_factory=list)
    overrides: dict[str, ConfigValue] = Field(default_factory=dict)
    command: str
    totalEpochs: int
    currentEpoch: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    logDir: str | None = None
    error: str | None = None
    errorTraceback: str | None = None


class TrainingRunPlanSummaryResponse(ApiResponseModel):
    totalRuns: int = 0
    completedRuns: int = 0
    runningRuns: int = 0
    pendingRuns: int = 0
    failedRuns: int = 0
    cancelledRuns: int = 0
    skippedRuns: int = 0
    totalEpochs: int = 0
    completedEpochs: int = 0
    remainingEpochs: int = 0


class SubmittedTrainingRunPlanSummaryRequest(ApiResponseModel):
    totalRuns: int = 0
    completedRuns: int = 0
    runningRuns: int = 0
    pendingRuns: int = 0
    failedRuns: int = 0
    cancelledRuns: int = 0
    skippedRuns: int = 0
    totalEpochs: int = 0
    completedEpochs: int = 0
    remainingEpochs: int = 0


class TrainingRunPlanResponse(ApiResponseModel):
    model: str
    preset: str
    presets: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    search: TrainingSearchResponse | None = None
    logFolder: str = ""
    isRandomSearch: bool = False
    runs: list[TrainingRunResponse] = Field(default_factory=list)
    summary: TrainingRunPlanSummaryResponse


class SubmittedTrainingRunPlanRequest(ApiResponseModel):
    model: str
    preset: str
    presets: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    search: TrainingSearchRequest | None = None
    logFolder: str = ""
    isRandomSearch: bool = False
    runs: list[SubmittedTrainingRunRequest] = Field(default_factory=list)
    summary: SubmittedTrainingRunPlanSummaryRequest


class TrainingResultLinkResponse(ApiResponseModel):
    preset: str | None = None
    dataset: str | None = None
    logDir: str | None = None


class TrainingJobResponse(ApiResponseModel):
    id: str
    status: str
    model: str
    preset: str
    presets: list[str] = Field(default_factory=list)
    datasets: list[str]
    overrides: dict[str, Any]
    search: TrainingSearchResponse | None = None
    plannedRunCount: int = 0
    runPlan: TrainingRunPlanResponse | None = None
    monitors: list[str]
    logFolder: str
    createdAt: str
    updatedAt: str
    exitCode: int | None = None
    pid: int
    currentPreset: str | None = None
    currentDataset: str | None = None
    epoch: int | None = None
    step: int | None = None
    metrics: dict[str, Any]
    logDir: str | None = None
    events: list[dict[str, Any]]
    logTail: list[str]
    resultLinks: list[TrainingResultLinkResponse]


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
    metrics: dict[str, Any]


class LogRunsResponse(ApiResponseModel):
    total: int = 0
    limit: int = 0
    offset: int = 0
    hasMore: bool = False
    runs: list[LogRunResponse]


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
