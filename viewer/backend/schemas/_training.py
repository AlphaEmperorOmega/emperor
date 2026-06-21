"""Training job and run-plan schemas."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import ConfigDict, Field

from viewer.backend.schemas._base import (
    ApiResponseModel,
    ConfigOverrides,
    ConfigValue,
    JsonObject,
)
from viewer.backend.training_limits import (
    MAX_TRAINING_DATASETS,
    MAX_TRAINING_MONITORS,
    MAX_TRAINING_PLANNED_RUNS,
    MAX_TRAINING_PRESETS,
    MAX_TRAINING_SEARCH_AXES,
)


class TrainingJobCreateRequest(ApiResponseModel):
    modelType: str
    model: str
    preset: str
    presets: list[str] | None = Field(default=None, max_length=MAX_TRAINING_PRESETS)
    datasets: list[str] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_DATASETS,
    )
    overrides: ConfigOverrides = Field(default_factory=dict)
    logFolder: str
    monitors: list[str] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_MONITORS,
    )
    search: TrainingSearchRequest | None = None
    runPlan: SubmittedTrainingRunPlanRequest | None = None


class TrainingRunPlanCreateRequest(ApiResponseModel):
    modelType: str
    model: str
    preset: str
    presets: list[str] | None = Field(default=None, max_length=MAX_TRAINING_PRESETS)
    datasets: list[str] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_DATASETS,
    )
    overrides: ConfigOverrides = Field(default_factory=dict)
    logFolder: str = ""
    search: TrainingSearchRequest | None = None


class TrainingSearchResponse(ApiResponseModel):
    mode: Literal["grid", "random"]
    values: dict[str, list[ConfigValue]]
    randomSamples: int | None = None


class TrainingSearchRequest(ApiResponseModel):
    mode: Literal["grid", "random"]
    values: dict[str, list[ConfigValue]] = Field(max_length=MAX_TRAINING_SEARCH_AXES)
    randomSamples: int | None = Field(
        default=None,
        ge=1,
        le=MAX_TRAINING_PLANNED_RUNS,
    )


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
    overrides: ConfigOverrides = Field(default_factory=dict)
    command: str
    totalEpochs: int
    currentEpoch: int = 0
    metrics: JsonObject = Field(default_factory=dict)
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
    overrides: ConfigOverrides = Field(default_factory=dict)
    command: str
    totalEpochs: int
    currentEpoch: int = 0
    metrics: JsonObject = Field(default_factory=dict)
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
    modelType: str
    model: str
    preset: str
    presets: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    overrides: ConfigOverrides = Field(default_factory=dict)
    search: TrainingSearchResponse | None = None
    logFolder: str = ""
    isRandomSearch: bool = False
    runs: list[TrainingRunResponse] = Field(default_factory=list)
    summary: TrainingRunPlanSummaryResponse


class SubmittedTrainingRunPlanRequest(ApiResponseModel):
    modelType: str
    model: str
    preset: str
    presets: list[str] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_PRESETS,
    )
    datasets: list[str] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_DATASETS,
    )
    overrides: ConfigOverrides = Field(default_factory=dict)
    search: TrainingSearchRequest | None = None
    logFolder: str = ""
    isRandomSearch: bool = False
    runs: list[SubmittedTrainingRunRequest] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_PLANNED_RUNS,
    )
    summary: SubmittedTrainingRunPlanSummaryRequest


class TrainingResultLinkResponse(ApiResponseModel):
    preset: str | None = None
    dataset: str | None = None
    logDir: str | None = None


class TrainingClusterGrowthAdditionResponse(ApiResponseModel):
    coord: list[int]
    step: int | None = None
    epoch: int | None = None


class TrainingClusterGrowthResponse(ApiResponseModel):
    node: str
    count: int = 0
    capacityTotal: int = 0
    additionCount: int = 0
    additions: list[TrainingClusterGrowthAdditionResponse] = Field(default_factory=list)


TrainingProgressStatus = Literal["running", "completed", "failed", "cancelled"]
TrainingProgressMetricMap = JsonObject
TrainingProgressParams = JsonObject


class TrainingProgressEventBaseResponse(ApiResponseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    timestamp: str | None = None
    status: TrainingProgressStatus | None = None
    jobId: str | None = None
    dataset: str | None = None
    preset: str | None = None
    presetKey: str | None = None
    logDir: str | None = None
    runId: str | None = None
    runIndex: int | None = None
    runTotal: int | None = None
    totalEpochs: int | None = None


class TrainingJobStartedProgressEventResponse(TrainingProgressEventBaseResponse):
    type: Literal["job_started"]
    status: Literal["running"] | None = None


class TrainingWorkerStartedProgressEventResponse(TrainingProgressEventBaseResponse):
    type: Literal["started"]
    status: Literal["running"] | None = None
    modelType: str | None = None
    model: str | None = None
    presets: list[str] | None = None
    datasets: list[str] | None = None
    monitors: list[str] | None = None


class TrainingWorkerCompletedProgressEventResponse(TrainingProgressEventBaseResponse):
    type: Literal["completed"]
    status: Literal["completed"] | None = None
    presets: list[str] | None = None


class TrainingCancelledProgressEventResponse(TrainingProgressEventBaseResponse):
    type: Literal["cancelled"]
    status: Literal["cancelled"] | None = None


class TrainingErrorProgressEventResponse(TrainingProgressEventBaseResponse):
    type: Literal["error"]
    status: Literal["failed"] | None = None
    error: str | None = None
    traceback: str | None = None


class TrainingDatasetStartedProgressEventResponse(TrainingProgressEventBaseResponse):
    type: Literal["dataset_started"]
    status: Literal["running"] | None = None
    params: TrainingProgressParams | None = None


class TrainingDatasetCompletedProgressEventResponse(TrainingProgressEventBaseResponse):
    type: Literal["dataset_completed"]
    status: Literal["running"] | None = None
    metrics: TrainingProgressMetricMap | None = None


class TrainingRunProgressEventResponse(TrainingProgressEventBaseResponse):
    type: Literal[
        "epoch_started",
        "step",
        "validation",
        "fit_completed",
        "test_completed",
    ]
    status: Literal["running"] | None = None
    epoch: int | None = None
    step: int | None = None
    batch: int | None = None
    metrics: TrainingProgressMetricMap | None = None


class TrainingClusterInitializedProgressEventResponse(
    TrainingProgressEventBaseResponse
):
    type: Literal["cluster_initialized"]
    node: str
    count: int
    capacity: list[int]
    coordinates: list[list[int]]
    coordinateCount: int | None = None
    coordinatesTruncated: bool | None = None


class TrainingNeuronAddedProgressEventResponse(TrainingProgressEventBaseResponse):
    type: Literal["neuron_added"]
    node: str
    coord: list[int]
    count: int
    capacity: list[int]
    epoch: int | None = None
    step: int | None = None


class TrainingNeuronsAddedProgressEventResponse(TrainingProgressEventBaseResponse):
    type: Literal["neurons_added"]
    node: str
    coordinates: list[list[int]]
    coordinateCount: int
    coordinatesTruncated: bool | None = None
    count: int
    capacity: list[int]
    epoch: int | None = None
    step: int | None = None


class UnknownTrainingProgressEventResponse(TrainingProgressEventBaseResponse):
    pass


TrainingProgressEventResponse: TypeAlias = (
    TrainingJobStartedProgressEventResponse
    | TrainingWorkerStartedProgressEventResponse
    | TrainingWorkerCompletedProgressEventResponse
    | TrainingCancelledProgressEventResponse
    | TrainingErrorProgressEventResponse
    | TrainingDatasetStartedProgressEventResponse
    | TrainingDatasetCompletedProgressEventResponse
    | TrainingRunProgressEventResponse
    | TrainingClusterInitializedProgressEventResponse
    | TrainingNeuronAddedProgressEventResponse
    | TrainingNeuronsAddedProgressEventResponse
    | UnknownTrainingProgressEventResponse
)


class TrainingJobResponse(ApiResponseModel):
    id: str
    status: str
    modelType: str
    model: str
    preset: str
    presets: list[str] = Field(default_factory=list)
    datasets: list[str]
    overrides: ConfigOverrides
    search: TrainingSearchResponse | None = None
    plannedRunCount: int = 0
    runPlan: TrainingRunPlanResponse | None = None
    monitors: list[str]
    logFolder: str
    createdAt: str
    updatedAt: str
    exitCode: int | None = None
    pid: int
    cancellationMode: Literal["strict-cgroup", "process-group", "unsupported"] = (
        "unsupported"
    )
    currentPreset: str | None = None
    currentDataset: str | None = None
    epoch: int | None = None
    step: int | None = None
    metrics: JsonObject
    logDir: str | None = None
    events: list[TrainingProgressEventResponse]
    eventCount: int = 0
    eventCounts: dict[str, int] = Field(default_factory=dict)
    eventsTruncated: bool = False
    clusterGrowth: list[TrainingClusterGrowthResponse] = Field(default_factory=list)
    logTail: list[str]
    resultLinks: list[TrainingResultLinkResponse]


class TrainingProgressEventsResponse(ApiResponseModel):
    jobId: str
    offset: int
    limit: int
    totalCount: int
    nextOffset: int | None = None
    events: list[TrainingProgressEventResponse]
