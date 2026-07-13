from __future__ import annotations

from typing import Annotated, Literal, Self, TypeAlias

from pydantic import ConfigDict, Field, model_validator

from workbench.backend.schemas._base import (
    ApiResponseModel,
    BoundedIdentifier,
    ConfigOverrides,
    ConfigValue,
    JsonObject,
)
from workbench.backend.training_jobs.limits import (
    MAX_TRAINING_DATASETS,
    MAX_TRAINING_MONITORS,
    MAX_TRAINING_PLANNED_RUNS,
    MAX_TRAINING_SEARCH_AXES,
    MAX_TRAINING_SEARCH_AXIS_VALUES,
)

TrainingSearchValues = Annotated[
    list[ConfigValue],
    Field(max_length=MAX_TRAINING_SEARCH_AXIS_VALUES),
]


class ConfigSnapshotRevisionResponse(ApiResponseModel):
    id: str
    semanticRevision: str = Field(min_length=64, max_length=64)


class TrainingJobCreateRequest(ApiResponseModel):
    modelType: BoundedIdentifier
    model: BoundedIdentifier
    preset: BoundedIdentifier
    presets: list[BoundedIdentifier] | None = Field(
        default=None,
        max_length=MAX_TRAINING_PLANNED_RUNS,
    )
    experimentTask: BoundedIdentifier | None = None
    datasets: list[BoundedIdentifier] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_DATASETS,
    )
    overrides: ConfigOverrides = Field(default_factory=dict)
    logFolder: BoundedIdentifier
    monitors: list[BoundedIdentifier] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_MONITORS,
    )
    search: TrainingSearchRequest | None = None
    runPlan: SubmittedTrainingRunPlanRequest | None = None
    snapshotIds: list[BoundedIdentifier] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_PLANNED_RUNS,
    )
    snapshotRevisions: list[ConfigSnapshotRevisionResponse] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_PLANNED_RUNS,
    )

    @model_validator(mode="after")
    def require_revision_only_snapshot_submission(self) -> Self:
        if self.snapshotIds and self.runPlan is not None:
            raise ValueError(
                "Snapshot Training Jobs accept snapshotIds and "
                "snapshotRevisions, not a submitted runPlan."
            )
        revision_ids = [revision.id for revision in self.snapshotRevisions]
        if len(revision_ids) != len(set(revision_ids)):
            raise ValueError("snapshotRevisions must contain unique ids.")
        if self.snapshotIds and set(revision_ids) != set(self.snapshotIds):
            raise ValueError(
                "Snapshot Training Jobs require backend-issued revisions for "
                "every snapshotId."
            )
        if self.snapshotRevisions and not self.snapshotIds:
            raise ValueError("snapshotRevisions require snapshotIds.")
        return self


class TrainingJobReconcileRequest(ApiResponseModel):
    action: Literal["mark-failed"]
    reason: str = Field(min_length=1, max_length=500)


class TrainingRunPlanCreateRequest(ApiResponseModel):
    modelType: BoundedIdentifier
    model: BoundedIdentifier
    preset: BoundedIdentifier
    presets: list[BoundedIdentifier] | None = Field(
        default=None,
        max_length=MAX_TRAINING_PLANNED_RUNS,
    )
    experimentTask: BoundedIdentifier | None = None
    datasets: list[BoundedIdentifier] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_DATASETS,
    )
    overrides: ConfigOverrides = Field(default_factory=dict)
    logFolder: BoundedIdentifier = ""
    monitors: list[BoundedIdentifier] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_MONITORS,
    )
    search: TrainingSearchRequest | None = None
    snapshotIds: list[BoundedIdentifier] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_PLANNED_RUNS,
    )


class TrainingSearchResponse(ApiResponseModel):
    mode: Literal["grid", "random"]
    values: dict[str, list[ConfigValue]]
    randomSamples: int | None = None


class TrainingSearchRequest(ApiResponseModel):
    mode: Literal["grid", "random"]
    values: dict[str, TrainingSearchValues] = Field(max_length=MAX_TRAINING_SEARCH_AXES)
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


class TrainingCommandsResponse(ApiResponseModel):
    posix: str = ""
    powershell: str = ""


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
    experimentTask: str = ""
    changes: list[TrainingRunChangeResponse] = Field(default_factory=list)
    overrides: ConfigOverrides = Field(default_factory=dict)
    command: str
    commandArgv: list[str] = Field(default_factory=list)
    commands: TrainingCommandsResponse = Field(default_factory=TrainingCommandsResponse)
    totalEpochs: int
    currentEpoch: int = 0
    metrics: JsonObject = Field(default_factory=dict)
    logDir: str | None = None
    error: str | None = None
    errorTraceback: str | None = None


class SubmittedTrainingRunRequest(ApiResponseModel):
    id: BoundedIdentifier
    preset: BoundedIdentifier
    snapshotId: BoundedIdentifier | None = None
    snapshotName: BoundedIdentifier | None = None
    dataset: BoundedIdentifier
    overrides: ConfigOverrides = Field(default_factory=dict)


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


class TrainingRunPlanResponse(ApiResponseModel):
    modelType: str
    model: str
    preset: str
    presets: list[str] = Field(default_factory=list)
    experimentTask: str = ""
    datasets: list[str] = Field(default_factory=list)
    overrides: ConfigOverrides = Field(default_factory=dict)
    search: TrainingSearchResponse | None = None
    logFolder: str = ""
    isRandomSearch: bool = False
    runs: list[TrainingRunResponse] = Field(default_factory=list)
    summary: TrainingRunPlanSummaryResponse
    snapshotRevisions: list[ConfigSnapshotRevisionResponse] = Field(
        default_factory=list
    )


class SubmittedTrainingRunPlanRequest(ApiResponseModel):
    runs: list[SubmittedTrainingRunRequest] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_PLANNED_RUNS,
    )
    snapshotRevisions: list[ConfigSnapshotRevisionResponse] = Field(
        default_factory=list,
        max_length=MAX_TRAINING_PLANNED_RUNS,
    )


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
    experimentTask: str | None = None
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
    experimentTask: str = ""
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
    logTailTruncated: bool = False
    resultLinks: list[TrainingResultLinkResponse]


class TrainingProgressEventsResponse(ApiResponseModel):
    jobId: str
    offset: int
    limit: int
    totalCount: int
    nextOffset: int | None = None
    events: list[TrainingProgressEventResponse]
