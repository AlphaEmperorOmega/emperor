from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from emperor_workbench.api.v1._base_contracts import (
    ApiResponseModel,
    BoundedIdentifier,
    ConfigOverrides,
    ConfigValue,
    JsonObject,
)
from emperor_workbench.run_plans import (
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


__all__ = [
    "ConfigSnapshotRevisionResponse",
    "SubmittedTrainingRunPlanRequest",
    "SubmittedTrainingRunRequest",
    "TrainingCommandsResponse",
    "TrainingRunChangeResponse",
    "TrainingRunPlanCreateRequest",
    "TrainingRunPlanResponse",
    "TrainingRunPlanSummaryResponse",
    "TrainingRunResponse",
    "TrainingSearchRequest",
    "TrainingSearchResponse",
]
