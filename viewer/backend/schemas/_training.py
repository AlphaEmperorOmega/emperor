"""Training job and run-plan schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from viewer.backend.schemas._base import ApiResponseModel, ConfigValue


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
