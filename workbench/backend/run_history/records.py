"""Frozen transport-neutral Run History values."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from workbench.backend.run_history.artifacts import RunArtifactObservation


@dataclass(frozen=True, slots=True)
class LogRun:
    id: str
    group: str | None
    experiment: str
    model: str
    preset: str
    dataset: str
    run_name: str
    timestamp: str | None
    version: str
    relative_path: str
    has_result: bool
    event_file_count: int
    checkpoint_count: int
    has_hparams: bool
    experiment_task: str | None = None
    has_layer_monitor_data: bool | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    path: Path = field(repr=False, compare=False, default=Path())
    artifacts: RunArtifactObservation | None = field(
        repr=False,
        compare=False,
        default=None,
    )


@dataclass(frozen=True, slots=True)
class LogRunFacetValue:
    value: str
    count: int


@dataclass(frozen=True, slots=True)
class LogRunModelFacet:
    model: str
    count: int


@dataclass(frozen=True, slots=True)
class LogRunExperimentFacets:
    experiment: str
    run_count: int
    datasets: tuple[LogRunFacetValue, ...]
    models: tuple[LogRunModelFacet, ...]
    presets: tuple[LogRunFacetValue, ...]


@dataclass(frozen=True, slots=True)
class LogRunFacets:
    experiments: tuple[LogRunExperimentFacets, ...]


@dataclass(frozen=True, slots=True)
class LogRunPage:
    runs: tuple[LogRun, ...]
    total: int
    limit: int
    offset: int
    has_more: bool
    facets: LogRunFacets


@dataclass(frozen=True, slots=True)
class LogCheckpoint:
    id: str
    run_id: str
    filename: str
    relative_path: str
    epoch: int | None
    step: int | None
    size_bytes: int
    modified_at: str


@dataclass(frozen=True, slots=True)
class LogRunArtifact:
    id: str
    kind: str
    label: str
    relative_path: str
    size_bytes: int
    modified_at: str


@dataclass(frozen=True, slots=True)
class LogRunArtifacts:
    run_id: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: tuple[LogRunArtifact, ...]
    checkpoints: tuple[LogCheckpoint, ...]
    truncation_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LogExperimentDeleteResult:
    experiment: str
    deleted_run_ids: tuple[str, ...]
    deleted_run_count: int
    deleted_relative_path: str


@dataclass(frozen=True, slots=True)
class LogRunDeleteFilters:
    experiments: tuple[str, ...]
    datasets: tuple[str, ...]
    models: tuple[str, ...]
    presets: tuple[str, ...]
    run_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LogRunDeleteCandidate:
    id: str
    experiment: str
    model: str
    preset: str
    dataset: str
    run_name: str
    version: str
    relative_path: str
    path: Path = field(repr=False, compare=False, default=Path())

    @classmethod
    def from_run(cls, run: LogRun) -> LogRunDeleteCandidate:
        return cls(
            id=run.id,
            experiment=run.experiment,
            model=run.model,
            preset=run.preset,
            dataset=run.dataset,
            run_name=run.run_name,
            version=run.version,
            relative_path=run.relative_path,
            path=run.path,
        )


@dataclass(frozen=True, slots=True)
class ActiveLogRunDeleteBlocker:
    id: str
    log_folder: str
    status: str


@dataclass(frozen=True, slots=True)
class LogRunDeletePlan:
    candidates: tuple[LogRunDeleteCandidate, ...]
    blocked_by_active_jobs: tuple[ActiveLogRunDeleteBlocker, ...] = ()

    @property
    def can_delete(self) -> bool:
        return bool(self.candidates) and not self.blocked_by_active_jobs


@dataclass(frozen=True, slots=True)
class LogRunDeleteResult:
    candidates: tuple[LogRunDeleteCandidate, ...]
    deleted_run_ids: tuple[str, ...]
    deleted_relative_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LogExperiment:
    experiment: str
    run_count: int
    relative_path: str


@dataclass(frozen=True, slots=True)
class LogExperimentPage:
    experiments: tuple[LogExperiment, ...]
    total: int
    limit: int
    offset: int
    has_more: bool


@dataclass(frozen=True, slots=True)
class LogArchiveImportResult:
    extracted_file_count: int
    skipped_file_count: int
    destination_root: str


@dataclass(frozen=True, slots=True)
class LogRunTags:
    run_id: str
    has_layer_monitor_data: bool | None
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
class LogScalarPoint:
    step: int
    wall_time: float
    value: float


@dataclass(frozen=True, slots=True)
class LogScalarSeries:
    run_id: str
    tag: str
    points: tuple[LogScalarPoint, ...]
    source_point_count: int | None = None
    truncated: bool | None = None


@dataclass(frozen=True, slots=True)
class LogImageSummary:
    run_id: str
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
class LogTextSummary:
    run_id: str
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
class LogMedia:
    images: tuple[LogImageSummary, ...]
    texts: tuple[LogTextSummary, ...]
    event_bytes: int | None = None
    skipped_event_files: int | None = None
    source_item_count: int | None = None
    returned_item_count: int | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class LogMonitorScalarSeries:
    tag: str
    label: str
    points: tuple[LogScalarPoint, ...]
    source_item_count: int | None = None
    returned_item_count: int | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class LogHistogramBucket:
    left: float
    right: float
    count: float


@dataclass(frozen=True, slots=True)
class LogHistogram:
    tag: str
    step: int
    wall_time: float
    buckets: tuple[LogHistogramBucket, ...]
    source_item_count: int | None = None
    returned_item_count: int | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class LogMonitorImage:
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
class LogMonitorData:
    job_id: str
    node_path: str
    preset: str | None
    dataset: str | None
    log_dir: str | None
    scalar_series: tuple[LogMonitorScalarSeries, ...]
    histograms: tuple[LogHistogram, ...]
    images: tuple[LogMonitorImage, ...]
    event_bytes: int | None = None
    skipped_event_files: int | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None
    source_item_count: int | None = None
    returned_item_count: int | None = None


@dataclass(frozen=True, slots=True)
class LogParameterChannelStatus:
    status: str
    metric: str | None
    last_step: int | None
    observed_points: int


@dataclass(frozen=True, slots=True)
class LogParameterNodeStatus:
    node_path: str
    weights: LogParameterChannelStatus
    bias: LogParameterChannelStatus


@dataclass(frozen=True, slots=True)
class LogParameterStatus:
    source_id: str
    preset: str | None
    dataset: str | None
    log_dir: str | None
    nodes: tuple[LogParameterNodeStatus, ...]
    event_bytes: int | None = None
    skipped_event_files: int | None = None
    truncated: bool | None = None
    truncation_reason: str | None = None
    source_item_count: int | None = None
    returned_item_count: int | None = None


__all__ = [
    "ActiveLogRunDeleteBlocker",
    "LogArchiveImportResult",
    "LogCheckpoint",
    "LogExperiment",
    "LogExperimentDeleteResult",
    "LogExperimentPage",
    "LogImageSummary",
    "LogHistogram",
    "LogHistogramBucket",
    "LogMedia",
    "LogMonitorData",
    "LogMonitorImage",
    "LogMonitorScalarSeries",
    "LogParameterChannelStatus",
    "LogParameterNodeStatus",
    "LogParameterStatus",
    "LogRun",
    "LogRunArtifact",
    "LogRunArtifacts",
    "LogRunDeleteCandidate",
    "LogRunDeleteFilters",
    "LogRunDeletePlan",
    "LogRunDeleteResult",
    "LogRunExperimentFacets",
    "LogRunFacetValue",
    "LogRunFacets",
    "LogRunModelFacet",
    "LogRunPage",
    "LogRunTags",
    "LogScalarPoint",
    "LogScalarSeries",
    "LogTextSummary",
]
