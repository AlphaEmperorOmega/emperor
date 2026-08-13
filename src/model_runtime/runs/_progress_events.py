from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias

from model_runtime.runs._value_policy import deep_freeze, deep_thaw


class RunProgressEventType(StrEnum):
    DATASET_STARTED = "dataset_started"
    DATASET_COMPLETED = "dataset_completed"
    ERROR = "error"
    EPOCH_STARTED = "epoch_started"
    STEP = "step"
    VALIDATION = "validation"
    FIT_COMPLETED = "fit_completed"
    TEST_COMPLETED = "test_completed"
    CLUSTER_INITIALIZED = "cluster_initialized"
    NEURON_ADDED = "neuron_added"
    NEURONS_ADDED = "neurons_added"


@dataclass(frozen=True, slots=True)
class DatasetStartedEvent:
    params: Mapping[str, Any]
    resumed_from: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", deep_freeze(self.params))
        if self.resumed_from is not None:
            object.__setattr__(
                self,
                "resumed_from",
                deep_freeze(self.resumed_from),
            )


@dataclass(frozen=True, slots=True)
class DatasetCompletedEvent:
    metrics: Mapping[str, Any]
    resumed_from: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", deep_freeze(self.metrics))
        if self.resumed_from is not None:
            object.__setattr__(
                self,
                "resumed_from",
                deep_freeze(self.resumed_from),
            )


@dataclass(frozen=True, slots=True)
class TrainingErrorEvent:
    error: str
    traceback: str


@dataclass(frozen=True, slots=True)
class EpochStartedEvent:
    epoch: int
    step: int


@dataclass(frozen=True, slots=True)
class StepEvent:
    epoch: int
    step: int
    batch: int
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", deep_freeze(self.metrics))


@dataclass(frozen=True, slots=True)
class ValidationEvent:
    epoch: int
    step: int
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", deep_freeze(self.metrics))


@dataclass(frozen=True, slots=True)
class FitCompletedEvent:
    epoch: int
    step: int
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", deep_freeze(self.metrics))


@dataclass(frozen=True, slots=True)
class TestCompletedEvent:
    epoch: int
    step: int
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", deep_freeze(self.metrics))


@dataclass(frozen=True, slots=True)
class ClusterInitializedEvent:
    node: str
    count: int
    capacity: Sequence[int]
    coordinates: Sequence[Sequence[int]]
    coordinate_count: int
    coordinates_truncated: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "capacity", tuple(self.capacity))
        object.__setattr__(
            self,
            "coordinates",
            tuple(tuple(coordinate) for coordinate in self.coordinates),
        )


@dataclass(frozen=True, slots=True)
class NeuronAddedEvent:
    coord: Sequence[int]
    node: str
    count: int
    capacity: Sequence[int]
    epoch: int
    step: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "coord", tuple(self.coord))
        object.__setattr__(self, "capacity", tuple(self.capacity))


@dataclass(frozen=True, slots=True)
class NeuronsAddedEvent:
    coordinates: Sequence[Sequence[int]]
    coordinate_count: int
    coordinates_truncated: bool
    node: str
    count: int
    capacity: Sequence[int]
    epoch: int
    step: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coordinates",
            tuple(tuple(coordinate) for coordinate in self.coordinates),
        )
        object.__setattr__(self, "capacity", tuple(self.capacity))


RunProgressEvent: TypeAlias = (
    DatasetStartedEvent
    | DatasetCompletedEvent
    | TrainingErrorEvent
    | EpochStartedEvent
    | StepEvent
    | ValidationEvent
    | FitCompletedEvent
    | TestCompletedEvent
    | ClusterInitializedEvent
    | NeuronAddedEvent
    | NeuronsAddedEvent
)
RunProgressEventInput: TypeAlias = RunProgressEvent | Mapping[str, Any]


class _RunProgressContext(Protocol):
    @property
    def experiment_task(self) -> str | None: ...

    @property
    def dataset(self) -> str: ...

    @property
    def preset(self) -> str: ...

    @property
    def preset_key(self) -> str: ...

    @property
    def log_dir(self) -> str | None: ...

    @property
    def run_id(self) -> str | None: ...

    @property
    def run_index(self) -> int | None: ...

    @property
    def run_total(self) -> int | None: ...

    @property
    def total_epochs(self) -> int: ...


@dataclass(frozen=True, slots=True)
class _ProjectedField:
    attribute: str
    wire_name: str
    omit_none: bool = False


@dataclass(frozen=True, slots=True)
class _EventProjection:
    event_type: RunProgressEventType
    status: str | None
    fields: tuple[_ProjectedField, ...]


def _field(
    attribute: str,
    wire_name: str | None = None,
    *,
    omit_none: bool = False,
) -> _ProjectedField:
    return _ProjectedField(attribute, wire_name or attribute, omit_none)


_METRIC_FIELDS = (_field("epoch"), _field("step"), _field("metrics"))
_GROWTH_FIELDS = (
    _field("node"),
    _field("count"),
    _field("capacity"),
    _field("epoch"),
    _field("step"),
)
_EVENT_PROJECTIONS: Mapping[type[object], _EventProjection] = MappingProxyType(
    {
        DatasetStartedEvent: _EventProjection(
            RunProgressEventType.DATASET_STARTED,
            "running",
            (
                _field("params"),
                _field("resumed_from", "resumedFrom", omit_none=True),
            ),
        ),
        DatasetCompletedEvent: _EventProjection(
            RunProgressEventType.DATASET_COMPLETED,
            "running",
            (
                _field("metrics"),
                _field("resumed_from", "resumedFrom", omit_none=True),
            ),
        ),
        TrainingErrorEvent: _EventProjection(
            RunProgressEventType.ERROR,
            "failed",
            (_field("error"), _field("traceback")),
        ),
        EpochStartedEvent: _EventProjection(
            RunProgressEventType.EPOCH_STARTED,
            "running",
            (_field("epoch"), _field("step")),
        ),
        StepEvent: _EventProjection(
            RunProgressEventType.STEP,
            "running",
            (
                _field("epoch"),
                _field("step"),
                _field("batch"),
                _field("metrics"),
            ),
        ),
        ValidationEvent: _EventProjection(
            RunProgressEventType.VALIDATION,
            "running",
            _METRIC_FIELDS,
        ),
        FitCompletedEvent: _EventProjection(
            RunProgressEventType.FIT_COMPLETED,
            "running",
            _METRIC_FIELDS,
        ),
        TestCompletedEvent: _EventProjection(
            RunProgressEventType.TEST_COMPLETED,
            "running",
            _METRIC_FIELDS,
        ),
        ClusterInitializedEvent: _EventProjection(
            RunProgressEventType.CLUSTER_INITIALIZED,
            None,
            (
                _field("node"),
                _field("count"),
                _field("capacity"),
                _field("coordinates"),
                _field("coordinate_count", "coordinateCount"),
                _field("coordinates_truncated", "coordinatesTruncated"),
            ),
        ),
        NeuronAddedEvent: _EventProjection(
            RunProgressEventType.NEURON_ADDED,
            None,
            (_field("coord"), *_GROWTH_FIELDS),
        ),
        NeuronsAddedEvent: _EventProjection(
            RunProgressEventType.NEURONS_ADDED,
            None,
            (
                _field("coordinates"),
                _field("coordinate_count", "coordinateCount"),
                _field("coordinates_truncated", "coordinatesTruncated"),
                *_GROWTH_FIELDS,
            ),
        ),
    }
)

MODEL_RUNTIME_PROGRESS_CONTEXT_FIELDS = (
    "experimentTask",
    "dataset",
    "preset",
    "presetKey",
    "logDir",
    "runId",
    "runIndex",
    "runTotal",
    "totalEpochs",
)
MODEL_RUNTIME_PROGRESS_EVENT_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        projection.event_type.value: (
            "type",
            *(("status",) if projection.status is not None else ()),
            *(field.wire_name for field in projection.fields),
        )
        for projection in _EVENT_PROJECTIONS.values()
    }
)
MODEL_RUNTIME_PROGRESS_OPTIONAL_FIELDS: Mapping[str, tuple[str, ...]] = (
    MappingProxyType(
        {
            projection.event_type.value: tuple(
                field.wire_name for field in projection.fields if field.omit_none
            )
            for projection in _EVENT_PROJECTIONS.values()
        }
    )
)
MODEL_RUNTIME_PROGRESS_EVENT_TYPES = tuple(MODEL_RUNTIME_PROGRESS_EVENT_FIELDS)


def _typed_event_payload(event: RunProgressEvent) -> dict[str, Any]:
    projection = _EVENT_PROJECTIONS.get(type(event))
    if projection is None:
        raise TypeError(f"Unsupported Runs progress event: {type(event).__name__}.")
    payload: dict[str, Any] = {"type": projection.event_type.value}
    if projection.status is not None:
        payload["status"] = projection.status
    for field in projection.fields:
        value = getattr(event, field.attribute)
        if value is not None or not field.omit_none:
            payload[field.wire_name] = deep_thaw(value)
    return payload


def project_run_progress_context(context: _RunProgressContext) -> dict[str, Any]:
    return {
        "experimentTask": context.experiment_task,
        "dataset": context.dataset,
        "preset": context.preset,
        "presetKey": context.preset_key,
        "logDir": context.log_dir,
        "runId": context.run_id,
        "runIndex": context.run_index,
        "runTotal": context.run_total,
        "totalEpochs": context.total_epochs,
    }


def project_run_progress_event(
    event: RunProgressEventInput,
    *,
    context: _RunProgressContext | None = None,
) -> dict[str, Any]:
    """Project a typed Run event or explicit legacy mapping to portable wire data."""
    payload = dict(event) if isinstance(event, Mapping) else _typed_event_payload(event)
    if context is not None:
        payload.update(project_run_progress_context(context))
    return payload


__all__ = [
    "ClusterInitializedEvent",
    "DatasetCompletedEvent",
    "DatasetStartedEvent",
    "EpochStartedEvent",
    "FitCompletedEvent",
    "MODEL_RUNTIME_PROGRESS_CONTEXT_FIELDS",
    "MODEL_RUNTIME_PROGRESS_EVENT_FIELDS",
    "MODEL_RUNTIME_PROGRESS_EVENT_TYPES",
    "MODEL_RUNTIME_PROGRESS_OPTIONAL_FIELDS",
    "NeuronAddedEvent",
    "NeuronsAddedEvent",
    "RunProgressEvent",
    "RunProgressEventInput",
    "RunProgressEventType",
    "StepEvent",
    "TestCompletedEvent",
    "TrainingErrorEvent",
    "ValidationEvent",
    "project_run_progress_context",
    "project_run_progress_event",
]
