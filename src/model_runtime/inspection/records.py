from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Literal, TypeVar, cast

from model_runtime.inspection.capture_limits import InspectionCaptureLimits
from model_runtime.packages.identity import ModelIdentity

GraphRole = Literal["architecture", "internal", "runtime"]
_SnapshotValue = TypeVar("_SnapshotValue")


def freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, Any], value)
        return MappingProxyType(
            {str(key): freeze_value(item) for key, item in mapping.items()}
        )
    if isinstance(value, (list, tuple)):
        sequence = cast(list[Any] | tuple[Any, ...], value)
        return tuple(freeze_value(item) for item in sequence)
    if isinstance(value, (set, frozenset)):
        values = cast(set[Any] | frozenset[Any], value)
        return frozenset(freeze_value(item) for item in values)
    return value


def _frozen_tuple(
    values: tuple[_SnapshotValue, ...],
) -> tuple[_SnapshotValue, ...]:
    return tuple(cast(_SnapshotValue, freeze_value(item)) for item in values)


def _validated_memory_limit(value: object | None) -> int | None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 1
    ):
        raise ValueError("Inspection memory limit must be a positive integer.")
    return value


def _validate_capture_limits(value: object) -> None:
    if not isinstance(value, InspectionCaptureLimits):
        raise TypeError("Inspection capture limits must be InspectionCaptureLimits.")


@dataclass(frozen=True)
class InspectionRequest:
    preset: str
    overrides: Mapping[str, Any] | ParsedOverrides = field(
        default_factory=dict[str, Any]
    )
    dataset: str | None = None
    experiment_task: str | None = None
    memory_limit_bytes: int | None = None
    capture_limits: InspectionCaptureLimits = field(
        default_factory=InspectionCaptureLimits
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "memory_limit_bytes",
            _validated_memory_limit(self.memory_limit_bytes),
        )
        _validate_capture_limits(self.capture_limits)
        if not isinstance(self.overrides, ParsedOverrides):
            object.__setattr__(self, "overrides", freeze_value(self.overrides))


@dataclass(frozen=True)
class ParsedOverrides:
    """Typed Runtime Defaults overrides with optional transient provenance.

    Package parsing may attach a process-local diagnostic identifying the Model
    Package that performed validation. That diagnostic is excluded from this
    public value's fields, equality, repr, pattern matching, and wire shape; it
    may be lost through reconstruction or ``dataclasses.replace``. It is never
    authorization and never skips validation against the selected Model Package.
    """

    _validated_identity: ClassVar[ModelIdentity | None] = None
    values: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_value(self.values))


def parsed_overrides_for_identity(
    values: Mapping[str, Any],
    identity: ModelIdentity,
) -> ParsedOverrides:
    record = ParsedOverrides(values)
    object.__setattr__(record, "_validated_identity", identity)
    return record


def parsed_overrides_identity(record: ParsedOverrides) -> ModelIdentity | None:
    identity = record.__dict__.get("_validated_identity")
    return identity if isinstance(identity, ModelIdentity) else None


SerializedConfigValue = bool | int | float | str | None


@dataclass(frozen=True)
class ConfigurationFieldCondition:
    key: str
    values: tuple[SerializedConfigValue, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _frozen_tuple(self.values))


@dataclass(frozen=True)
class ConfigurationField:
    key: str
    flag: str
    section_path: tuple[str, ...]
    description: str
    value_type: str
    default: SerializedConfigValue
    nullable: bool
    choices: tuple[SerializedConfigValue, ...]
    applicable_when: tuple[ConfigurationFieldCondition, ...]
    maximum: int | float | None = None
    locked: bool = False
    locked_value: SerializedConfigValue = None
    locked_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "section_path",
            _frozen_tuple(self.section_path),
        )
        object.__setattr__(self, "choices", _frozen_tuple(self.choices))
        object.__setattr__(
            self,
            "applicable_when",
            _frozen_tuple(self.applicable_when),
        )


@dataclass(frozen=True)
class ConfigurationSchema:
    identity: ModelIdentity
    fields: tuple[ConfigurationField, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _frozen_tuple(self.fields))


@dataclass(frozen=True)
class SearchAxis:
    key: str
    search_key: str
    section: str
    value_type: str
    values: tuple[SerializedConfigValue, ...]
    locked: bool = False
    locked_value: SerializedConfigValue = None
    locked_reason: str = ""
    locked_by_presets: tuple[str, ...] = ()
    lock_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _frozen_tuple(self.values))
        object.__setattr__(
            self,
            "locked_by_presets",
            _frozen_tuple(self.locked_by_presets),
        )
        object.__setattr__(
            self,
            "lock_reasons",
            _frozen_tuple(self.lock_reasons),
        )


@dataclass(frozen=True)
class SearchSpace:
    identity: ModelIdentity
    preset: str | None
    axes: tuple[SearchAxis, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", _frozen_tuple(self.axes))


@dataclass(frozen=True)
class GraphConfigurationField:
    key: str
    value: Any
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", freeze_value(self.value))


@dataclass(frozen=True)
class GraphConfiguration:
    type_name: str
    fields: tuple[GraphConfigurationField, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _frozen_tuple(self.fields))


@dataclass(frozen=True)
class GraphNode:
    id: str
    type_name: str
    description: str | None
    path: str
    graph_role: GraphRole
    parameter_count: int
    parameter_size_bytes: int
    details: Mapping[str, Any]
    configuration: GraphConfiguration | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", freeze_value(self.details))


@dataclass(frozen=True)
class GraphEdge:
    id: str
    source: str
    target: str


@dataclass(frozen=True)
class ModelGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", _frozen_tuple(self.nodes))
        object.__setattr__(self, "edges", _frozen_tuple(self.edges))


@dataclass(frozen=True)
class InspectionResult:
    identity: ModelIdentity
    preset: str
    parameter_count: int
    parameter_size_bytes: int
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", _frozen_tuple(self.nodes))
        object.__setattr__(self, "edges", _frozen_tuple(self.edges))


__all__ = [
    "ConfigurationField",
    "ConfigurationFieldCondition",
    "ConfigurationSchema",
    "GraphConfiguration",
    "GraphConfigurationField",
    "GraphEdge",
    "GraphNode",
    "GraphRole",
    "InspectionRequest",
    "InspectionResult",
    "ModelGraph",
    "ParsedOverrides",
    "SearchAxis",
    "SearchSpace",
    "SerializedConfigValue",
    "freeze_value",
]
