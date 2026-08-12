from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast

from model_runtime.inspection.capture_limits import InspectionCaptureLimits
from model_runtime.packages.identity import ModelIdentity

GraphRole = Literal["architecture", "internal", "runtime"]


def freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, Any], value)
        return MappingProxyType(
            {str(key): freeze_value(item) for key, item in mapping.items()}
        )
    if isinstance(value, (list, tuple)):
        sequence = cast(list[Any] | tuple[Any, ...], value)
        return tuple(freeze_value(item) for item in sequence)
    return value


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
    values: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_value(self.values))


SerializedConfigValue = bool | int | float | str | None


@dataclass(frozen=True)
class ConfigurationFieldCondition:
    key: str
    values: tuple[SerializedConfigValue, ...]


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


@dataclass(frozen=True)
class ConfigurationSchema:
    identity: ModelIdentity
    fields: tuple[ConfigurationField, ...]


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


@dataclass(frozen=True)
class SearchSpace:
    identity: ModelIdentity
    preset: str | None
    axes: tuple[SearchAxis, ...]


@dataclass(frozen=True)
class GraphConfigurationField:
    key: str
    value: Any
    description: str | None = None


@dataclass(frozen=True)
class GraphConfiguration:
    type_name: str
    fields: tuple[GraphConfigurationField, ...]


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


@dataclass(frozen=True)
class InspectionResult:
    identity: ModelIdentity
    preset: str
    parameter_count: int
    parameter_size_bytes: int
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


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
