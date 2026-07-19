from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeVar

from model_runtime.packages.identity import ModelIdentity

SearchMode = Literal["grid", "random"]
RunParameterSource = Literal["override", "search"]
_SampleValue = TypeVar("_SampleValue")


class RandomSource(Protocol):
    def sample(
        self,
        population: Sequence[_SampleValue],
        k: int,
    ) -> list[_SampleValue]: ...

    def randrange(self, stop: int) -> int: ...


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class SearchAxisSelection:
    key: str
    values: tuple[Any, ...] | None = None
    allow_custom_values: bool = False

    def __post_init__(self) -> None:
        if self.values is not None:
            object.__setattr__(
                self,
                "values",
                tuple(_freeze_value(value) for value in self.values),
            )


@dataclass(frozen=True, slots=True)
class SearchSpec:
    mode: SearchMode
    axes: tuple[SearchAxisSelection, ...] | None = None
    random_samples: int | None = None

    def __post_init__(self) -> None:
        if self.axes is not None:
            object.__setattr__(self, "axes", tuple(self.axes))


@dataclass(frozen=True, slots=True)
class RunRequest:
    presets: tuple[str, ...]
    datasets: tuple[str, ...]
    experiment_task: str | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    search: SearchSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "presets", tuple(self.presets))
        object.__setattr__(self, "datasets", tuple(self.datasets))
        object.__setattr__(self, "overrides", _freeze_value(self.overrides))


@dataclass(frozen=True, slots=True)
class PlanningBudget:
    max_axes: int | None = None
    max_values_per_axis: int | None = None
    max_materialized_runs: int | None = None


@dataclass(frozen=True, slots=True)
class RunParameter:
    key: str
    value: Any
    source: RunParameterSource

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze_value(self.value))


@dataclass(frozen=True, slots=True)
class RunSpec:
    id: str
    experiment_task: str
    preset: str
    dataset: str
    parameters: tuple[RunParameter, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))

    @property
    def overrides(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {parameter.key: parameter.value for parameter in self.parameters}
        )


@dataclass(frozen=True, slots=True)
class RunPlan:
    identity: ModelIdentity
    presets: tuple[str, ...]
    experiment_task: str
    datasets: tuple[str, ...]
    overrides: Mapping[str, Any]
    search: SearchSpec | None
    runs: tuple[RunSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "presets", tuple(self.presets))
        object.__setattr__(self, "datasets", tuple(self.datasets))
        object.__setattr__(self, "overrides", _freeze_value(self.overrides))
        object.__setattr__(self, "runs", tuple(self.runs))


@dataclass(frozen=True, slots=True)
class SubmittedRun:
    id: str | None
    preset: str
    dataset: str
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", _freeze_value(self.overrides))


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    experiment_task: str
    preset: str
    dataset: str
    log_dir: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_value(self.payload))


__all__ = [
    "PlanningBudget",
    "RandomSource",
    "RunParameter",
    "RunParameterSource",
    "RunPlan",
    "RunRequest",
    "RunResult",
    "RunSpec",
    "SearchAxisSelection",
    "SearchMode",
    "SearchSpec",
    "SubmittedRun",
]
