from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeVar, cast

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
        mapping = cast(Mapping[object, Any], value)
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in mapping.items()}
        )
    if isinstance(value, (list, tuple)):
        sequence = cast(list[Any] | tuple[Any, ...], value)
        return tuple(_freeze_value(item) for item in sequence)
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
class PresetSearch:
    preset: str
    search: SearchSpec | None


@dataclass(frozen=True, slots=True)
class RunRequest:
    presets: tuple[str, ...]
    datasets: tuple[str, ...]
    experiment_task: str | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict[str, Any])
    search: SearchSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "presets", tuple(self.presets))
        object.__setattr__(self, "datasets", tuple(self.datasets))
        object.__setattr__(self, "overrides", _freeze_value(self.overrides))


@dataclass(frozen=True, slots=True)
class PlanningBudget:
    max_axes: int | None = 16
    max_values_per_axis: int | None = 50
    max_materialized_runs: int | None = 2_000

    def __post_init__(self) -> None:
        limits = {
            "max_axes": self.max_axes,
            "max_values_per_axis": self.max_values_per_axis,
            "max_materialized_runs": self.max_materialized_runs,
        }
        for name, value in limits.items():
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None.")

    @classmethod
    def unlimited(cls) -> PlanningBudget:
        """Create an explicit opt-in budget with no planning ceilings."""
        return cls(
            max_axes=None,
            max_values_per_axis=None,
            max_materialized_runs=None,
        )


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
    preset_searches: tuple[PresetSearch, ...] = ()

    def __post_init__(self) -> None:
        presets = tuple(self.presets)
        preset_searches = tuple(self.preset_searches)
        if not preset_searches:
            preset_searches = tuple(
                PresetSearch(preset=preset, search=self.search) for preset in presets
            )
        if tuple(entry.preset for entry in preset_searches) != presets:
            raise ValueError(
                "Run Plan per-preset Search provenance must match presets in order."
            )
        summary = next(
            (entry.search for entry in preset_searches if entry.search is not None),
            None,
        )
        object.__setattr__(self, "presets", presets)
        object.__setattr__(self, "datasets", tuple(self.datasets))
        object.__setattr__(self, "overrides", _freeze_value(self.overrides))
        object.__setattr__(self, "search", summary)
        object.__setattr__(self, "runs", tuple(self.runs))
        object.__setattr__(self, "preset_searches", preset_searches)

    def search_for_preset(self, preset: str) -> SearchSpec | None:
        for entry in self.preset_searches:
            if entry.preset == preset:
                return entry.search
        raise KeyError(f"Run Plan does not contain preset {preset!r}.")


@dataclass(frozen=True, slots=True)
class SubmittedRun:
    id: str | None
    preset: str
    dataset: str
    overrides: Mapping[str, Any] = field(default_factory=dict[str, Any])

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
    "PresetSearch",
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
