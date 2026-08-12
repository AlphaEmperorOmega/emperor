from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import cast

from model_runtime.packages.configuration import iter_supported_config_keys


def _runtime_default_mapping(value: object) -> dict[object, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("Runtime Defaults values must be a mapping")
    return dict(cast(Mapping[object, object], value))


def validate_runtime_default_values(
    values: Mapping[str, object] | None,
    *,
    package: str,
    config_module: ModuleType,
) -> dict[str, object]:
    try:
        resolved = _runtime_default_mapping(values)
    except TypeError as exc:
        raise TypeError(f"{package}: {exc}") from exc
    if any(not isinstance(key, str) for key in resolved):
        raise TypeError(f"{package}: Runtime Defaults keys must be strings")
    normalized = {cast(str, key): value for key, value in resolved.items()}
    accepted = {key.lower() for key in iter_supported_config_keys(config_module)}
    unknown = sorted(set(normalized) - accepted)
    if unknown:
        fields = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"{package}: unknown Runtime Defaults field(s): {fields}")
    return normalized


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeOptions:
    """Immutable package-local construction values produced from flat defaults."""

    _values: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_values", MappingProxyType(dict(self._values)))

    def _as_construction_kwargs(self) -> dict[str, object]:
        return dict(self._values)


__all__ = ["ResolvedRuntimeOptions", "validate_runtime_default_values"]
