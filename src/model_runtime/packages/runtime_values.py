from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, ModuleType

from model_runtime.packages.configuration import iter_supported_config_keys


def validate_runtime_default_values(
    values: Mapping[str, object] | None,
    *,
    package: str,
    config_module: ModuleType,
) -> dict[str, object]:
    resolved = dict(values or {})
    if any(not isinstance(key, str) for key in resolved):
        raise TypeError(f"{package}: Runtime Defaults keys must be strings")
    accepted = {key.lower() for key in iter_supported_config_keys(config_module)}
    unknown = sorted(set(resolved) - accepted)
    if unknown:
        fields = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"{package}: unknown Runtime Defaults field(s): {fields}")
    return resolved


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeOptions:
    """Immutable package-local construction values produced from flat defaults."""

    _values: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_values", MappingProxyType(dict(self._values)))

    def _as_construction_kwargs(self) -> dict[str, object]:
        return dict(self._values)


__all__ = ["ResolvedRuntimeOptions", "validate_runtime_default_values"]
