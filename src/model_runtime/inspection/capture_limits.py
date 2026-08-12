from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, cast

from model_runtime.inspection.errors import InspectionError


@dataclass(frozen=True, slots=True)
class InspectionCaptureLimits:
    """Finite limits for graph and executed shape-trace capture."""

    maximum_graph_modules: int = 8_192
    maximum_graph_relationships: int = 8_192
    maximum_graph_depth: int = 512
    maximum_graph_nodes: int = 8_192
    maximum_graph_edges: int = 8_192
    maximum_configuration_fields: int = 512
    maximum_neuron_coordinates: int = 512
    maximum_terminal_connections: int = 1_024
    maximum_parameter_registrations: int = 1_000_000
    maximum_parameter_memberships: int = 4_000_000
    maximum_module_calls: int = 4_096
    maximum_methods: int = 4_096
    maximum_trace_events: int = 65_536
    maximum_variable_events: int = 16_384
    maximum_tensor_observations: int = 65_536
    maximum_tensor_nesting_depth: int = 64
    maximum_output_bytes: int = 16 * 1024**2

    def __post_init__(self) -> None:
        for limit_field in fields(self):
            value = getattr(self, limit_field.name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{limit_field.name} must be a positive integer.")


def _scalar_output_bytes(value: object) -> int | None:
    size: int | None = None
    if value is None:
        size = 4
    elif isinstance(value, bool):
        size = 5
    elif isinstance(value, int):
        size = len(str(value))
    elif isinstance(value, float):
        size = max(32, len(repr(value)))
    elif isinstance(value, str):
        size = len(json.dumps(value, ensure_ascii=True).encode("utf-8"))
    elif isinstance(value, bytes):
        size = 2 + len(value) * 12
    return size


def _output_entries(
    value: object,
) -> tuple[Iterator[tuple[str | None, object]], bool] | None:
    if is_dataclass(value) and not isinstance(value, type):
        entries = (
            (item.name, cast(object, getattr(value, item.name)))
            for item in fields(value)
        )
        return entries, True
    if isinstance(value, Mapping):
        mapping_value = cast(Mapping[object, object], value)
        entries = ((str(key), item) for key, item in mapping_value.items())
        return entries, True
    if isinstance(value, Sequence):
        sequence_value = cast(Sequence[object], value)
        entries = ((None, item) for item in sequence_value)
        return entries, False
    return None


class _OutputSizeEstimator:
    def __init__(self, active: set[int]) -> None:
        self._active = active

    def estimate(self, value: object, cutoff: int) -> int:
        if cutoff < 0:
            return 1
        scalar_size = _scalar_output_bytes(value)
        if scalar_size is not None:
            return scalar_size
        return self._composite_size(value, cutoff)

    def _composite_size(self, value: object, cutoff: int) -> int:
        value_id = id(value)
        if value_id in self._active:
            return cutoff + 1
        self._active.add(value_id)
        try:
            output_entries = _output_entries(value)
            if output_entries is None:
                return len(json.dumps(str(value), ensure_ascii=True).encode("utf-8"))
            entries, mapping = output_entries
            return self._container_size(entries, mapping=mapping, cutoff=cutoff)
        finally:
            self._active.remove(value_id)

    def _container_size(
        self,
        entries: Iterator[tuple[str | None, object]],
        *,
        mapping: bool,
        cutoff: int,
    ) -> int:
        total = 2
        for index, (key, item) in enumerate(entries):
            if index:
                total += 1
            if mapping:
                assert key is not None
                total += self.estimate(key, cutoff - total) + 1
            total += self.estimate(item, cutoff - total)
            if total > cutoff:
                return cutoff + 1
        return total


def _estimated_output_bytes(
    value: object,
    cutoff: int,
    active: set[int] | None = None,
) -> int:
    visited = set[int]() if active is None else active
    return _OutputSizeEstimator(visited).estimate(value, cutoff)


def estimated_output_bytes(value: object, cutoff: int) -> int:
    """Conservatively estimate encoded output size up to a caller-owned cutoff."""

    return _estimated_output_bytes(value, cutoff)


@dataclass(slots=True)
class InspectionCapture:
    limits: InspectionCaptureLimits
    output_bytes: int = 0
    _counts: dict[str, int] = field(default_factory=dict[str, int])

    def increment(
        self,
        counter: str,
        *,
        label: str,
        maximum: int,
        amount: int = 1,
    ) -> int:
        value = self._counts.get(counter, 0) + amount
        if value > maximum:
            raise InspectionError(f"Inspection {label} limit of {maximum} exceeded.")
        self._counts[counter] = value
        return value

    def reserve_output(self, value: Any) -> None:
        remaining = self.limits.maximum_output_bytes - self.output_bytes
        estimated = estimated_output_bytes(value, remaining)
        if estimated > remaining:
            raise InspectionError(
                "Inspection output byte limit of "
                f"{self.limits.maximum_output_bytes} exceeded."
            )
        self.output_bytes += estimated

    def ensure_total_output(self, value: object) -> None:
        """Validate a complete record independently of fragment reservations."""

        maximum = self.limits.maximum_output_bytes
        if estimated_output_bytes(value, maximum) > maximum:
            raise InspectionError(
                f"Inspection output byte limit of {maximum} exceeded."
            )


__all__ = ["InspectionCaptureLimits", "estimated_output_bytes"]
