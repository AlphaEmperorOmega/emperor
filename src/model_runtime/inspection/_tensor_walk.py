from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Generic, TypeVar, cast

from torch import Tensor, nn

from model_runtime.inspection.capture_limits import InspectionCapture
from model_runtime.inspection.errors import InspectionError

ObservationT = TypeVar("ObservationT")
TensorObservationFactory = Callable[[str, Tensor, int], ObservationT]


@dataclass(frozen=True, slots=True)
class TensorWalkStart:
    seen: set[int] | None = None
    depth: int = 0


class _TensorWalker(Generic[ObservationT]):
    def __init__(
        self,
        capture: InspectionCapture | None,
        seen: set[int] | None,
        observation_factory: TensorObservationFactory[ObservationT],
    ) -> None:
        self._capture = capture
        self._seen = set[int]() if seen is None else seen
        self._observation_factory = observation_factory
        self._observations: list[ObservationT] = []

    def collect(
        self,
        value: object,
        name: str,
        depth: int,
    ) -> tuple[ObservationT, ...]:
        self._visit(value, name, depth)
        return tuple(self._observations)

    def _reserve_visit(self, depth: int) -> None:
        if self._capture is not None:
            if depth > self._capture.limits.maximum_tensor_nesting_depth:
                raise InspectionError(
                    "Inspection tensor nesting depth limit of "
                    f"{self._capture.limits.maximum_tensor_nesting_depth} exceeded."
                )
            self._capture.increment(
                "tensor_observations",
                label="tensor observation",
                maximum=self._capture.limits.maximum_tensor_observations,
            )

    def _visit(self, value: object, name: str, depth: int) -> None:
        self._reserve_visit(depth)
        if self._visit_leaf(value, name):
            return

        value_id = id(value)
        if value_id in self._seen:
            return
        if is_dataclass(value) and not isinstance(value, type):
            self._seen.add(value_id)
            self._visit_dataclass(value, name, depth)
            return
        if isinstance(value, Mapping):
            self._seen.add(value_id)
            self._visit_mapping(
                cast(Mapping[object, object], value),
                name,
                depth,
            )
            return
        if isinstance(value, Sequence):
            self._seen.add(value_id)
            self._visit_sequence(cast(Sequence[object], value), name, depth)

    def _visit_leaf(self, value: object, name: str) -> bool:
        if isinstance(value, Tensor):
            self._observations.append(self._observation_factory(name, value, id(value)))
            return True
        return value is None or isinstance(
            value,
            (str, bytes, int, float, bool, type, nn.Module),
        )

    def _visit_dataclass(self, value: object, name: str, depth: int) -> None:
        for data_field in fields(cast(Any, value)):
            self._visit(
                getattr(value, data_field.name),
                f"{name}.{data_field.name}",
                depth + 1,
            )

    def _visit_mapping(
        self,
        value: Mapping[object, object],
        name: str,
        depth: int,
    ) -> None:
        for key, item in value.items():
            self._visit(
                item,
                _mapping_path(name, key),
                depth + 1,
            )

    def _visit_sequence(
        self,
        value: Sequence[object],
        name: str,
        depth: int,
    ) -> None:
        for index, item in enumerate(value):
            self._visit(item, f"{name}[{index}]", depth + 1)


def _mapping_path(name: str, key: object) -> str:
    if isinstance(key, str) and key.isidentifier():
        return f"{name}.{key}"
    return f"{name}[{key!r}]"


def tensor_observations(
    value: object,
    name: str,
    capture: InspectionCapture | None,
    observation_factory: TensorObservationFactory[ObservationT],
    start: TensorWalkStart | None = None,
) -> tuple[ObservationT, ...]:
    selected_start = TensorWalkStart() if start is None else start
    return _TensorWalker(capture, selected_start.seen, observation_factory).collect(
        value,
        name,
        selected_start.depth,
    )


__all__ = ["TensorObservationFactory", "TensorWalkStart", "tensor_observations"]
