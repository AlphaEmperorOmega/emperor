from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from torch.nn.parameter import is_lazy

from model_runtime.inspection.capture_limits import InspectionCaptureLimits
from model_runtime.inspection.errors import InspectionError


class GraphModule(Protocol):
    def named_children(self) -> Iterator[tuple[object, GraphModule]]: ...

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Any]]: ...


@dataclass(slots=True)
class _SubtreeParameters:
    parameters: dict[int, tuple[int, int]]
    count: int
    size_bytes: int
    consumable: bool = False


@dataclass(frozen=True, slots=True)
class GraphParameterAccounting:
    children_by_module_id: dict[int, tuple[tuple[str, GraphModule], ...]]
    direct_parameters_by_module_id: dict[int, dict[str, Any]]
    statistics_by_module_id: dict[int, tuple[int, int]]


class _ModuleInventory:
    def __init__(self, root: GraphModule, limits: InspectionCaptureLimits) -> None:
        self._limits = limits
        self.children_by_module_id: dict[int, tuple[tuple[str, GraphModule], ...]] = {}
        self.reference_counts: dict[int, int] = {id(root): 1}
        self._relationship_count = 0

    @classmethod
    def collect(
        cls,
        root: GraphModule,
        limits: InspectionCaptureLimits,
    ) -> _ModuleInventory:
        inventory = cls(root, limits)
        inventory._collect_module(root, depth=1)
        return inventory

    def _reserve_relationship(self) -> None:
        self._relationship_count += 1
        if self._relationship_count > self._limits.maximum_graph_edges:
            raise InspectionError(
                "Inspection graph edge limit of "
                f"{self._limits.maximum_graph_edges} exceeded."
            )
        if self._relationship_count > self._limits.maximum_graph_relationships:
            raise InspectionError(
                "Inspection graph relationship limit of "
                f"{self._limits.maximum_graph_relationships} exceeded."
            )

    @staticmethod
    def _registered_child_name(value: object) -> str:
        if not isinstance(value, str) or not value or "." in value:
            raise InspectionError(
                "Inspection graph requires non-empty registered module names "
                "without '.'."
            )
        return value

    def _collect_module(self, module: GraphModule, depth: int) -> None:
        module_id = id(module)
        if module_id in self.children_by_module_id:
            return
        if depth > self._limits.maximum_graph_depth:
            raise InspectionError(
                "Inspection graph depth limit of "
                f"{self._limits.maximum_graph_depth} exceeded."
            )
        if len(self.children_by_module_id) >= self._limits.maximum_graph_modules:
            raise InspectionError(
                "Inspection graph module limit of "
                f"{self._limits.maximum_graph_modules} exceeded."
            )

        self.children_by_module_id[module_id] = ()
        children: list[tuple[str, GraphModule]] = []
        for raw_child_name, child in module.named_children():
            self._reserve_relationship()
            child_name = self._registered_child_name(raw_child_name)
            children.append((child_name, child))
            child_id = id(child)
            self.reference_counts[child_id] = self.reference_counts.get(child_id, 0) + 1
            self._collect_module(child, depth + 1)
        self.children_by_module_id[module_id] = tuple(children)


class _ParameterAccumulator:
    def __init__(
        self,
        inventory: _ModuleInventory,
        limits: InspectionCaptureLimits,
    ) -> None:
        self._inventory = inventory
        self._limits = limits
        self._direct_parameters_by_module_id: dict[int, dict[str, Any]] = {}
        self._statistics_by_module_id: dict[int, tuple[int, int]] = {}
        self._subtrees_by_module_id: dict[int, _SubtreeParameters] = {}
        self._active_module_ids: set[int] = set()
        self._parameter_registration_count = 0
        self._parameter_membership_count = 0

    def build(self, root: GraphModule) -> GraphParameterAccounting:
        self._inspect_subtree(root)
        return GraphParameterAccounting(
            children_by_module_id=self._inventory.children_by_module_id,
            direct_parameters_by_module_id=self._direct_parameters_by_module_id,
            statistics_by_module_id=self._statistics_by_module_id,
        )

    def _reserve_registration(self) -> None:
        self._parameter_registration_count += 1
        if (
            self._parameter_registration_count
            > self._limits.maximum_parameter_registrations
        ):
            raise InspectionError(
                "Inspection parameter registration limit of "
                f"{self._limits.maximum_parameter_registrations} exceeded."
            )

    def _reserve_memberships(self, amount: int = 1) -> None:
        self._parameter_membership_count += amount
        if (
            self._parameter_membership_count
            > self._limits.maximum_parameter_memberships
        ):
            raise InspectionError(
                "Inspection parameter membership limit of "
                f"{self._limits.maximum_parameter_memberships} exceeded."
            )

    def _registered_parameters(self, module: GraphModule) -> dict[str, Any]:
        parameters: dict[str, Any] = {}
        for parameter_name, parameter in module.named_parameters(
            recurse=False,
            remove_duplicate=False,
        ):
            self._reserve_registration()
            parameters[parameter_name] = parameter
        self._direct_parameters_by_module_id[id(module)] = parameters
        return parameters

    def _direct_subtree(self, module: GraphModule) -> _SubtreeParameters:
        aggregate = _SubtreeParameters({}, 0, 0)
        for parameter in self._registered_parameters(module).values():
            parameter_id = id(parameter)
            if parameter_id in aggregate.parameters or is_lazy(parameter):
                continue
            self._reserve_memberships()
            parameter_count = parameter.numel()
            parameter_size = parameter_count * parameter.element_size()
            aggregate.parameters[parameter_id] = (parameter_count, parameter_size)
            aggregate.count += parameter_count
            aggregate.size_bytes += parameter_size
        return aggregate

    def _copied_memberships(
        self,
        parameters: dict[int, tuple[int, int]],
    ) -> dict[int, tuple[int, int]]:
        self._reserve_memberships(len(parameters))
        return dict(parameters)

    def _merge_child(
        self,
        aggregate: _SubtreeParameters,
        child: _SubtreeParameters,
    ) -> _SubtreeParameters:
        child_parameters = child.parameters
        child_count = child.count
        child_size_bytes = child.size_bytes
        if len(aggregate.parameters) < len(child_parameters):
            previous = aggregate
            aggregate = _SubtreeParameters(
                parameters=(
                    child_parameters
                    if child.consumable
                    else self._copied_memberships(child_parameters)
                ),
                count=child_count,
                size_bytes=child_size_bytes,
            )
            child_parameters = previous.parameters
            child_count = previous.count
            child_size_bytes = previous.size_bytes

        duplicate_count = 0
        duplicate_size_bytes = 0
        for parameter_id, statistics in child_parameters.items():
            self._reserve_memberships()
            if parameter_id in aggregate.parameters:
                duplicate_count += statistics[0]
                duplicate_size_bytes += statistics[1]
                continue
            aggregate.parameters[parameter_id] = statistics
        aggregate.count += child_count - duplicate_count
        aggregate.size_bytes += child_size_bytes - duplicate_size_bytes
        return aggregate

    def _inspect_subtree(self, module: GraphModule) -> _SubtreeParameters:
        module_id = id(module)
        cached = self._subtrees_by_module_id.get(module_id)
        if cached is not None:
            return cached
        if module_id in self._active_module_ids:
            return _SubtreeParameters({}, 0, 0)

        self._active_module_ids.add(module_id)
        aggregate = self._direct_subtree(module)
        for _child_name, child in self._inventory.children_by_module_id[module_id]:
            aggregate = self._merge_child(aggregate, self._inspect_subtree(child))

        aggregate.consumable = self._inventory.reference_counts[module_id] == 1
        self._subtrees_by_module_id[module_id] = aggregate
        self._statistics_by_module_id[module_id] = (
            aggregate.count,
            aggregate.size_bytes,
        )
        self._active_module_ids.remove(module_id)
        return aggregate


def _unique_registered_parameters(module: GraphModule) -> Iterator[Any]:
    seen_parameter_ids: set[int] = set()
    for _name, parameter in module.named_parameters(
        recurse=True,
        remove_duplicate=False,
    ):
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            continue
        seen_parameter_ids.add(parameter_id)
        yield parameter


def parameter_count(module: GraphModule) -> int:
    return sum(
        parameter.numel()
        for parameter in _unique_registered_parameters(module)
        if not is_lazy(parameter)
    )


def parameter_size_bytes(module: GraphModule) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in _unique_registered_parameters(module)
        if not is_lazy(parameter)
    )


def parameter_accounting(
    module: GraphModule,
    limits: InspectionCaptureLimits,
) -> GraphParameterAccounting:
    inventory = _ModuleInventory.collect(module, limits)
    return _ParameterAccumulator(inventory, limits).build(module)


__all__ = [
    "GraphModule",
    "GraphParameterAccounting",
    "parameter_accounting",
    "parameter_count",
    "parameter_size_bytes",
]
