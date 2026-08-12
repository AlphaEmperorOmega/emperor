from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model_runtime.inspection._graph_accounting import (
    GraphModule as _GraphModule,
)
from model_runtime.inspection._graph_accounting import (
    GraphParameterAccounting as _ParameterAccounting,
)
from model_runtime.inspection._graph_accounting import (
    parameter_accounting as _parameter_accounting,
)
from model_runtime.inspection._graph_accounting import (
    parameter_count,
    parameter_size_bytes,
)
from model_runtime.inspection._graph_component_semantics import (
    ARCHITECTURE_ROLE,
    INTERNAL_ROLE,
    RUNTIME_ROLE,
)
from model_runtime.inspection._graph_construction import (
    ROOT_NODE_ID,
    ROOT_NODE_PATH,
    GraphConstruction,
    construct_model_graph,
)
from model_runtime.inspection._graph_semantics import ModuleSemanticAdapter
from model_runtime.inspection.capture_limits import (
    InspectionCapture,
    InspectionCaptureLimits,
)
from model_runtime.inspection.records import GraphNode, GraphRole, ModelGraph


def _selected_capture_limits(value: object) -> InspectionCaptureLimits:
    if value is None:
        return InspectionCaptureLimits()
    if not isinstance(value, InspectionCaptureLimits):
        raise TypeError("Inspection graph limits must be InspectionCaptureLimits.")
    return value


def module_details(
    module: _GraphModule,
    direct_parameters: dict[str, Any] | None = None,
    *,
    limits: InspectionCaptureLimits | None = None,
) -> dict[str, Any]:
    selected_limits = _selected_capture_limits(limits)
    return ModuleSemanticAdapter(
        module,
        direct_parameters,
        selected_limits,
    ).details()


def graph_role(module: _GraphModule) -> GraphRole:
    return ModuleSemanticAdapter(module).graph_role()


@dataclass(frozen=True, slots=True)
class _GraphNodeFactory:
    accounting: _ParameterAccounting
    limits: InspectionCaptureLimits
    capture: InspectionCapture

    def __call__(
        self,
        node_id: str,
        path: str,
        module: _GraphModule,
    ) -> GraphNode:
        parameter_count_value, parameter_size = self.accounting.statistics_by_module_id[
            id(module)
        ]
        semantic_facts = ModuleSemanticAdapter(
            module,
            self.accounting.direct_parameters_by_module_id[id(module)],
            self.limits,
        ).facts()
        node = GraphNode(
            id=node_id,
            type_name=semantic_facts.type_name,
            description=semantic_facts.description,
            path=path,
            graph_role=semantic_facts.graph_role,
            parameter_count=parameter_count_value,
            parameter_size_bytes=parameter_size,
            details=semantic_facts.details,
            configuration=semantic_facts.configuration,
        )
        self.capture.reserve_output(node)
        return node


def inspect_model_graph(
    module: _GraphModule,
    *,
    limits: InspectionCaptureLimits | None = None,
    _capture: InspectionCapture | None = None,
) -> ModelGraph:
    selected_limits = _selected_capture_limits(
        _capture.limits if limits is None and _capture is not None else limits
    )
    if _capture is not None and _capture.limits != selected_limits:
        raise ValueError("Inspection graph limits must match the shared capture.")
    capture = _capture or InspectionCapture(selected_limits)
    accounting = _parameter_accounting(module, selected_limits)
    return construct_model_graph(
        GraphConstruction(
            root=module,
            children_by_module_id=accounting.children_by_module_id,
            limits=selected_limits,
            capture=capture,
            node_factory=_GraphNodeFactory(
                accounting,
                selected_limits,
                capture,
            ),
        )
    )


__all__ = [
    "ARCHITECTURE_ROLE",
    "INTERNAL_ROLE",
    "ROOT_NODE_ID",
    "ROOT_NODE_PATH",
    "RUNTIME_ROLE",
    "graph_role",
    "inspect_model_graph",
    "module_details",
    "parameter_count",
    "parameter_size_bytes",
]
