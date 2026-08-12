from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from model_runtime.inspection._graph_accounting import GraphModule
from model_runtime.inspection.capture_limits import (
    InspectionCapture,
    InspectionCaptureLimits,
)
from model_runtime.inspection.errors import InspectionError
from model_runtime.inspection.records import GraphEdge, GraphNode, ModelGraph

ROOT_NODE_ID = "__root__"
ROOT_NODE_PATH = "model"

GraphNodeFactory = Callable[[str, str, GraphModule], GraphNode]


@dataclass(frozen=True, slots=True)
class GraphConstruction:
    root: GraphModule
    children_by_module_id: Mapping[
        int,
        tuple[tuple[str, GraphModule], ...],
    ]
    limits: InspectionCaptureLimits
    capture: InspectionCapture
    node_factory: GraphNodeFactory


class _UniqueIdAllocator:
    def __init__(self) -> None:
        self._allocated_ids: set[str] = set()
        self._next_suffix_by_base: dict[str, int] = {}

    def allocate(self, base_id: str) -> str:
        if base_id not in self._allocated_ids:
            self._allocated_ids.add(base_id)
            self._next_suffix_by_base.setdefault(base_id, 2)
            return base_id

        suffix = self._next_suffix_by_base.get(base_id, 2)
        candidate_id = f"{base_id}#{suffix}"
        while candidate_id in self._allocated_ids:
            suffix += 1
            candidate_id = f"{base_id}#{suffix}"
        self._next_suffix_by_base[base_id] = suffix + 1
        self._allocated_ids.add(candidate_id)
        return candidate_id


class _GraphBuilder:
    def __init__(self, construction: GraphConstruction) -> None:
        self._construction = construction
        self._nodes: list[GraphNode] = []
        self._edges: list[GraphEdge] = []
        self._node_ids = _UniqueIdAllocator()
        self._edge_ids = _UniqueIdAllocator()
        self._canonical_node_by_module_id: dict[int, str] = {}
        self._active_module_ids: set[int] = set()

    def build(self) -> ModelGraph:
        root_id = self._append_node(
            self._construction.root,
            ROOT_NODE_ID,
            ROOT_NODE_PATH,
        )
        self._visit(self._construction.root, root_id, "", depth=1)
        graph = ModelGraph(nodes=tuple(self._nodes), edges=tuple(self._edges))
        self._construction.capture.ensure_total_output(graph)
        return graph

    def _append_node(
        self,
        module: GraphModule,
        node_id: str,
        path: str,
    ) -> str:
        allocated_node_id = self._node_ids.allocate(node_id)
        self._construction.capture.increment(
            "graph_nodes",
            label="graph node",
            maximum=self._construction.limits.maximum_graph_nodes,
        )
        node = self._construction.node_factory(allocated_node_id, path, module)
        self._nodes.append(node)
        self._canonical_node_by_module_id[id(module)] = allocated_node_id
        return allocated_node_id

    def _append_edge(self, parent_id: str, child_id: str) -> None:
        self._construction.capture.increment(
            "graph_edges",
            label="graph edge",
            maximum=self._construction.limits.maximum_graph_edges,
        )
        edge = GraphEdge(
            id=self._edge_ids.allocate(f"{parent_id}-{child_id}"),
            source=parent_id,
            target=child_id,
        )
        self._construction.capture.reserve_output(edge)
        self._edges.append(edge)

    def _append_child(
        self,
        child: GraphModule,
        child_id: str,
        child_path: str,
        parent_id: str,
    ) -> str | None:
        canonical_node_id = self._canonical_node_by_module_id.get(id(child))
        if canonical_node_id is not None:
            self._append_edge(parent_id, canonical_node_id)
            return None
        allocated_child_id = self._append_node(child, child_id, child_path)
        self._append_edge(parent_id, allocated_child_id)
        return allocated_child_id

    def _reject_active_module(self, module: GraphModule) -> None:
        if id(module) in self._active_module_ids:
            raise InspectionError(
                "Inspection graph contains a registered module cycle."
            )

    @contextmanager
    def _activated(self, module: GraphModule) -> Generator[None]:
        self._reject_active_module(module)
        module_id = id(module)
        self._active_module_ids.add(module_id)
        try:
            yield
        finally:
            self._active_module_ids.remove(module_id)

    def _visit(
        self,
        parent: GraphModule,
        parent_id: str,
        parent_path: str,
        depth: int,
    ) -> None:
        with self._activated(parent):
            self._reject_excessive_depth(depth)
            for child_name, child in self._children(parent):
                self._reject_active_module(child)
                child_path = _child_path(parent_path, child_name)
                if _is_transparent_graph_container(parent, child_name, child):
                    self._visit_transparent(child, parent_id, child_path, depth + 1)
                    continue
                allocated_child_id = self._append_child(
                    child,
                    child_path,
                    child_path,
                    parent_id,
                )
                if allocated_child_id is not None:
                    self._visit(child, allocated_child_id, child_path, depth + 1)

    def _visit_transparent(
        self,
        container: GraphModule,
        visible_parent_id: str,
        container_path: str,
        depth: int,
    ) -> None:
        with self._activated(container):
            for child_name, child in self._children(container):
                self._reject_active_module(child)
                child_path = _child_path(container_path, child_name)
                allocated_child_id = self._append_child(
                    child,
                    child_path,
                    child_path,
                    visible_parent_id,
                )
                if allocated_child_id is not None:
                    self._visit(child, allocated_child_id, child_path, depth + 1)

    def _children(
        self,
        module: GraphModule,
    ) -> tuple[tuple[str, GraphModule], ...]:
        return self._construction.children_by_module_id[id(module)]

    def _reject_excessive_depth(self, depth: int) -> None:
        if depth > self._construction.limits.maximum_graph_depth:
            raise InspectionError(
                "Inspection graph depth limit of "
                f"{self._construction.limits.maximum_graph_depth} exceeded."
            )


def _child_path(parent_path: str, child_name: str) -> str:
    return child_name if not parent_path else f"{parent_path}.{child_name}"


def _is_transparent_graph_container(
    parent: GraphModule,
    child_name: str,
    child: GraphModule,
) -> bool:
    return (
        type(parent).__name__ == "LayerStack"
        and child_name == "layers"
        and type(child).__name__ == "ModuleList"
    )


def construct_model_graph(construction: GraphConstruction) -> ModelGraph:
    return _GraphBuilder(construction).build()


__all__ = [
    "GraphConstruction",
    "ROOT_NODE_ID",
    "ROOT_NODE_PATH",
    "construct_model_graph",
]
