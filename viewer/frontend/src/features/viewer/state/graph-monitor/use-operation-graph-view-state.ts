import { useCallback, useEffect, useMemo, useState } from "react";
import { type OperationGraphNode, type OperationGraphResponse } from "@/lib/api";
import {
  type GraphScope,
  OPERATION_GROUP_NODE_PREFIX,
} from "@/lib/graph";

// See use-graph-view-state: lazily load the dagre-backed operation layout so
// `dagre` stays out of the first-load bundle.
type LayoutOperationGraphFn =
  typeof import("@/lib/graph/operation-layout").layoutOperationGraph;
const EMPTY_OPERATION_LAYOUT: ReturnType<LayoutOperationGraphFn> = {
  nodes: [],
  edges: [],
};

export function useOperationGraphViewState(
  graph: OperationGraphResponse | undefined,
) {
  const [operationGraphScope, setOperationGraphScope] =
    useState<GraphScope>("opened");
  const [expandedOperationGroupIds, setExpandedOperationGroupIds] =
    useState<Set<string>>(new Set());
  const [selectedOperationNodeId, setSelectedOperationNodeId] =
    useState<string | null>(null);

  const toggleOperationGroup = useCallback((groupId: string) => {
    setExpandedOperationGroupIds((current) => {
      const next = new Set(current);
      if (next.has(groupId)) {
        next.delete(groupId);
      } else {
        next.add(groupId);
      }
      return next;
    });
  }, []);

  const [layoutOperationGraph, setLayoutOperationGraph] =
    useState<LayoutOperationGraphFn | null>(null);
  useEffect(() => {
    let active = true;
    void import("@/lib/graph/operation-layout").then((module) => {
      if (active) {
        setLayoutOperationGraph(() => module.layoutOperationGraph);
      }
    });
    return () => {
      active = false;
    };
  }, []);

  const baseLayout = useMemo(() => {
    if (!layoutOperationGraph) {
      return EMPTY_OPERATION_LAYOUT;
    }
    return layoutOperationGraph(graph, {
      scope: operationGraphScope,
      expandedGroupIds: expandedOperationGroupIds,
      selectedNodeId: null,
      onSelectNode: setSelectedOperationNodeId,
      onToggleGroup: toggleOperationGroup,
    });
  }, [
    layoutOperationGraph,
    expandedOperationGroupIds,
    graph,
    operationGraphScope,
    toggleOperationGroup,
  ]);

  const operationNodes = useMemo(
    () =>
      baseLayout.nodes.map((node) =>
        node.selected === (node.id === selectedOperationNodeId)
          ? node
          : { ...node, selected: node.id === selectedOperationNodeId },
      ),
    [baseLayout, selectedOperationNodeId],
  );

  const visibleNodeIds = useMemo(
    () => new Set(operationNodes.map((node) => node.id)),
    [operationNodes],
  );

  useEffect(() => {
    if (!selectedOperationNodeId || visibleNodeIds.has(selectedOperationNodeId)) {
      return;
    }
    setSelectedOperationNodeId(null);
  }, [selectedOperationNodeId, visibleNodeIds]);

  const selectedOperationNode = useMemo<OperationGraphNode | undefined>(() => {
    if (!selectedOperationNodeId || selectedOperationNodeId.startsWith(OPERATION_GROUP_NODE_PREFIX)) {
      return undefined;
    }
    return graph?.nodes.find((node) => node.id === selectedOperationNodeId);
  }, [graph, selectedOperationNodeId]);

  const collapseOperationGraphNodes = useCallback(() => {
    setExpandedOperationGroupIds(new Set());
    setSelectedOperationNodeId(null);
  }, []);

  const resetOperationGraphSelectionAndExpansion = useCallback(() => {
    setSelectedOperationNodeId(null);
    setExpandedOperationGroupIds(new Set());
  }, []);

  return {
    operationGraphScope,
    setOperationGraphScope,
    expandedOperationGroupIds,
    selectedOperationNodeId,
    setSelectedOperationNodeId,
    selectedOperationNode,
    operationNodes,
    operationEdges: baseLayout.edges,
    collapseOperationGraphNodes,
    resetOperationGraphSelectionAndExpansion,
  };
}
