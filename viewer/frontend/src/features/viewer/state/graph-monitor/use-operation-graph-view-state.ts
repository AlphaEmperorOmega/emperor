import { useCallback, useEffect, useMemo, useState } from "react";
import { type OperationGraphNode, type OperationGraphResponse } from "@/lib/api";
import {
  type GraphScope,
  OPERATION_GROUP_NODE_PREFIX,
  layoutOperationGraph,
} from "@/lib/graph";

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

  const baseLayout = useMemo(
    () =>
      layoutOperationGraph(graph, {
        scope: operationGraphScope,
        expandedGroupIds: expandedOperationGroupIds,
        selectedNodeId: null,
        onSelectNode: setSelectedOperationNodeId,
        onToggleGroup: toggleOperationGroup,
      }),
    [expandedOperationGroupIds, graph, operationGraphScope, toggleOperationGroup],
  );

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
