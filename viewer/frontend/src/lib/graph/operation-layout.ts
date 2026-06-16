import dagre from "dagre";
import { MarkerType, type Edge, type Node } from "@xyflow/react";
import {
  type OperationGraphNode,
  type OperationGraphResponse,
} from "@/lib/api";
import {
  GRAPH_HORIZONTAL_CARD_GAP,
  GRAPH_VERTICAL_CARD_GAP,
} from "@/lib/graph/constants";
import { OPERATION_GROUP_NODE_PREFIX } from "@/lib/graph/operation-graph-constants";
import {
  type GraphScope,
  type OperationFlowNodeData,
} from "@/lib/graph/types";

const OPERATION_GROUP_WIDTH = 260;
const OPERATION_GROUP_HEIGHT = 92;
const OPERATION_NODE_WIDTH = 292;
const OPERATION_NODE_HEIGHT = 132;

function operationGroupNodeId(groupId: string) {
  return `${OPERATION_GROUP_NODE_PREFIX}${groupId}`;
}

function operationGroupLabel(groupId: string) {
  if (groupId === "__inputs__") {
    return "Inputs";
  }
  if (groupId === "__outputs__") {
    return "Outputs";
  }
  if (groupId === "__root__") {
    return "Model";
  }
  return groupId.split(".").at(-1) || groupId;
}

function operationGroupSubtitle(groupId: string) {
  if (groupId === "__inputs__") {
    return "synthetic inputs";
  }
  if (groupId === "__outputs__") {
    return "export outputs";
  }
  if (groupId === "__root__") {
    return "root module";
  }
  return groupId;
}

function operationGroupIds(graph: OperationGraphResponse) {
  const seen = new Set<string>();
  const groups: string[] = [];
  for (const node of graph.nodes) {
    if (!node.groupId || seen.has(node.groupId)) {
      continue;
    }
    seen.add(node.groupId);
    groups.push(node.groupId);
  }
  return groups;
}

function operationCountsByGroup(graph: OperationGraphResponse) {
  const counts = new Map<string, number>();
  for (const node of graph.nodes) {
    if (!node.groupId) {
      continue;
    }
    counts.set(node.groupId, (counts.get(node.groupId) ?? 0) + 1);
  }
  return counts;
}

function visibleNodeId(
  node: OperationGraphNode,
  expandedGroupIds: Set<string>,
  scope: GraphScope,
) {
  if (!node.groupId || scope === "entire" || expandedGroupIds.has(node.groupId)) {
    return node.id;
  }
  return operationGroupNodeId(node.groupId);
}

export function layoutOperationGraph(
  graph: OperationGraphResponse | undefined,
  options: {
    scope: GraphScope;
    expandedGroupIds: Set<string>;
    selectedNodeId: string | null;
    onSelectNode: (nodeId: string) => void;
    onToggleGroup: (groupId: string) => void;
  },
) {
  if (!graph || graph.status !== "ok") {
    return { nodes: [] as Array<Node<OperationFlowNodeData>>, edges: [] as Edge[] };
  }

  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));
  const visibleNodes = new Map<string, Node<OperationFlowNodeData>>();
  const groupCounts = operationCountsByGroup(graph);

  if (options.scope === "opened") {
    for (const groupId of operationGroupIds(graph)) {
      if (options.expandedGroupIds.has(groupId)) {
        continue;
      }
      const nodeId = operationGroupNodeId(groupId);
      visibleNodes.set(nodeId, {
        id: nodeId,
        type: "operationNode",
        position: { x: 0, y: 0 },
        selected: options.selectedNodeId === nodeId,
        style: { width: OPERATION_GROUP_WIDTH, height: OPERATION_GROUP_HEIGHT },
        data: {
          kind: "group",
          groupId,
          label: operationGroupLabel(groupId),
          subtitle: operationGroupSubtitle(groupId),
          operationCount: groupCounts.get(groupId) ?? 0,
          height: OPERATION_GROUP_HEIGHT,
          isExpanded: false,
          onActivateNode: () => {
            options.onSelectNode(nodeId);
            options.onToggleGroup(groupId);
          },
          onToggleExpansion: () => options.onToggleGroup(groupId),
        },
      });
    }
  }

  for (const operation of graph.nodes) {
    if (
      operation.groupId &&
      options.scope === "opened" &&
      !options.expandedGroupIds.has(operation.groupId)
    ) {
      continue;
    }
    visibleNodes.set(operation.id, {
      id: operation.id,
      type: "operationNode",
      position: { x: 0, y: 0 },
      selected: options.selectedNodeId === operation.id,
      style: { width: OPERATION_NODE_WIDTH, height: OPERATION_NODE_HEIGHT },
      data: {
        kind: "operation",
        nodeId: operation.id,
        label: operation.label,
        opKind: operation.opKind,
        target: operation.target,
        modulePath: operation.modulePath,
        groupId: operation.groupId,
        details: operation.details,
        height: OPERATION_NODE_HEIGHT,
        onActivateNode: () => options.onSelectNode(operation.id),
      },
    });
  }

  const visibleEdges = new Map<string, Edge>();
  for (const edge of graph.edges) {
    const source = nodesById.get(edge.source);
    const target = nodesById.get(edge.target);
    if (!source || !target) {
      continue;
    }
    const sourceId = visibleNodeId(source, options.expandedGroupIds, options.scope);
    const targetId = visibleNodeId(target, options.expandedGroupIds, options.scope);
    if (sourceId === targetId) {
      continue;
    }
    if (!visibleNodes.has(sourceId) || !visibleNodes.has(targetId)) {
      continue;
    }
    const edgeId = `${sourceId}-${targetId}`;
    if (visibleEdges.has(edgeId)) {
      continue;
    }
    visibleEdges.set(edgeId, {
      id: edgeId,
      source: sourceId,
      target: targetId,
      markerEnd: { type: MarkerType.ArrowClosed, color: "#76e4f7" },
      style: { stroke: "#67e8f9", strokeWidth: 1.8 },
    });
  }

  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({
    rankdir: "LR",
    nodesep: GRAPH_VERTICAL_CARD_GAP,
    ranksep: GRAPH_HORIZONTAL_CARD_GAP,
  });

  for (const node of visibleNodes.values()) {
    const width =
      node.data.kind === "group" ? OPERATION_GROUP_WIDTH : OPERATION_NODE_WIDTH;
    dagreGraph.setNode(node.id, { width, height: node.data.height });
  }
  for (const edge of visibleEdges.values()) {
    dagreGraph.setEdge(edge.source, edge.target);
  }
  dagre.layout(dagreGraph);

  const nodes = Array.from(visibleNodes.values()).map((node) => {
    const position = dagreGraph.node(node.id) ?? { x: 0, y: 0 };
    const width =
      node.data.kind === "group" ? OPERATION_GROUP_WIDTH : OPERATION_NODE_WIDTH;
    return {
      ...node,
      position: {
        x: position.x - width / 2,
        y: position.y - node.data.height / 2,
      },
    };
  });

  return { nodes, edges: Array.from(visibleEdges.values()) };
}
