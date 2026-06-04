import { type InspectResponse } from "@/lib/api";
import { type GraphDetailMode, type GraphNavigation, type GraphScope } from "@/lib/graph/types";

export function filterGraphByDetail(
  graph: InspectResponse | undefined,
  detailMode: GraphDetailMode,
): InspectResponse | undefined {
  if (!graph || detailMode === "full") {
    return graph;
  }

  const visibleNodeIds = new Set(
    graph.nodes
      .filter((node) => node.graphRole === "architecture")
      .map((node) => node.id),
  );
  const parentByChildId = new Map<string, string>();

  for (const edge of graph.edges) {
    if (!parentByChildId.has(edge.target)) {
      parentByChildId.set(edge.target, edge.source);
    }
  }

  const edges: InspectResponse["edges"] = [];
  const edgeIds = new Set<string>();

  for (const node of graph.nodes) {
    if (!visibleNodeIds.has(node.id)) {
      continue;
    }

    let ancestorId = parentByChildId.get(node.id);
    while (ancestorId && !visibleNodeIds.has(ancestorId)) {
      ancestorId = parentByChildId.get(ancestorId);
    }

    if (!ancestorId) {
      continue;
    }

    const edgeId = `${ancestorId}-${node.id}`;
    if (edgeIds.has(edgeId)) {
      continue;
    }
    edgeIds.add(edgeId);
    edges.push({
      id: edgeId,
      source: ancestorId,
      target: node.id,
    });
  }

  return {
    ...graph,
    nodes: graph.nodes.filter((node) => visibleNodeIds.has(node.id)),
    edges,
  };
}

export function filterGraphByExpansion(
  graph: InspectResponse | undefined,
  navigation: GraphNavigation,
  expandedNodeIds: Set<string>,
  scope: GraphScope,
): InspectResponse | undefined {
  if (!graph || scope === "entire") {
    return graph;
  }

  const visibleNodeIds = new Set<string>();

  const addVisibleNode = (nodeId: string) => {
    if (visibleNodeIds.has(nodeId)) {
      return;
    }
    visibleNodeIds.add(nodeId);

    const shouldShowChildren =
      navigation.rootIds.has(nodeId) || expandedNodeIds.has(nodeId);
    if (!shouldShowChildren) {
      return;
    }

    for (const childId of navigation.childrenById.get(nodeId) ?? []) {
      addVisibleNode(childId);
    }
  };

  for (const rootId of navigation.rootIds) {
    addVisibleNode(rootId);
  }

  return {
    ...graph,
    nodes: graph.nodes.filter((node) => visibleNodeIds.has(node.id)),
    edges: graph.edges.filter(
      (edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target),
    ),
  };
}
