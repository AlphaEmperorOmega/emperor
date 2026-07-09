import { type InspectResponse } from "@/lib/api";
import { type GraphNavigation, type HierarchyNode } from "@/lib/graph/types";

export function buildHierarchy(graph: InspectResponse | undefined) {
  if (!graph) {
    return [];
  }

  const nodesById = new Map(
    graph.nodes.map((node) => [node.id, { node, children: [] as HierarchyNode[] }]),
  );
  const childIds = new Set<string>();

  for (const edge of graph.edges) {
    const parent = nodesById.get(edge.source);
    const child = nodesById.get(edge.target);
    if (!parent || !child) {
      continue;
    }
    parent.children.push(child);
    childIds.add(edge.target);
  }

  return graph.nodes
    .filter((node) => !childIds.has(node.id))
    .map((node) => nodesById.get(node.id))
    .filter((node): node is HierarchyNode => Boolean(node));
}

export function buildGraphNavigation(graph: InspectResponse | undefined): GraphNavigation {
  const childrenById = new Map<string, string[]>();
  const parentById = new Map<string, string>();
  const childIds = new Set<string>();

  if (!graph) {
    return { childrenById, parentById, rootIds: new Set() };
  }

  for (const node of graph.nodes) {
    childrenById.set(node.id, []);
  }

  for (const edge of graph.edges) {
    if (!childrenById.has(edge.source) || !childrenById.has(edge.target)) {
      continue;
    }
    childrenById.get(edge.source)?.push(edge.target);
    if (!parentById.has(edge.target)) {
      parentById.set(edge.target, edge.source);
    }
    childIds.add(edge.target);
  }

  const rootIds = new Set(
    graph.nodes
      .filter((node) => !childIds.has(node.id))
      .map((node) => node.id),
  );

  if (rootIds.size === 0 && graph.nodes[0]) {
    rootIds.add(graph.nodes[0].id);
  }

  return { childrenById, parentById, rootIds };
}

export function ancestorNodeIds(nodeId: string, navigation: GraphNavigation) {
  if (navigation.rootIds.has(nodeId)) {
    return [];
  }
  if (!navigation.childrenById.has(nodeId)) {
    return [];
  }

  const ancestors: string[] = [];
  const visited = new Set<string>([nodeId]);
  let parentId = navigation.parentById.get(nodeId);

  while (parentId) {
    if (visited.has(parentId)) {
      return [];
    }
    visited.add(parentId);
    ancestors.push(parentId);
    if (navigation.rootIds.has(parentId)) {
      return ancestors.reverse();
    }
    parentId = navigation.parentById.get(parentId);
  }

  return ancestors.reverse();
}

export function expandableSubtreeNodeIds(nodeId: string, navigation: GraphNavigation) {
  if (!navigation.childrenById.has(nodeId)) {
    return [];
  }

  const expandableNodeIds: string[] = [];
  const visitedNodeIds = new Set<string>();

  const visitNode = (currentNodeId: string) => {
    if (visitedNodeIds.has(currentNodeId)) {
      return;
    }
    visitedNodeIds.add(currentNodeId);

    const children = navigation.childrenById.get(currentNodeId) ?? [];
    if (children.length === 0) {
      return;
    }

    expandableNodeIds.push(currentNodeId);
    for (const childId of children) {
      visitNode(childId);
    }
  };

  visitNode(nodeId);
  return expandableNodeIds;
}
