import { type GraphNode, type InspectResponse } from "@/lib/api";

const LINEAR_MONITOR_TYPES = new Set(["LinearLayer", "AdaptiveLinearLayer"]);
const NUMERIC_STACK_ENTRY = /^\d+$/;

export type LinearMonitorComparisonScope = "same-stack" | "all-layers";

export type LinearMonitorComparisonCandidateGroups = Record<
  LinearMonitorComparisonScope,
  GraphNode[]
>;

function isEligibleLinearMonitorNode(node: GraphNode) {
  return node.graphRole === "architecture" && LINEAR_MONITOR_TYPES.has(node.typeName);
}

function stackIndex(node: GraphNode) {
  const segment = node.path.split(".").at(-1) ?? "";
  return NUMERIC_STACK_ENTRY.test(segment) ? Number(segment) : undefined;
}

function comparisonSortIndex(
  node: GraphNode,
  nodesById: Map<string, GraphNode>,
  parentIdByChildId: Map<string, string>,
) {
  const parentId = parentIdByChildId.get(node.id);
  const parent = parentId ? nodesById.get(parentId) : undefined;
  return stackIndex(node) ?? (parent ? stackIndex(parent) : undefined) ?? 0;
}

function hasRuntimeOrInternalAncestor(
  node: GraphNode,
  nodesById: Map<string, GraphNode>,
  parentIdByChildId: Map<string, string>,
) {
  const visited = new Set<string>();
  let parentId = parentIdByChildId.get(node.id);

  while (parentId && !visited.has(parentId)) {
    visited.add(parentId);
    const parent = nodesById.get(parentId);
    if (!parent) {
      return false;
    }
    if (parent.graphRole !== "architecture") {
      return true;
    }
    parentId = parentIdByChildId.get(parent.id);
  }

  return false;
}

export function createLinearMonitorTargetResolver(
  graph: InspectResponse | undefined,
): (node: GraphNode | undefined) => GraphNode | undefined {
  if (!graph) {
    return () => undefined;
  }

  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));
  const nodesByPath = new Map(graph.nodes.map((node) => [node.path, node]));
  const rootIds = new Set(graph.nodes.map((node) => node.id));
  const childIdsBySourceId = new Map<string, Set<string>>();

  for (const edge of graph.edges) {
    rootIds.delete(edge.target);
    const childIds = childIdsBySourceId.get(edge.source) ?? new Set<string>();
    childIds.add(edge.target);
    childIdsBySourceId.set(edge.source, childIds);
  }

  return (candidate: GraphNode | undefined) => {
    if (!candidate) {
      return undefined;
    }

    const node = nodesById.get(candidate.id) ?? nodesByPath.get(candidate.path);
    if (!node || rootIds.has(node.id) || node.graphRole !== "architecture") {
      return undefined;
    }

    if (isEligibleLinearMonitorNode(node)) {
      return node;
    }

    if (node.typeName !== "Layer") {
      return undefined;
    }

    const child = nodesByPath.get(`${node.path}.model`);
    if (!child || !childIdsBySourceId.get(node.id)?.has(child.id)) {
      return undefined;
    }

    return isEligibleLinearMonitorNode(child) ? child : undefined;
  };
}

export function resolveLinearMonitorTarget(
  graph: InspectResponse | undefined,
  node: GraphNode | undefined,
) {
  return createLinearMonitorTargetResolver(graph)(node);
}

export function buildLinearMonitorComparisonCandidates(
  graph: InspectResponse | undefined,
  node: GraphNode | undefined,
) {
  if (!graph) {
    return [];
  }

  const nodesById = new Map(graph.nodes.map((graphNode) => [graphNode.id, graphNode]));
  const childIdsByParentId = new Map<string, string[]>();
  const parentIdByChildId = new Map<string, string>();

  for (const edge of graph.edges) {
    childIdsByParentId.set(edge.source, [
      ...(childIdsByParentId.get(edge.source) ?? []),
      edge.target,
    ]);
    if (!parentIdByChildId.has(edge.target)) {
      parentIdByChildId.set(edge.target, edge.source);
    }
  }

  const resolveTarget = createLinearMonitorTargetResolver(graph);
  const primaryTarget = resolveTarget(node);
  if (!primaryTarget) {
    return [];
  }

  const targetParent = parentIdByChildId.get(primaryTarget.id);
  const targetParentNode = targetParent ? nodesById.get(targetParent) : undefined;
  const hasNumericLayerWrapper = Boolean(
    targetParentNode &&
      stackIndex(targetParentNode) !== undefined &&
      primaryTarget.path === `${targetParentNode.path}.model`,
  );
  const stackEntry =
    targetParentNode && hasNumericLayerWrapper
      ? targetParentNode
      : stackIndex(primaryTarget) !== undefined
        ? primaryTarget
        : undefined;
  const stackParentId = stackEntry ? parentIdByChildId.get(stackEntry.id) : undefined;
  if (!stackEntry || !stackParentId) {
    return [];
  }

  const candidates = new Map<string, GraphNode>();
  for (const siblingId of childIdsByParentId.get(stackParentId) ?? []) {
    const sibling = nodesById.get(siblingId);
    if (!sibling || sibling.id === stackEntry.id || stackIndex(sibling) === undefined) {
      continue;
    }

    const target = resolveTarget(sibling);
    if (!target || target.id === primaryTarget.id || target.path === primaryTarget.path) {
      continue;
    }
    candidates.set(target.path, target);
  }

  return [...candidates.values()].sort((left, right) => {
    const leftIndex = comparisonSortIndex(left, nodesById, parentIdByChildId);
    const rightIndex = comparisonSortIndex(right, nodesById, parentIdByChildId);
    return leftIndex - rightIndex || left.path.localeCompare(right.path);
  });
}

function buildAllLinearMonitorComparisonCandidates(
  graph: InspectResponse | undefined,
  node: GraphNode | undefined,
) {
  if (!graph) {
    return [];
  }

  const nodesById = new Map(graph.nodes.map((graphNode) => [graphNode.id, graphNode]));
  const parentIdByChildId = new Map<string, string>();
  for (const edge of graph.edges) {
    if (!parentIdByChildId.has(edge.target)) {
      parentIdByChildId.set(edge.target, edge.source);
    }
  }

  const resolveTarget = createLinearMonitorTargetResolver(graph);
  const primaryTarget = resolveTarget(node);
  if (!primaryTarget) {
    return [];
  }

  const candidates = new Map<string, GraphNode>();
  for (const candidate of graph.nodes) {
    const target = resolveTarget(candidate);
    if (
      !target ||
      target.id === primaryTarget.id ||
      target.path === primaryTarget.path ||
      hasRuntimeOrInternalAncestor(target, nodesById, parentIdByChildId)
    ) {
      continue;
    }
    if (!candidates.has(target.path)) {
      candidates.set(target.path, target);
    }
  }

  return [...candidates.values()];
}

export function buildLinearMonitorComparisonCandidateGroups(
  graph: InspectResponse | undefined,
  node: GraphNode | undefined,
): LinearMonitorComparisonCandidateGroups {
  return {
    "same-stack": buildLinearMonitorComparisonCandidates(graph, node),
    "all-layers": buildAllLinearMonitorComparisonCandidates(graph, node),
  };
}
