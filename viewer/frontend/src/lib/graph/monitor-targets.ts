import { type GraphNode, type InspectResponse } from "@/lib/api";

const MONITOR_TARGET_TYPES = {
  linear: new Set(["LinearLayer", "AdaptiveLinearLayer"]),
  attention: new Set([
    "SelfAttention",
    "IndependentAttention",
    "MixtureOfAttentionHeads",
  ]),
  "recurrent-layer": new Set(["RecurrentLayer"]),
  "layer-controller": new Set(["Layer"]),
  parametric: new Set(["ParametricLayer"]),
} as const;

const NUMERIC_STACK_ENTRY = /^\d+$/;

export type MonitorName = keyof typeof MONITOR_TARGET_TYPES;
export type MonitorComparisonScope = "same-stack" | "all-layers";
export type LinearMonitorComparisonScope = MonitorComparisonScope;

export type ResolvedMonitorTarget = {
  monitorName: MonitorName;
  node: GraphNode;
};

export type MonitorComparisonCandidateGroups = Record<
  MonitorComparisonScope,
  GraphNode[]
>;
export type LinearMonitorComparisonCandidateGroups =
  MonitorComparisonCandidateGroups;

export type MonitorTargetResolver = (
  node: GraphNode | undefined,
) => ResolvedMonitorTarget | undefined;
export type LinearMonitorTargetResolver = (
  node: GraphNode | undefined,
) => GraphNode | undefined;

type MonitorAvailabilityPredicate = (target: ResolvedMonitorTarget) => boolean;

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

function isDirectMonitorType(node: GraphNode, monitorName: MonitorName) {
  return (MONITOR_TARGET_TYPES[monitorName] as ReadonlySet<string>).has(
    node.typeName,
  );
}

function directChildByPath(
  node: GraphNode,
  childPath: string,
  nodesByPath: Map<string, GraphNode>,
  childIdsBySourceId: Map<string, Set<string>>,
) {
  const child = nodesByPath.get(childPath);
  if (!child || !childIdsBySourceId.get(node.id)?.has(child.id)) {
    return undefined;
  }
  return child;
}

function candidateTargetsForNode(
  node: GraphNode,
  nodesByPath: Map<string, GraphNode>,
  childIdsBySourceId: Map<string, Set<string>>,
): ResolvedMonitorTarget[] {
  const targets: ResolvedMonitorTarget[] = [];

  if (isDirectMonitorType(node, "linear")) {
    targets.push({ monitorName: "linear", node });
  }
  if (isDirectMonitorType(node, "attention")) {
    targets.push({ monitorName: "attention", node });
  }
  if (isDirectMonitorType(node, "recurrent-layer")) {
    targets.push({ monitorName: "recurrent-layer", node });
  }
  if (isDirectMonitorType(node, "parametric")) {
    targets.push({ monitorName: "parametric", node });
  }

  if (node.typeName === "Layer") {
    const modelChild = directChildByPath(
      node,
      `${node.path}.model`,
      nodesByPath,
      childIdsBySourceId,
    );
    if (modelChild && isDirectMonitorType(modelChild, "linear")) {
      targets.push({ monitorName: "linear", node: modelChild });
    }
    if (modelChild && isDirectMonitorType(modelChild, "parametric")) {
      targets.push({ monitorName: "parametric", node: modelChild });
    }
    targets.push({ monitorName: "layer-controller", node });
  }

  return targets;
}

export function createMonitorTargetResolver(
  graph: InspectResponse | undefined,
  isAvailable?: MonitorAvailabilityPredicate,
): MonitorTargetResolver {
  if (!graph) {
    return () => undefined;
  }

  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));
  const nodesByPath = new Map(graph.nodes.map((node) => [node.path, node]));
  const rootIds = new Set(graph.nodes.map((node) => node.id));
  const childIdsBySourceId = new Map<string, Set<string>>();
  const parentIdByChildId = new Map<string, string>();

  for (const edge of graph.edges) {
    rootIds.delete(edge.target);
    const childIds = childIdsBySourceId.get(edge.source) ?? new Set<string>();
    childIds.add(edge.target);
    childIdsBySourceId.set(edge.source, childIds);
    if (!parentIdByChildId.has(edge.target)) {
      parentIdByChildId.set(edge.target, edge.source);
    }
  }

  return (candidate: GraphNode | undefined) => {
    if (!candidate) {
      return undefined;
    }

    const node = nodesById.get(candidate.id) ?? nodesByPath.get(candidate.path);
    if (
      !node ||
      rootIds.has(node.id) ||
      node.graphRole !== "architecture" ||
      hasRuntimeOrInternalAncestor(node, nodesById, parentIdByChildId)
    ) {
      return undefined;
    }

    const targets = candidateTargetsForNode(
      node,
      nodesByPath,
      childIdsBySourceId,
    ).filter(
      (target) =>
        target.node.graphRole === "architecture" &&
        !hasRuntimeOrInternalAncestor(
          target.node,
          nodesById,
          parentIdByChildId,
        ),
    );

    return isAvailable ? targets.find(isAvailable) : targets[0];
  };
}

export function createMonitorTargetNodeResolver(
  graph: InspectResponse | undefined,
  isAvailable?: MonitorAvailabilityPredicate,
): LinearMonitorTargetResolver {
  const resolveTarget = createMonitorTargetResolver(graph, isAvailable);
  return (node) => resolveTarget(node)?.node;
}

export function createLinearMonitorTargetResolver(
  graph: InspectResponse | undefined,
): LinearMonitorTargetResolver {
  const resolveTarget = createMonitorTargetResolver(
    graph,
    (target) => target.monitorName === "linear",
  );
  return (node) => resolveTarget(node)?.node;
}

export function resolveMonitorTarget(
  graph: InspectResponse | undefined,
  node: GraphNode | undefined,
  monitorName?: MonitorName,
) {
  return createMonitorTargetResolver(
    graph,
    monitorName ? (target) => target.monitorName === monitorName : undefined,
  )(node);
}

export function resolveLinearMonitorTarget(
  graph: InspectResponse | undefined,
  node: GraphNode | undefined,
) {
  return createLinearMonitorTargetResolver(graph)(node);
}

export function buildMonitorComparisonCandidates(
  graph: InspectResponse | undefined,
  node: GraphNode | undefined,
  monitorName?: MonitorName,
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

  const primary = resolveMonitorTarget(graph, node, monitorName);
  if (!primary) {
    return [];
  }
  const resolveTarget = createMonitorTargetResolver(
    graph,
    (target) => target.monitorName === primary.monitorName,
  );
  const primaryTarget = primary.node;

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
    if (
      !target ||
      target.node.id === primaryTarget.id ||
      target.node.path === primaryTarget.path
    ) {
      continue;
    }
    candidates.set(target.node.path, target.node);
  }

  return [...candidates.values()].sort((left, right) => {
    const leftIndex = comparisonSortIndex(left, nodesById, parentIdByChildId);
    const rightIndex = comparisonSortIndex(right, nodesById, parentIdByChildId);
    return leftIndex - rightIndex || left.path.localeCompare(right.path);
  });
}

export function buildLinearMonitorComparisonCandidates(
  graph: InspectResponse | undefined,
  node: GraphNode | undefined,
) {
  return buildMonitorComparisonCandidates(graph, node, "linear");
}

function buildAllMonitorComparisonCandidates(
  graph: InspectResponse | undefined,
  node: GraphNode | undefined,
  monitorName?: MonitorName,
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

  const primary = resolveMonitorTarget(graph, node, monitorName);
  if (!primary) {
    return [];
  }
  const resolveTarget = createMonitorTargetResolver(
    graph,
    (target) => target.monitorName === primary.monitorName,
  );
  const primaryTarget = primary.node;

  const candidates = new Map<string, GraphNode>();
  for (const candidate of graph.nodes) {
    const target = resolveTarget(candidate);
    if (
      !target ||
      target.node.id === primaryTarget.id ||
      target.node.path === primaryTarget.path ||
      hasRuntimeOrInternalAncestor(target.node, nodesById, parentIdByChildId)
    ) {
      continue;
    }
    if (!candidates.has(target.node.path)) {
      candidates.set(target.node.path, target.node);
    }
  }

  return [...candidates.values()];
}

export function buildMonitorComparisonCandidateGroups(
  graph: InspectResponse | undefined,
  node: GraphNode | undefined,
  monitorName?: MonitorName,
): MonitorComparisonCandidateGroups {
  const primary = resolveMonitorTarget(graph, node, monitorName);
  return {
    "same-stack": buildMonitorComparisonCandidates(
      graph,
      primary?.node ?? node,
      primary?.monitorName ?? monitorName,
    ),
    "all-layers": buildAllMonitorComparisonCandidates(
      graph,
      primary?.node ?? node,
      primary?.monitorName ?? monitorName,
    ),
  };
}

export function buildLinearMonitorComparisonCandidateGroups(
  graph: InspectResponse | undefined,
  node: GraphNode | undefined,
): LinearMonitorComparisonCandidateGroups {
  return buildMonitorComparisonCandidateGroups(graph, node, "linear");
}
