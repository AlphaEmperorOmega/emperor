import type { GraphNode, InspectResponse } from "@/lib/api";
import {
  detailText,
  nodeBadges,
  nodeDimsText,
  structureNodeLabel,
} from "@/lib/graph/formatting";
import { lastPathSegment } from "@/lib/graph/helpers";

const VIRTUAL_ROOT_ID = "__parameter_focus_root__";

export type ParameterFocusNodeSummary = {
  id: string;
  label: string;
  path: string;
  typeName: string;
  graphRole: GraphNode["graphRole"];
  parameterCount: number;
  parameterSizeBytes: number;
  childCount: number;
  hasParameters: boolean;
  dimText?: string;
  badges: Array<{ key: string; value: string }>;
  isVirtualRoot?: boolean;
};

export type ParameterFocusData = {
  focusedNode: ParameterFocusNodeSummary | null;
  focusNodeId: string | null;
  ancestors: ParameterFocusNodeSummary[];
  immediateChildren: ParameterFocusNodeSummary[];
  zeroParameterChildren: ParameterFocusNodeSummary[];
  totalParameterCount: number;
  focusedParameterCount: number;
  hasParameters: boolean;
};

type GraphIndexes = {
  nodesById: Map<string, GraphNode>;
  childrenById: Map<string, string[]>;
  parentById: Map<string, string>;
  rootIds: string[];
};

function finiteNonNegative(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : 0;
}

function buildGraphIndexes(graph: InspectResponse): GraphIndexes {
  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));
  const childrenById = new Map(graph.nodes.map((node) => [node.id, [] as string[]]));
  const parentById = new Map<string, string>();
  const childIds = new Set<string>();
  const edgeKeys = new Set<string>();

  for (const edge of graph.edges) {
    if (
      edge.source === edge.target ||
      !nodesById.has(edge.source) ||
      !nodesById.has(edge.target)
    ) {
      continue;
    }

    const edgeKey = `${edge.source}\0${edge.target}`;
    if (edgeKeys.has(edgeKey)) {
      continue;
    }
    edgeKeys.add(edgeKey);
    childrenById.get(edge.source)?.push(edge.target);
    if (!parentById.has(edge.target)) {
      parentById.set(edge.target, edge.source);
    }
    childIds.add(edge.target);
  }

  const rootIds = graph.nodes
    .filter((node) => !childIds.has(node.id))
    .map((node) => node.id);

  return {
    nodesById,
    childrenById,
    parentById,
    rootIds: rootIds.length > 0 ? rootIds : graph.nodes[0] ? [graph.nodes[0].id] : [],
  };
}

function pathSegments(path: string) {
  return path.split(".").filter(Boolean);
}

function uniqueLabelForDuplicate(node: GraphNode, pathSegmentCount: number) {
  const segments = pathSegments(node.path);
  const suffix = segments.slice(Math.max(0, segments.length - pathSegmentCount)).join(".");
  return `${suffix || lastPathSegment(node.path) || node.id}: ${node.typeName}`;
}

function uniqueParameterLabels(nodes: GraphNode[]) {
  const labelsById = new Map<string, string>();
  const baseGroups = new Map<string, GraphNode[]>();

  for (const node of nodes) {
    const label = structureNodeLabel(node);
    const group = baseGroups.get(label) ?? [];
    group.push(node);
    baseGroups.set(label, group);
  }

  for (const [label, group] of baseGroups) {
    if (group.length === 1) {
      labelsById.set(group[0].id, label);
      continue;
    }

    let segmentCount = 2;
    let labels = group.map((node) => uniqueLabelForDuplicate(node, segmentCount));
    while (new Set(labels).size < labels.length && segmentCount < 8) {
      segmentCount += 1;
      labels = group.map((node) => uniqueLabelForDuplicate(node, segmentCount));
    }

    labels.forEach((candidate, index) => {
      const node = group[index];
      labelsById.set(
        node.id,
        labels.filter((other) => other === candidate).length === 1
          ? candidate
          : `${candidate} · ${node.id}`,
      );
    });
  }

  return labelsById;
}

function summarizeNode(
  node: GraphNode,
  childrenById: Map<string, string[]>,
  label = structureNodeLabel(node),
): ParameterFocusNodeSummary {
  const parameterCount = finiteNonNegative(node.parameterCount);
  return {
    id: node.id,
    label,
    path: node.path,
    typeName: node.typeName,
    graphRole: node.graphRole,
    parameterCount,
    parameterSizeBytes: finiteNonNegative(node.parameterSizeBytes),
    childCount: childrenById.get(node.id)?.length ?? 0,
    hasParameters: parameterCount > 0,
    dimText: nodeDimsText(node.details, node.config),
    badges: nodeBadges(node.details).map(([key, value]) => ({
      key,
      value: detailText(value),
    })),
  };
}

function virtualRootSummary(
  graph: InspectResponse,
  childCount: number,
  totalParameterCount: number,
): ParameterFocusNodeSummary {
  return {
    id: VIRTUAL_ROOT_ID,
    label: `${graph.model} / ${graph.preset}`,
    path: graph.model,
    typeName: "Root",
    graphRole: "architecture",
    parameterCount: totalParameterCount,
    parameterSizeBytes: finiteNonNegative(graph.parameterSizeBytes),
    childCount,
    hasParameters: totalParameterCount > 0,
    badges: [],
    isVirtualRoot: true,
  };
}

function ancestorSummaries(
  nodeId: string,
  indexes: GraphIndexes,
  labelsByParentId: Map<string, Map<string, string>>,
) {
  const ancestors: ParameterFocusNodeSummary[] = [];
  const visited = new Set<string>([nodeId]);
  let parentId = indexes.parentById.get(nodeId);

  while (parentId) {
    if (visited.has(parentId)) {
      return [];
    }
    visited.add(parentId);
    const parent = indexes.nodesById.get(parentId);
    if (!parent) {
      break;
    }
    ancestors.push(
      summarizeNode(
        parent,
        indexes.childrenById,
        labelsByParentId.get(indexes.parentById.get(parent.id) ?? "")?.get(parent.id) ??
          structureNodeLabel(parent),
      ),
    );
    parentId = indexes.parentById.get(parentId);
  }

  return ancestors.reverse();
}

function childNodes(childIds: string[], nodesById: Map<string, GraphNode>) {
  return childIds
    .map((childId) => nodesById.get(childId))
    .filter((node): node is GraphNode => Boolean(node));
}

function labelsByParent(indexes: GraphIndexes) {
  const result = new Map<string, Map<string, string>>();
  result.set(
    VIRTUAL_ROOT_ID,
    uniqueParameterLabels(childNodes(indexes.rootIds, indexes.nodesById)),
  );
  for (const [parentId, childIds] of indexes.childrenById) {
    result.set(parentId, uniqueParameterLabels(childNodes(childIds, indexes.nodesById)));
  }
  return result;
}

export function fallbackParameterFocusNodeId(
  requestedFocusNodeId: string | null | undefined,
  visibleGraph: InspectResponse | undefined,
  sourceGraph: InspectResponse | undefined = visibleGraph,
) {
  if (!requestedFocusNodeId || !visibleGraph || visibleGraph.nodes.length === 0) {
    return null;
  }

  const visibleNodeIds = new Set(visibleGraph.nodes.map((node) => node.id));
  if (visibleNodeIds.has(requestedFocusNodeId)) {
    return requestedFocusNodeId;
  }
  if (!sourceGraph) {
    return null;
  }

  const parentById = buildGraphIndexes(sourceGraph).parentById;
  const visited = new Set<string>([requestedFocusNodeId]);
  let parentId = parentById.get(requestedFocusNodeId);

  while (parentId) {
    if (visited.has(parentId)) {
      return null;
    }
    visited.add(parentId);
    if (visibleNodeIds.has(parentId)) {
      return parentId;
    }
    parentId = parentById.get(parentId);
  }

  return null;
}

export function buildParameterFocusData(
  graph: InspectResponse | undefined,
  focusNodeId: string | null = null,
): ParameterFocusData {
  if (!graph || graph.nodes.length === 0) {
    return {
      focusedNode: null,
      focusNodeId: null,
      ancestors: [],
      immediateChildren: [],
      zeroParameterChildren: [],
      totalParameterCount: 0,
      focusedParameterCount: 0,
      hasParameters: false,
    };
  }

  const indexes = buildGraphIndexes(graph);
  const labelsByParentId = labelsByParent(indexes);
  const resolvedFocusNodeId = indexes.nodesById.has(focusNodeId ?? "")
    ? focusNodeId
    : null;
  const focusedGraphNode = resolvedFocusNodeId
    ? indexes.nodesById.get(resolvedFocusNodeId)
    : undefined;
  const totalParameterCount =
    finiteNonNegative(graph.parameterCount) ||
    graph.nodes.reduce((total, node) => total + finiteNonNegative(node.parameterCount), 0);
  const rootChildIds =
    indexes.rootIds.length === 1 &&
    (indexes.childrenById.get(indexes.rootIds[0])?.length ?? 0) > 0
      ? indexes.childrenById.get(indexes.rootIds[0]) ?? []
      : indexes.rootIds;
  const rootFocusNode =
    indexes.rootIds.length === 1 ? indexes.nodesById.get(indexes.rootIds[0]) : undefined;
  const focusedNode = focusedGraphNode
    ? summarizeNode(
        focusedGraphNode,
        indexes.childrenById,
        labelsByParentId
          .get(indexes.parentById.get(focusedGraphNode.id) ?? VIRTUAL_ROOT_ID)
          ?.get(focusedGraphNode.id) ?? structureNodeLabel(focusedGraphNode),
      )
    : rootFocusNode
      ? summarizeNode(rootFocusNode, indexes.childrenById)
      : virtualRootSummary(graph, rootChildIds.length, totalParameterCount);
  const focusedParameterCount =
    focusedGraphNode || rootFocusNode
      ? finiteNonNegative((focusedGraphNode ?? rootFocusNode)?.parameterCount) ||
        totalParameterCount
      : totalParameterCount;
  const focusedChildIds = focusedGraphNode
    ? indexes.childrenById.get(focusedGraphNode.id) ?? []
    : rootChildIds;
  const focusedChildNodes = childNodes(focusedChildIds, indexes.nodesById);
  const focusedChildLabels = focusedGraphNode
    ? labelsByParentId.get(focusedGraphNode.id) ?? new Map<string, string>()
    : indexes.rootIds.length === 1
      ? labelsByParentId.get(indexes.rootIds[0]) ?? new Map<string, string>()
      : labelsByParentId.get(VIRTUAL_ROOT_ID) ?? new Map<string, string>();
  const immediateChildren = focusedChildNodes.map((node) =>
    summarizeNode(
      node,
      indexes.childrenById,
      focusedChildLabels.get(node.id) ?? structureNodeLabel(node),
    ),
  );
  const zeroParameterChildren = immediateChildren.filter((child) => !child.hasParameters);

  return {
    focusedNode,
    focusNodeId: resolvedFocusNodeId,
    ancestors: focusedGraphNode
      ? ancestorSummaries(focusedGraphNode.id, indexes, labelsByParentId)
      : [],
    immediateChildren,
    zeroParameterChildren,
    totalParameterCount,
    focusedParameterCount,
    hasParameters: totalParameterCount > 0,
  };
}
