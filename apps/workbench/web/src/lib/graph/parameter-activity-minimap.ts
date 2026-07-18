import type { InspectResponse } from "@/lib/api/inspection";
import {
  ancestorNodeIds,
  buildGraphNavigation,
} from "@/lib/graph/navigation";
import {
  type GraphNavigation,
  type GraphParameterActivity,
} from "@/lib/graph/types";

export type ParameterActivityMinimapModel = {
  graph: InspectResponse | undefined;
  navigation: GraphNavigation;
  parameterNodeIds: Set<string>;
  relevantNodeIds: Set<string>;
  initialExpandedNodeIds: Set<string>;
  expandableNodeIds: Set<string>;
  parameterNodeCount: number;
};

export type ParameterActivityMinimapInput = {
  graph?: InspectResponse;
  activityByNodePath?: Map<string, GraphParameterActivity>;
};

function emptyMinimapModel(): ParameterActivityMinimapModel {
  return {
    graph: undefined,
    navigation: buildGraphNavigation(undefined),
    parameterNodeIds: new Set(),
    relevantNodeIds: new Set(),
    initialExpandedNodeIds: new Set(),
    expandableNodeIds: new Set(),
    parameterNodeCount: 0,
  };
}

function expandableNodeIds(navigation: GraphNavigation) {
  const expandable = new Set<string>();

  for (const [nodeId, children] of navigation.childrenById) {
    if (children.length > 0) {
      expandable.add(nodeId);
    }
  }

  return expandable;
}

export function deriveParameterActivityMinimapModel({
  graph,
  activityByNodePath,
}: ParameterActivityMinimapInput): ParameterActivityMinimapModel {
  if (!graph || !activityByNodePath || activityByNodePath.size === 0) {
    return emptyMinimapModel();
  }

  const fullNavigation = buildGraphNavigation(graph);
  const nodesByPath = new Map(graph.nodes.map((node) => [node.path, node]));
  const parameterNodeIds = new Set<string>();
  const relevantNodeIds = new Set<string>();

  for (const nodePath of activityByNodePath.keys()) {
    const node = nodesByPath.get(nodePath);
    if (!node) {
      continue;
    }

    parameterNodeIds.add(node.id);
    relevantNodeIds.add(node.id);

    for (const ancestorId of ancestorNodeIds(node.id, fullNavigation)) {
      relevantNodeIds.add(ancestorId);
    }
  }

  if (parameterNodeIds.size === 0) {
    return emptyMinimapModel();
  }

  const relevantGraph: InspectResponse = {
    ...graph,
    nodes: graph.nodes.filter((node) => relevantNodeIds.has(node.id)),
    edges: graph.edges.filter(
      (edge) =>
        relevantNodeIds.has(edge.source) && relevantNodeIds.has(edge.target),
    ),
  };
  const navigation = buildGraphNavigation(relevantGraph);
  const expandableNodeIdsSet = expandableNodeIds(navigation);
  const initialExpandedNodeIds = new Set(
    [...navigation.rootIds].filter((nodeId) =>
      expandableNodeIdsSet.has(nodeId),
    ),
  );

  return {
    graph: relevantGraph,
    navigation,
    parameterNodeIds,
    relevantNodeIds,
    initialExpandedNodeIds,
    expandableNodeIds: expandableNodeIdsSet,
    parameterNodeCount: parameterNodeIds.size,
  };
}

export function filterParameterActivityMinimapGraphByExpansion(
  model: ParameterActivityMinimapModel,
  expandedNodeIds: Set<string>,
): InspectResponse | undefined {
  if (!model.graph) {
    return undefined;
  }

  const visibleNodeIds = new Set<string>();
  const visit = (nodeId: string) => {
    if (visibleNodeIds.has(nodeId)) {
      return;
    }
    visibleNodeIds.add(nodeId);

    if (!expandedNodeIds.has(nodeId)) {
      return;
    }

    for (const childId of model.navigation.childrenById.get(nodeId) ?? []) {
      visit(childId);
    }
  };

  for (const rootId of model.navigation.rootIds) {
    visit(rootId);
  }

  return {
    ...model.graph,
    nodes: model.graph.nodes.filter((node) => visibleNodeIds.has(node.id)),
    edges: model.graph.edges.filter(
      (edge) =>
        visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target),
    ),
  };
}

export function expandAllParameterActivityMinimapNodes(
  model: ParameterActivityMinimapModel,
) {
  return new Set(model.expandableNodeIds);
}

export function collapseParameterActivityMinimapNodes(
  model: ParameterActivityMinimapModel,
) {
  return new Set(model.initialExpandedNodeIds);
}
