import { type Node } from "@xyflow/react";
import type { GraphNode, InspectResponse } from "@/lib/api/inspection";
import { buildChildSummaries } from "@/lib/graph/child-summaries";
import { buildClusterDiagrams } from "@/lib/graph/cluster-diagrams";
import { graphCardGeometry } from "@/lib/graph/constants";
import { buildExpertDiagrams } from "@/lib/graph/expert-diagrams";
import {
  filterGraphByDetail,
  filterGraphByExpansion,
} from "@/lib/graph/filtering";
import { graphNodeHeight } from "@/lib/graph/format/height";
import { nodeSubtitle, nodeTitle } from "@/lib/graph/format/text";
import { buildGraphNavigation } from "@/lib/graph/navigation";
import { buildStackDiagrams } from "@/lib/graph/stack-diagrams";
import {
  type ChildSummary,
  type ClusterDiagram,
  type ExpertDiagram,
  type GraphDetailMode,
  type GraphNavigation,
  type GraphParameterActivity,
  type GraphScope,
  type StackDiagram,
  type WorkbenchNodeData,
} from "@/lib/graph/types";

export type GraphDisplayModel = {
  graph: InspectResponse | undefined;
  graphDetailMode: GraphDetailMode;
  fullNavigation: GraphNavigation;
  fullNodeIds: Set<string>;
  fullClusterNodeIds: Set<string>;
  detailGraph: InspectResponse | undefined;
  detailNavigation: GraphNavigation;
  detailNodeIds: Set<string>;
  childSummariesById: Map<string, ChildSummary[]>;
  sourceNodesById: Map<string, GraphNode>;
  expertDiagramsById: Map<string, ExpertDiagram>;
  stackDiagramsById: Map<string, StackDiagram>;
  clusterDiagramsById: Map<string, ClusterDiagram>;
};

export type GraphDisplayCard = {
  id: string;
  width: number;
  height: number;
  data: WorkbenchNodeData;
};

export type GraphDisplayProjection = {
  graph: InspectResponse | undefined;
  cards: GraphDisplayCard[];
  edges: InspectResponse["edges"];
};

export type ProjectGraphDisplayOptions = {
  graphScope: GraphScope;
  expandedGraphNodeIds: Set<string>;
  expandedDetailNodeIds: Set<string>;
  canOpenMonitor?: (node: GraphNode) => boolean;
  parameterActivityForNode?: (
    node: GraphNode,
  ) => GraphParameterActivity | undefined;
  onActivateNode: (nodeId: string) => void;
  onToggleExpansion: (nodeId: string) => void;
  onOpenMonitor?: (node: GraphNode) => void;
  onToggleDetails: (nodeId: string) => void;
};

export function deriveGraphDisplayModel(
  graph: InspectResponse | undefined,
  graphDetailMode: GraphDetailMode,
): GraphDisplayModel {
  const fullNavigation = buildGraphNavigation(graph);
  const detailGraph = filterGraphByDetail(graph, graphDetailMode);
  const detailNavigation = buildGraphNavigation(detailGraph);

  return {
    graph,
    graphDetailMode,
    fullNavigation,
    fullNodeIds: new Set((graph?.nodes ?? []).map((node) => node.id)),
    fullClusterNodeIds: new Set(
      (graph?.nodes ?? [])
        .filter((node) => node.typeName === "NeuronCluster")
        .map((node) => node.id),
    ),
    detailGraph,
    detailNavigation,
    detailNodeIds: new Set((detailGraph?.nodes ?? []).map((node) => node.id)),
    childSummariesById: buildChildSummaries(detailGraph, detailNavigation),
    sourceNodesById: new Map(
      (detailGraph?.nodes ?? []).map((node) => [node.id, node]),
    ),
    expertDiagramsById: buildExpertDiagrams(detailGraph, detailNavigation),
    stackDiagramsById: buildStackDiagrams(detailGraph, detailNavigation),
    clusterDiagramsById: buildClusterDiagrams(detailGraph),
  };
}

function decorateChildSummaries(
  summaries: ChildSummary[],
  model: GraphDisplayModel,
  parameterActivityForNode:
    | ((node: GraphNode) => GraphParameterActivity | undefined)
    | undefined,
) {
  if (!parameterActivityForNode) {
    return summaries;
  }

  let changed = false;
  const decorated = summaries.map((summary) => {
    const sourceNode = summary.sourceNodeId
      ? model.sourceNodesById.get(summary.sourceNodeId)
      : undefined;
    const parameterActivity = sourceNode
      ? parameterActivityForNode(sourceNode)
      : undefined;
    if (!parameterActivity || parameterActivity === summary.parameterActivity) {
      return summary;
    }

    changed = true;
    return { ...summary, parameterActivity };
  });

  return changed ? decorated : summaries;
}

function graphDisplayCard(
  node: GraphNode,
  model: GraphDisplayModel,
  options: ProjectGraphDisplayOptions,
): GraphDisplayCard {
  const childCount =
    model.detailNavigation.childrenById.get(node.id)?.length ?? 0;
  const childSummaries = decorateChildSummaries(
    model.childSummariesById.get(node.id) ?? [],
    model,
    options.parameterActivityForNode,
  );
  const expertDiagram = model.expertDiagramsById.get(node.id);
  const stackDiagram = model.stackDiagramsById.get(node.id);
  const clusterDiagram = model.clusterDiagramsById.get(node.id);
  const isRootNode = model.detailNavigation.rootIds.has(node.id);
  const isExpanded =
    isRootNode || options.expandedGraphNodeIds.has(node.id);
  const isDetailsExpanded = options.expandedDetailNodeIds.has(node.id);
  const canToggleExpansion =
    options.graphScope === "opened" && !isRootNode && childCount > 0;
  const canOpenMonitor = Boolean(
    options.onOpenMonitor && options.canOpenMonitor?.(node),
  );
  const height =
    model.graphDetailMode === "simple"
      ? graphCardGeometry.simpleHeight
      : graphNodeHeight({
          details: node.details,
          config: node.config,
          childSummaries,
          expertDiagram,
          stackDiagram,
          clusterDiagram,
          isDetailsExpanded,
        });

  return {
    id: node.id,
    width: graphCardGeometry.width,
    height,
    data: {
      nodeId: node.id,
      label: nodeTitle(node),
      typeName: node.typeName,
      description: node.description,
      subtitle: nodeSubtitle(node),
      path: node.path,
      graphRole: node.graphRole,
      parameterCount: node.parameterCount,
      parameterSizeBytes: node.parameterSizeBytes,
      details: node.details,
      config: node.config,
      childCount,
      childSummaries,
      expertDiagram,
      stackDiagram,
      clusterDiagram,
      graphDetailMode: model.graphDetailMode,
      height,
      isRootNode,
      isExpanded,
      canToggleExpansion,
      canOpenMonitor,
      parameterActivity: options.parameterActivityForNode?.(node),
      isDetailsExpanded,
      onActivateNode: () => options.onActivateNode(node.id),
      onToggleExpansion: () => options.onToggleExpansion(node.id),
      onOpenMonitor: canOpenMonitor
        ? () => options.onOpenMonitor?.(node)
        : undefined,
      onToggleDetails: () => options.onToggleDetails(node.id),
    },
  };
}

export function projectGraphDisplay(
  model: GraphDisplayModel,
  options: ProjectGraphDisplayOptions,
): GraphDisplayProjection {
  const graph = filterGraphByExpansion(
    model.detailGraph,
    model.detailNavigation,
    options.expandedGraphNodeIds,
    options.graphScope,
  );

  return {
    graph,
    cards: (graph?.nodes ?? []).map((node) =>
      graphDisplayCard(node, model, options),
    ),
    edges: graph?.edges ?? [],
  };
}

export function decorateGraphSelection(
  nodes: Array<Node<WorkbenchNodeData>>,
  selectedNodeId: string | null,
) {
  return nodes.map((node) => {
    const selected = node.id === selectedNodeId;
    return node.selected === selected ? node : { ...node, selected };
  });
}
