import dagre from "dagre";
import { MarkerType, type Edge, type Node } from "@xyflow/react";
import { type GraphNode, type InspectResponse } from "@/lib/api";
import {
  GRAPH_HORIZONTAL_CARD_GAP,
  GRAPH_VERTICAL_CARD_GAP,
  NODE_WIDTH,
  SIMPLE_NODE_HEIGHT,
} from "@/lib/graph/constants";
import { graphNodeHeight, nodeSubtitle, nodeTitle } from "@/lib/graph/formatting";
import {
  type ChildSummary,
  type ClusterDiagram,
  type ExpertDiagram,
  type GraphDetailMode,
  type GraphNavigation,
  type GraphParameterActivity,
  type StackDiagram,
  type ViewerNodeData,
} from "@/lib/graph/types";

export function layoutGraph(
  graph: InspectResponse | undefined,
  options: {
    graphDetailMode: GraphDetailMode;
    navigation: GraphNavigation;
    childSummariesById: Map<string, ChildSummary[]>;
    childSummarySourceNodesById?: Map<string, GraphNode>;
    expertDiagramsById?: Map<string, ExpertDiagram>;
    stackDiagramsById?: Map<string, StackDiagram>;
    clusterDiagramsById?: Map<string, ClusterDiagram>;
    expandedGraphNodeIds: Set<string>;
    expandedDetailNodeIds: Set<string>;
    enableExpansion: boolean;
    selectedNodeId: string | null;
    canOpenMonitor?: (node: GraphNode) => boolean;
    parameterActivityForNode?: (node: GraphNode) => GraphParameterActivity | undefined;
    onActivateNode: (nodeId: string) => void;
    onToggleExpansion: (nodeId: string) => void;
    onOpenMonitor?: (node: GraphNode) => void;
    onToggleDetails: (nodeId: string) => void;
  },
) {
  if (!graph) {
    return { nodes: [], edges: [] };
  }

  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({
    rankdir: "LR",
    nodesep: GRAPH_VERTICAL_CARD_GAP,
    ranksep: GRAPH_HORIZONTAL_CARD_GAP,
  });

  const isSimpleMode = options.graphDetailMode === "simple";
  const childSummarySourceNodesById =
    options.childSummarySourceNodesById ??
    new Map(graph.nodes.map((node) => [node.id, node]));

  const decorateChildSummaries = (childSummaries: ChildSummary[]) => {
    if (!options.parameterActivityForNode) {
      return childSummaries;
    }

    let changed = false;
    const decorated = childSummaries.map((summary) => {
      if (!summary.sourceNodeId) {
        return summary;
      }

      const sourceNode = childSummarySourceNodesById.get(summary.sourceNodeId);
      const parameterActivity = sourceNode
        ? options.parameterActivityForNode?.(sourceNode)
        : undefined;
      if (!parameterActivity || parameterActivity === summary.parameterActivity) {
        return summary;
      }

      changed = true;
      return { ...summary, parameterActivity };
    });

    return changed ? decorated : childSummaries;
  };

  // Compute each node's height once and reuse it for both the dagre layout and
  // the React Flow node style (previously graphNodeHeight ran twice per node).
  const heightById = new Map<string, number>();
  graph.nodes.forEach((node) => {
    const childCount = options.navigation.childrenById.get(node.id)?.length ?? 0;
    const childSummaries = options.childSummariesById.get(node.id) ?? [];
    const expertDiagram = options.expertDiagramsById?.get(node.id);
    const stackDiagram = options.stackDiagramsById?.get(node.id);
    const clusterDiagram = options.clusterDiagramsById?.get(node.id);
    const isRoot = options.navigation.rootIds.has(node.id);
    const canToggleExpansion = options.enableExpansion && !isRoot && childCount > 0;
    heightById.set(
      node.id,
      isSimpleMode
        ? SIMPLE_NODE_HEIGHT
        : graphNodeHeight({
            title: nodeTitle(node),
            parameterCount: node.parameterCount,
            parameterSizeBytes: node.parameterSizeBytes,
            childCount,
            graphDetailMode: options.graphDetailMode,
            canToggleExpansion,
            isRootNode: isRoot,
            details: node.details,
            config: node.config,
            childSummaries,
            expertDiagram,
            stackDiagram,
            clusterDiagram,
            isDetailsExpanded: options.expandedDetailNodeIds.has(node.id),
          }),
    );
  });

  graph.nodes.forEach((node) => {
    dagreGraph.setNode(node.id, {
      width: NODE_WIDTH,
      height: heightById.get(node.id) ?? 0,
    });
  });
  graph.edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });
  dagre.layout(dagreGraph);

  const nodes: Array<Node<ViewerNodeData>> = graph.nodes.map((node) => {
    const position = dagreGraph.node(node.id);
    const childCount = options.navigation.childrenById.get(node.id)?.length ?? 0;
    const childSummaries = decorateChildSummaries(
      options.childSummariesById.get(node.id) ?? [],
    );
    const expertDiagram = options.expertDiagramsById?.get(node.id);
    const stackDiagram = options.stackDiagramsById?.get(node.id);
    const clusterDiagram = options.clusterDiagramsById?.get(node.id);
    const isRoot = options.navigation.rootIds.has(node.id);
    const isExpanded = isRoot || options.expandedGraphNodeIds.has(node.id);
    const isDetailsExpanded = options.expandedDetailNodeIds.has(node.id);
    const height = heightById.get(node.id) ?? 0;
    const canToggleExpansion = options.enableExpansion && !isRoot && childCount > 0;
    const canOpenMonitor = Boolean(
      options.onOpenMonitor && options.canOpenMonitor?.(node),
    );
    const title = nodeTitle(node);

    return {
      id: node.id,
      type: "viewerNode",
      position: {
        x: position.x - NODE_WIDTH / 2,
        y: position.y - height / 2,
      },
      selected: options.selectedNodeId === node.id,
      style: { width: NODE_WIDTH, height },
      data: {
        nodeId: node.id,
        label: title,
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
        graphDetailMode: options.graphDetailMode,
        height,
        isRootNode: isRoot,
        isExpanded,
        canToggleExpansion,
        canOpenMonitor,
        parameterActivity: options.parameterActivityForNode?.(node),
        isDetailsExpanded,
        onActivateNode: () => options.onActivateNode(node.id),
        onToggleExpansion: () => options.onToggleExpansion(node.id),
        onOpenMonitor: canOpenMonitor ? () => options.onOpenMonitor?.(node) : undefined,
        onToggleDetails: () => options.onToggleDetails(node.id),
      },
    };
  });

  const edges: Edge[] = graph.edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    markerEnd: { type: MarkerType.ArrowClosed, color: "#7c8dff" },
    style: { stroke: "#8b5cf6", strokeWidth: 2 },
  }));

  return { nodes, edges };
}
