import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type GraphNode, type InspectResponse } from "@/lib/api";
import {
  type GraphDetailMode,
  type GraphParameterActivity,
  type GraphScope,
  ancestorNodeIds,
  buildChildSummaries,
  buildClusterDiagrams,
  buildExpertDiagrams,
  buildGraphNavigation,
  buildStackDiagrams,
  expandableSubtreeNodeIds,
  filterGraphByDetail,
  filterGraphByExpansion,
} from "@/lib/graph";

// The dagre-backed layout is loaded only after graph data exists. Keeping the
// promise at module scope deduplicates imports when graph state changes while
// the layout chunk is in flight.
type LayoutGraphFn = typeof import("@/lib/graph/layout").layoutGraph;
const EMPTY_GRAPH_LAYOUT: ReturnType<LayoutGraphFn> = { nodes: [], edges: [] };
let layoutGraphPromise: Promise<LayoutGraphFn> | null = null;

function loadLayoutGraph() {
  layoutGraphPromise ??= import("@/lib/graph/layout").then(
    (module) => module.layoutGraph,
  );
  return layoutGraphPromise;
}

type GraphViewStateOptions = {
  canOpenMonitor?: (node: GraphNode) => boolean;
  onOpenMonitor?: (node: GraphNode) => void;
  resolveMonitorTarget?: (node: GraphNode) => GraphNode | undefined;
  resolveParameterActivityTarget?: (node: GraphNode) => GraphNode | undefined;
  parameterActivityByNodePath?: Map<string, GraphParameterActivity>;
  inspectionTransition?: {
    revision: number;
    cause: "target-changed" | "inspection-refreshed";
  };
};

export function useGraphViewState(
  graph: InspectResponse | undefined,
  options: GraphViewStateOptions = {},
) {
  const {
    canOpenMonitor,
    onOpenMonitor,
    resolveMonitorTarget,
    resolveParameterActivityTarget,
    parameterActivityByNodePath,
    inspectionTransition,
  } = options;
  const [graphDetailMode, setGraphDetailMode] = useState<GraphDetailMode>("basic");
  const [graphScope, setGraphScope] = useState<GraphScope>("opened");
  const [expandedGraphNodeIds, setExpandedGraphNodeIds] = useState<Set<string>>(new Set());
  const [expandedDetailNodeIds, setExpandedDetailNodeIds] = useState<Set<string>>(new Set());
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [cluster3dNodeId, setCluster3dNodeId] = useState<string | null>(null);

  const fullGraphNavigation = useMemo(() => buildGraphNavigation(graph), [graph]);
  const fullNodeIds = useMemo(
    () => new Set((graph?.nodes ?? []).map((node) => node.id)),
    [graph],
  );
  const graphForDetail = useMemo(
    () => filterGraphByDetail(graph, graphDetailMode),
    [graph, graphDetailMode],
  );
  const graphNavigation = useMemo(() => buildGraphNavigation(graphForDetail), [graphForDetail]);
  const childSummariesById = useMemo(
    () => buildChildSummaries(graphForDetail, graphNavigation),
    [graphForDetail, graphNavigation],
  );
  const childSummarySourceNodesById = useMemo(
    () => new Map((graphForDetail?.nodes ?? []).map((node) => [node.id, node])),
    [graphForDetail],
  );
  const expertDiagramsById = useMemo(
    () => buildExpertDiagrams(graphForDetail, graphNavigation),
    [graphForDetail, graphNavigation],
  );
  const stackDiagramsById = useMemo(
    () => buildStackDiagrams(graphForDetail, graphNavigation),
    [graphForDetail, graphNavigation],
  );
  const clusterDiagramsById = useMemo(
    () => buildClusterDiagrams(graphForDetail),
    [graphForDetail],
  );
  const detailNodeIds = useMemo(
    () => new Set((graphForDetail?.nodes ?? []).map((node) => node.id)),
    [graphForDetail],
  );
  const fullClusterNodeIds = useMemo(
    () =>
      new Set(
        (graph?.nodes ?? [])
          .filter((node) => node.typeName === "NeuronCluster")
          .map((node) => node.id),
      ),
    [graph],
  );

  const activateGraphNode = useCallback((nodeId: string) => {
    setSelectedNodeId(nodeId);
    if (graphScope !== "opened") {
      return;
    }
    if (graphNavigation.rootIds.has(nodeId)) {
      return;
    }
    if ((graphNavigation.childrenById.get(nodeId)?.length ?? 0) === 0) {
      return;
    }
    setExpandedGraphNodeIds((current) => {
      const next = new Set(current);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  }, [graphNavigation, graphScope]);

  const toggleGraphNodeExpansion = useCallback((nodeId: string) => {
    if (graphScope !== "opened") {
      return;
    }
    if (graphNavigation.rootIds.has(nodeId)) {
      return;
    }
    if ((graphNavigation.childrenById.get(nodeId)?.length ?? 0) === 0) {
      return;
    }
    setExpandedGraphNodeIds((current) => {
      const next = new Set(current);
      if (next.has(nodeId)) {
        for (const subtreeNodeId of expandableSubtreeNodeIds(nodeId, graphNavigation)) {
          next.delete(subtreeNodeId);
        }
      } else {
        for (const subtreeNodeId of expandableSubtreeNodeIds(nodeId, graphNavigation)) {
          next.add(subtreeNodeId);
        }
      }
      return next;
    });
  }, [graphNavigation, graphScope]);

  const toggleNodeDetails = useCallback((nodeId: string) => {
    setExpandedDetailNodeIds((current) => {
      const next = new Set(current);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  }, []);

  const monitorTargetForNode = useCallback((node: GraphNode) => {
    return resolveMonitorTarget ? resolveMonitorTarget(node) : node;
  }, [resolveMonitorTarget]);

  const parameterActivityTargetForNode = useCallback((node: GraphNode) => {
    return resolveParameterActivityTarget
      ? resolveParameterActivityTarget(node)
      : monitorTargetForNode(node);
  }, [monitorTargetForNode, resolveParameterActivityTarget]);

  const canOpenGraphNodeMonitor = useCallback((node: GraphNode) => {
    const monitorTarget = monitorTargetForNode(node);
    return Boolean(monitorTarget && canOpenMonitor?.(monitorTarget));
  }, [canOpenMonitor, monitorTargetForNode]);

  const openGraphNodeMonitor = useCallback((node: GraphNode) => {
    const monitorTarget = monitorTargetForNode(node);
    if (!monitorTarget) {
      return;
    }
    setSelectedNodeId(node.id);
    onOpenMonitor?.(monitorTarget);
  }, [monitorTargetForNode, onOpenMonitor]);

  const parameterActivityForNode = useCallback((node: GraphNode) => {
    const monitorTarget = parameterActivityTargetForNode(node);
    return monitorTarget
      ? parameterActivityByNodePath?.get(monitorTarget.path)
      : undefined;
  }, [parameterActivityByNodePath, parameterActivityTargetForNode]);

  const revealGraphNode = useCallback((nodeId: string) => {
    if (!detailNodeIds.has(nodeId)) {
      return;
    }

    setSelectedNodeId(nodeId);
    const ancestors = ancestorNodeIds(nodeId, graphNavigation);
    if (ancestors.length === 0) {
      return;
    }

    setExpandedGraphNodeIds((current) => {
      let changed = false;
      const next = new Set(current);
      for (const ancestorId of ancestors) {
        if (!next.has(ancestorId)) {
          next.add(ancestorId);
          changed = true;
        }
      }
      return changed ? next : current;
    });
  }, [detailNodeIds, graphNavigation]);

  const revealGraphNodeInFull = useCallback((nodeId: string) => {
    if (!fullNodeIds.has(nodeId)) {
      return;
    }

    setGraphDetailMode("full");
    setSelectedNodeId(nodeId);
    const ancestors = ancestorNodeIds(nodeId, fullGraphNavigation);
    if (ancestors.length === 0) {
      return;
    }

    setExpandedGraphNodeIds((current) => {
      let changed = false;
      const next = new Set(current);
      for (const ancestorId of ancestors) {
        if (!next.has(ancestorId)) {
          next.add(ancestorId);
          changed = true;
        }
      }
      return changed ? next : current;
    });
  }, [fullGraphNavigation, fullNodeIds]);

  const openCluster3d = useCallback((nodeId?: string) => {
    const nextNodeId = nodeId ?? selectedNodeId;
    if (!nextNodeId || !fullClusterNodeIds.has(nextNodeId)) {
      return;
    }
    setSelectedNodeId(nextNodeId);
    setCluster3dNodeId(nextNodeId);
  }, [fullClusterNodeIds, selectedNodeId]);

  const closeCluster3d = useCallback(() => {
    setCluster3dNodeId(null);
  }, []);

  const graphForDisplay = useMemo(
    () => filterGraphByExpansion(graphForDetail, graphNavigation, expandedGraphNodeIds, graphScope),
    [expandedGraphNodeIds, graphForDetail, graphNavigation, graphScope],
  );

  const [layoutGraph, setLayoutGraph] = useState<LayoutGraphFn | null>(null);
  const hasGraphToLayout = Boolean(graphForDisplay?.nodes.length);
  useEffect(() => {
    if (!hasGraphToLayout || layoutGraph) {
      return;
    }
    let active = true;
    void loadLayoutGraph().then((loadedLayoutGraph) => {
      if (active) {
        setLayoutGraph(() => loadedLayoutGraph);
      }
    });
    return () => {
      active = false;
    };
  }, [hasGraphToLayout, layoutGraph]);

  // Structural layout pass: runs the dagre layout. Deliberately excludes
  // selectedNodeId so that selecting a node does NOT trigger a relayout. The
  // per-node handlers are useCallback-stable (keyed on navigation/scope), so
  // keeping them here does not couple layout to selection.
  const baseLayout = useMemo(() => {
    if (!layoutGraph) {
      return EMPTY_GRAPH_LAYOUT;
    }
    return layoutGraph(graphForDisplay, {
      graphDetailMode,
      navigation: graphNavigation,
      childSummariesById,
      childSummarySourceNodesById,
      expertDiagramsById,
      stackDiagramsById,
      clusterDiagramsById,
      expandedGraphNodeIds,
      expandedDetailNodeIds,
      enableExpansion: graphScope === "opened",
      selectedNodeId: null,
      canOpenMonitor: canOpenMonitor ? canOpenGraphNodeMonitor : undefined,
      parameterActivityForNode,
      onActivateNode: activateGraphNode,
      onToggleExpansion: toggleGraphNodeExpansion,
      onOpenMonitor: onOpenMonitor ? openGraphNodeMonitor : undefined,
      onToggleDetails: toggleNodeDetails,
    });
  }, [
    layoutGraph,
    activateGraphNode,
    expandedDetailNodeIds,
    expandedGraphNodeIds,
    graphDetailMode,
    graphForDisplay,
    graphNavigation,
    graphScope,
    canOpenMonitor,
    canOpenGraphNodeMonitor,
    parameterActivityForNode,
    openGraphNodeMonitor,
    onOpenMonitor,
    childSummariesById,
    childSummarySourceNodesById,
    clusterDiagramsById,
    expertDiagramsById,
    stackDiagramsById,
    toggleGraphNodeExpansion,
    toggleNodeDetails,
  ]);

  const edges = baseLayout.edges;

  // Cheap decoration pass: apply the `selected` flag without relayout. Only the
  // node whose selection changed gets a new object; all others keep their
  // reference so React Flow and the memoized node view skip re-rendering them.
  const nodes = useMemo(
    () =>
      baseLayout.nodes.map((node) =>
        node.selected === (node.id === selectedNodeId)
          ? node
          : { ...node, selected: node.id === selectedNodeId },
      ),
    [baseLayout, selectedNodeId],
  );

  const selectedNode = useMemo(
    () =>
      graphForDisplay?.nodes.find((node) => node.id === selectedNodeId) ??
      graphForDisplay?.nodes[0],
    [graphForDisplay, selectedNodeId],
  );

  useEffect(() => {
    if (!selectedNodeId || !graphForDisplay) {
      return;
    }
    if (!graphForDisplay.nodes.some((node) => node.id === selectedNodeId)) {
      setSelectedNodeId(null);
    }
  }, [graphForDisplay, selectedNodeId]);

  useEffect(() => {
    if (!cluster3dNodeId) {
      return;
    }
    if (!fullClusterNodeIds.has(cluster3dNodeId)) {
      setCluster3dNodeId(null);
      return;
    }
    if (!selectedNodeId) {
      setCluster3dNodeId(null);
      return;
    }
    if (
      selectedNodeId !== cluster3dNodeId &&
      !ancestorNodeIds(selectedNodeId, fullGraphNavigation).includes(cluster3dNodeId)
    ) {
      setCluster3dNodeId(null);
    }
  }, [cluster3dNodeId, fullClusterNodeIds, fullGraphNavigation, selectedNodeId]);

  const collapseGraphNodes = useCallback(() => {
    setExpandedGraphNodeIds(new Set());
    setSelectedNodeId(null);
    setCluster3dNodeId(null);
  }, []);

  const resetGraphExpansion = useCallback(() => {
    setExpandedGraphNodeIds(new Set());
    setExpandedDetailNodeIds(new Set());
  }, []);

  const resetGraphSelectionAndExpansion = useCallback(() => {
    setSelectedNodeId(null);
    setCluster3dNodeId(null);
    resetGraphExpansion();
  }, [resetGraphExpansion]);
  const appliedInspectionTransitionRevisionRef = useRef(
    inspectionTransition?.revision ?? 0,
  );
  useEffect(() => {
    const revision = inspectionTransition?.revision ?? 0;
    if (revision === appliedInspectionTransitionRevisionRef.current) {
      return;
    }
    appliedInspectionTransitionRevisionRef.current = revision;
    resetGraphSelectionAndExpansion();
  }, [inspectionTransition, resetGraphSelectionAndExpansion]);

  return {
    graphDetailMode,
    setGraphDetailMode,
    graphScope,
    setGraphScope,
    expandedGraphNodeIds,
    selectedNodeId,
    setSelectedNodeId: setSelectedNodeId as (nodeId: string | null) => void,
    cluster3dNodeId,
    openCluster3d,
    closeCluster3d,
    graphForDetail,
    nodes,
    edges,
    selectedNode: selectedNode as GraphNode | undefined,
    revealGraphNode,
    revealGraphNodeInFull,
    collapseGraphNodes,
    clearForConnectionChange: resetGraphSelectionAndExpansion,
  };
}
