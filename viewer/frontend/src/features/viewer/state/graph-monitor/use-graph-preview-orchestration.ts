import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  type LogRun,
  type LogRunTags,
  type TrainingJob,
} from "@/lib/api";
import { useGraphViewState } from "@/features/viewer/state/graph-monitor/use-graph-view-state";
import {
  usePreviewInspectionState,
} from "@/features/viewer/state/graph-monitor/use-preview-inspection";
import {
  useOperationGraphViewState,
} from "@/features/viewer/state/graph-monitor/use-operation-graph-view-state";
import {
  useMonitorSourceOrchestration,
} from "@/features/viewer/state/graph-monitor/use-monitor-source-orchestration";

type GraphResetHandlers = {
  resetGraphSelectionAndExpansion: () => void;
  resetGraphExpansion: () => void;
};

type GraphPreviewController = ReturnType<typeof useGraphPreviewController>;

type GraphPreviewOrchestrationInput = {
  controller: GraphPreviewController;
  activeTrainingJob: TrainingJob | undefined;
  historicalMonitorRuns: LogRun[];
  selectedHistoricalExperiment: string;
  selectedHistoricalDataset: string;
  selectedHistoricalPreset: string;
  logRunTags?: LogRunTags[];
  filteredHistoricalRunIds: string[];
  targetModel: string;
  targetPreset: string;
  targetDatasets: string[];
};

export function useGraphPreviewController() {
  const {
    graph,
    operationGraph,
    previewRequest,
    previewRequestKey,
    operationGraphRequestKey,
    operationGraphInFlightRequestKey,
    operationGraphFailedRequestKey,
    clearPreview,
    requestPreview,
    requestOperationGraph,
    resetOperationGraphFailure,
    previewInspection,
    operationInspection,
  } = usePreviewInspectionState();
  const graphResetHandlersRef = useRef<GraphResetHandlers>({
    resetGraphSelectionAndExpansion: () => {},
    resetGraphExpansion: () => {},
  });
  const resetGraphSelectionAndExpansion = useCallback(() => {
    graphResetHandlersRef.current.resetGraphSelectionAndExpansion();
  }, []);
  const resetGraphExpansion = useCallback(() => {
    graphResetHandlersRef.current.resetGraphExpansion();
  }, []);
  const bindGraphResetHandlers = useCallback((handlers: GraphResetHandlers) => {
    graphResetHandlersRef.current = handlers;
  }, []);

  return {
    graph,
    operationGraph,
    previewRequest,
    previewRequestKey,
    operationGraphRequestKey,
    operationGraphInFlightRequestKey,
    operationGraphFailedRequestKey,
    clearPreview,
    requestPreview,
    requestOperationGraph,
    resetOperationGraphFailure,
    previewInspection,
    operationInspection,
    resetGraphSelectionAndExpansion,
    resetGraphExpansion,
    bindGraphResetHandlers,
  };
}

export function useGraphPreviewOrchestration({
  controller,
  activeTrainingJob,
  historicalMonitorRuns,
  selectedHistoricalExperiment,
  selectedHistoricalDataset,
  selectedHistoricalPreset,
  logRunTags,
  filteredHistoricalRunIds,
  targetModel,
  targetPreset,
  targetDatasets,
}: GraphPreviewOrchestrationInput) {
  const { graph, previewInspection, bindGraphResetHandlers } = controller;
  const {
    operationGraph,
    previewRequest,
    previewRequestKey,
    operationGraphRequestKey,
    operationGraphInFlightRequestKey,
    operationGraphFailedRequestKey,
    requestOperationGraph,
    resetOperationGraphFailure,
    operationInspection,
  } = controller;
  const targetGraph =
    graph && graph.model === targetModel && graph.preset === targetPreset
      ? graph
      : undefined;
  const targetOperationGraph =
    operationGraph &&
    operationGraph.model === targetModel &&
    operationGraph.preset === targetPreset
      ? operationGraph
      : undefined;
  const [graphKind, setGraphKind] = useState<"module" | "operation">("module");
  const monitorSource = useMonitorSourceOrchestration({
    graph: targetGraph,
    activeTrainingJob,
    historicalMonitorRuns,
    selectedHistoricalExperiment,
    selectedHistoricalDataset,
    selectedHistoricalPreset,
    logRunTags,
    filteredHistoricalRunIds,
    targetPreset,
    targetDatasets,
  });
  const {
    graphMonitorNode,
    openGraphNodeMonitor,
    closeGraphNodeMonitor,
    resolveMonitorTargetNode,
    canOpenGraphNodeMonitor,
    parameterActivityByNodePath,
    deriveSelectedMonitorSourceState,
  } = monitorSource;
  const graphState = useGraphViewState(targetGraph, {
    canOpenMonitor: canOpenGraphNodeMonitor,
    onOpenMonitor: openGraphNodeMonitor,
    resolveMonitorTarget: resolveMonitorTargetNode,
    parameterActivityByNodePath,
  });
  const {
    closeCluster3d,
    previewVisualizationMode,
  } = graphState;
  const isOperationMode =
    previewVisualizationMode === "graph" && graphKind === "operation";
  const operationGraphState = useOperationGraphViewState(targetOperationGraph);
  const {
    resetGraphExpansion,
    resetGraphSelectionAndExpansion,
  } = graphState;
  const {
    resetOperationGraphSelectionAndExpansion,
  } = operationGraphState;

  useEffect(() => {
    if (previewVisualizationMode === "graph" && graphKind === "module") {
      return;
    }
    closeCluster3d();
  }, [closeCluster3d, graphKind, previewVisualizationMode]);

  useEffect(() => {
    if (!isOperationMode || !previewRequest || !previewRequestKey) {
      return;
    }
    if (operationGraphRequestKey === previewRequestKey) {
      return;
    }
    if (operationGraphInFlightRequestKey === previewRequestKey) {
      return;
    }
    if (operationGraphFailedRequestKey === previewRequestKey) {
      return;
    }
    if (operationInspection.isBuilding) {
      return;
    }
    requestOperationGraph(previewRequest);
  }, [
    isOperationMode,
    operationGraphFailedRequestKey,
    operationGraphInFlightRequestKey,
    operationGraphRequestKey,
    operationInspection.isBuilding,
    previewRequest,
    previewRequestKey,
    requestOperationGraph,
  ]);
  useEffect(() => {
    if (isOperationMode || !operationGraphFailedRequestKey) {
      return;
    }
    resetOperationGraphFailure(operationGraphFailedRequestKey);
  }, [
    isOperationMode,
    operationGraphFailedRequestKey,
    resetOperationGraphFailure,
  ]);
  useLayoutEffect(() => {
    bindGraphResetHandlers({
      resetGraphSelectionAndExpansion: () => {
        resetGraphSelectionAndExpansion();
        resetOperationGraphSelectionAndExpansion();
      },
      resetGraphExpansion: () => {
        resetGraphExpansion();
        resetOperationGraphSelectionAndExpansion();
      },
    });
  }, [
    bindGraphResetHandlers,
    resetGraphExpansion,
    resetGraphSelectionAndExpansion,
    resetOperationGraphSelectionAndExpansion,
  ]);

  const monitorSourceState = useMemo(
    () => deriveSelectedMonitorSourceState(graphState.selectedNode),
    [deriveSelectedMonitorSourceState, graphState.selectedNode],
  );
  const {
    selectedMonitorNode,
    selectedMonitorComparisonCandidateGroups,
    selectedLogRunHasMonitorTags,
    graphMonitorComparisonCandidateGroups,
    graphMonitorSource,
  } = monitorSourceState;

  return {
    graph: {
      graph: targetGraph,
      graphKind,
      setGraphKind,
      operationGraph: targetOperationGraph,
      operationInspection,
      graphForDetail: graphState.graphForDetail,
      nodes: graphState.nodes,
      edges: graphState.edges,
      operationNodes: operationGraphState.operationNodes,
      operationEdges: operationGraphState.operationEdges,
      previewVisualizationMode,
      setPreviewVisualizationMode: graphState.setPreviewVisualizationMode,
      graphDetailMode: graphState.graphDetailMode,
      setGraphDetailMode: graphState.setGraphDetailMode,
      graphScope: graphState.graphScope,
      setGraphScope: graphState.setGraphScope,
      operationGraphScope: operationGraphState.operationGraphScope,
      setOperationGraphScope: operationGraphState.setOperationGraphScope,
      expandedGraphNodeIds: graphState.expandedGraphNodeIds,
      expandedOperationGroupIds: operationGraphState.expandedOperationGroupIds,
      selectedNodeId: graphState.selectedNodeId,
      setSelectedNodeId: graphState.setSelectedNodeId,
      cluster3dNodeId: graphState.cluster3dNodeId,
      openCluster3d: graphState.openCluster3d,
      closeCluster3d: graphState.closeCluster3d,
      selectedOperationNodeId: operationGraphState.selectedOperationNodeId,
      setSelectedOperationNodeId: operationGraphState.setSelectedOperationNodeId,
      parameterFocusNodeId: graphState.parameterFocusNodeId,
      setParameterFocusNodeId: graphState.setParameterFocusNodeId,
      selectedNode: graphState.selectedNode,
      selectedOperationNode: operationGraphState.selectedOperationNode,
      selectedMonitorNode,
      selectedMonitorComparisonCandidateGroups,
      collapseGraphNodes: graphState.collapseGraphNodes,
      collapseOperationGraphNodes:
        operationGraphState.collapseOperationGraphNodes,
      revealGraphNode: graphState.revealGraphNode,
      revealGraphNodeInFull: graphState.revealGraphNodeInFull,
      previewInspection,
    },
    history: {
      selectedLogRunHasMonitorTags,
    },
    training: {
      graphMonitorNode,
      openGraphNodeMonitor,
      closeGraphNodeMonitor,
      graphMonitorSource,
      graphMonitorComparisonCandidateGroups,
    },
  };
}
