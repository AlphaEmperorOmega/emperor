import { useCallback, useLayoutEffect, useMemo, useRef } from "react";
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
  targetPreset: string;
  targetDatasets: string[];
};

export function useGraphPreviewController() {
  const { graph, requestPreview, previewInspection } = usePreviewInspectionState();
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
    requestPreview,
    previewInspection,
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
  targetPreset,
  targetDatasets,
}: GraphPreviewOrchestrationInput) {
  const { graph, previewInspection, bindGraphResetHandlers } = controller;
  const monitorSource = useMonitorSourceOrchestration({
    graph,
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
  const graphState = useGraphViewState(graph, {
    canOpenMonitor: canOpenGraphNodeMonitor,
    onOpenMonitor: openGraphNodeMonitor,
    resolveMonitorTarget: resolveMonitorTargetNode,
    parameterActivityByNodePath,
  });
  useLayoutEffect(() => {
    bindGraphResetHandlers({
      resetGraphSelectionAndExpansion: graphState.resetGraphSelectionAndExpansion,
      resetGraphExpansion: graphState.resetGraphExpansion,
    });
  }, [
    bindGraphResetHandlers,
    graphState.resetGraphExpansion,
    graphState.resetGraphSelectionAndExpansion,
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
      graph,
      graphForDetail: graphState.graphForDetail,
      nodes: graphState.nodes,
      edges: graphState.edges,
      previewVisualizationMode: graphState.previewVisualizationMode,
      setPreviewVisualizationMode: graphState.setPreviewVisualizationMode,
      graphDetailMode: graphState.graphDetailMode,
      setGraphDetailMode: graphState.setGraphDetailMode,
      graphScope: graphState.graphScope,
      setGraphScope: graphState.setGraphScope,
      expandedGraphNodeIds: graphState.expandedGraphNodeIds,
      selectedNodeId: graphState.selectedNodeId,
      setSelectedNodeId: graphState.setSelectedNodeId,
      parameterTreemapFocusNodeId: graphState.parameterTreemapFocusNodeId,
      setParameterTreemapFocusNodeId: graphState.setParameterTreemapFocusNodeId,
      selectedNode: graphState.selectedNode,
      selectedMonitorNode,
      selectedMonitorComparisonCandidateGroups,
      collapseGraphNodes: graphState.collapseGraphNodes,
      revealGraphNode: graphState.revealGraphNode,
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
