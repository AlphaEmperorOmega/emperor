import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
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
import { type ActiveMonitorJob } from "@/types/monitor";

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

type StableActiveMonitorJobRef = {
  signature: string;
  job: ActiveMonitorJob | undefined;
};

function activeMonitorJobSignature(job: TrainingJob | undefined) {
  if (!job) {
    return "inactive";
  }
  return JSON.stringify({
    id: job.id,
    status: job.status,
    monitors: job.monitors,
    preset: job.preset,
    presets: job.presets,
    datasets: job.datasets,
    logFolder: job.logFolder,
    currentPreset: job.currentPreset ?? null,
    currentDataset: job.currentDataset ?? null,
  });
}

function activeMonitorJobValue(job: TrainingJob): ActiveMonitorJob {
  return {
    id: job.id,
    status: job.status,
    monitors: job.monitors,
    preset: job.preset,
    presets: job.presets,
    datasets: job.datasets,
    logFolder: job.logFolder,
    currentPreset: job.currentPreset,
    currentDataset: job.currentDataset,
  };
}

function useStableActiveMonitorJob(job: TrainingJob | undefined) {
  const signature = activeMonitorJobSignature(job);
  const stableJobRef = useRef<StableActiveMonitorJobRef | null>(null);
  if (!stableJobRef.current || stableJobRef.current.signature !== signature) {
    stableJobRef.current = {
      signature,
      job: job ? activeMonitorJobValue(job) : undefined,
    };
  }
  return stableJobRef.current.job;
}

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

  return useMemo(
    () => ({
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
    }),
    [
      bindGraphResetHandlers,
      clearPreview,
      graph,
      operationGraph,
      operationGraphFailedRequestKey,
      operationGraphInFlightRequestKey,
      operationGraphRequestKey,
      operationInspection,
      previewInspection,
      previewRequest,
      previewRequestKey,
      requestOperationGraph,
      requestPreview,
      resetGraphExpansion,
      resetGraphSelectionAndExpansion,
      resetOperationGraphFailure,
    ],
  );
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
  const activeMonitorJob = useStableActiveMonitorJob(activeTrainingJob);
  const monitorSource = useMonitorSourceOrchestration({
    graph: targetGraph,
    activeTrainingJob: activeMonitorJob,
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
    isParameterStatusPartiallyLoading,
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

  const graphSlice = useMemo(
    () => ({
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
      isParameterStatusPartiallyLoading,
      collapseGraphNodes: graphState.collapseGraphNodes,
      collapseOperationGraphNodes:
        operationGraphState.collapseOperationGraphNodes,
      revealGraphNode: graphState.revealGraphNode,
      revealGraphNodeInFull: graphState.revealGraphNodeInFull,
      previewInspection,
    }),
    [
      graphKind,
      graphState.cluster3dNodeId,
      graphState.closeCluster3d,
      graphState.collapseGraphNodes,
      graphState.edges,
      graphState.expandedGraphNodeIds,
      graphState.graphDetailMode,
      graphState.graphForDetail,
      graphState.graphScope,
      graphState.nodes,
      graphState.openCluster3d,
      graphState.parameterFocusNodeId,
      graphState.revealGraphNode,
      graphState.revealGraphNodeInFull,
      graphState.selectedNode,
      graphState.selectedNodeId,
      graphState.setGraphDetailMode,
      graphState.setGraphScope,
      graphState.setParameterFocusNodeId,
      graphState.setPreviewVisualizationMode,
      graphState.setSelectedNodeId,
      isParameterStatusPartiallyLoading,
      operationGraphState.collapseOperationGraphNodes,
      operationGraphState.expandedOperationGroupIds,
      operationGraphState.operationEdges,
      operationGraphState.operationGraphScope,
      operationGraphState.operationNodes,
      operationGraphState.selectedOperationNode,
      operationGraphState.selectedOperationNodeId,
      operationGraphState.setOperationGraphScope,
      operationGraphState.setSelectedOperationNodeId,
      operationInspection,
      previewInspection,
      previewVisualizationMode,
      selectedMonitorComparisonCandidateGroups,
      selectedMonitorNode,
      targetGraph,
      targetOperationGraph,
    ],
  );
  const historySlice = useMemo(
    () => ({
      selectedLogRunHasMonitorTags,
    }),
    [selectedLogRunHasMonitorTags],
  );
  const graphMonitorSlice = useMemo(
    () => ({
      graphMonitorNode,
      openGraphNodeMonitor,
      closeGraphNodeMonitor,
      graphMonitorSource,
      graphMonitorComparisonCandidateGroups,
    }),
    [
      closeGraphNodeMonitor,
      graphMonitorComparisonCandidateGroups,
      graphMonitorNode,
      graphMonitorSource,
      openGraphNodeMonitor,
    ],
  );

  return useMemo(
    () => ({
      graph: graphSlice,
      history: historySlice,
      graphMonitor: graphMonitorSlice,
    }),
    [graphMonitorSlice, graphSlice, historySlice],
  );
}
