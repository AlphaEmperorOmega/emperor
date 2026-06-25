import {
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
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
  targetModelType: string;
  targetModel: string;
  targetPreset: string;
  targetDatasets: string[];
  targetMode: "preset" | "snapshot" | "experiment";
  targetId: string;
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
    previewRequest,
    previewRequestKey,
    clearPreview,
    requestPreview,
    previewInspection,
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
      previewRequest,
      previewRequestKey,
      clearPreview,
      requestPreview,
      previewInspection,
      resetGraphSelectionAndExpansion,
      resetGraphExpansion,
      bindGraphResetHandlers,
    }),
    [
      bindGraphResetHandlers,
      clearPreview,
      graph,
      previewInspection,
      previewRequest,
      previewRequestKey,
      requestPreview,
      resetGraphExpansion,
      resetGraphSelectionAndExpansion,
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
  targetModelType,
  targetModel,
  targetPreset,
  targetDatasets,
  targetMode,
  targetId,
}: GraphPreviewOrchestrationInput) {
  const {
    graph,
    previewInspection,
    previewRequest,
    bindGraphResetHandlers,
  } = controller;
  const targetDataset = targetDatasets[0] ?? "";
  const previewRequestMatchesTarget =
    !previewRequest ||
    (previewRequest.modelType === targetModelType &&
      previewRequest.model === targetModel &&
      previewRequest.preset === targetPreset &&
      (previewRequest.dataset ?? "") === targetDataset &&
      (previewRequest.targetMode ?? "preset") === targetMode &&
      (previewRequest.targetId ?? previewRequest.preset) === targetId &&
      (targetMode !== "experiment" || previewRequest.logRunId === targetId));
  const targetGraph =
    graph &&
    graph.modelType === targetModelType &&
    graph.model === targetModel &&
    graph.preset === targetPreset &&
    previewRequestMatchesTarget
      ? graph
      : undefined;
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
    resolveParameterActivityTargetNode,
    canOpenGraphNodeMonitor,
    parameterActivityByNodePath,
    isParameterStatusPartiallyLoading,
    deriveSelectedMonitorSourceState,
  } = monitorSource;
  const graphState = useGraphViewState(targetGraph, {
    canOpenMonitor: canOpenGraphNodeMonitor,
    onOpenMonitor: openGraphNodeMonitor,
    resolveMonitorTarget: resolveMonitorTargetNode,
    resolveParameterActivityTarget: resolveParameterActivityTargetNode,
    parameterActivityByNodePath,
  });
  const {
    resetGraphExpansion,
    resetGraphSelectionAndExpansion,
  } = graphState;
  useLayoutEffect(() => {
    bindGraphResetHandlers({
      resetGraphSelectionAndExpansion: () => {
        resetGraphSelectionAndExpansion();
      },
      resetGraphExpansion: () => {
        resetGraphExpansion();
      },
    });
  }, [
    bindGraphResetHandlers,
    resetGraphExpansion,
    resetGraphSelectionAndExpansion,
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
      graphForDetail: graphState.graphForDetail,
      nodes: graphState.nodes,
      edges: graphState.edges,
      graphDetailMode: graphState.graphDetailMode,
      setGraphDetailMode: graphState.setGraphDetailMode,
      graphScope: graphState.graphScope,
      setGraphScope: graphState.setGraphScope,
      expandedGraphNodeIds: graphState.expandedGraphNodeIds,
      selectedNodeId: graphState.selectedNodeId,
      setSelectedNodeId: graphState.setSelectedNodeId,
      cluster3dNodeId: graphState.cluster3dNodeId,
      openCluster3d: graphState.openCluster3d,
      closeCluster3d: graphState.closeCluster3d,
      selectedNode: graphState.selectedNode,
      selectedMonitorNode,
      selectedMonitorComparisonCandidateGroups,
      isParameterStatusPartiallyLoading,
      collapseGraphNodes: graphState.collapseGraphNodes,
      revealGraphNode: graphState.revealGraphNode,
      revealGraphNodeInFull: graphState.revealGraphNodeInFull,
      previewInspection,
    }),
    [
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
      graphState.revealGraphNode,
      graphState.revealGraphNodeInFull,
      graphState.selectedNode,
      graphState.selectedNodeId,
      graphState.setGraphDetailMode,
      graphState.setGraphScope,
      graphState.setSelectedNodeId,
      isParameterStatusPartiallyLoading,
      previewInspection,
      selectedMonitorComparisonCandidateGroups,
      selectedMonitorNode,
      targetGraph,
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
