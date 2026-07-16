import { useCallback, useMemo } from "react";
import type { InspectResponse } from "@/lib/api/inspection";
import type { LogRun, LogRunTags } from "@/lib/api/logs";
import type { TrainingJob } from "@/lib/api/training-jobs";
import { useGraphViewState } from "@/features/workbench/state/graph-monitor/use-graph-view-state";
import {
  useMonitorSourceOrchestration,
} from "@/features/workbench/state/graph-monitor/use-monitor-source-orchestration";
import { type ActiveMonitorJob } from "@/types/monitor";

type GraphPreviewOrchestrationInput = {
  inspection: {
    graph: InspectResponse | undefined;
    status: {
      isBuilding: boolean;
      isError: boolean;
      error: Error | null;
    };
    transition: {
      revision: number;
      cause: "target-changed" | "inspection-refreshed";
    };
  };
  activeTrainingJob: TrainingJob | undefined;
  protectedReadsEnabled?: boolean;
  historicalMonitorRuns: LogRun[];
  selectedHistoricalExperiment: string;
  selectedHistoricalDataset: string;
  selectedHistoricalPreset: string;
  logRunTags?: LogRunTags[];
  filteredHistoricalRunIds: string[];
  targetPreset: string;
  targetDatasets: string[];
};

function useStableActiveMonitorJob(job: TrainingJob | undefined) {
  const active = job !== undefined;
  const id = job?.id;
  const status = job?.status;
  const preset = job?.preset;
  const logFolder = job?.logFolder;
  const currentPreset = job?.currentPreset;
  const currentDataset = job?.currentDataset;
  const monitorsSignature = JSON.stringify(job?.monitors ?? []);
  const presetsSignature = JSON.stringify(job?.presets ?? []);
  const datasetsSignature = JSON.stringify(job?.datasets ?? []);
  return useMemo<ActiveMonitorJob | undefined>(
    () =>
      active
        ? {
            id: id!,
            status: status!,
            monitors: JSON.parse(monitorsSignature) as string[],
            preset: preset!,
            presets: JSON.parse(presetsSignature) as string[],
            datasets: JSON.parse(datasetsSignature) as string[],
            logFolder: logFolder!,
            currentPreset: currentPreset ?? null,
            currentDataset: currentDataset ?? null,
          }
        : undefined,
    [
      active,
      currentDataset,
      currentPreset,
      datasetsSignature,
      id,
      logFolder,
      monitorsSignature,
      preset,
      presetsSignature,
      status,
    ],
  );
}

export function useGraphPreviewOrchestration({
  inspection,
  activeTrainingJob,
  protectedReadsEnabled = true,
  historicalMonitorRuns,
  selectedHistoricalExperiment,
  selectedHistoricalDataset,
  selectedHistoricalPreset,
  logRunTags,
  filteredHistoricalRunIds,
  targetPreset,
  targetDatasets,
}: GraphPreviewOrchestrationInput) {
  const { graph, status: previewInspection, transition } = inspection;
  const targetGraph = graph;
  const activeMonitorJob = useStableActiveMonitorJob(activeTrainingJob);
  const monitorSource = useMonitorSourceOrchestration({
    graph: targetGraph,
    activeTrainingJob: activeMonitorJob,
    protectedReadsEnabled,
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
    isParameterStatusLoading,
    isParameterStatusPathMismatch,
    deriveSelectedMonitorSourceState,
  } = monitorSource;
  const graphState = useGraphViewState(targetGraph, {
    canOpenMonitor: canOpenGraphNodeMonitor,
    onOpenMonitor: openGraphNodeMonitor,
    resolveMonitorTarget: resolveMonitorTargetNode,
    resolveParameterActivityTarget: resolveParameterActivityTargetNode,
    parameterActivityByNodePath,
    inspectionTransition: transition,
  });
  const clearGraphViewForConnectionChange = graphState.clearForConnectionChange;
  const clearForConnectionChange = useCallback(() => {
    clearGraphViewForConnectionChange();
    closeGraphNodeMonitor();
  }, [clearGraphViewForConnectionChange, closeGraphNodeMonitor]);
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
      isParameterStatusLoading,
      isParameterStatusPathMismatch,
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
      isParameterStatusLoading,
      isParameterStatusPathMismatch,
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
      closeGraphNodeMonitor,
      graphMonitorSource,
      graphMonitorComparisonCandidateGroups,
    }),
    [
      closeGraphNodeMonitor,
      graphMonitorComparisonCandidateGroups,
      graphMonitorNode,
      graphMonitorSource,
    ],
  );

  return useMemo(
    () => ({
      graph: graphSlice,
      history: historySlice,
      graphMonitor: graphMonitorSlice,
      clearForConnectionChange,
    }),
    [clearForConnectionChange, graphMonitorSlice, graphSlice, historySlice],
  );
}
