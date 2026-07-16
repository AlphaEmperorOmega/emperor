import { useCallback, useMemo, useState } from "react";
import type { GraphNode, InspectResponse } from "@/lib/api/inspection";
import type { LogRun, LogRunTags } from "@/lib/api/logs";
import { deriveMonitorSource } from "@/features/workbench/state/graph-monitor/graph-monitor-selectors";
import { useExperimentMonitorParameterActivity } from "@/features/workbench/state/graph-monitor/use-experiment-monitor-parameter-activity";
import { type ActiveMonitorJob } from "@/types/monitor";

export type MonitorSourceOrchestrationInput = {
  graph?: InspectResponse;
  activeTrainingJob: ActiveMonitorJob | undefined;
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

export function useMonitorSourceOrchestration({
  graph,
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
}: MonitorSourceOrchestrationInput) {
  const [graphMonitorNode, setGraphMonitorNode] = useState<
    GraphNode | undefined
  >();
  const baseMonitorSourceState = useMemo(
    () =>
      deriveMonitorSource({
        graph,
        activeTrainingJob,
        historicalMonitorRuns,
        selectedHistoricalExperiment,
        selectedHistoricalDataset,
        selectedHistoricalPreset,
        logRunTags,
        filteredHistoricalRunIds,
      }),
    [
      activeTrainingJob,
      filteredHistoricalRunIds,
      graph,
      historicalMonitorRuns,
      logRunTags,
      selectedHistoricalDataset,
      selectedHistoricalExperiment,
      selectedHistoricalPreset,
    ],
  );
  const {
    monitorTargetResolver,
    linearMonitorTargetResolver,
    graphMonitorSource,
  } = baseMonitorSourceState;
  const parameterActivity = useExperimentMonitorParameterActivity({
    graph,
    source: graphMonitorSource,
    protectedReadsEnabled,
    fallbackPreset: targetPreset,
    fallbackDataset: targetDatasets[0],
    linearMonitorTargetResolver,
  });
  const resolveMonitorTargetNode = useCallback(
    (node: GraphNode) => monitorTargetResolver(node)?.node,
    [monitorTargetResolver],
  );
  const resolveParameterActivityTargetNode = useCallback(
    (node: GraphNode) => linearMonitorTargetResolver(node),
    [linearMonitorTargetResolver],
  );
  const canOpenGraphNodeMonitor = useCallback(
    (monitorTarget: GraphNode) => Boolean(monitorTargetResolver(monitorTarget)),
    [monitorTargetResolver],
  );
  const closeGraphNodeMonitor = useCallback(() => {
    setGraphMonitorNode(undefined);
  }, []);
  const deriveSelectedMonitorSourceState = useCallback(
    (selectedNode: GraphNode | undefined) =>
      deriveMonitorSource({
        graph,
        selectedNode,
        graphMonitorNode,
        activeTrainingJob,
        historicalMonitorRuns,
        selectedHistoricalExperiment,
        selectedHistoricalDataset,
        selectedHistoricalPreset,
        logRunTags,
        filteredHistoricalRunIds,
        monitorTargetResolver,
        linearMonitorTargetResolver,
      }),
    [
      activeTrainingJob,
      filteredHistoricalRunIds,
      graph,
      graphMonitorNode,
      historicalMonitorRuns,
      linearMonitorTargetResolver,
      logRunTags,
      monitorTargetResolver,
      selectedHistoricalDataset,
      selectedHistoricalExperiment,
      selectedHistoricalPreset,
    ],
  );

  return {
    graphMonitorNode,
    openGraphNodeMonitor: setGraphMonitorNode,
    closeGraphNodeMonitor,
    resolveMonitorTargetNode,
    resolveParameterActivityTargetNode,
    canOpenGraphNodeMonitor,
    parameterActivityByNodePath: parameterActivity.activityByNodePath,
    isParameterStatusLoading: parameterActivity.isLoading,
    isParameterStatusPathMismatch: parameterActivity.isPathMismatch,
    deriveSelectedMonitorSourceState,
  };
}
