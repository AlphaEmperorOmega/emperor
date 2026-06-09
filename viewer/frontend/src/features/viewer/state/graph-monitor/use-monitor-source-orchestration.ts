import { useCallback, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchLogParameterStatus,
  fetchMonitorParameterStatus,
  type GraphNode,
  type InspectResponse,
  type LogParameterStatusResponse,
  type LogRun,
  type LogRunTags,
  type ParameterStatus,
  type TrainingJob,
} from "@/lib/api";
import { monitorQueryKeys } from "@/lib/query-keys";
import {
  deriveMonitorSource,
  deriveParameterActivityByNodePath,
} from "@/features/viewer/state/graph-monitor/graph-monitor-selectors";

const runningTrainingStatuses = new Set(["running", "queued"]);

export type MonitorSourceOrchestrationInput = {
  graph?: InspectResponse;
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

export function useMonitorSourceOrchestration({
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
    graphMonitorSource: parameterStatusSource,
  } = baseMonitorSourceState;
  const parameterStatusActiveJob =
    parameterStatusSource?.kind === "active-job"
      ? parameterStatusSource.job
      : undefined;
  const parameterStatusHistoricalRuns = useMemo(() => {
    if (parameterStatusSource?.kind === "historical-run-group") {
      return parameterStatusSource.runs;
    }
    if (parameterStatusSource?.kind === "historical-run") {
      return [parameterStatusSource.run];
    }
    return [];
  }, [parameterStatusSource]);
  const parameterStatusHistoricalRunIds = useMemo(
    () => parameterStatusHistoricalRuns.map((run) => run.id),
    [parameterStatusHistoricalRuns],
  );
  const parameterStatusPreset =
    parameterStatusActiveJob?.currentPreset ??
    parameterStatusActiveJob?.preset ??
    targetPreset;
  const parameterStatusDataset =
    parameterStatusActiveJob?.currentDataset ??
    parameterStatusActiveJob?.datasets[0] ??
    targetDatasets[0];
  const parameterStatusQuery = useQuery<
    ParameterStatus | LogParameterStatusResponse
  >({
    queryKey: parameterStatusActiveJob
      ? monitorQueryKeys.activeJobParameterStatus(
          parameterStatusActiveJob.id,
          parameterStatusPreset || undefined,
          parameterStatusDataset || undefined,
        )
      : monitorQueryKeys.historicalParameterStatus(parameterStatusHistoricalRunIds),
    queryFn: () => {
      if (parameterStatusActiveJob) {
        return fetchMonitorParameterStatus({
          jobId: parameterStatusActiveJob.id,
          preset: parameterStatusPreset || undefined,
          dataset: parameterStatusDataset || undefined,
        });
      }
      return fetchLogParameterStatus({ runIds: parameterStatusHistoricalRunIds });
    },
    enabled: parameterStatusActiveJob
      ? parameterStatusActiveJob.monitors.includes("linear")
      : parameterStatusHistoricalRunIds.length > 0,
    retry: false,
    refetchInterval:
      parameterStatusActiveJob &&
      runningTrainingStatuses.has(parameterStatusActiveJob.status)
        ? 1500
        : false,
  });
  const parameterActivityByNodePath = useMemo(
    () =>
      deriveParameterActivityByNodePath({
        graph,
        source: parameterStatusSource,
        status: parameterStatusQuery.data,
        linearMonitorTargetResolver,
      }),
    [
      graph,
      linearMonitorTargetResolver,
      parameterStatusQuery.data,
      parameterStatusSource,
    ],
  );
  const resolveMonitorTargetNode = useCallback(
    (node: GraphNode) => monitorTargetResolver(node)?.node,
    [monitorTargetResolver],
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
    canOpenGraphNodeMonitor,
    parameterActivityByNodePath,
    deriveSelectedMonitorSourceState,
  };
}
