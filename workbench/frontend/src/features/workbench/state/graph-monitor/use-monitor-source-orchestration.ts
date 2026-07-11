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
} from "@/lib/api";
import { monitorQueryKeys } from "@/lib/query-keys";
import {
  deriveMonitorSource,
  deriveParameterActivityByNodePath,
  deriveParameterStatusPathMismatch,
} from "@/features/workbench/state/graph-monitor/graph-monitor-selectors";
import { type ActiveMonitorJob } from "@/types/monitor";

const runningTrainingStatuses = new Set(["running", "queued"]);
const PARAMETER_STATUS_STALE_TIME_MS = 5 * 60_000;

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
  const activeParameterStatusEnabled = Boolean(
    parameterStatusActiveJob?.monitors.includes("linear"),
  );
  const historicalParameterStatusEnabled = Boolean(
    !parameterStatusActiveJob && parameterStatusHistoricalRunIds.length > 0,
  );
  const activeParameterStatusQuery = useQuery<ParameterStatus>({
    queryKey: parameterStatusActiveJob
      ? monitorQueryKeys.activeJobParameterStatus(
          parameterStatusActiveJob.id,
          parameterStatusPreset || undefined,
          parameterStatusDataset || undefined,
        )
      : (["monitor-parameter-status", "inactive-active-job"] as const),
    queryFn: ({ signal }) => {
      if (!parameterStatusActiveJob) {
        throw new Error("No active training job selected");
      }
      return fetchMonitorParameterStatus(
        {
          jobId: parameterStatusActiveJob.id,
          preset: parameterStatusPreset || undefined,
          dataset: parameterStatusDataset || undefined,
        },
        { signal },
      );
    },
    enabled: protectedReadsEnabled && activeParameterStatusEnabled,
    retry: false,
    staleTime: PARAMETER_STATUS_STALE_TIME_MS,
    refetchInterval:
      parameterStatusActiveJob &&
      runningTrainingStatuses.has(parameterStatusActiveJob.status)
        ? 1500
        : false,
  });
  const historicalParameterStatusQuery = useQuery<LogParameterStatusResponse>({
    queryKey: monitorQueryKeys.historicalParameterStatus(
      parameterStatusHistoricalRunIds,
    ),
    queryFn: ({ signal }) =>
      fetchLogParameterStatus(
        { runIds: parameterStatusHistoricalRunIds },
        { signal },
      ),
    enabled: protectedReadsEnabled && historicalParameterStatusEnabled,
    retry: false,
    staleTime: PARAMETER_STATUS_STALE_TIME_MS,
  });

  const historicalParameterStatusData:
    | LogParameterStatusResponse
    | undefined = !parameterStatusActiveJob
    ? historicalParameterStatusQuery.data
    : undefined;
  const parameterStatusData: ParameterStatus | LogParameterStatusResponse | undefined =
    parameterStatusActiveJob
      ? activeParameterStatusQuery.data
      : historicalParameterStatusData &&
          historicalParameterStatusData.runs.length > 0
        ? historicalParameterStatusData
        : undefined;
  const isHistoricalParameterStatusLoading = Boolean(
    !parameterStatusActiveJob &&
      historicalParameterStatusEnabled &&
      !historicalParameterStatusQuery.data &&
      (historicalParameterStatusQuery.isFetching ||
        historicalParameterStatusQuery.isLoading),
  );
  const isActiveParameterStatusLoading = Boolean(
    activeParameterStatusEnabled &&
      !activeParameterStatusQuery.data &&
      (activeParameterStatusQuery.isFetching ||
        activeParameterStatusQuery.isLoading),
  );
  const isParameterStatusLoading =
    isActiveParameterStatusLoading || isHistoricalParameterStatusLoading;
  const parameterActivityByNodePath = useMemo(
    () =>
      deriveParameterActivityByNodePath({
        graph,
        source: parameterStatusSource,
        status: parameterStatusData,
        statusLoading: isParameterStatusLoading,
        linearMonitorTargetResolver,
      }),
    [
      graph,
      isParameterStatusLoading,
      linearMonitorTargetResolver,
      parameterStatusData,
      parameterStatusSource,
    ],
  );
  const isParameterStatusPathMismatch = useMemo(
    () =>
      deriveParameterStatusPathMismatch({
        graph,
        source: parameterStatusSource,
        status: parameterStatusData,
        statusLoading: isParameterStatusLoading,
        linearMonitorTargetResolver,
      }),
    [
      graph,
      isParameterStatusLoading,
      linearMonitorTargetResolver,
      parameterStatusData,
      parameterStatusSource,
    ],
  );
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
    parameterActivityByNodePath,
    isParameterStatusLoading,
    isParameterStatusPathMismatch,
    deriveSelectedMonitorSourceState,
  };
}
