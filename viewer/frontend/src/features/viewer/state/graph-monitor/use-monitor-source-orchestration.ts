import { useCallback, useEffect, useMemo, useState } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";
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
} from "@/features/viewer/state/graph-monitor/graph-monitor-selectors";
import { type ActiveMonitorJob } from "@/types/monitor";

const runningTrainingStatuses = new Set(["running", "queued"]);
const HISTORICAL_PARAMETER_STATUS_REQUEST_CONCURRENCY = 2;

type ProgressQuery = {
  isSuccess: boolean;
  isError: boolean;
  isFetching: boolean;
  isLoading: boolean;
};

function settledQueryCount(queries: ProgressQuery[]) {
  return queries.filter((query) => query.isSuccess || query.isError).length;
}

function isQueryWindowLoading({
  enabledCount,
  queries,
  total,
}: {
  enabledCount: number;
  queries: ProgressQuery[];
  total: number;
}) {
  const settled = settledQueryCount(queries);
  return (
    total > 0 &&
    settled < total &&
    (enabledCount < total ||
      queries.some((query) => query.isFetching || query.isLoading))
  );
}

export type MonitorSourceOrchestrationInput = {
  graph?: InspectResponse;
  activeTrainingJob: ActiveMonitorJob | undefined;
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
  const parameterStatusHistoricalRunKey = useMemo(
    () => parameterStatusHistoricalRunIds.join("\n"),
    [parameterStatusHistoricalRunIds],
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
  const [
    historicalParameterStatusEnabledCount,
    setHistoricalParameterStatusEnabledCount,
  ] = useState(() =>
    historicalParameterStatusEnabled
      ? Math.min(
          parameterStatusHistoricalRunIds.length,
          HISTORICAL_PARAMETER_STATUS_REQUEST_CONCURRENCY,
        )
      : 0,
  );

  useEffect(() => {
    setHistoricalParameterStatusEnabledCount(
      historicalParameterStatusEnabled
        ? Math.min(
            parameterStatusHistoricalRunIds.length,
            HISTORICAL_PARAMETER_STATUS_REQUEST_CONCURRENCY,
          )
        : 0,
    );
  }, [
    historicalParameterStatusEnabled,
    parameterStatusHistoricalRunIds.length,
    parameterStatusHistoricalRunKey,
  ]);

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
    enabled: activeParameterStatusEnabled,
    retry: false,
    refetchInterval:
      parameterStatusActiveJob &&
      runningTrainingStatuses.has(parameterStatusActiveJob.status)
        ? 1500
        : false,
  });
  const historicalParameterStatusQueries = useQueries({
    queries: parameterStatusHistoricalRuns.map((run, index) => ({
      queryKey: monitorQueryKeys.historicalParameterStatus([run.id]),
      queryFn: ({ signal }) =>
        fetchLogParameterStatus({ runIds: [run.id] }, { signal }),
      enabled:
        historicalParameterStatusEnabled &&
        index < historicalParameterStatusEnabledCount,
      retry: false,
    })),
  });
  const historicalParameterStatusSettledCount = settledQueryCount(
    historicalParameterStatusQueries,
  );

  useEffect(() => {
    if (!historicalParameterStatusEnabled) {
      return;
    }
    const nextEnabledCount = Math.min(
      parameterStatusHistoricalRunIds.length,
      historicalParameterStatusSettledCount +
        HISTORICAL_PARAMETER_STATUS_REQUEST_CONCURRENCY,
    );
    setHistoricalParameterStatusEnabledCount((current) =>
      nextEnabledCount > current ? nextEnabledCount : current,
    );
  }, [
    historicalParameterStatusEnabled,
    historicalParameterStatusSettledCount,
    parameterStatusHistoricalRunIds.length,
  ]);

  const historicalParameterStatusData:
    | LogParameterStatusResponse
    | undefined = !parameterStatusActiveJob
    ? {
        runs: parameterStatusHistoricalRuns.flatMap((_, index) => {
          return historicalParameterStatusQueries[index]?.data?.runs ?? [];
        }),
      }
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
      isQueryWindowLoading({
        enabledCount: historicalParameterStatusEnabledCount,
        queries: historicalParameterStatusQueries,
        total: parameterStatusHistoricalRunIds.length,
      }),
  );
  const isActiveParameterStatusLoading = Boolean(
    activeParameterStatusEnabled &&
      !activeParameterStatusQuery.data &&
      (activeParameterStatusQuery.isFetching ||
        activeParameterStatusQuery.isLoading),
  );
  const isParameterStatusLoading =
    isActiveParameterStatusLoading || isHistoricalParameterStatusLoading;
  const isParameterStatusPartiallyLoading = Boolean(
    !parameterStatusActiveJob &&
      historicalParameterStatusData &&
      historicalParameterStatusData.runs.length > 0 &&
      isHistoricalParameterStatusLoading,
  );
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
    isParameterStatusPartiallyLoading,
    isParameterStatusPathMismatch,
    deriveSelectedMonitorSourceState,
  };
}
