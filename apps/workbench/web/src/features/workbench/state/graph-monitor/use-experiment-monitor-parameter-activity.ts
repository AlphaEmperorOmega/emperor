import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  type LogParameterStatusResponse,
  type ParameterStatus,
} from "@/lib/api/monitor-data";
import { type InspectResponse } from "@/lib/api/inspection";
import { monitorQueryKeys } from "@/lib/query-keys";
import {
  deriveParameterActivityByNodePath,
  deriveParameterStatusPathMismatch,
} from "@/features/workbench/state/graph-monitor/graph-monitor-selectors";
import {
  createLinearMonitorTargetResolver,
  type LinearMonitorTargetResolver,
} from "@/lib/graph/monitor-targets";
import { type MonitorChartsSource } from "@/types/monitor";
import { createLazyFunction } from "@/lib/lazy-value";

const RUNNING_TRAINING_STATUSES = new Set(["running", "queued"]);
const PARAMETER_STATUS_STALE_TIME_MS = 5 * 60_000;
const INACTIVE_PARAMETER_STATUS_QUERY_KEY = [
  "experiment-monitor-parameter-activity",
  "inactive",
] as const;
type MonitorDataApi = typeof import("@/lib/api/monitor-data");
const fetchLogParameterStatus: MonitorDataApi["fetchLogParameterStatus"] =
  createLazyFunction(() =>
    import("@/lib/api/monitor-data").then(
      (module) => module.fetchLogParameterStatus,
    ),
  );
const fetchMonitorParameterStatus: MonitorDataApi["fetchMonitorParameterStatus"] =
  createLazyFunction(() =>
    import("@/lib/api/monitor-data").then(
      (module) => module.fetchMonitorParameterStatus,
    ),
  );

export type ExperimentMonitorParameterActivityInput = {
  graph?: InspectResponse;
  source?: MonitorChartsSource;
  enabled?: boolean;
  protectedReadsEnabled?: boolean;
  fallbackPreset?: string;
  fallbackDataset?: string;
  linearMonitorTargetResolver?: LinearMonitorTargetResolver;
};

function historicalRunIds(source: MonitorChartsSource | undefined) {
  if (source?.kind === "historical-run-group") {
    return source.runs.map((run) => run.id);
  }
  if (source?.kind === "historical-run") {
    return [source.run.id];
  }
  return [];
}

export function useExperimentMonitorParameterActivity({
  graph,
  source,
  enabled = true,
  protectedReadsEnabled = true,
  fallbackPreset = "",
  fallbackDataset = "",
  linearMonitorTargetResolver: suppliedLinearMonitorTargetResolver,
}: ExperimentMonitorParameterActivityInput) {
  const activeJob = source?.kind === "active-job" ? source.job : undefined;
  const historicalIds = useMemo(() => historicalRunIds(source), [source]);
  const preset =
    activeJob?.currentPreset ?? activeJob?.preset ?? fallbackPreset;
  const dataset =
    activeJob?.currentDataset ?? activeJob?.datasets[0] ?? fallbackDataset;
  const sourceReady = activeJob
    ? activeJob.monitors.includes("linear")
    : historicalIds.length > 0;
  const queryEnabled = protectedReadsEnabled && enabled && sourceReady;
  const statusQuery = useQuery<ParameterStatus | LogParameterStatusResponse>({
    queryKey: activeJob
      ? monitorQueryKeys.activeJobParameterStatus(
          activeJob.id,
          preset || undefined,
          dataset || undefined,
        )
      : historicalIds.length > 0
        ? monitorQueryKeys.historicalParameterStatus(historicalIds)
        : INACTIVE_PARAMETER_STATUS_QUERY_KEY,
    queryFn: ({ signal }) => {
      if (activeJob) {
        return fetchMonitorParameterStatus(
          {
            jobId: activeJob.id,
            preset: preset || undefined,
            dataset: dataset || undefined,
          },
          { signal },
        );
      }
      if (historicalIds.length > 0) {
        return fetchLogParameterStatus({ runIds: historicalIds }, { signal });
      }
      throw new Error("No Experiment Monitor parameter-activity source selected");
    },
    enabled: queryEnabled,
    retry: false,
    staleTime: PARAMETER_STATUS_STALE_TIME_MS,
    refetchInterval:
      activeJob && RUNNING_TRAINING_STATUSES.has(activeJob.status) ? 1500 : false,
  });
  const isLoading = Boolean(
    queryEnabled &&
      !statusQuery.data &&
      (statusQuery.isFetching || statusQuery.isLoading),
  );
  const linearMonitorTargetResolver = useMemo(
    () =>
      suppliedLinearMonitorTargetResolver ??
      createLinearMonitorTargetResolver(graph),
    [graph, suppliedLinearMonitorTargetResolver],
  );
  const activityByNodePath = useMemo(
    () =>
      deriveParameterActivityByNodePath({
        graph,
        source,
        status: statusQuery.data,
        statusLoading: isLoading,
        linearMonitorTargetResolver,
      }),
    [
      graph,
      isLoading,
      linearMonitorTargetResolver,
      source,
      statusQuery.data,
    ],
  );
  const isPathMismatch = useMemo(
    () =>
      deriveParameterStatusPathMismatch({
        graph,
        source,
        status: statusQuery.data,
        statusLoading: isLoading,
        linearMonitorTargetResolver,
      }),
    [
      graph,
      isLoading,
      linearMonitorTargetResolver,
      source,
      statusQuery.data,
    ],
  );

  return {
    activityByNodePath,
    isLoading,
    isPathMismatch,
    isError: statusQuery.isError,
    error: statusQuery.error,
  };
}
