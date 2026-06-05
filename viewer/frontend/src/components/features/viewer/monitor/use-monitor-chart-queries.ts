import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchLogRunMonitorData,
  fetchMonitorData,
  type MonitorData,
} from "@/lib/api";
import { hasHistoricalMonitorData, hasMonitorData } from "@/lib/monitor/grouping";
import { monitorQueryKeys } from "@/lib/query-keys";
import {
  type HistoricalMonitorRunData,
  type MonitorChartsSource,
  type MonitorQueryData,
} from "@/types/monitor";

const runningStatuses = new Set(["running", "queued"]);

type UseMonitorChartQueriesInput = {
  source: MonitorChartsSource;
  nodePath: string;
  dataset: string;
  preset: string;
  comparisonNodePath?: string;
};

export function useMonitorChartQueries({
  source,
  nodePath,
  dataset,
  preset,
  comparisonNodePath,
}: UseMonitorChartQueriesInput) {
  const activeJob = source.kind === "active-job" ? source.job : undefined;
  const historicalRun = source.kind === "historical-run" ? source.run : undefined;
  const historicalRunGroup =
    source.kind === "historical-run-group" ? source : undefined;
  const historicalRuns = useMemo(
    () => historicalRunGroup?.runs ?? (historicalRun ? [historicalRun] : []),
    [historicalRun, historicalRunGroup?.runs],
  );
  const historicalRunIds = useMemo(
    () => historicalRuns.map((run) => run.id),
    [historicalRuns],
  );
  const isRunning = activeJob ? runningStatuses.has(activeJob.status) : false;
  const monitorCount = activeJob?.monitors.length;
  const isComparing = Boolean(comparisonNodePath);

  const monitorQuery = useQuery<MonitorQueryData>({
    queryKey: activeJob
      ? monitorQueryKeys.activeJob(activeJob.id, nodePath, preset, dataset)
      : historicalRunGroup
        ? monitorQueryKeys.historicalRunGroup(historicalRunIds, nodePath)
        : monitorQueryKeys.historicalRun(historicalRun?.id, nodePath),
    queryFn: () => {
      if (activeJob) {
        return fetchMonitorData({
          jobId: activeJob.id,
          nodePath,
          preset: preset || undefined,
          dataset: dataset || undefined,
        });
      }
      if (historicalRunGroup) {
        return Promise.all(
          historicalRuns.map(async (run) => ({
            run,
            data: await fetchLogRunMonitorData({
              runId: run.id,
              nodePath,
            }),
          })),
        );
      }
      return fetchLogRunMonitorData({
        runId: historicalRun?.id ?? "",
        nodePath,
      });
    },
    enabled: activeJob
      ? activeJob.monitors.length > 0
      : historicalRunGroup
        ? historicalRuns.length > 0
        : Boolean(historicalRun),
    retry: false,
    refetchInterval: isRunning ? 1500 : false,
  });

  const comparisonQuery = useQuery<MonitorQueryData>({
    queryKey: activeJob
      ? monitorQueryKeys.activeJob(
          activeJob.id,
          comparisonNodePath,
          preset,
          dataset,
        )
      : historicalRunGroup
        ? monitorQueryKeys.historicalRunGroup(historicalRunIds, comparisonNodePath)
        : monitorQueryKeys.historicalRun(historicalRun?.id, comparisonNodePath),
    queryFn: () => {
      if (!comparisonNodePath) {
        throw new Error("No comparison node selected");
      }
      if (activeJob) {
        return fetchMonitorData({
          jobId: activeJob.id,
          nodePath: comparisonNodePath,
          preset: preset || undefined,
          dataset: dataset || undefined,
        });
      }
      if (historicalRunGroup) {
        return Promise.all(
          historicalRuns.map(async (run) => ({
            run,
            data: await fetchLogRunMonitorData({
              runId: run.id,
              nodePath: comparisonNodePath,
            }),
          })),
        );
      }
      return fetchLogRunMonitorData({
        runId: historicalRun?.id ?? "",
        nodePath: comparisonNodePath,
      });
    },
    enabled:
      Boolean(comparisonNodePath) &&
      (activeJob
        ? activeJob.monitors.length > 0
        : historicalRunGroup
          ? historicalRuns.length > 0
          : Boolean(historicalRun)),
    retry: false,
    refetchInterval: isRunning ? 1500 : false,
  });

  const monitorQueryData = monitorQuery.data;
  const comparisonQueryData = comparisonQuery.data;
  const historicalData: HistoricalMonitorRunData[] | undefined = Array.isArray(
    monitorQueryData,
  )
    ? monitorQueryData
    : undefined;
  const historicalComparisonData: HistoricalMonitorRunData[] | undefined =
    Array.isArray(comparisonQueryData) ? comparisonQueryData : undefined;
  const data: MonitorData | undefined = Array.isArray(monitorQueryData)
    ? undefined
    : monitorQueryData;
  const comparisonData: MonitorData | undefined = Array.isArray(comparisonQueryData)
    ? undefined
    : comparisonQueryData;
  const hasData = historicalRunGroup
    ? isComparing
      ? hasHistoricalMonitorData(historicalData) ||
        hasHistoricalMonitorData(historicalComparisonData)
      : hasHistoricalMonitorData(historicalData)
    : isComparing
      ? hasMonitorData(data) || hasMonitorData(comparisonData)
      : hasMonitorData(data);
  const isFetching = monitorQuery.isFetching || (isComparing && comparisonQuery.isFetching);
  const isLoading = monitorQuery.isLoading || (isComparing && comparisonQuery.isLoading);
  const comparisonLoading = comparisonQuery.isLoading || comparisonQuery.isFetching;
  const emptyMessage = useMemo(() => {
    if (monitorCount === 0) {
      return {
        title: "No monitor selected",
        detail: "Start a training job with at least one optional monitor enabled.",
      };
    }
    if (monitorQuery.isLoading || (isComparing && comparisonQuery.isLoading)) {
      return {
        title: "Loading monitor data",
        detail: "Reading TensorBoard event files for the selected node.",
      };
    }
    if (isRunning) {
      return {
        title: "No data yet",
        detail: "The job is running, but this node has not emitted matching monitor tags yet.",
      };
    }
    if (historicalRunGroup) {
      return {
        title: "No tags for this node",
        detail:
          "No scalar, histogram, or image tags matched this node path in the latest filtered historical runs.",
      };
    }
    if (historicalRun) {
      return {
        title: "No tags for this node",
        detail:
          "No scalar, histogram, or image tags matched this node path in the selected historical run.",
      };
    }
    return {
      title: "No tags for this node",
      detail: "No scalar, histogram, or image tags matched this node path in the selected run.",
    };
  }, [
    comparisonQuery.isLoading,
    historicalRunGroup,
    historicalRun,
    isComparing,
    isRunning,
    monitorCount,
    monitorQuery.isLoading,
  ]);

  const refetch = () => {
    void monitorQuery.refetch();
    if (isComparing) {
      void comparisonQuery.refetch();
    }
  };

  return {
    data,
    comparisonData,
    historicalData,
    historicalComparisonData,
    hasData,
    isComparing,
    isFetching,
    isLoading,
    comparisonLoading,
    emptyMessage,
    error: monitorQuery.error ?? (isComparing ? comparisonQuery.error : null),
    refetch,
  };
}
