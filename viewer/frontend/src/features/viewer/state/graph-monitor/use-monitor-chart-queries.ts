import { useEffect, useMemo, useState } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";
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
} from "@/types/monitor";

const runningStatuses = new Set(["running", "queued"]);
const HISTORICAL_MONITOR_REQUEST_CONCURRENCY = 2;

type QueryProgress = {
  loaded: number;
  failed: number;
  total: number;
  isLoading: boolean;
};

type ProgressQuery = {
  isSuccess: boolean;
  isError: boolean;
  isFetching: boolean;
  isLoading: boolean;
};

function settledQueryCount(queries: ProgressQuery[]) {
  return queries.filter((query) => query.isSuccess || query.isError).length;
}

function queryProgress({
  enabledCount,
  queries,
  total,
}: {
  enabledCount: number;
  queries: ProgressQuery[];
  total: number;
}): QueryProgress {
  const loaded = queries.filter((query) => query.isSuccess).length;
  const failed = queries.filter((query) => query.isError).length;
  const settled = loaded + failed;
  return {
    loaded,
    failed,
    total,
    isLoading:
      total > 0 &&
      settled < total &&
      (enabledCount < total ||
        queries.some((query) => query.isFetching || query.isLoading)),
  };
}

function dataWasTruncated(data: MonitorData | undefined) {
  return Boolean(
    data?.truncated ||
      data?.scalarSeries.some((series) => series.truncated) ||
      data?.histograms.some((histogram) => histogram.truncated) ||
      data?.images.some((image) => image.truncated),
  );
}

function historicalDataWasTruncated(results: HistoricalMonitorRunData[] | undefined) {
  return Boolean(results?.some((result) => dataWasTruncated(result.data)));
}

function truncationReason(data: MonitorData | undefined) {
  return (
    data?.truncationReason ??
    data?.scalarSeries.find((series) => series.truncationReason)?.truncationReason ??
    data?.histograms.find((histogram) => histogram.truncationReason)?.truncationReason ??
    data?.images.find((image) => image.truncationReason)?.truncationReason ??
    null
  );
}

function historicalTruncationReason(
  results: HistoricalMonitorRunData[] | undefined,
) {
  return (
    results
      ?.map((result) => truncationReason(result.data))
      .find((reason): reason is string => Boolean(reason)) ?? null
  );
}

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
  const historicalRunQueryKey = useMemo(
    () => `${nodePath}\n${historicalRunIds.join("\n")}`,
    [historicalRunIds, nodePath],
  );
  const historicalComparisonQueryKey = useMemo(
    () => `${comparisonNodePath ?? ""}\n${historicalRunIds.join("\n")}`,
    [comparisonNodePath, historicalRunIds],
  );
  const isRunning = activeJob ? runningStatuses.has(activeJob.status) : false;
  const monitorCount = activeJob?.monitors.length;
  const isComparing = Boolean(comparisonNodePath);
  const historicalRunGroupEnabled = Boolean(
    historicalRunGroup && historicalRuns.length > 0,
  );
  const historicalComparisonEnabled =
    historicalRunGroupEnabled && Boolean(comparisonNodePath);
  const [historicalEnabledCount, setHistoricalEnabledCount] = useState(() =>
    historicalRunGroupEnabled
      ? Math.min(historicalRuns.length, HISTORICAL_MONITOR_REQUEST_CONCURRENCY)
      : 0,
  );
  const [historicalComparisonEnabledCount, setHistoricalComparisonEnabledCount] =
    useState(() =>
      historicalComparisonEnabled
        ? Math.min(historicalRuns.length, HISTORICAL_MONITOR_REQUEST_CONCURRENCY)
        : 0,
    );

  useEffect(() => {
    setHistoricalEnabledCount(
      historicalRunGroupEnabled
        ? Math.min(historicalRuns.length, HISTORICAL_MONITOR_REQUEST_CONCURRENCY)
        : 0,
    );
  }, [historicalRunGroupEnabled, historicalRunQueryKey, historicalRuns.length]);

  useEffect(() => {
    setHistoricalComparisonEnabledCount(
      historicalComparisonEnabled
        ? Math.min(historicalRuns.length, HISTORICAL_MONITOR_REQUEST_CONCURRENCY)
        : 0,
    );
  }, [
    historicalComparisonEnabled,
    historicalComparisonQueryKey,
    historicalRuns.length,
  ]);

  const monitorQuery = useQuery<MonitorData>({
    queryKey: activeJob
      ? monitorQueryKeys.activeJob(activeJob.id, nodePath, preset, dataset)
      : monitorQueryKeys.historicalRun(historicalRun?.id, nodePath),
    queryFn: ({ signal }) => {
      if (activeJob) {
        return fetchMonitorData(
          {
            jobId: activeJob.id,
            nodePath,
            preset: preset || undefined,
            dataset: dataset || undefined,
          },
          { signal },
        );
      }
      return fetchLogRunMonitorData(
        {
          runId: historicalRun?.id ?? "",
          nodePath,
        },
        { signal },
      );
    },
    enabled: activeJob
      ? activeJob.monitors.length > 0
      : !historicalRunGroup && Boolean(historicalRun),
    retry: false,
    refetchInterval: isRunning ? 1500 : false,
  });

  const historicalMonitorQueries = useQueries({
    queries: historicalRuns.map((run, index) => ({
      queryKey: monitorQueryKeys.historicalRun(run.id, nodePath),
      queryFn: ({ signal }) =>
        fetchLogRunMonitorData(
          {
            runId: run.id,
            nodePath,
          },
          { signal },
        ),
      enabled: historicalRunGroupEnabled && index < historicalEnabledCount,
      retry: false,
    })),
  });
  const historicalSettledCount = settledQueryCount(historicalMonitorQueries);

  useEffect(() => {
    if (!historicalRunGroupEnabled) {
      return;
    }
    const nextEnabledCount = Math.min(
      historicalRuns.length,
      historicalSettledCount + HISTORICAL_MONITOR_REQUEST_CONCURRENCY,
    );
    setHistoricalEnabledCount((current) =>
      nextEnabledCount > current ? nextEnabledCount : current,
    );
  }, [
    historicalRunGroupEnabled,
    historicalRuns.length,
    historicalSettledCount,
  ]);

  const comparisonQuery = useQuery<MonitorData>({
    queryKey: activeJob
      ? monitorQueryKeys.activeJob(
          activeJob.id,
          comparisonNodePath,
          preset,
          dataset,
        )
      : monitorQueryKeys.historicalRun(historicalRun?.id, comparisonNodePath),
    queryFn: ({ signal }) => {
      if (!comparisonNodePath) {
        throw new Error("No comparison node selected");
      }
      if (activeJob) {
        return fetchMonitorData(
          {
            jobId: activeJob.id,
            nodePath: comparisonNodePath,
            preset: preset || undefined,
            dataset: dataset || undefined,
          },
          { signal },
        );
      }
      return fetchLogRunMonitorData(
        {
          runId: historicalRun?.id ?? "",
          nodePath: comparisonNodePath,
        },
        { signal },
      );
    },
    enabled:
      Boolean(comparisonNodePath) &&
      (activeJob
        ? activeJob.monitors.length > 0
        : !historicalRunGroup && Boolean(historicalRun)),
    retry: false,
    refetchInterval: isRunning ? 1500 : false,
  });

  const historicalComparisonQueries = useQueries({
    queries: historicalRuns.map((run, index) => ({
      queryKey: monitorQueryKeys.historicalRun(run.id, comparisonNodePath),
      queryFn: ({ signal }) =>
        fetchLogRunMonitorData(
          {
            runId: run.id,
            nodePath: comparisonNodePath ?? "",
          },
          { signal },
        ),
      enabled:
        historicalComparisonEnabled &&
        index < historicalComparisonEnabledCount,
      retry: false,
    })),
  });
  const historicalComparisonSettledCount = settledQueryCount(
    historicalComparisonQueries,
  );

  useEffect(() => {
    if (!historicalComparisonEnabled) {
      return;
    }
    const nextEnabledCount = Math.min(
      historicalRuns.length,
      historicalComparisonSettledCount +
        HISTORICAL_MONITOR_REQUEST_CONCURRENCY,
    );
    setHistoricalComparisonEnabledCount((current) =>
      nextEnabledCount > current ? nextEnabledCount : current,
    );
  }, [
    historicalComparisonEnabled,
    historicalComparisonSettledCount,
    historicalRuns.length,
  ]);

  const historicalData: HistoricalMonitorRunData[] | undefined = historicalRunGroup
    ? historicalRuns.flatMap((run, index) => {
        const data = historicalMonitorQueries[index]?.data;
        return data ? [{ run, data }] : [];
      })
    : undefined;
  const historicalComparisonData: HistoricalMonitorRunData[] | undefined =
    historicalRunGroup && comparisonNodePath
      ? historicalRuns.flatMap((run, index) => {
          const data = historicalComparisonQueries[index]?.data;
          return data ? [{ run, data }] : [];
        })
      : undefined;
  const data: MonitorData | undefined = historicalRunGroup
    ? undefined
    : monitorQuery.data;
  const comparisonData: MonitorData | undefined = historicalRunGroup
    ? undefined
    : comparisonQuery.data;
  const historicalProgress = historicalRunGroup
    ? queryProgress({
        enabledCount: historicalEnabledCount,
        queries: historicalMonitorQueries,
        total: historicalRuns.length,
      })
    : null;
  const historicalComparisonProgress =
    historicalRunGroup && comparisonNodePath
      ? queryProgress({
          enabledCount: historicalComparisonEnabledCount,
          queries: historicalComparisonQueries,
          total: historicalRuns.length,
        })
      : null;
  const hasData = historicalRunGroup
    ? isComparing
      ? hasHistoricalMonitorData(historicalData) ||
        hasHistoricalMonitorData(historicalComparisonData)
      : hasHistoricalMonitorData(historicalData)
    : isComparing
      ? hasMonitorData(data) || hasMonitorData(comparisonData)
      : hasMonitorData(data);
  const historicalFetching = Boolean(
    historicalProgress?.isLoading ||
      historicalMonitorQueries.some((query) => query.isFetching),
  );
  const historicalComparisonFetching = Boolean(
    historicalComparisonProgress?.isLoading ||
      historicalComparisonQueries.some((query) => query.isFetching),
  );
  const isFetching = historicalRunGroup
    ? historicalFetching || (isComparing && historicalComparisonFetching)
    : monitorQuery.isFetching || (isComparing && comparisonQuery.isFetching);
  const isLoading = !hasData
    ? historicalRunGroup
      ? historicalFetching || (isComparing && historicalComparisonFetching)
      : monitorQuery.isLoading || (isComparing && comparisonQuery.isLoading)
    : false;
  const comparisonLoading = historicalRunGroup
    ? historicalComparisonFetching
    : comparisonQuery.isLoading || comparisonQuery.isFetching;
  const emptyMessage = useMemo(() => {
    if (monitorCount === 0) {
      return {
        title: "No monitor selected",
        detail: "Start a training job with at least one optional monitor enabled.",
      };
    }
    if (isLoading) {
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
    if (
      dataWasTruncated(data) ||
      dataWasTruncated(comparisonData) ||
      historicalDataWasTruncated(historicalData) ||
      historicalDataWasTruncated(historicalComparisonData)
    ) {
      return {
        title: "Monitor payload truncated",
        detail:
          truncationReason(data) ??
          truncationReason(comparisonData) ??
          historicalTruncationReason(historicalData) ??
          historicalTruncationReason(historicalComparisonData) ??
          "Some oversized event data was skipped to keep the viewer responsive.",
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
    comparisonData,
    data,
    historicalComparisonData,
    historicalData,
    historicalRunGroup,
    historicalRun,
    isLoading,
    isRunning,
    monitorCount,
  ]);

  const refetch = () => {
    if (historicalRunGroup) {
      setHistoricalEnabledCount(
        Math.min(historicalRuns.length, HISTORICAL_MONITOR_REQUEST_CONCURRENCY),
      );
      historicalMonitorQueries.slice(0, HISTORICAL_MONITOR_REQUEST_CONCURRENCY)
        .forEach((query) => {
          void query.refetch();
        });
      if (isComparing) {
        setHistoricalComparisonEnabledCount(
          Math.min(
            historicalRuns.length,
            HISTORICAL_MONITOR_REQUEST_CONCURRENCY,
          ),
        );
        historicalComparisonQueries
          .slice(0, HISTORICAL_MONITOR_REQUEST_CONCURRENCY)
          .forEach((query) => {
            void query.refetch();
          });
      }
    } else {
      void monitorQuery.refetch();
      if (isComparing) {
        void comparisonQuery.refetch();
      }
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
    historicalProgress,
    historicalComparisonProgress,
    emptyMessage,
    error:
      monitorQuery.error ??
      historicalMonitorQueries.find((query) => query.error)?.error ??
      (isComparing
        ? comparisonQuery.error ??
          historicalComparisonQueries.find((query) => query.error)?.error
        : null),
    refetch,
  };
}
