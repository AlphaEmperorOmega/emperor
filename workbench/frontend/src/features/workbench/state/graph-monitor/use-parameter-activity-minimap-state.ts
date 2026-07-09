import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchLogParameterStatus,
  type InspectResponse,
  type LogParameterStatusResponse,
  type LogRun,
} from "@/lib/api";
import { monitorQueryKeys } from "@/lib/query-keys";
import {
  deriveParameterActivityByNodePath,
  deriveParameterStatusPathMismatch,
} from "@/features/workbench/state/graph-monitor/graph-monitor-selectors";
import { deriveParameterActivityMinimapModel } from "@/lib/graph";
import { createLinearMonitorTargetResolver } from "@/lib/graph/monitor-targets";
import { type MonitorEligibility } from "@/lib/historical-monitor-runs";
import {
  type GraphParameterActivity,
} from "@/lib/graph/types";
import { type MonitorChartsSource } from "@/types/monitor";

type TargetMode = "preset" | "snapshot" | "experiment";

export type ParameterActivityMinimapState = {
  shouldRenderButton: boolean;
  canOpen: boolean;
  disabledReason?: string;
  activityByNodePath?: Map<string, GraphParameterActivity>;
  isStatusLoading: boolean;
  isPathMismatch: boolean;
  selectedRunSource?: MonitorChartsSource;
  parameterNodeCount: number;
};

type ParameterActivityMinimapStateInput = {
  graph?: InspectResponse;
  selectedTargetMode: TargetMode;
  selectedExperimentRunId: string;
  selectedLogRun?: LogRun;
  selectedLogRunMonitorEligibility?: MonitorEligibility;
};

export function useParameterActivityMinimapState({
  graph,
  selectedTargetMode,
  selectedExperimentRunId,
  selectedLogRun,
  selectedLogRunMonitorEligibility,
}: ParameterActivityMinimapStateInput): ParameterActivityMinimapState {
  const shouldRenderButton = Boolean(
    selectedTargetMode === "experiment" && selectedExperimentRunId && graph,
  );
  const selectedRunSource: MonitorChartsSource | undefined = useMemo(
    () =>
      selectedLogRun && selectedLogRun.id === selectedExperimentRunId
        ? { kind: "historical-run", run: selectedLogRun }
        : undefined,
    [selectedExperimentRunId, selectedLogRun],
  );
  const statusQueryEnabled = Boolean(
    shouldRenderButton &&
      selectedRunSource &&
      selectedLogRunMonitorEligibility === "eligible",
  );
  const linearMonitorTargetResolver = useMemo(
    () => createLinearMonitorTargetResolver(graph),
    [graph],
  );
  const statusQuery = useQuery<LogParameterStatusResponse>({
    queryKey: selectedExperimentRunId
      ? monitorQueryKeys.historicalParameterStatus([selectedExperimentRunId])
      : (["monitor-parameter-status", "inactive-minimap-run"] as const),
    queryFn: ({ signal }) =>
      fetchLogParameterStatus(
        { runIds: [selectedExperimentRunId] },
        { signal },
      ),
    enabled: statusQueryEnabled,
    retry: false,
  });
  const isStatusLoading = Boolean(
    statusQueryEnabled &&
      !statusQuery.data &&
      (statusQuery.isFetching || statusQuery.isLoading),
  );
  const activityByNodePath = useMemo(
    () =>
      deriveParameterActivityByNodePath({
        graph,
        source: selectedRunSource,
        status: statusQuery.data,
        statusLoading: isStatusLoading,
        linearMonitorTargetResolver,
      }),
    [
      graph,
      isStatusLoading,
      linearMonitorTargetResolver,
      selectedRunSource,
      statusQuery.data,
    ],
  );
  const isPathMismatch = useMemo(
    () =>
      deriveParameterStatusPathMismatch({
        graph,
        source: selectedRunSource,
        status: statusQuery.data,
        statusLoading: isStatusLoading,
        linearMonitorTargetResolver,
      }),
    [
      graph,
      isStatusLoading,
      linearMonitorTargetResolver,
      selectedRunSource,
      statusQuery.data,
    ],
  );
  const parameterNodeCount = useMemo(
    () =>
      deriveParameterActivityMinimapModel({
        graph,
        activityByNodePath,
      }).parameterNodeCount,
    [activityByNodePath, graph],
  );
  const disabledReason = !shouldRenderButton
    ? undefined
    : !selectedRunSource
      ? "Selected Training Run is not ready"
      : selectedLogRunMonitorEligibility === "checking"
        ? "Checking monitor data for this Training Run"
        : selectedLogRunMonitorEligibility === "ineligible"
          ? "No monitor data for this Training Run"
          : statusQuery.isError
            ? "Could not load parameter activity for this Training Run"
            : isPathMismatch
              ? "Monitor paths do not match this graph"
              : parameterNodeCount === 0
                ? "No parameter-bearing components found"
                : undefined;

  return {
    shouldRenderButton,
    canOpen: shouldRenderButton && !disabledReason,
    disabledReason,
    activityByNodePath,
    isStatusLoading,
    isPathMismatch,
    selectedRunSource,
    parameterNodeCount,
  };
}
