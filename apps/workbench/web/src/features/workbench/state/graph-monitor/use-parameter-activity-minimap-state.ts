import { useMemo } from "react";
import type { InspectResponse } from "@/lib/api/inspection";
import type { LogRun } from "@/lib/api/logs";
import { useExperimentMonitorParameterActivity } from "@/features/workbench/state/graph-monitor/use-experiment-monitor-parameter-activity";
import { deriveParameterActivityMinimapModel } from "@/lib/graph";
import { type MonitorEligibility } from "@/lib/historical-monitor-runs";
import { type GraphParameterActivity } from "@/lib/graph/types";
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
  protectedReadsEnabled?: boolean;
};

function parameterActivityMinimapDisabledReason({
  shouldRenderButton,
  selectedRunSource,
  monitorEligibility,
  hasQueryError,
  isPathMismatch,
  parameterNodeCount,
}: {
  shouldRenderButton: boolean;
  selectedRunSource: MonitorChartsSource | undefined;
  monitorEligibility: MonitorEligibility | undefined;
  hasQueryError: boolean;
  isPathMismatch: boolean;
  parameterNodeCount: number;
}) {
  if (!shouldRenderButton) {
    return undefined;
  }
  if (!selectedRunSource) {
    return "Selected Training Run is not ready";
  }
  if (monitorEligibility === "checking") {
    return "Checking monitor data for this Training Run";
  }
  if (monitorEligibility === "ineligible") {
    return "No monitor data for this Training Run";
  }
  if (hasQueryError) {
    return "Could not load parameter activity for this Training Run";
  }
  if (isPathMismatch) {
    return "Monitor paths do not match this graph";
  }
  if (parameterNodeCount === 0) {
    return "No parameter-bearing components found";
  }
  return undefined;
}

export function useParameterActivityMinimapState({
  graph,
  selectedTargetMode,
  selectedExperimentRunId,
  selectedLogRun,
  selectedLogRunMonitorEligibility,
  protectedReadsEnabled = true,
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
  const parameterActivity = useExperimentMonitorParameterActivity({
    graph,
    source: selectedRunSource,
    enabled: statusQueryEnabled,
    protectedReadsEnabled,
  });
  const activityByNodePath = parameterActivity.activityByNodePath;
  const isStatusLoading = parameterActivity.isLoading;
  const isPathMismatch = parameterActivity.isPathMismatch;
  const parameterNodeCount = useMemo(
    () =>
      deriveParameterActivityMinimapModel({
        graph,
        activityByNodePath,
      }).parameterNodeCount,
    [activityByNodePath, graph],
  );
  const disabledReason = parameterActivityMinimapDisabledReason({
    shouldRenderButton,
    selectedRunSource,
    monitorEligibility: selectedLogRunMonitorEligibility,
    hasQueryError: parameterActivity.isError,
    isPathMismatch,
    parameterNodeCount,
  });

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
