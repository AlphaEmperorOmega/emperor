import { type ReactNode, useCallback, useEffect, useRef } from "react";
import { createWorkbenchContext } from "@/features/workbench/providers/create-context";
import { useModelPackageInspection } from "@/features/workbench/providers/workbench-providers";
import {
  isWorkbenchProtectedAccessReady,
  useRegisterWorkbenchConnectionReset,
  useWorkbenchCapabilities,
  useWorkbenchConnection,
} from "@/features/workbench/providers/workbench-connection-provider";
import { useActiveTrainingJob } from "@/features/workbench/providers/training-provider";
import {
  useLogsWorkspaceState,
  type LogsBrowser,
  type LogsChartsInput,
  type LogsDeletion,
  type LogsRunDetail,
} from "@/features/workbench/state/logs/_use-logs-workspace-state";
import { useLogsChartViewModel } from "@/features/workbench/state/logs/_logs-chart-state";

export type {
  LogsBrowser,
  LogsBrowserFilter,
  LogsBrowserFilterKey,
  LogsDeletion,
} from "@/features/workbench/state/logs/_use-logs-workspace-state";
export type LogsCharts = ReturnType<typeof useLogsChartViewModel>;
export type {
  LogBestRunViewModel,
  LogMetricChartLayoutGroupKey,
  LogMetricGroupScalarQueryState,
  LogsChartEmptyState,
  ScalarChartGridMode,
} from "@/features/workbench/state/logs/_logs-chart-state";

const [LogsBrowserProvider, useLogsBrowser] =
  createWorkbenchContext<LogsBrowser>("LogsBrowserContext");
const [LogsChartsContextProvider, useLogsCharts] =
  createWorkbenchContext<LogsCharts>("LogsChartsContext");
const [LogsRunDetailProvider, useLogRunDetail] =
  createWorkbenchContext<LogsRunDetail>("LogsRunDetailContext");
const [LogsDeletionProvider, useLogsDeletion] =
  createWorkbenchContext<LogsDeletion>("LogsDeletionContext");

export { useLogRunDetail, useLogsBrowser, useLogsCharts, useLogsDeletion };

const noStartedExperiments: readonly string[] = [];

export function LogsChartsProvider({
  input,
  children,
}: {
  input: LogsChartsInput;
  children: ReactNode;
}) {
  const charts = useLogsChartViewModel(input);
  return (
    <LogsChartsContextProvider value={charts}>
      {children}
    </LogsChartsContextProvider>
  );
}

export function LogsWorkspaceProvider({
  enabled,
  startedExperiments = noStartedExperiments,
  children,
}: {
  enabled: boolean;
  startedExperiments?: readonly string[];
  children: ReactNode;
}) {
  const { capabilities } = useWorkbenchCapabilities();
  const workbenchConnection = useWorkbenchConnection();
  const protectedAccessReady = isWorkbenchProtectedAccessReady(
    workbenchConnection,
  );
  const { target, options } = useModelPackageInspection();
  const { activeTrainingJob } = useActiveTrainingJob();
  const targetPreset =
    target.kind === "historical-run"
      ? target.preset
      : options.presets.find((preset) => preset.name === target.preset)?.label ??
        target.preset;
  const workspace = useLogsWorkspaceState({
    enabled: enabled && protectedAccessReady,
    logDeletionEnabled:
      capabilities.logDeletionEnabled && protectedAccessReady,
    targetScope: {
      modelType: target.modelPackage.modelType,
      model: target.modelPackage.model,
      preset: targetPreset,
      datasets: target.datasets,
    },
  });
  const includeStartedExperiment =
    workspace.commands.includeStartedExperiment;
  const deliveredStartedExperimentsRef = useRef(new Set<string>());
  const clearLogsForConnectionChange =
    workspace.commands.clearForConnectionChange;
  const clearForConnectionChange = useCallback(() => {
    deliveredStartedExperimentsRef.current.clear();
    clearLogsForConnectionChange();
  }, [clearLogsForConnectionChange]);
  useRegisterWorkbenchConnectionReset(clearForConnectionChange);

  useEffect(() => {
    const observedExperiments = activeTrainingJob?.logFolder
      ? [...startedExperiments, activeTrainingJob.logFolder]
      : startedExperiments;
    for (const experiment of observedExperiments) {
      if (deliveredStartedExperimentsRef.current.has(experiment)) {
        continue;
      }
      deliveredStartedExperimentsRef.current.add(experiment);
      includeStartedExperiment(experiment);
    }
  }, [
    activeTrainingJob?.logFolder,
    includeStartedExperiment,
    startedExperiments,
  ]);

  const workspaceContent = (
    <LogsRunDetailProvider value={workspace.detail}>
      <LogsDeletionProvider value={workspace.deletion}>
        {children}
      </LogsDeletionProvider>
    </LogsRunDetailProvider>
  );
  return (
    <LogsBrowserProvider value={workspace.browser}>
      {enabled ? (
        <LogsChartsProvider input={workspace.charts}>
          {workspaceContent}
        </LogsChartsProvider>
      ) : (
        workspaceContent
      )}
    </LogsBrowserProvider>
  );
}
