import { type ReactNode, useEffect } from "react";
import { createWorkbenchContext } from "@/features/workbench/providers/create-context";
import {
  useActiveTrainingJob,
  useTargetConfig,
} from "@/features/workbench/providers/workbench-providers";
import {
  useLogsWorkspaceState,
  type LogsWorkspaceState,
} from "@/features/workbench/state/logs/use-logs-workspace-state";

const [LogsWorkspaceProviderBase, useLogsWorkspace] =
  createWorkbenchContext<LogsWorkspaceState>("LogsWorkspaceContext");

export { useLogsWorkspace };

export function LogsWorkspaceProvider({
  enabled,
  children,
}: {
  enabled: boolean;
  children: ReactNode;
}) {
  const {
    selectedModelType,
    selectedModel,
    selectedPreset,
    selectedPresetMeta,
    selectedDatasets,
    capabilities,
  } = useTargetConfig();
  const { activeTrainingJob } = useActiveTrainingJob();
  const state = useLogsWorkspaceState({
    enabled,
    logDeletionEnabled: capabilities.logDeletionEnabled,
    targetScope: {
      modelType: selectedModelType,
      model: selectedModel,
      preset: selectedPresetMeta?.label ?? selectedPreset,
      datasets: selectedDatasets,
    },
  });
  const includeStartedExperiment = state.includeStartedExperiment;

  useEffect(() => {
    if (activeTrainingJob?.logFolder) {
      includeStartedExperiment(activeTrainingJob.logFolder);
    }
  }, [activeTrainingJob?.logFolder, includeStartedExperiment]);

  return <LogsWorkspaceProviderBase value={state}>{children}</LogsWorkspaceProviderBase>;
}
