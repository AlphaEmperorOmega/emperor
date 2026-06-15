import { type ReactNode, useEffect } from "react";
import { createViewerContext } from "@/features/viewer/providers/create-context";
import {
  useTargetConfig,
  useTraining,
} from "@/features/viewer/providers/viewer-providers";
import {
  useLogsWorkspaceState,
  type LogsWorkspaceState,
} from "@/features/viewer/state/logs/use-logs-workspace-state";

const [LogsWorkspaceProviderBase, useLogsWorkspace] =
  createViewerContext<LogsWorkspaceState>("LogsWorkspaceContext");

export { useLogsWorkspace };

export function LogsWorkspaceProvider({
  enabled,
  children,
}: {
  enabled: boolean;
  children: ReactNode;
}) {
  const {
    selectedModel,
    selectedPreset,
    selectedPresetMeta,
    selectedDatasets,
    capabilities,
  } = useTargetConfig();
  const { activeTrainingJob } = useTraining();
  const state = useLogsWorkspaceState({
    enabled,
    logDeletionEnabled: capabilities.logDeletionEnabled,
    targetScope: {
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
