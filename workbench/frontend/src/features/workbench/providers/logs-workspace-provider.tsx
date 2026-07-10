import { type ReactNode, useEffect } from "react";
import { createWorkbenchContext } from "@/features/workbench/providers/create-context";
import {
  useActiveTrainingJob,
  useModelTargetConfig,
  useTargetCatalog,
} from "@/features/workbench/providers/workbench-providers";
import {
  useLogsWorkspaceState,
  type LogsWorkspaceState,
} from "@/features/workbench/state/logs/use-logs-workspace-state";

const [LogsWorkspaceProviderBase, useLogsWorkspace] =
  createWorkbenchContext<LogsWorkspaceState>("LogsWorkspaceContext");
const noStartedExperiments: readonly string[] = [];

export { useLogsWorkspace };

export function LogsWorkspaceProvider({
  enabled,
  startedExperiments = noStartedExperiments,
  children,
}: {
  enabled: boolean;
  startedExperiments?: readonly string[];
  children: ReactNode;
}) {
  const { capabilities } = useTargetCatalog();
  const {
    selectedModelType,
    selectedModel,
    selectedPreset,
    selectedPresetMeta,
    selectedDatasets,
  } = useModelTargetConfig();
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
    for (const experiment of startedExperiments) {
      includeStartedExperiment(experiment);
    }
    if (activeTrainingJob?.logFolder) {
      includeStartedExperiment(activeTrainingJob.logFolder);
    }
  }, [activeTrainingJob?.logFolder, includeStartedExperiment, startedExperiments]);

  return <LogsWorkspaceProviderBase value={state}>{children}</LogsWorkspaceProviderBase>;
}
