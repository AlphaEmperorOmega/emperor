import { type ReactNode } from "react";
import {
  useWorkbenchState,
  type ActiveTrainingJobContextValue,
  type ApiConnectionContextValue,
  type GraphViewContextValue,
  type GraphMonitorContextValue,
  type HistoricalRunsContextValue,
  type TargetConfigContextValue,
} from "@/features/workbench/state/use-workbench-state";
import { createWorkbenchContext } from "@/features/workbench/providers/create-context";
import {
  useActiveTrainingJobProgress,
  type ActiveTrainingJobProgress,
} from "@/features/workbench/state/training/use-training-job-controller";
import { type WorkbenchWorkspace } from "@/types/workbench";

const [TargetConfigProvider, useTargetConfig] =
  createWorkbenchContext<TargetConfigContextValue>("TargetConfigContext");
const [GraphViewProvider, useGraphView] =
  createWorkbenchContext<GraphViewContextValue>("GraphViewContext");
const [HistoricalRunsProvider, useHistoricalRuns] =
  createWorkbenchContext<HistoricalRunsContextValue>("HistoricalRunsContext");
const [ActiveTrainingJobProvider, useActiveTrainingJob] =
  createWorkbenchContext<ActiveTrainingJobContextValue>("ActiveTrainingJobContext");
const [ActiveTrainingJobProgressProvider, useActiveTrainingJobProgressState] =
  createWorkbenchContext<ActiveTrainingJobProgress>(
    "ActiveTrainingJobProgressContext",
  );
const [GraphMonitorProvider, useGraphMonitor] =
  createWorkbenchContext<GraphMonitorContextValue>("GraphMonitorContext");
const [ApiConnectionProvider, useApiConnection] =
  createWorkbenchContext<ApiConnectionContextValue>("ApiConnectionContext");

export {
  useTargetConfig,
  useGraphView,
  useHistoricalRuns,
  useActiveTrainingJob,
  useActiveTrainingJobProgressState,
  useGraphMonitor,
  useApiConnection,
};

function ActiveTrainingJobProgressController({
  children,
}: {
  children: ReactNode;
}) {
  const { activeJobId, onJobChange } = useActiveTrainingJob();
  const progress = useActiveTrainingJobProgress({ activeJobId, onJobChange });

  return (
    <ActiveTrainingJobProgressProvider value={progress}>
      {children}
    </ActiveTrainingJobProgressProvider>
  );
}

export function useTargetHeaderState() {
  const {
    selectedModelType,
    selectedModel,
    selectedPreset,
    apiOnline,
    overrideCount,
    presetOwnedFieldCount,
    resetOverrides,
  } = useTargetConfig();

  return {
    selectedModelType,
    selectedModel,
    selectedPreset,
    apiOnline,
    overrideCount,
    presetOwnedFieldCount,
    resetOverrides,
  };
}

export function useTargetSelectorState() {
  const {
    selectedModelType,
    selectedModel,
    selectedTargetMode,
    activateTargetPresetMode,
    activateTargetSnapshotMode,
    activateTargetExperimentMode,
    selectedPreset,
    selectedPresetMeta,
    selectedSnapshotId,
    selectedConfigSnapshot,
    selectedExperimentRunId,
    selectedExperimentTask,
    experimentTaskOptions,
    selectedDatasets,
    activeOverrides,
    effectivePresetOverrides,
    configSections,
    capabilities,
    selectModelType,
    selectModel,
    selectTargetPreset,
    selectTargetSnapshot,
    selectExperimentTask,
    preparePresetSnapshotDraft,
    prepareSelectedSnapshotEdit,
    toggleDataset,
    selectAllDatasets,
    selectFirstDataset,
    allConfigSnapshots,
    models,
    presets,
    datasets,
    targetMonitors,
    targetMonitorsLoading,
    isSchemaReady,
  } = useTargetConfig();

  return {
    selectedModelType,
    selectedModel,
    selectedTargetMode,
    activateTargetPresetMode,
    activateTargetSnapshotMode,
    activateTargetExperimentMode,
    selectedPreset,
    selectedPresetMeta,
    selectedSnapshotId,
    selectedConfigSnapshot,
    selectedExperimentRunId,
    selectedExperimentTask,
    experimentTaskOptions,
    selectedDatasets,
    activeOverrides,
    effectivePresetOverrides,
    configSections,
    configSnapshotsEnabled: capabilities.configSnapshotsEnabled,
    isSchemaReady,
    selectModelType,
    selectModel,
    selectPreset: selectTargetPreset,
    selectSnapshot: selectTargetSnapshot,
    selectExperimentTask,
    preparePresetSnapshotDraft,
    prepareSelectedSnapshotEdit,
    toggleDataset,
    selectAllDatasets,
    selectFirstDataset,
    snapshots: allConfigSnapshots,
    models,
    presets,
    datasets,
    targetMonitors,
    targetMonitorsLoading,
  };
}

export function useTargetConfigSummaryState() {
  const {
    fieldCount,
    overrideCount,
    allConfigSnapshotCount,
    selectedModel,
    selectedPreset,
    selectedTargetMode,
    isSchemaReady,
    schemaLoading,
  } = useTargetConfig();

  return {
    fieldCount,
    overrideCount,
    configSnapshotCount: allConfigSnapshotCount,
    canOpenFullConfig: Boolean(
      selectedModel && selectedPreset && isSchemaReady,
    ),
    showFullConfigButton: selectedTargetMode !== "experiment",
    isSchemaLoading: schemaLoading,
  };
}

export function useConfigSnapshotLibraryState() {
  const {
    configSnapshotLibrary,
    configSnapshotLibraryCount,
    libraryLoading,
    isLibraryError,
    libraryError,
    loadConfigSnapshot,
  } = useTargetConfig();

  return {
    snapshots: configSnapshotLibrary,
    snapshotCount: configSnapshotLibraryCount,
    isLoading: libraryLoading,
    isError: isLibraryError,
    error: libraryError,
    loadConfigSnapshot,
  };
}

export function useTargetQueryStatusState() {
  const {
    isModelsError,
    modelsError,
    isPresetsError,
    presetsError,
    isDatasetsError,
    datasetsError,
    isSchemaError,
    schemaError,
  } = useTargetConfig();

  return {
    modelsQuery: {
      isError: isModelsError,
      error: modelsError,
    },
    presetsQuery: {
      isError: isPresetsError,
      error: presetsError,
    },
    datasetsQuery: {
      isError: isDatasetsError,
      error: datasetsError,
    },
    schemaQuery: {
      isError: isSchemaError,
      error: schemaError,
    },
  };
}

export type WorkbenchProvidersProps = {
  /** Wired to the logs workspace so a new job's folder appears in its run list. */
  onJobStarted?: (logFolder: string) => void;
  activeWorkspace?: WorkbenchWorkspace;
  children: ReactNode;
};

/**
 * Runs the workbench orchestration engine once and distributes its four domain
 * slices through nested contexts, so panels read exactly the slice they need
 * instead of receiving it drilled down through props.
 */
export function WorkbenchProviders({
  activeWorkspace,
  onJobStarted,
  children,
}: WorkbenchProvidersProps) {
  const { target, graph, history, activeJob, graphMonitor, apiConnection } =
    useWorkbenchState({
      activeWorkspace,
      onJobStarted,
    });
  return (
    <TargetConfigProvider value={target}>
      <GraphViewProvider value={graph}>
        <HistoricalRunsProvider value={history}>
          <ActiveTrainingJobProvider value={activeJob}>
            <ActiveTrainingJobProgressController>
              <GraphMonitorProvider value={graphMonitor}>
                <ApiConnectionProvider value={apiConnection}>
                  {children}
                </ApiConnectionProvider>
              </GraphMonitorProvider>
            </ActiveTrainingJobProgressController>
          </ActiveTrainingJobProvider>
        </HistoricalRunsProvider>
      </GraphViewProvider>
    </TargetConfigProvider>
  );
}
