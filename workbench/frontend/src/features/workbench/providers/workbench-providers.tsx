import { type ReactNode, useMemo } from "react";
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
  type ModelTargetContextValue,
  type TargetCatalogContextValue,
  type TargetSnapshotsContextValue,
  type TrainingTargetContextValue,
  useTargetContextSlices,
} from "@/features/workbench/providers/target-context-slices";
import {
  useActiveTrainingJobProgress,
  type ActiveTrainingJobProgress,
} from "@/features/workbench/state/training/use-training-job-controller";
import { type WorkbenchWorkspace } from "@/types/workbench";

const [TargetCatalogProvider, useTargetCatalog] =
  createWorkbenchContext<TargetCatalogContextValue>("TargetCatalogContext");
const [ModelTargetProvider, useModelTargetConfig] =
  createWorkbenchContext<ModelTargetContextValue>("ModelTargetContext");
const [TrainingTargetProvider, useTrainingTargetState] =
  createWorkbenchContext<TrainingTargetContextValue>("TrainingTargetContext");
const [TargetSnapshotsProvider, useTargetSnapshots] =
  createWorkbenchContext<TargetSnapshotsContextValue>("TargetSnapshotsContext");
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
  useTargetCatalog,
  useModelTargetConfig,
  useTargetSnapshots,
  useGraphView,
  useHistoricalRuns,
  useActiveTrainingJob,
  useActiveTrainingJobProgressState,
  useGraphMonitor,
  useApiConnection,
};

/** Compatibility Interface for the few dialogs that intentionally coordinate
 * Model, Training, and snapshot state. Hot-path consumers use a focused hook. */
export function useTargetConfig(): TargetConfigContextValue {
  const catalog = useTargetCatalog();
  const model = useModelTargetConfig();
  const training = useTrainingTargetState();
  const snapshots = useTargetSnapshots();

  return useMemo(
    () => ({ ...catalog, ...model, ...training, ...snapshots }),
    [catalog, model, snapshots, training],
  ) as TargetConfigContextValue;
}

export function useTrainingTargetConfig() {
  const catalog = useTargetCatalog();
  const training = useTrainingTargetState();

  return useMemo(
    () => ({ ...catalog, ...training }),
    [catalog, training],
  );
}

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
  const { apiOnline } = useTargetCatalog();
  const {
    selectedModelType,
    selectedModel,
    selectedPreset,
    overrideCount,
    presetOwnedFieldCount,
    resetOverrides,
  } = useModelTargetConfig();

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
  const { capabilities, models } = useTargetCatalog();
  const {
    selectedSnapshotId,
    selectedConfigSnapshot,
    selectTargetSnapshot,
    prepareSelectedSnapshotEdit,
    preparePresetSnapshotDraft,
    allConfigSnapshots,
  } = useTargetSnapshots();
  const {
    selectedModelType,
    selectedModel,
    selectedTargetMode,
    activateTargetPresetMode,
    activateTargetSnapshotMode,
    activateTargetExperimentMode,
    selectedPreset,
    selectedPresetMeta,
    selectedExperimentRunId,
    selectedExperimentTask,
    experimentTaskOptions,
    selectedDatasets,
    activeOverrides,
    effectivePresetOverrides,
    configSections,
    selectModelType,
    selectModel,
    selectTargetPreset,
    selectExperimentTask,
    toggleDataset,
    selectAllDatasets,
    selectFirstDataset,
    presets,
    datasets,
    targetMonitors,
    targetMonitorsLoading,
    isSchemaReady,
  } = useModelTargetConfig();

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
  const { allConfigSnapshotCount } = useTargetSnapshots();
  const {
    fieldCount,
    overrideCount,
    selectedModel,
    selectedPreset,
    selectedTargetMode,
    isSchemaReady,
    schemaLoading,
  } = useModelTargetConfig();

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
  } = useTargetSnapshots();

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
  const { isModelsError, modelsError } = useTargetCatalog();
  const {
    isPresetsError,
    presetsError,
    isDatasetsError,
    datasetsError,
    isSchemaError,
    schemaError,
  } = useModelTargetConfig();

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
  const targetSlices = useTargetContextSlices(target);
  return (
    <TargetCatalogProvider value={targetSlices.catalog}>
      <ModelTargetProvider value={targetSlices.model}>
        <TrainingTargetProvider value={targetSlices.training}>
          <TargetSnapshotsProvider value={targetSlices.snapshots}>
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
          </TargetSnapshotsProvider>
        </TrainingTargetProvider>
      </ModelTargetProvider>
    </TargetCatalogProvider>
  );
}
