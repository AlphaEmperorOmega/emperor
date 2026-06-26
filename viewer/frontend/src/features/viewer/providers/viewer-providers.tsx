import { type ReactNode } from "react";
import {
  useViewerState,
  type ActiveTrainingJobContextValue,
  type ApiConnectionContextValue,
  type GraphViewContextValue,
  type GraphMonitorContextValue,
  type HistoricalRunsContextValue,
  type TargetConfigContextValue,
} from "@/features/viewer/state/use-viewer-state";
import { createViewerContext } from "@/features/viewer/providers/create-context";
import {
  useActiveTrainingJobProgress,
  type ActiveTrainingJobProgress,
} from "@/features/viewer/state/training/use-training-job-controller";
import { type ViewerWorkspace } from "@/types/viewer";

const [TargetConfigProvider, useTargetConfig] =
  createViewerContext<TargetConfigContextValue>("TargetConfigContext");
const [GraphViewProvider, useGraphView] =
  createViewerContext<GraphViewContextValue>("GraphViewContext");
const [HistoricalRunsProvider, useHistoricalRuns] =
  createViewerContext<HistoricalRunsContextValue>("HistoricalRunsContext");
const [ActiveTrainingJobProvider, useActiveTrainingJob] =
  createViewerContext<ActiveTrainingJobContextValue>("ActiveTrainingJobContext");
const [ActiveTrainingJobProgressProvider, useActiveTrainingJobProgressState] =
  createViewerContext<ActiveTrainingJobProgress>(
    "ActiveTrainingJobProgressContext",
  );
const [GraphMonitorProvider, useGraphMonitor] =
  createViewerContext<GraphMonitorContextValue>("GraphMonitorContext");
const [ApiConnectionProvider, useApiConnection] =
  createViewerContext<ApiConnectionContextValue>("ApiConnectionContext");

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
    selectedDatasets,
    activeOverrides,
    effectivePresetOverrides,
    configSections,
    capabilities,
    selectModelType,
    selectModel,
    selectTargetPreset,
    selectTargetSnapshot,
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

export function useCompareTargetState() {
  const {
    selectedModelType,
    selectedModel,
    selectedPreset,
    selectModel,
    selectPreset,
    models,
    modelsLoading,
    isModelsError,
    modelsError,
  } = useTargetConfig();

  return {
    selectedModelType,
    selectedModel,
    selectedPreset,
    selectModel,
    selectPreset,
    catalog: {
      models,
      isLoading: modelsLoading,
      isError: isModelsError,
      error: modelsError,
    },
  };
}

export type ViewerProvidersProps = {
  /** Wired to the logs workspace so a new job's folder appears in its run list. */
  onJobStarted?: (logFolder: string) => void;
  activeWorkspace?: ViewerWorkspace;
  children: ReactNode;
};

/**
 * Runs the viewer orchestration engine once and distributes its four domain
 * slices through nested contexts, so panels read exactly the slice they need
 * instead of receiving it drilled down through props.
 */
export function ViewerProviders({
  activeWorkspace,
  onJobStarted,
  children,
}: ViewerProvidersProps) {
  const { target, graph, history, activeJob, graphMonitor, apiConnection } =
    useViewerState({
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
