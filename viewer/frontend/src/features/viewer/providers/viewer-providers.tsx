import { type ReactNode } from "react";
import {
  useViewerState,
  type GraphViewContextValue,
  type HistoricalRunsContextValue,
  type TargetConfigContextValue,
  type TrainingContextValue,
} from "@/features/viewer/state/use-viewer-state";
import { createViewerContext } from "@/features/viewer/providers/create-context";

const [TargetConfigProvider, useTargetConfig] =
  createViewerContext<TargetConfigContextValue>("TargetConfigContext");
const [GraphViewProvider, useGraphView] =
  createViewerContext<GraphViewContextValue>("GraphViewContext");
const [HistoricalRunsProvider, useHistoricalRuns] =
  createViewerContext<HistoricalRunsContextValue>("HistoricalRunsContext");
const [TrainingProvider, useTraining] =
  createViewerContext<TrainingContextValue>("TrainingContext");

export { useTargetConfig, useGraphView, useHistoricalRuns, useTraining };

export function useTargetHeaderState() {
  const {
    selectedModel,
    selectedPreset,
    apiOnline,
    overrideCount,
    presetOwnedFieldCount,
    resetOverrides,
    updatePreview,
  } = useTargetConfig();

  return {
    selectedModel,
    selectedPreset,
    apiOnline,
    overrideCount,
    presetOwnedFieldCount,
    resetOverrides,
    updatePreview,
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
    modelsQuery,
    presetsQuery,
    datasetsQuery,
    schemaQuery,
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
    configSnapshotsEnabled: capabilities.configSnapshotsEnabled,
    isSchemaReady: schemaQuery.isSuccess,
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
    models: modelsQuery.data?.models ?? [],
    presets: presetsQuery.data?.presets ?? [],
    datasets: datasetsQuery.data?.datasets ?? [],
  };
}

export function useTargetConfigSummaryState() {
  const {
    fieldCount,
    overrideCount,
    allConfigSnapshotCount,
    selectedModel,
    selectedPreset,
    schemaQuery,
  } = useTargetConfig();

  return {
    fieldCount,
    overrideCount,
    configSnapshotCount: allConfigSnapshotCount,
    canOpenFullConfig: Boolean(
      selectedModel && selectedPreset && schemaQuery.isSuccess,
    ),
    isSchemaLoading: schemaQuery.isLoading,
  };
}

export function useConfigSnapshotLibraryState() {
  const {
    configSnapshotLibrary,
    configSnapshotLibraryCount,
    configSnapshotLibraryQuery,
    loadConfigSnapshot,
  } = useTargetConfig();

  return {
    snapshots: configSnapshotLibrary,
    snapshotCount: configSnapshotLibraryCount,
    isLoading: configSnapshotLibraryQuery.isLoading,
    isError: configSnapshotLibraryQuery.isError,
    error: configSnapshotLibraryQuery.error,
    loadConfigSnapshot,
  };
}

export function useTargetQueryStatusState() {
  const { modelsQuery, presetsQuery, datasetsQuery, schemaQuery } = useTargetConfig();

  return {
    modelsQuery: {
      isError: modelsQuery.isError,
      error: modelsQuery.error,
    },
    presetsQuery: {
      isError: presetsQuery.isError,
      error: presetsQuery.error,
    },
    datasetsQuery: {
      isError: datasetsQuery.isError,
      error: datasetsQuery.error,
    },
    schemaQuery: {
      isError: schemaQuery.isError,
      error: schemaQuery.error,
    },
  };
}

export function useCompareTargetState() {
  const {
    selectedModel,
    selectedPreset,
    selectModel,
    selectPreset,
    modelsQuery,
  } = useTargetConfig();

  return {
    selectedModel,
    selectedPreset,
    selectModel,
    selectPreset,
    catalog: {
      models: modelsQuery.data?.models ?? [],
      isLoading: modelsQuery.isLoading,
      isError: modelsQuery.isError,
      error: modelsQuery.error,
    },
  };
}

export type ViewerProvidersProps = {
  /** Wired to the logs workspace so a new job's folder appears in its run list. */
  onJobStarted?: (logFolder: string) => void;
  children: ReactNode;
};

/**
 * Runs the viewer orchestration engine once and distributes its four domain
 * slices through nested contexts, so panels read exactly the slice they need
 * instead of receiving it drilled down through props.
 */
export function ViewerProviders({ onJobStarted, children }: ViewerProvidersProps) {
  const { target, graph, history, training } = useViewerState({ onJobStarted });
  return (
    <TargetConfigProvider value={target}>
      <GraphViewProvider value={graph}>
        <HistoricalRunsProvider value={history}>
          <TrainingProvider value={training}>{children}</TrainingProvider>
        </HistoricalRunsProvider>
      </GraphViewProvider>
    </TargetConfigProvider>
  );
}
