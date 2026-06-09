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
    selectedPreset,
    selectedPresetMeta,
    selectedDatasets,
    selectModelType,
    selectModel,
    selectPreset,
    toggleDataset,
    selectAllDatasets,
    selectFirstDataset,
    modelsQuery,
    presetsQuery,
    datasetsQuery,
  } = useTargetConfig();

  return {
    selectedModelType,
    selectedModel,
    selectedPreset,
    selectedPresetMeta,
    selectedDatasets,
    selectModelType,
    selectModel,
    selectPreset,
    toggleDataset,
    selectAllDatasets,
    selectFirstDataset,
    models: modelsQuery.data?.models ?? [],
    presets: presetsQuery.data?.presets ?? [],
    datasets: datasetsQuery.data?.datasets ?? [],
  };
}

export function useTargetConfigSummaryState() {
  const {
    fieldCount,
    overrideCount,
    configSnapshotCount,
    selectedModel,
    selectedPreset,
    schemaQuery,
  } = useTargetConfig();

  return {
    fieldCount,
    overrideCount,
    configSnapshotCount,
    canOpenFullConfig: Boolean(
      selectedModel && selectedPreset && schemaQuery.isSuccess,
    ),
    isSchemaLoading: schemaQuery.isLoading,
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
