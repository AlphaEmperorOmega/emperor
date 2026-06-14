import {
  useGraphPreviewController,
  useGraphPreviewOrchestration,
} from "@/features/viewer/state/graph-monitor/use-graph-preview-orchestration";
import {
  useTargetConfigState,
} from "@/features/viewer/state/target/use-target-config-state";
import {
  useHistoricalRunsState,
  useHistoricalRunSelectionState,
} from "@/features/viewer/state/use-historical-runs-state";
import {
  useActiveTrainingJobState,
} from "@/features/viewer/state/training/use-active-training-job-state";

type GraphPreviewControllerState = ReturnType<typeof useGraphPreviewController>;
type GraphPreviewState = ReturnType<typeof useGraphPreviewOrchestration>;
type TargetConfigState = ReturnType<typeof useTargetConfigState>;
type HistoricalRunSelectionState = ReturnType<
  typeof useHistoricalRunSelectionState
>;
type HistoricalRunsState = ReturnType<typeof useHistoricalRunsState>;
type ActiveTrainingJobState = ReturnType<typeof useActiveTrainingJobState>;

export type ViewerStateOptions = {
  /** Notifies the logs workspace when a training job starts writing to a folder. */
  onJobStarted?: (logFolder: string) => void;
};

function targetConfigCascadeRules(
  graphPreview: GraphPreviewControllerState,
  historicalRunSelection: HistoricalRunSelectionState,
): Parameters<typeof useTargetConfigState>[0] {
  const clearHistoricalSelection =
    historicalRunSelection.clearHistoricalSelectionForTarget;

  return {
    requestPreview: graphPreview.requestPreview,
    clearPreview: graphPreview.clearPreview,
    resetGraphSelectionAndExpansion: graphPreview.resetGraphSelectionAndExpansion,
    resetGraphExpansion: graphPreview.resetGraphExpansion,
    onModelSelected: clearHistoricalSelection,
    onTargetPresetSelected: clearHistoricalSelection,
    onTargetSnapshotSelected: clearHistoricalSelection,
  };
}

function graphPreviewCompositionInput({
  graphPreview,
  targetConfig,
  historicalRuns,
  activeTrainingJobState,
}: {
  graphPreview: GraphPreviewControllerState;
  targetConfig: TargetConfigState;
  historicalRuns: HistoricalRunsState;
  activeTrainingJobState: ActiveTrainingJobState;
}): Parameters<typeof useGraphPreviewOrchestration>[0] {
  const { selectedModel, selectedPreset, selectedDatasets } = targetConfig.selection;
  const historicalGraphPreview = historicalRuns.graphPreview;

  return {
    controller: graphPreview,
    activeTrainingJob: activeTrainingJobState.activeTrainingJob,
    historicalMonitorRuns: historicalGraphPreview.historicalMonitorRuns,
    selectedHistoricalExperiment:
      historicalGraphPreview.selectedHistoricalExperiment,
    selectedHistoricalDataset: historicalGraphPreview.selectedHistoricalDataset,
    selectedHistoricalPreset: historicalGraphPreview.selectedHistoricalRunPreset,
    logRunTags: historicalGraphPreview.logRunTags,
    filteredHistoricalRunIds: historicalGraphPreview.filteredHistoricalRunIds,
    targetModel: selectedModel,
    targetPreset: selectedPreset,
    targetDatasets: selectedDatasets,
  };
}

function composeProviderSlices({
  targetConfig,
  historicalRuns,
  activeTrainingJobState,
  graphPreviewState,
}: {
  targetConfig: TargetConfigState;
  historicalRuns: HistoricalRunsState;
  activeTrainingJobState: ActiveTrainingJobState;
  graphPreviewState: GraphPreviewState;
}) {
  return {
    target: targetConfig.target,
    graph: graphPreviewState.graph,
    history: {
      ...historicalRuns.history,
      selectedLogRunHasMonitorTags:
        graphPreviewState.history.selectedLogRunHasMonitorTags,
    },
    training: {
      ...activeTrainingJobState,
      ...graphPreviewState.training,
    },
  };
}

/**
 * Composition module behind the viewer context providers.
 * Cross-slice cascade rules are named here; domain state stays in extracted hooks.
 */
export function useViewerState(options: ViewerStateOptions = {}) {
  const { onJobStarted } = options;

  const graphPreview = useGraphPreviewController();
  const historicalRunSelection = useHistoricalRunSelectionState();

  const targetConfig = useTargetConfigState(
    targetConfigCascadeRules(graphPreview, historicalRunSelection),
  );
  const { selectedModel } = targetConfig.selection;

  const activeTrainingJobState = useActiveTrainingJobState({ onJobStarted });
  const historicalRuns = useHistoricalRunsState({
    selectedModel,
    syncSelectedLogRun: targetConfig.syncSelectedLogRun,
    selection: historicalRunSelection,
  });
  const graphPreviewState = useGraphPreviewOrchestration(
    graphPreviewCompositionInput({
      graphPreview,
      targetConfig,
      historicalRuns,
      activeTrainingJobState,
    }),
  );

  return composeProviderSlices({
    targetConfig,
    historicalRuns,
    activeTrainingJobState,
    graphPreviewState,
  });
}

export type ViewerState = ReturnType<typeof useViewerState>;
export type TargetConfigContextValue = ViewerState["target"];
export type GraphViewContextValue = ViewerState["graph"];
export type HistoricalRunsContextValue = ViewerState["history"];
export type TrainingContextValue = ViewerState["training"];
