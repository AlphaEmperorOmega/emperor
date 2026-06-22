import { useMemo } from "react";
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
import {
  useViewerApiConnectionSwitch,
} from "@/features/viewer/state/use-viewer-api-connection";
import { type ViewerWorkspace } from "@/types/viewer";

type GraphPreviewControllerState = ReturnType<typeof useGraphPreviewController>;
type TargetConfigState = ReturnType<typeof useTargetConfigState>;
type HistoricalRunSelectionState = ReturnType<
  typeof useHistoricalRunSelectionState
>;
type HistoricalRunsState = ReturnType<typeof useHistoricalRunsState>;
type ActiveTrainingJobState = ReturnType<typeof useActiveTrainingJobState>;

export type ViewerStateOptions = {
  /** Notifies the logs workspace when a training job starts writing to a folder. */
  onJobStarted?: (logFolder: string) => void;
  activeWorkspace?: ViewerWorkspace;
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

/**
 * Composition module behind the viewer context providers.
 * Cross-slice cascade rules are named here; domain state stays in extracted hooks.
 */
export function useViewerState(options: ViewerStateOptions = {}) {
  const { activeWorkspace = "model", onJobStarted } = options;

  const graphPreview = useGraphPreviewController();
  const historicalRunSelection = useHistoricalRunSelectionState();
  const cascadeRules = useMemo(
    () => targetConfigCascadeRules(graphPreview, historicalRunSelection),
    [graphPreview, historicalRunSelection],
  );

  const targetConfig = useTargetConfigState(cascadeRules);
  const { selectedModel } = targetConfig.selection;

  const activeTrainingJobState = useActiveTrainingJobState({ onJobStarted });
  const historicalTagsEnabled =
    activeWorkspace === "logs" ||
    targetConfig.target.selectedTargetMode === "experiment" ||
    historicalRunSelection.selectedLogRunId !== null;
  const historicalRuns = useHistoricalRunsState({
    selectedModel,
    tagsEnabled: historicalTagsEnabled,
    syncSelectedLogRun: targetConfig.syncSelectedLogRun,
    selection: historicalRunSelection,
  });
  const apiConnection = useViewerApiConnectionSwitch(graphPreview);
  const graphPreviewInput = useMemo(
    () =>
      graphPreviewCompositionInput({
        graphPreview,
        targetConfig,
        historicalRuns,
        activeTrainingJobState,
      }),
    [activeTrainingJobState, graphPreview, historicalRuns, targetConfig],
  );
  const graphPreviewState = useGraphPreviewOrchestration(graphPreviewInput);

  const history = useMemo(
    () => ({
      ...historicalRuns.history,
      selectedLogRunHasMonitorTags:
        graphPreviewState.history.selectedLogRunHasMonitorTags,
    }),
    [
      graphPreviewState.history.selectedLogRunHasMonitorTags,
      historicalRuns.history,
    ],
  );
  const activeJob = activeTrainingJobState;
  const graphMonitor = graphPreviewState.graphMonitor;

  return useMemo(
    () => ({
      target: targetConfig.target,
      graph: graphPreviewState.graph,
      history,
      activeJob,
      graphMonitor,
      apiConnection,
    }),
    [
      activeJob,
      apiConnection,
      graphMonitor,
      graphPreviewState.graph,
      history,
      targetConfig.target,
    ],
  );
}

export type ViewerState = ReturnType<typeof useViewerState>;
export type TargetConfigContextValue = ViewerState["target"];
export type GraphViewContextValue = ViewerState["graph"];
export type HistoricalRunsContextValue = ViewerState["history"];
export type ActiveTrainingJobContextValue = ViewerState["activeJob"];
export type GraphMonitorContextValue = ViewerState["graphMonitor"];
export type ApiConnectionContextValue = ViewerState["apiConnection"];
