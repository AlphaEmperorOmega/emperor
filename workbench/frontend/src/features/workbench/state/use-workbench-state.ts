import { useMemo } from "react";
import {
  useGraphPreviewController,
  useGraphPreviewOrchestration,
} from "@/features/workbench/state/graph-monitor/use-graph-preview-orchestration";
import {
  useTargetConfigState,
} from "@/features/workbench/state/target/use-target-config-state";
import {
  useHistoricalRunsState,
  useHistoricalRunSelectionState,
} from "@/features/workbench/state/use-historical-runs-state";
import {
  useActiveTrainingJobState,
} from "@/features/workbench/state/training/use-active-training-job-state";
import {
  useWorkbenchApiConnectionSwitch,
} from "@/features/workbench/state/use-workbench-api-connection";
import { type WorkbenchWorkspace } from "@/types/workbench";

type GraphPreviewControllerState = ReturnType<typeof useGraphPreviewController>;
type TargetConfigState = ReturnType<typeof useTargetConfigState>;
type HistoricalRunSelectionState = ReturnType<
  typeof useHistoricalRunSelectionState
>;
type HistoricalRunsState = ReturnType<typeof useHistoricalRunsState>;
type ActiveTrainingJobState = ReturnType<typeof useActiveTrainingJobState>;

export type WorkbenchStateOptions = {
  /** Notifies the logs workspace when a training job starts writing to a folder. */
  onJobStarted?: (logFolder: string) => void;
  activeWorkspace?: WorkbenchWorkspace;
  snapshotLibraryEnabled?: boolean;
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
  const {
    selectedModelType,
    selectedTargetMode,
    selectedSnapshotId,
    selectedExperimentRunId,
    selectedExperimentPreset,
    selectedExperimentDataset,
  } = targetConfig.target;
  const historicalGraphPreview = historicalRuns.graphPreview;
  const targetMode =
    selectedTargetMode === "snapshot" && selectedSnapshotId
      ? "snapshot"
      : selectedTargetMode === "experiment"
        ? "experiment"
        : "preset";
  const targetId =
    targetMode === "snapshot"
      ? selectedSnapshotId
      : targetMode === "experiment"
        ? selectedExperimentRunId
        : selectedPreset;
  const targetPreset =
    targetMode === "experiment" && selectedExperimentPreset
      ? selectedExperimentPreset
      : selectedPreset;
  const targetDatasets =
    targetMode === "experiment" && selectedExperimentDataset
      ? [selectedExperimentDataset]
      : selectedDatasets;

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
    targetModelType: selectedModelType,
    targetModel: selectedModel,
    targetPreset,
    targetDatasets,
    targetMode,
    targetId,
  };
}

/**
 * Composition module behind the workbench context providers.
 * Cross-slice cascade rules are named here; domain state stays in extracted hooks.
 */
export function useWorkbenchState(options: WorkbenchStateOptions = {}) {
  const {
    onJobStarted,
    activeWorkspace = "model",
    snapshotLibraryEnabled = false,
  } = options;

  const graphPreview = useGraphPreviewController();
  const historicalRunSelection = useHistoricalRunSelectionState();
  const cascadeRules = useMemo(
    () => targetConfigCascadeRules(graphPreview, historicalRunSelection),
    [graphPreview, historicalRunSelection],
  );

  const targetConfig = useTargetConfigState({
    ...cascadeRules,
    activeWorkspace,
    snapshotLibraryEnabled,
  });
  const { selectedModel } = targetConfig.selection;
  const { selectedExperimentTask, selectedModelType } = targetConfig.target;

  const activeTrainingJobState = useActiveTrainingJobState({ onJobStarted });
  const historicalTagsEnabled =
    targetConfig.target.selectedTargetMode === "experiment" ||
    historicalRunSelection.selectedLogRunId !== null;
  const historicalRunsEnabled =
    activeWorkspace === "logs" ||
    targetConfig.target.selectedTargetMode === "experiment";
  const historicalRuns = useHistoricalRunsState({
    selectedModelType,
    selectedModel,
    selectedExperimentTask,
    runsEnabled: historicalRunsEnabled,
    tagsEnabled: historicalTagsEnabled,
    syncSelectedLogRun: targetConfig.syncSelectedLogRun,
    clearSelectedExperimentRun: targetConfig.clearSelectedExperimentRun,
    selection: historicalRunSelection,
  });
  const apiConnection = useWorkbenchApiConnectionSwitch(graphPreview);
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

export type WorkbenchState = ReturnType<typeof useWorkbenchState>;
export type TargetConfigContextValue = WorkbenchState["target"];
export type GraphViewContextValue = WorkbenchState["graph"];
export type HistoricalRunsContextValue = WorkbenchState["history"];
export type ActiveTrainingJobContextValue = WorkbenchState["activeJob"];
export type GraphMonitorContextValue = WorkbenchState["graphMonitor"];
export type ApiConnectionContextValue = WorkbenchState["apiConnection"];
