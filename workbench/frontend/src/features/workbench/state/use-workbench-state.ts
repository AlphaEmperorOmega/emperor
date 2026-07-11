import { useCallback, useMemo } from "react";
import {
  useGraphPreviewOrchestration,
} from "@/features/workbench/state/graph-monitor/use-graph-preview-orchestration";
import {
  useModelPackageInspectionState,
} from "@/features/workbench/state/target/use-model-package-inspection-state";
import {
  useHistoricalRunsState,
  useHistoricalRunSelectionState,
} from "@/features/workbench/state/use-historical-runs-state";
import { type TrainingJob } from "@/lib/api";
import { type WorkbenchWorkspace } from "@/types/workbench";

type ModelPackageInspectionState = ReturnType<
  typeof useModelPackageInspectionState
>;
type HistoricalRunSelectionState = ReturnType<
  typeof useHistoricalRunSelectionState
>;
type HistoricalRunsState = ReturnType<typeof useHistoricalRunsState>;
export type WorkbenchStateOptions = {
  activeWorkspace?: WorkbenchWorkspace;
  activeTrainingJob?: TrainingJob;
  protectedReadsEnabled?: boolean;
};

function modelPackageInspectionCascadeRules(
  historicalRunSelection: HistoricalRunSelectionState,
): Parameters<typeof useModelPackageInspectionState>[0] {
  const clearHistoricalSelection =
    historicalRunSelection.clearHistoricalSelectionForTarget;

  return {
    onModelSelected: clearHistoricalSelection,
    onTargetPresetSelected: clearHistoricalSelection,
    onTargetSnapshotSelected: clearHistoricalSelection,
  };
}

function graphPreviewCompositionInput({
  inspectionState,
  historicalRuns,
  activeTrainingJob,
  protectedReadsEnabled,
}: {
  inspectionState: ModelPackageInspectionState;
  historicalRuns: HistoricalRunsState;
  activeTrainingJob: TrainingJob | undefined;
  protectedReadsEnabled: boolean;
}): Parameters<typeof useGraphPreviewOrchestration>[0] {
  const target = inspectionState.contexts.model.target;
  const historicalGraphPreview = historicalRuns.graphPreview;

  return {
    inspection: inspectionState.inspection,
    activeTrainingJob,
    protectedReadsEnabled,
    historicalMonitorRuns: historicalGraphPreview.historicalMonitorRuns,
    selectedHistoricalExperiment:
      historicalGraphPreview.selectedHistoricalExperiment,
    selectedHistoricalDataset: historicalGraphPreview.selectedHistoricalDataset,
    selectedHistoricalPreset: historicalGraphPreview.selectedHistoricalRunPreset,
    logRunTags: historicalGraphPreview.logRunTags,
    filteredHistoricalRunIds: historicalGraphPreview.filteredHistoricalRunIds,
    targetPreset: target.preset,
    targetDatasets: target.datasets,
  };
}

/**
 * Composition module behind the workbench context providers.
 * Cross-slice cascade rules are named here; domain state stays in extracted hooks.
 */
export function useWorkbenchState(options: WorkbenchStateOptions = {}) {
  const {
    activeWorkspace = "model",
    activeTrainingJob,
    protectedReadsEnabled = true,
  } = options;

  const historicalRunSelection = useHistoricalRunSelectionState();
  const cascadeRules = useMemo(
    () => modelPackageInspectionCascadeRules(historicalRunSelection),
    [historicalRunSelection],
  );

  const inspectionState = useModelPackageInspectionState({
    ...cascadeRules,
    protectedReadsEnabled,
  });
  const modelPackageInspection = inspectionState.contexts.model;
  const { browser, target } = modelPackageInspection;
  const selectedModel = browser.selectedModel;
  const selectedModelType = browser.selectedModelType;
  const selectedExperimentTask = browser.selectedExperimentTask;
  const selectedTargetBrowserMode = browser.mode;

  const historicalTagsEnabled =
    selectedTargetBrowserMode === "experiment" ||
    target.kind === "historical-run" ||
    historicalRunSelection.selectedLogRunId !== null;
  const historicalRunsEnabled =
    activeWorkspace === "logs" ||
    selectedTargetBrowserMode === "experiment" ||
    target.kind === "historical-run";
  const historicalRuns = useHistoricalRunsState({
    selectedModelType,
    selectedModel,
    selectedExperimentTask,
    runsEnabled: protectedReadsEnabled && historicalRunsEnabled,
    tagsEnabled: protectedReadsEnabled && historicalTagsEnabled,
    syncSelectedLogRun: inspectionState.selectHistoricalRunTarget,
    selection: historicalRunSelection,
  });
  const graphPreviewInput = useMemo(
    () =>
      graphPreviewCompositionInput({
        inspectionState,
        historicalRuns,
        activeTrainingJob,
        protectedReadsEnabled,
      }),
    [
      activeTrainingJob,
      historicalRuns,
      inspectionState,
      protectedReadsEnabled,
    ],
  );
  const graphPreviewState = useGraphPreviewOrchestration(graphPreviewInput);
  const clearGraphForConnectionChange = graphPreviewState.clearForConnectionChange;
  const clearForConnectionChange = useCallback(() => {
    historicalRunSelection.clearHistoricalSelectionForTarget();
    clearGraphForConnectionChange();
    inspectionState.inspection.clearForConnectionChange();
  }, [
    clearGraphForConnectionChange,
    historicalRunSelection,
    inspectionState.inspection,
  ]);

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
  const graphMonitor = graphPreviewState.graphMonitor;

  return useMemo(
    () => ({
      targetContexts: inspectionState.contexts,
      graph: graphPreviewState.graph,
      history,
      graphMonitor,
      clearForConnectionChange,
    }),
    [
      clearForConnectionChange,
      graphMonitor,
      graphPreviewState.graph,
      history,
      inspectionState.contexts,
    ],
  );
}

export type WorkbenchState = ReturnType<typeof useWorkbenchState>;
export type GraphViewContextValue = WorkbenchState["graph"];
export type HistoricalRunsContextValue = WorkbenchState["history"];
export type GraphMonitorContextValue = WorkbenchState["graphMonitor"];
