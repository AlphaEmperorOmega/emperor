import { useCallback, useMemo } from "react";
import {
  useGraphPreviewOrchestration,
} from "@/features/workbench/state/graph-monitor/use-graph-preview-orchestration";
import {
  useModelPackageInspectionState,
} from "@/features/workbench/state/target/use-model-package-inspection-state";
import type { TrainingJob } from "@/lib/api/training-jobs";
import { type WorkbenchWorkspace } from "@/types/workbench";

type ModelPackageInspectionState = ReturnType<
  typeof useModelPackageInspectionState
>;
export type WorkbenchStateOptions = {
  activeWorkspace?: WorkbenchWorkspace;
  activeTrainingJob?: TrainingJob;
  protectedReadsEnabled?: boolean;
};

function graphPreviewCompositionInput({
  inspectionState,
  activeTrainingJob,
  protectedReadsEnabled,
}: {
  inspectionState: ModelPackageInspectionState;
  activeTrainingJob: TrainingJob | undefined;
  protectedReadsEnabled: boolean;
}): Parameters<typeof useGraphPreviewOrchestration>[0] {
  const target = inspectionState.contexts.model.target;
  const historicalGraphPreview = inspectionState.historical.graphFacts;

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
 * Composition module behind the Workbench context providers. Inspection owns
 * target-scoped history; this layer only adapts its read-only graph facts.
 */
export function useWorkbenchState(options: WorkbenchStateOptions = {}) {
  const {
    activeWorkspace = "model",
    activeTrainingJob,
    protectedReadsEnabled = true,
  } = options;

  const inspectionState = useModelPackageInspectionState({
    historicalRunsEnabled: activeWorkspace === "logs",
    protectedReadsEnabled,
  });
  const graphPreviewInput = useMemo(
    () =>
      graphPreviewCompositionInput({
        inspectionState,
        activeTrainingJob,
        protectedReadsEnabled,
      }),
    [
      activeTrainingJob,
      inspectionState,
      protectedReadsEnabled,
    ],
  );
  const graphPreviewState = useGraphPreviewOrchestration(graphPreviewInput);
  const clearGraphForConnectionChange = graphPreviewState.clearForConnectionChange;
  const clearForConnectionChange = useCallback(() => {
    clearGraphForConnectionChange();
    inspectionState.inspection.clearForConnectionChange();
  }, [
    clearGraphForConnectionChange,
    inspectionState.inspection,
  ]);

  const history = useMemo(
    () => ({
      ...inspectionState.historical.browsing,
      selectedLogRunHasMonitorTags:
        graphPreviewState.history.selectedLogRunHasMonitorTags,
    }),
    [
      graphPreviewState.history.selectedLogRunHasMonitorTags,
      inspectionState.historical.browsing,
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
