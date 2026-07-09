import { LogsChartPanel } from "@/features/workbench/components/logs/logs-chart-panel";
import { LogsSidebar } from "@/features/workbench/components/logs/logs-sidebar";
import { useLogsWorkspace } from "@/features/workbench/providers/logs-workspace-provider";
import { useLogsChartViewModel } from "@/features/workbench/state/logs/logs-chart-view-model";
import { type LogsWorkspaceState } from "@/features/workbench/state/logs/use-logs-workspace-state";

export function LogsSidebarPanel({ state }: { state: LogsWorkspaceState }) {
  return (
    <LogsSidebar
      runs={state.runs}
      runsQuery={state.runsQuery}
      experimentsQuery={state.experimentsQuery}
      tagsQuery={state.tagsQuery}
      logDeletionEnabled={state.logDeletionEnabled}
      experimentOptions={state.experimentOptions}
      datasetOptions={state.datasetOptions}
      modelOptions={state.modelOptions}
      presetOptions={state.presetOptions}
      tagOptions={state.tagOptions}
      selectedExperiments={state.selectedExperiments}
      selectedDatasets={state.selectedDatasets}
      selectedModels={state.selectedModels}
      selectedPresets={state.selectedPresets}
      selectedTags={state.selectedTags}
      toggleExperiment={state.toggleExperiment}
      toggleDataset={state.toggleDataset}
      toggleModel={state.toggleModel}
      togglePreset={state.togglePreset}
      toggleTag={state.toggleTag}
      selectAllExperiments={state.selectAllExperiments}
      selectNoExperiments={state.selectNoExperiments}
      selectAllDatasets={state.selectAllDatasets}
      selectNoDatasets={state.selectNoDatasets}
      selectAllModels={state.selectAllModels}
      selectNoModels={state.selectNoModels}
      selectAllPresets={state.selectAllPresets}
      selectNoPresets={state.selectNoPresets}
      selectAllTags={state.selectAllTags}
      selectNoTags={state.selectNoTags}
      refreshLogLists={state.refreshLogLists}
      resetDeleteExperiment={state.resetDeleteExperiment}
      deleteExperiment={state.deleteExperiment}
      deleteExperimentError={state.deleteExperimentError}
      isDeletingExperiment={state.isDeletingExperiment}
      resetRunDelete={state.resetRunDelete}
      createRunDeletePlan={state.createRunDeletePlan}
      runDeletePlan={state.runDeletePlan}
      runDeletePlanError={state.runDeletePlanError}
      isPlanningRunDelete={state.isPlanningRunDelete}
      deleteRuns={state.deleteRuns}
      runDeleteError={state.runDeleteError}
      isDeletingRunDelete={state.isDeletingRunDelete}
      loadedScalarTagRunCount={state.loadedScalarTagRunCount}
      totalScalarTagRunCount={state.totalScalarTagRunCount}
      canLoadMoreScalarTags={state.canLoadMoreScalarTags}
      isLoadingMoreScalarTags={state.isLoadingMoreScalarTags}
      loadMoreScalarTags={state.loadMoreScalarTags}
    />
  );
}

export function LogsGraphPreviewPanel({ state }: { state: LogsWorkspaceState }) {
  const chart = useLogsChartViewModel(state);

  return <LogsChartPanel {...chart} />;
}

export function ConnectedLogsSidebarPanel() {
  const state = useLogsWorkspace();
  return <LogsSidebarPanel state={state} />;
}

export function ConnectedLogsGraphPreviewPanel() {
  const state = useLogsWorkspace();
  return <LogsGraphPreviewPanel state={state} />;
}
