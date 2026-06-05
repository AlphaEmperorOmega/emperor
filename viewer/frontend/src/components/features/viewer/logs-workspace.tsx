import { useMemo, useState } from "react";
import {
  LogsChartPanel,
  type LogsChartEmptyState,
  type ScalarChartGridMode,
} from "@/components/features/viewer/logs/logs-chart-panel";
import { LogsSidebar } from "@/components/features/viewer/logs/logs-sidebar";
import { type LogScalarSeries } from "@/lib/api";
import { useLogScalarsQuery } from "@/hooks/use-log-queries";
import { logQueryKeys } from "@/lib/query-keys";
import { type LogsWorkspaceState } from "@/components/features/viewer/state/use-logs-workspace-state";

export function LogsSidebarPanel({ state }: { state: LogsWorkspaceState }) {
  return (
    <LogsSidebar
      runs={state.runs}
      visibleRuns={state.visibleRuns}
      runsQuery={state.runsQuery}
      experimentsQuery={state.experimentsQuery}
      tagsQuery={state.tagsQuery}
      experimentOptions={state.experimentOptions}
      datasetOptions={state.datasetOptions}
      modelOptions={state.modelOptions}
      presetOptions={state.presetOptions}
      runOptions={state.runOptions}
      tagOptions={state.tagOptions}
      selectedExperiments={state.selectedExperiments}
      selectedDatasets={state.selectedDatasets}
      selectedModels={state.selectedModels}
      selectedPresets={state.selectedPresets}
      selectedRunIds={state.selectedRunIds}
      selectedTags={state.selectedTags}
      toggleExperiment={state.toggleExperiment}
      toggleDataset={state.toggleDataset}
      toggleModel={state.toggleModel}
      togglePreset={state.togglePreset}
      toggleRun={state.toggleRun}
      toggleTag={state.toggleTag}
      selectAllExperiments={state.selectAllExperiments}
      selectNoExperiments={state.selectNoExperiments}
      selectAllDatasets={state.selectAllDatasets}
      selectNoDatasets={state.selectNoDatasets}
      selectAllModels={state.selectAllModels}
      selectNoModels={state.selectNoModels}
      selectAllPresets={state.selectAllPresets}
      selectNoPresets={state.selectNoPresets}
      selectAllRuns={state.selectAllRuns}
      selectNoRuns={state.selectNoRuns}
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
    />
  );
}

export function LogsGraphPreviewPanel({ state }: { state: LogsWorkspaceState }) {
  const [scalarChartGridMode, setScalarChartGridMode] =
    useState<ScalarChartGridMode>("full");
  const scalarQuery = useLogScalarsQuery({
    runIds: state.visibleRunIds,
    tags: state.selectedTagList,
    enabled: state.enabled,
    queryKey: logQueryKeys.scalarsForRunsAndTags(
      state.visibleRunIds,
      state.selectedTagList,
    ),
  });

  const runsById = useMemo(
    () => new Map(state.visibleRuns.map((run) => [run.id, run])),
    [state.visibleRuns],
  );
  const seriesByTag = useMemo(() => {
    const byTag = new Map<string, LogScalarSeries[]>();
    for (const series of scalarQuery.data?.series ?? []) {
      if (series.points.length === 0) {
        continue;
      }
      byTag.set(series.tag, [...(byTag.get(series.tag) ?? []), series]);
    }
    return byTag;
  }, [scalarQuery.data]);
  const selectedSeriesCount = Array.from(seriesByTag.values()).reduce(
    (total, series) => total + series.length,
    0,
  );
  const hasEventFiles = state.visibleRuns.some((run) => run.eventFileCount > 0);

  let emptyTitle = "";
  let emptyDetail = "";
  if (state.runsQuery.isLoading) {
    emptyTitle = "Scanning logs";
    emptyDetail = "Reading historical run folders.";
  } else if (state.visibleRuns.length === 0) {
    emptyTitle = "No runs selected";
    emptyDetail = "Use the sidebar filters to include at least one historical run.";
  } else if (state.tagsQuery.isLoading) {
    emptyTitle = "Reading TensorBoard tags";
    emptyDetail = "Collecting scalar tags from the selected runs.";
  } else if (state.tagOptions.length === 0) {
    emptyTitle = "No TensorBoard scalars";
    emptyDetail = "The selected runs do not contain scalar event data.";
  } else if (state.selectedTagList.length === 0) {
    emptyTitle = "No scalar tags selected";
    emptyDetail = "Select one or more scalar tags to draw historical charts.";
  } else if (scalarQuery.isLoading) {
    emptyTitle = "Loading scalar points";
    emptyDetail = "Reading TensorBoard scalar series for the selected runs.";
  } else if (selectedSeriesCount === 0 && hasEventFiles) {
    emptyTitle = "No scalar points for selection";
    emptyDetail = "The selected runs have event files, but none contain the checked scalar tags.";
  } else if (selectedSeriesCount === 0) {
    emptyTitle = "No TensorBoard scalars";
    emptyDetail = "The selected runs do not contain scalar event data.";
  }

  const emptyState: LogsChartEmptyState | null = emptyTitle
    ? {
        title: emptyTitle,
        detail: emptyDetail,
        busy: state.runsQuery.isLoading || state.tagsQuery.isLoading || scalarQuery.isLoading,
      }
    : null;

  return (
    <LogsChartPanel
      selectedTagList={state.selectedTagList}
      seriesByTag={seriesByTag}
      runsById={runsById}
      runOrder={state.visibleRunIds}
      visibleRunCount={state.visibleRuns.length}
      selectedTagCount={state.selectedTagList.length}
      gridMode={scalarChartGridMode}
      onGridModeChange={setScalarChartGridMode}
      isFetching={scalarQuery.isFetching}
      isRefreshDisabled={!scalarQuery.isSuccess && !scalarQuery.isError}
      onRefresh={() => {
        void scalarQuery.refetch();
      }}
      isError={scalarQuery.isError}
      error={scalarQuery.error}
      emptyState={emptyState}
      onSelectRun={state.setSelectedDetailRunId}
    />
  );
}
