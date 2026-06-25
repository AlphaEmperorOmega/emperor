import { type LogRun, type LogRunTags } from "@/lib/api";
import {
  filterHistoricalRuns,
  historicalDatasetOptions as buildHistoricalDatasetOptions,
  historicalExperimentRunOptions,
  historicalPresetOptions as buildHistoricalPresetOptions,
  latestHistoricalMonitorRuns,
  logRunHasLayerMonitorData,
  sortLogRunsNewestFirst,
  type HistoricalRunOption,
} from "@/lib/historical-monitor-runs";

export type DatasetSelectionInput = {
  logRuns?: LogRun[];
  modelRunTags?: LogRunTags[];
  includeRunsWithoutMonitorTags?: boolean;
  selectedModelType?: string;
  selectedModel: string;
  selectedHistoricalExperimentFilter?: string;
  selectedHistoricalDatasetFilter?: string;
  selectedHistoricalPreset: string;
  selectedLogRunId: string | null;
};

export type DatasetSelectionState = {
  modelLogRuns: LogRun[];
  historicalExperimentOptions: HistoricalRunOption[];
  historicalDatasetOptions: HistoricalRunOption[];
  historicalPresetOptions: HistoricalRunOption[];
  visibleHistoricalRuns: LogRun[];
  selectedHistoricalExperiment: string;
  selectedHistoricalDataset: string;
  selectedHistoricalRunPreset: string;
  filteredHistoricalRuns: LogRun[];
  historicalMonitorRuns: LogRun[];
  filteredHistoricalRunIds: string[];
  selectedLogRun: LogRun | undefined;
};

function monitorEligibleRunIds({
  modelLogRuns,
  modelRunTags,
  includeRunsWithoutMonitorTags,
}: {
  modelLogRuns: LogRun[];
  modelRunTags: LogRunTags[] | undefined;
  includeRunsWithoutMonitorTags: boolean | undefined;
}) {
  if (modelRunTags === undefined && includeRunsWithoutMonitorTags) {
    return new Set(modelLogRuns.map((run) => run.id));
  }

  const runIds = new Set<string>();
  for (const tags of modelRunTags ?? []) {
    if (logRunHasLayerMonitorData(tags)) {
      runIds.add(tags.runId);
    }
  }
  return runIds;
}

export function deriveDatasetSelectionState(
  input: DatasetSelectionInput,
): DatasetSelectionState {
  const selectedHistoricalExperimentFilter =
    input.selectedHistoricalExperimentFilter ?? "";
  const selectedHistoricalDatasetFilter =
    input.selectedHistoricalDatasetFilter ?? "";
  const modelLogRuns = sortLogRunsNewestFirst(
    (input.logRuns ?? []).filter(
      (run) =>
        run.model === input.selectedModel &&
        (!input.selectedModelType || run.modelType === input.selectedModelType),
    ),
  );
  const historicalExperimentOptions = historicalExperimentRunOptions(modelLogRuns);
  const historicalDatasetOptions = selectedHistoricalExperimentFilter
    ? buildHistoricalDatasetOptions(
        modelLogRuns,
        selectedHistoricalExperimentFilter,
      )
    : [];
  const datasetFilteredHistoricalRuns =
    selectedHistoricalExperimentFilter && selectedHistoricalDatasetFilter
      ? filterHistoricalRuns(
          modelLogRuns,
          selectedHistoricalExperimentFilter,
          selectedHistoricalDatasetFilter,
        )
      : [];
  const historicalPresetOptions =
    selectedHistoricalExperimentFilter && selectedHistoricalDatasetFilter
      ? buildHistoricalPresetOptions(datasetFilteredHistoricalRuns)
      : [];
  const visibleHistoricalRuns = filterHistoricalRuns(
    modelLogRuns,
    selectedHistoricalExperimentFilter,
    selectedHistoricalDatasetFilter,
    input.selectedHistoricalPreset,
  );
  const eligibleRunIds = monitorEligibleRunIds({
    modelLogRuns,
    modelRunTags: input.modelRunTags,
    includeRunsWithoutMonitorTags: input.includeRunsWithoutMonitorTags,
  });
  const eligibleRuns = modelLogRuns.filter((run) =>
    eligibleRunIds.has(run.id),
  );
  const selectedModelLogRun = input.selectedLogRunId
    ? modelLogRuns.find((run) => run.id === input.selectedLogRunId)
    : undefined;
  const selectedLogRun = selectedModelLogRun;
  const hasCompleteCascade = Boolean(
    selectedHistoricalExperimentFilter &&
      selectedHistoricalDatasetFilter &&
      input.selectedHistoricalPreset,
  );
  const selectedHistoricalExperiment =
    selectedLogRun?.experiment ??
    (hasCompleteCascade ? selectedHistoricalExperimentFilter : "");
  const selectedHistoricalDataset =
    selectedLogRun?.dataset ??
    (hasCompleteCascade ? selectedHistoricalDatasetFilter : "");
  const selectedHistoricalRunPreset =
    selectedLogRun?.preset ??
    (hasCompleteCascade ? input.selectedHistoricalPreset : "");
  const hasMonitorSelection = Boolean(
    selectedHistoricalExperiment &&
      selectedHistoricalDataset &&
      selectedHistoricalRunPreset,
  );
  const filteredHistoricalRuns = hasMonitorSelection
    ? filterHistoricalRuns(
        eligibleRuns,
        selectedHistoricalExperiment,
        selectedHistoricalDataset,
        selectedHistoricalRunPreset,
      )
    : [];
  const historicalMonitorRuns = latestHistoricalMonitorRuns(filteredHistoricalRuns);
  const filteredHistoricalRunIds = filteredHistoricalRuns.map((run) => run.id);

  return {
    modelLogRuns,
    historicalExperimentOptions,
    historicalDatasetOptions,
    historicalPresetOptions,
    visibleHistoricalRuns,
    selectedHistoricalExperiment,
    selectedHistoricalDataset,
    selectedHistoricalRunPreset,
    filteredHistoricalRuns,
    historicalMonitorRuns,
    filteredHistoricalRunIds,
    selectedLogRun,
  };
}
