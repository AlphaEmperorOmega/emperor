import { type LogRun, type LogRunTags } from "@/lib/api";
import {
  filterHistoricalRuns,
  filterLogRunsByExperimentTask,
  historicalDatasetOptions as buildHistoricalDatasetOptions,
  historicalExperimentRunOptions,
  historicalPresetOptions as buildHistoricalPresetOptions,
  latestHistoricalMonitorRuns,
  logRunHasLayerMonitorData,
  monitorEligibilityDescription,
  sortLogRunsNewestFirst,
  type HistoricalRunOption,
  type MonitorEligibility,
} from "@/lib/historical-monitor-runs";

export type DatasetSelectionInput = {
  logRuns?: LogRun[];
  modelRunTags?: LogRunTags[];
  includeRunsWithoutMonitorTags?: boolean;
  selectedModelType?: string;
  selectedModel: string;
  selectedExperimentTask?: string;
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
  selectedLogRunMonitorEligibility: MonitorEligibility | undefined;
};

function tagsByRunId(modelRunTags: LogRunTags[] | undefined) {
  return new Map((modelRunTags ?? []).map((tags) => [tags.runId, tags]));
}

function monitorEligibilityForRun({
  run,
  modelRunTagsByRunId,
}: {
  run: LogRun;
  modelRunTagsByRunId: Map<string, LogRunTags>;
}): MonitorEligibility {
  const tags = modelRunTagsByRunId.get(run.id);
  if (tags) {
    return logRunHasLayerMonitorData(tags) ? "eligible" : "ineligible";
  }
  if (run.hasLayerMonitorData === true) {
    return "eligible";
  }
  if (run.hasLayerMonitorData === false) {
    return "ineligible";
  }
  return "checking";
}

function monitorEligibleRuns({
  modelLogRuns,
  modelRunTags,
}: {
  modelLogRuns: LogRun[];
  modelRunTags: LogRunTags[] | undefined;
}) {
  const modelRunTagsByRunId = tagsByRunId(modelRunTags);
  return modelLogRuns.filter(
    (run) =>
      monitorEligibilityForRun({ run, modelRunTagsByRunId }) === "eligible",
  );
}

export function deriveDatasetSelectionState(
  input: DatasetSelectionInput,
): DatasetSelectionState {
  const selectedHistoricalExperimentFilter =
    input.selectedHistoricalExperimentFilter ?? "";
  const selectedHistoricalDatasetFilter =
    input.selectedHistoricalDatasetFilter ?? "";
  const modelLogRuns = sortLogRunsNewestFirst(
    filterLogRunsByExperimentTask(
      (input.logRuns ?? []).filter(
        (run) =>
          run.model === input.selectedModel &&
          (!input.selectedModelType || run.modelType === input.selectedModelType),
      ),
      input.selectedExperimentTask ?? "",
    ),
  );
  const modelRunTagsByRunId = tagsByRunId(input.modelRunTags);
  const getMonitorEligibility = (run: LogRun) =>
    monitorEligibilityForRun({ run, modelRunTagsByRunId });
  const eligibleRuns = monitorEligibleRuns({
    modelLogRuns,
    modelRunTags: input.modelRunTags,
  });
  const eligibleRunIds = new Set(eligibleRuns.map((run) => run.id));
  const historicalExperimentOptions = historicalExperimentRunOptions(
    modelLogRuns,
    getMonitorEligibility,
  );
  const historicalDatasetOptions = selectedHistoricalExperimentFilter
    ? buildHistoricalDatasetOptions(
        modelLogRuns,
        selectedHistoricalExperimentFilter,
        getMonitorEligibility,
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
      ? buildHistoricalPresetOptions(
          datasetFilteredHistoricalRuns,
          getMonitorEligibility,
        )
      : [];
  const visibleHistoricalRuns = filterHistoricalRuns(
    modelLogRuns,
    selectedHistoricalExperimentFilter,
    selectedHistoricalDatasetFilter,
    input.selectedHistoricalPreset,
  );
  const selectedModelLogRun = input.selectedLogRunId
    ? modelLogRuns.find((run) => run.id === input.selectedLogRunId)
    : undefined;
  const selectedLogRun = selectedModelLogRun;
  const selectedLogRunMonitorEligibility = selectedLogRun
    ? getMonitorEligibility(selectedLogRun)
    : undefined;
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
        modelLogRuns,
        selectedHistoricalExperiment,
        selectedHistoricalDataset,
        selectedHistoricalRunPreset,
      )
    : [];
  const eligibleFilteredHistoricalRuns = filteredHistoricalRuns.filter((run) =>
    eligibleRunIds.has(run.id),
  );
  const historicalMonitorRuns = latestHistoricalMonitorRuns(
    eligibleFilteredHistoricalRuns,
  );
  const filteredHistoricalRunIds = eligibleFilteredHistoricalRuns.map((run) => run.id);

  return {
    modelLogRuns,
    historicalExperimentOptions: historicalExperimentOptions.map((option) => ({
      ...option,
      description: option.description ?? monitorEligibilityDescription(option.monitorEligibility),
    })),
    historicalDatasetOptions: historicalDatasetOptions.map((option) => ({
      ...option,
      description: option.description ?? monitorEligibilityDescription(option.monitorEligibility),
    })),
    historicalPresetOptions: historicalPresetOptions.map((option) => ({
      ...option,
      description: option.description ?? monitorEligibilityDescription(option.monitorEligibility),
    })),
    visibleHistoricalRuns,
    selectedHistoricalExperiment,
    selectedHistoricalDataset,
    selectedHistoricalRunPreset,
    filteredHistoricalRuns,
    historicalMonitorRuns,
    filteredHistoricalRunIds,
    selectedLogRun,
    selectedLogRunMonitorEligibility,
  };
}
