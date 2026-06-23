import { type LogRun, type LogRunTags } from "@/lib/api";
import {
  experimentsWithLayerMonitorData,
  filterHistoricalRuns,
  historicalPresetOptions as buildHistoricalPresetOptions,
  latestHistoricalMonitorRuns,
  sortLogRunsNewestFirst,
  type HistoricalRunOption,
} from "@/lib/historical-monitor-runs";

export type DatasetSelectionInput = {
  logRuns?: LogRun[];
  modelRunTags?: LogRunTags[];
  includeRunsWithoutMonitorTags?: boolean;
  selectedModel: string;
  selectedHistoricalPreset: string;
  selectedLogRunId: string | null;
};

export type DatasetSelectionState = {
  modelLogRuns: LogRun[];
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

export function deriveDatasetSelectionState(
  input: DatasetSelectionInput,
): DatasetSelectionState {
  const modelLogRuns = sortLogRunsNewestFirst(
    (input.logRuns ?? []).filter((run) => run.model === input.selectedModel),
  );
  const layerMonitorExperiments =
    input.modelRunTags === undefined && input.includeRunsWithoutMonitorTags
      ? new Set(modelLogRuns.map((run) => run.experiment))
      : experimentsWithLayerMonitorData(
          modelLogRuns,
          input.modelRunTags ?? [],
        );
  const eligibleRuns = modelLogRuns.filter((run) =>
    layerMonitorExperiments.has(run.experiment),
  );
  const selectedModelLogRun = input.selectedLogRunId
    ? modelLogRuns.find((run) => run.id === input.selectedLogRunId)
    : undefined;
  const historicalPresetOptions = buildHistoricalPresetOptions(eligibleRuns);
  const filteredVisibleHistoricalRuns = eligibleRuns.filter(
    (run) =>
      !input.selectedHistoricalPreset || run.preset === input.selectedHistoricalPreset,
  );
  const visibleHistoricalRuns =
    selectedModelLogRun &&
    !filteredVisibleHistoricalRuns.some((run) => run.id === selectedModelLogRun.id)
      ? sortLogRunsNewestFirst([
          ...filteredVisibleHistoricalRuns,
          selectedModelLogRun,
        ])
      : filteredVisibleHistoricalRuns;
  const selectedLogRun = selectedModelLogRun;
  const selectedHistoricalExperiment = selectedLogRun?.experiment ?? "";
  const selectedHistoricalDataset = selectedLogRun?.dataset ?? "";
  const selectedHistoricalRunPreset = selectedLogRun?.preset ?? "";
  const selectedLogRunIsMonitorEligible = selectedLogRun
    ? eligibleRuns.some((run) => run.id === selectedLogRun.id)
    : false;
  const filteredHistoricalRuns = selectedLogRun
    ? selectedLogRunIsMonitorEligible
      ? filterHistoricalRuns(
          eligibleRuns,
          selectedHistoricalExperiment,
          selectedHistoricalDataset,
          selectedHistoricalRunPreset,
        )
      : []
    : [];
  const historicalMonitorRuns = latestHistoricalMonitorRuns(filteredHistoricalRuns);
  const filteredHistoricalRunIds = filteredHistoricalRuns.map((run) => run.id);

  return {
    modelLogRuns,
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
