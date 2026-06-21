import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import { type LogRun, type LogRunTags } from "@/lib/api";
import { useLogRunsQuery, useLogTagsQuery } from "@/features/viewer/state/logs/use-log-queries";
import {
  deriveDatasetSelectionState,
} from "@/features/viewer/state/logs/historical-run-selection";
import { logQueryKeys } from "@/lib/query-keys";

type HistoricalRunSelectionState = {
  selectedLogRunId: string | null;
  selectedHistoricalPreset: string;
  setSelectedHistoricalPreset: Dispatch<SetStateAction<string>>;
  setSelectedLogRunId: Dispatch<SetStateAction<string | null>>;
  clearHistoricalSelectionForTarget: () => void;
  selectLogRun: (runId: string) => void;
};

type HistoricalRunsStateInput = {
  selectedModel: string;
  tagsEnabled?: boolean;
  syncSelectedLogRun: (selectedLogRun: LogRun) => void;
  selection: HistoricalRunSelectionState;
};

type HistoricalRunsProviderSlice = {
  visibleHistoricalRuns: LogRun[];
  historicalMonitorRuns: LogRun[];
  historicalPresetOptions: ReturnType<
    typeof deriveDatasetSelectionState
  >["historicalPresetOptions"];
  selectedHistoricalPreset: string;
  setSelectedHistoricalPreset: Dispatch<SetStateAction<string>>;
  selectedHistoricalExperiment: string;
  selectedHistoricalDataset: string;
  selectedHistoricalRunPreset: string;
  selectedLogRunId: string | null;
  selectLogRun: (runId: string) => void;
  logRunTagsLoading: boolean;
  experimentsLoading: boolean;
  experimentsError: Error | null;
};

type HistoricalRunsGraphPreviewState = {
  historicalMonitorRuns: LogRun[];
  selectedHistoricalExperiment: string;
  selectedHistoricalDataset: string;
  selectedHistoricalRunPreset: string;
  logRunTags?: LogRunTags[];
  filteredHistoricalRunIds: string[];
};

type HistoricalRunsState = {
  history: HistoricalRunsProviderSlice;
  graphPreview: HistoricalRunsGraphPreviewState;
};

export function useHistoricalRunSelectionState(): HistoricalRunSelectionState {
  const [selectedLogRunId, setSelectedLogRunId] = useState<string | null>(null);
  const [selectedHistoricalPreset, setSelectedHistoricalPreset] = useState("");

  const clearHistoricalSelectionForTarget = useCallback(() => {
    setSelectedHistoricalPreset("");
    setSelectedLogRunId(null);
  }, []);

  const selectLogRun = useCallback((runId: string) => {
    setSelectedLogRunId((current) => (current === runId ? current : runId));
  }, []);

  return useMemo(
    () => ({
      selectedLogRunId,
      selectedHistoricalPreset,
      setSelectedHistoricalPreset,
      setSelectedLogRunId,
      clearHistoricalSelectionForTarget,
      selectLogRun,
    }),
    [
      clearHistoricalSelectionForTarget,
      selectLogRun,
      selectedHistoricalPreset,
      selectedLogRunId,
    ],
  );
}

export function useHistoricalRunsState({
  selectedModel,
  tagsEnabled = true,
  syncSelectedLogRun,
  selection,
}: HistoricalRunsStateInput): HistoricalRunsState {
  const {
    selectedLogRunId,
    selectedHistoricalPreset,
    setSelectedHistoricalPreset,
    setSelectedLogRunId,
    selectLogRun,
  } = selection;

  const logRunsQuery = useLogRunsQuery();
  const modelLogRunIds = useMemo(
    () =>
      (logRunsQuery.data?.runs ?? [])
        .filter((run) => run.model === selectedModel)
        .map((run) => run.id),
    [logRunsQuery.data?.runs, selectedModel],
  );
  const modelRunTagsQuery = useLogTagsQuery({
    runIds: modelLogRunIds,
    enabled: tagsEnabled,
    queryKey: logQueryKeys.modelRunTags(modelLogRunIds),
  });
  const datasetSelectionState = useMemo(
    () =>
      deriveDatasetSelectionState({
        logRuns: logRunsQuery.data?.runs,
        modelRunTags: modelRunTagsQuery.data?.runs,
        selectedModel,
        selectedHistoricalPreset,
        selectedLogRunId,
      }),
    [
      logRunsQuery.data?.runs,
      modelRunTagsQuery.data?.runs,
      selectedHistoricalPreset,
      selectedLogRunId,
      selectedModel,
    ],
  );
  const {
    modelLogRuns,
    historicalPresetOptions,
    visibleHistoricalRuns,
    selectedHistoricalExperiment,
    selectedHistoricalDataset,
    selectedHistoricalRunPreset,
    historicalMonitorRuns,
    filteredHistoricalRunIds,
    selectedLogRun,
  } = datasetSelectionState;
  const logRunTagsQuery = useLogTagsQuery({
    runIds: filteredHistoricalRunIds,
    enabled: tagsEnabled && selectedLogRunId !== null,
    queryKey: logQueryKeys.filteredHistoricalRunTags(filteredHistoricalRunIds),
  });

  useEffect(() => {
    if (!selectedModel) {
      setSelectedHistoricalPreset("");
      return;
    }
    setSelectedHistoricalPreset((current) =>
      current && historicalPresetOptions.some((option) => option.value === current)
        ? current
        : "",
    );
  }, [historicalPresetOptions, selectedModel, setSelectedHistoricalPreset]);

  useEffect(() => {
    if (!selectedModel) {
      setSelectedLogRunId(null);
      return;
    }
    if (logRunsQuery.isLoading && !logRunsQuery.data) {
      return;
    }
    setSelectedLogRunId((current) =>
      current && modelLogRuns.some((run) => run.id === current)
        ? current
        : null,
    );
  }, [
    logRunsQuery.data,
    logRunsQuery.isLoading,
    modelLogRuns,
    selectedModel,
    setSelectedLogRunId,
  ]);

  useEffect(() => {
    if (!selectedModel || !selectedLogRun) {
      return;
    }
    syncSelectedLogRun(selectedLogRun);
  }, [selectedLogRun, selectedModel, syncSelectedLogRun]);

  const history = useMemo(
    () => ({
      visibleHistoricalRuns,
      historicalMonitorRuns,
      historicalPresetOptions,
      selectedHistoricalPreset,
      setSelectedHistoricalPreset,
      selectedHistoricalExperiment,
      selectedHistoricalDataset,
      selectedHistoricalRunPreset,
      selectedLogRunId,
      selectLogRun,
      logRunTagsLoading: logRunTagsQuery.isLoading,
      experimentsLoading:
        logRunsQuery.isLoading || (tagsEnabled && modelRunTagsQuery.isLoading),
      experimentsError: logRunsQuery.error ?? (tagsEnabled ? modelRunTagsQuery.error : null),
    }),
    [
      historicalMonitorRuns,
      historicalPresetOptions,
      logRunTagsQuery.isLoading,
      logRunsQuery.error,
      logRunsQuery.isLoading,
      modelRunTagsQuery.error,
      modelRunTagsQuery.isLoading,
      selectLogRun,
      selectedHistoricalDataset,
      selectedHistoricalExperiment,
      selectedHistoricalPreset,
      selectedHistoricalRunPreset,
      selectedLogRunId,
      setSelectedHistoricalPreset,
      tagsEnabled,
      visibleHistoricalRuns,
    ],
  );
  const graphPreview = useMemo(
    () => ({
      historicalMonitorRuns,
      selectedHistoricalExperiment,
      selectedHistoricalDataset,
      selectedHistoricalRunPreset,
      logRunTags: logRunTagsQuery.data?.runs,
      filteredHistoricalRunIds,
    }),
    [
      filteredHistoricalRunIds,
      historicalMonitorRuns,
      logRunTagsQuery.data?.runs,
      selectedHistoricalDataset,
      selectedHistoricalExperiment,
      selectedHistoricalRunPreset,
    ],
  );

  return useMemo(
    () => ({
      history,
      graphPreview,
    }),
    [graphPreview, history],
  );
}
