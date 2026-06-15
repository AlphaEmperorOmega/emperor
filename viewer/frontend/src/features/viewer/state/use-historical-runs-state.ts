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
} from "@/features/viewer/state/graph-monitor/graph-monitor-selectors";
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
  logRunsQuery: ReturnType<typeof useLogRunsQuery>;
  logRunTagsQuery: ReturnType<typeof useLogTagsQuery>;
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

  return {
    selectedLogRunId,
    selectedHistoricalPreset,
    setSelectedHistoricalPreset,
    setSelectedLogRunId,
    clearHistoricalSelectionForTarget,
    selectLogRun,
  };
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
    setSelectedLogRunId((current) =>
      current && visibleHistoricalRuns.some((run) => run.id === current)
        ? current
        : null,
    );
  }, [selectedModel, setSelectedLogRunId, visibleHistoricalRuns]);

  useEffect(() => {
    if (!selectedModel || !selectedLogRun) {
      return;
    }
    syncSelectedLogRun(selectedLogRun);
  }, [selectedLogRun, selectedModel, syncSelectedLogRun]);

  return {
    history: {
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
      logRunsQuery,
      logRunTagsQuery,
      experimentsLoading:
        logRunsQuery.isLoading || (tagsEnabled && modelRunTagsQuery.isLoading),
      experimentsError: logRunsQuery.error ?? (tagsEnabled ? modelRunTagsQuery.error : null),
    },
    graphPreview: {
      historicalMonitorRuns,
      selectedHistoricalExperiment,
      selectedHistoricalDataset,
      selectedHistoricalRunPreset,
      logRunTags: logRunTagsQuery.data?.runs,
      filteredHistoricalRunIds,
    },
  };
}
