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
  selectedHistoricalExperimentFilter: string;
  setSelectedHistoricalExperimentFilter: (experiment: string) => void;
  selectedHistoricalDatasetFilter: string;
  setSelectedHistoricalDatasetFilter: (dataset: string) => void;
  selectedHistoricalPreset: string;
  setSelectedHistoricalPreset: (preset: string) => void;
  setSelectedLogRunId: Dispatch<SetStateAction<string | null>>;
  clearHistoricalSelectionForTarget: () => void;
  selectLogRun: (runId: string) => void;
};

type HistoricalRunsStateInput = {
  selectedModelType: string;
  selectedModel: string;
  tagsEnabled?: boolean;
  syncSelectedLogRun: (selectedLogRun: LogRun) => void;
  clearSelectedExperimentRun: () => void;
  selection: HistoricalRunSelectionState;
};

type HistoricalRunsProviderSlice = {
  visibleHistoricalRuns: LogRun[];
  historicalMonitorRuns: LogRun[];
  historicalExperimentOptions: ReturnType<
    typeof deriveDatasetSelectionState
  >["historicalExperimentOptions"];
  historicalDatasetOptions: ReturnType<
    typeof deriveDatasetSelectionState
  >["historicalDatasetOptions"];
  historicalPresetOptions: ReturnType<
    typeof deriveDatasetSelectionState
  >["historicalPresetOptions"];
  selectedHistoricalExperimentFilter: string;
  setSelectedHistoricalExperimentFilter: (experiment: string) => void;
  selectedHistoricalDatasetFilter: string;
  setSelectedHistoricalDatasetFilter: (dataset: string) => void;
  selectedHistoricalPreset: string;
  setSelectedHistoricalPreset: (preset: string) => void;
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
  const [
    selectedHistoricalExperimentFilter,
    setSelectedHistoricalExperimentFilterValue,
  ] = useState("");
  const [
    selectedHistoricalDatasetFilter,
    setSelectedHistoricalDatasetFilterValue,
  ] = useState("");
  const [selectedHistoricalPreset, setSelectedHistoricalPreset] = useState("");

  const clearHistoricalSelectionForTarget = useCallback(() => {
    setSelectedHistoricalExperimentFilterValue("");
    setSelectedHistoricalDatasetFilterValue("");
    setSelectedHistoricalPreset("");
    setSelectedLogRunId(null);
  }, []);

  const setSelectedHistoricalExperimentFilter = useCallback(
    (experiment: string) => {
      if (experiment === selectedHistoricalExperimentFilter) {
        return;
      }
      setSelectedHistoricalExperimentFilterValue(experiment);
      setSelectedHistoricalDatasetFilterValue("");
      setSelectedHistoricalPreset("");
      setSelectedLogRunId(null);
    },
    [selectedHistoricalExperimentFilter],
  );

  const setSelectedHistoricalDatasetFilter = useCallback(
    (dataset: string) => {
      if (dataset === selectedHistoricalDatasetFilter) {
        return;
      }
      setSelectedHistoricalDatasetFilterValue(dataset);
      setSelectedHistoricalPreset("");
      setSelectedLogRunId(null);
    },
    [selectedHistoricalDatasetFilter],
  );

  const setSelectedHistoricalPresetFilter = useCallback(
    (preset: string) => {
      if (preset === selectedHistoricalPreset) {
        return;
      }
      setSelectedHistoricalPreset(preset);
      setSelectedLogRunId(null);
    },
    [selectedHistoricalPreset],
  );

  const selectLogRun = useCallback((runId: string) => {
    setSelectedLogRunId((current) => (current === runId ? current : runId));
  }, []);

  return useMemo(
    () => ({
      selectedLogRunId,
      selectedHistoricalExperimentFilter,
      setSelectedHistoricalExperimentFilter,
      selectedHistoricalDatasetFilter,
      setSelectedHistoricalDatasetFilter,
      selectedHistoricalPreset,
      setSelectedHistoricalPreset: setSelectedHistoricalPresetFilter,
      setSelectedLogRunId,
      clearHistoricalSelectionForTarget,
      selectLogRun,
    }),
    [
      clearHistoricalSelectionForTarget,
      selectLogRun,
      selectedHistoricalDatasetFilter,
      selectedHistoricalExperimentFilter,
      selectedHistoricalPreset,
      selectedLogRunId,
      setSelectedHistoricalDatasetFilter,
      setSelectedHistoricalExperimentFilter,
      setSelectedHistoricalPresetFilter,
    ],
  );
}

export function useHistoricalRunsState({
  selectedModelType,
  selectedModel,
  tagsEnabled = true,
  syncSelectedLogRun,
  clearSelectedExperimentRun,
  selection,
}: HistoricalRunsStateInput): HistoricalRunsState {
  const {
    selectedLogRunId,
    selectedHistoricalExperimentFilter,
    setSelectedHistoricalExperimentFilter,
    selectedHistoricalDatasetFilter,
    setSelectedHistoricalDatasetFilter,
    selectedHistoricalPreset,
    setSelectedHistoricalPreset,
    setSelectedLogRunId,
    selectLogRun,
  } = selection;

  const logRunsQuery = useLogRunsQuery();
  const modelLogRunIds = useMemo(
    () =>
      (logRunsQuery.data?.runs ?? [])
        .filter(
          (run) =>
            run.modelType === selectedModelType && run.model === selectedModel,
        )
        .map((run) => run.id),
    [logRunsQuery.data?.runs, selectedModel, selectedModelType],
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
        includeRunsWithoutMonitorTags: !tagsEnabled,
        selectedModelType,
        selectedModel,
        selectedHistoricalExperimentFilter,
        selectedHistoricalDatasetFilter,
        selectedHistoricalPreset,
        selectedLogRunId,
      }),
    [
      logRunsQuery.data?.runs,
      modelRunTagsQuery.data?.runs,
      selectedHistoricalDatasetFilter,
      selectedHistoricalExperimentFilter,
      selectedHistoricalPreset,
      selectedLogRunId,
      selectedModel,
      selectedModelType,
      tagsEnabled,
    ],
  );
  const {
    historicalExperimentOptions,
    historicalDatasetOptions,
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
    enabled: tagsEnabled && filteredHistoricalRunIds.length > 0,
    queryKey: logQueryKeys.filteredHistoricalRunTags(filteredHistoricalRunIds),
  });
  const modelRunTagsLoading =
    tagsEnabled && modelRunTagsQuery.isLoading && !modelRunTagsQuery.data;

  useEffect(() => {
    if (!selectedModel) {
      setSelectedHistoricalExperimentFilter("");
      return;
    }
    if (modelRunTagsLoading) {
      return;
    }
    if (
      selectedHistoricalExperimentFilter &&
      !historicalExperimentOptions.some(
        (option) => option.value === selectedHistoricalExperimentFilter,
      )
    ) {
      setSelectedHistoricalExperimentFilter("");
    }
  }, [
    historicalExperimentOptions,
    modelRunTagsLoading,
    selectedHistoricalExperimentFilter,
    selectedModel,
    setSelectedHistoricalExperimentFilter,
  ]);

  useEffect(() => {
    if (!selectedHistoricalExperimentFilter) {
      setSelectedHistoricalDatasetFilter("");
      return;
    }
    if (modelRunTagsLoading) {
      return;
    }
    if (
      selectedHistoricalDatasetFilter &&
      !historicalDatasetOptions.some(
        (option) => option.value === selectedHistoricalDatasetFilter,
      )
    ) {
      setSelectedHistoricalDatasetFilter("");
    }
  }, [
    historicalDatasetOptions,
    modelRunTagsLoading,
    selectedHistoricalDatasetFilter,
    selectedHistoricalExperimentFilter,
    setSelectedHistoricalDatasetFilter,
  ]);

  useEffect(() => {
    if (!selectedHistoricalExperimentFilter || !selectedHistoricalDatasetFilter) {
      setSelectedHistoricalPreset("");
      return;
    }
    if (modelRunTagsLoading) {
      return;
    }
    if (
      selectedHistoricalPreset &&
      !historicalPresetOptions.some(
        (option) => option.value === selectedHistoricalPreset,
      )
    ) {
      setSelectedHistoricalPreset("");
    }
  }, [
    historicalPresetOptions,
    modelRunTagsLoading,
    selectedHistoricalDatasetFilter,
    selectedHistoricalExperimentFilter,
    selectedHistoricalPreset,
    setSelectedHistoricalPreset,
  ]);

  useEffect(() => {
    if (!selectedModel) {
      setSelectedLogRunId(null);
      return;
    }
    if (logRunsQuery.isLoading && !logRunsQuery.data) {
      return;
    }
    if (modelRunTagsLoading) {
      return;
    }
    setSelectedLogRunId((current) =>
      current && selectedLogRun?.id === current
        ? current
        : null,
    );
  }, [
    logRunsQuery.data,
    logRunsQuery.isLoading,
    modelRunTagsLoading,
    selectedModel,
    selectedLogRun,
    setSelectedLogRunId,
  ]);

  useEffect(() => {
    if (
      !selectedModel ||
      !selectedHistoricalExperimentFilter ||
      !selectedHistoricalDatasetFilter ||
      !selectedHistoricalPreset
    ) {
      return;
    }
    if (modelRunTagsLoading) {
      return;
    }
    const resolvedRun = visibleHistoricalRuns[0];
    setSelectedLogRunId((current) =>
      current === (resolvedRun?.id ?? null) ? current : resolvedRun?.id ?? null,
    );
  }, [
    selectedHistoricalDatasetFilter,
    selectedHistoricalExperimentFilter,
    selectedHistoricalPreset,
    selectedModel,
    modelRunTagsLoading,
    setSelectedLogRunId,
    visibleHistoricalRuns,
  ]);

  useEffect(() => {
    if (selectedLogRunId === null) {
      clearSelectedExperimentRun();
    }
  }, [clearSelectedExperimentRun, selectedLogRunId]);

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
      historicalExperimentOptions,
      historicalDatasetOptions,
      historicalPresetOptions,
      selectedHistoricalExperimentFilter,
      setSelectedHistoricalExperimentFilter,
      selectedHistoricalDatasetFilter,
      setSelectedHistoricalDatasetFilter,
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
      historicalDatasetOptions,
      historicalExperimentOptions,
      historicalPresetOptions,
      logRunTagsQuery.isLoading,
      logRunsQuery.error,
      logRunsQuery.isLoading,
      modelRunTagsQuery.error,
      modelRunTagsQuery.isLoading,
      selectLogRun,
      selectedHistoricalDatasetFilter,
      selectedHistoricalDataset,
      selectedHistoricalExperimentFilter,
      selectedHistoricalExperiment,
      selectedHistoricalPreset,
      selectedHistoricalRunPreset,
      selectedLogRunId,
      setSelectedHistoricalDatasetFilter,
      setSelectedHistoricalExperimentFilter,
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
