import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import { type LogRun, type LogRunTags, type ModelIdentity } from "@/lib/api";
import { useLogRunsQuery, useLogTagsQuery } from "@/features/workbench/state/logs/use-log-queries";
import {
  deriveDatasetSelectionState,
} from "@/features/workbench/state/logs/historical-run-selection";
import { logQueryKeys } from "@/lib/query-keys";

const HISTORICAL_RUN_PAGE_LIMIT = 500;

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
  selectedExperimentTask?: string;
  runsEnabled?: boolean;
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
  selectedLogRun?: LogRun;
  selectLogRun: (runId: string) => void;
  logRunTagsLoading: boolean;
  experimentsLoading: boolean;
  experimentsError: Error | null;
  selectedLogRunMonitorEligibility:
    ReturnType<typeof deriveDatasetSelectionState>["selectedLogRunMonitorEligibility"];
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
  selectedExperimentTask = "",
  runsEnabled = true,
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

  const logRunFilters = useMemo(
    () =>
      selectedModel && selectedModelType
        ? {
            models: [
              {
                modelType: selectedModelType,
                model: selectedModel,
              } satisfies ModelIdentity,
            ],
          }
        : undefined,
    [selectedModel, selectedModelType],
  );
  const logRunsQuery = useLogRunsQuery({
    enabled: runsEnabled && Boolean(selectedModel && selectedModelType),
    filters: logRunFilters,
    pagination: { limit: HISTORICAL_RUN_PAGE_LIMIT, offset: 0 },
    projection: "summary",
    keepPreviousData: false,
  });
  const selectedRunCandidateState = useMemo(
    () =>
      deriveDatasetSelectionState({
        logRuns: logRunsQuery.data?.runs,
        modelRunTags: undefined,
        selectedModelType,
        selectedModel,
        selectedExperimentTask,
        selectedHistoricalExperimentFilter,
        selectedHistoricalDatasetFilter,
        selectedHistoricalPreset,
        selectedLogRunId,
      }),
    [
      logRunsQuery.data?.runs,
      selectedHistoricalDatasetFilter,
      selectedHistoricalExperimentFilter,
      selectedHistoricalPreset,
      selectedExperimentTask,
      selectedLogRunId,
      selectedModel,
      selectedModelType,
    ],
  );
  const selectedHistoricalRunIdsForTags = useMemo(
    () => selectedRunCandidateState.filteredHistoricalRuns.map((run) => run.id),
    [selectedRunCandidateState.filteredHistoricalRuns],
  );
  const logRunTagsQuery = useLogTagsQuery({
    runIds: selectedHistoricalRunIdsForTags,
    enabled: tagsEnabled && selectedHistoricalRunIdsForTags.length > 0,
    queryKey: logQueryKeys.filteredHistoricalRunTags(
      selectedHistoricalRunIdsForTags,
    ),
  });
  const selectedRunTags = useMemo(() => {
    if (!tagsEnabled || selectedHistoricalRunIdsForTags.length === 0) {
      return undefined;
    }
    const selectedRunIds = new Set(selectedHistoricalRunIdsForTags);
    return (logRunTagsQuery.data?.runs ?? []).filter((tags) =>
      selectedRunIds.has(tags.runId),
    );
  }, [
    logRunTagsQuery.data?.runs,
    selectedHistoricalRunIdsForTags,
    tagsEnabled,
  ]);
  const datasetSelectionState = useMemo(
    () =>
      deriveDatasetSelectionState({
        logRuns: logRunsQuery.data?.runs,
        modelRunTags: selectedRunTags,
        selectedModelType,
        selectedModel,
        selectedExperimentTask,
        selectedHistoricalExperimentFilter,
        selectedHistoricalDatasetFilter,
        selectedHistoricalPreset,
        selectedLogRunId,
      }),
    [
      logRunsQuery.data?.runs,
      selectedRunTags,
      selectedHistoricalDatasetFilter,
      selectedHistoricalExperimentFilter,
      selectedHistoricalPreset,
      selectedExperimentTask,
      selectedLogRunId,
      selectedModel,
      selectedModelType,
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
    selectedLogRunMonitorEligibility,
  } = datasetSelectionState;
  useEffect(() => {
    if (!selectedModel) {
      setSelectedHistoricalExperimentFilter("");
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
    selectedHistoricalExperimentFilter,
    selectedModel,
    setSelectedHistoricalExperimentFilter,
  ]);

  useEffect(() => {
    if (!selectedHistoricalExperimentFilter) {
      setSelectedHistoricalDatasetFilter("");
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
    selectedHistoricalDatasetFilter,
    selectedHistoricalExperimentFilter,
    setSelectedHistoricalDatasetFilter,
  ]);

  useEffect(() => {
    if (!selectedHistoricalExperimentFilter || !selectedHistoricalDatasetFilter) {
      setSelectedHistoricalPreset("");
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
    setSelectedLogRunId((current) =>
      current && selectedLogRun?.id === current
        ? current
        : null,
    );
  }, [
    logRunsQuery.data,
    logRunsQuery.isLoading,
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
    const resolvedRun = visibleHistoricalRuns[0];
    setSelectedLogRunId((current) =>
      current === (resolvedRun?.id ?? null) ? current : resolvedRun?.id ?? null,
    );
  }, [
    selectedHistoricalDatasetFilter,
    selectedHistoricalExperimentFilter,
    selectedHistoricalPreset,
    selectedModel,
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
      selectedLogRun,
      selectLogRun,
      logRunTagsLoading: logRunTagsQuery.isLoading,
      experimentsLoading: logRunsQuery.isLoading,
      experimentsError: logRunsQuery.error,
      selectedLogRunMonitorEligibility,
    }),
    [
      historicalMonitorRuns,
      historicalDatasetOptions,
      historicalExperimentOptions,
      historicalPresetOptions,
      logRunTagsQuery.isLoading,
      logRunsQuery.error,
      logRunsQuery.isLoading,
      selectLogRun,
      selectedHistoricalDatasetFilter,
      selectedHistoricalDataset,
      selectedHistoricalExperimentFilter,
      selectedHistoricalExperiment,
      selectedHistoricalPreset,
      selectedHistoricalRunPreset,
      selectedLogRun,
      selectedLogRunId,
      selectedLogRunMonitorEligibility,
      setSelectedHistoricalDatasetFilter,
      setSelectedHistoricalExperimentFilter,
      setSelectedHistoricalPreset,
      visibleHistoricalRuns,
    ],
  );
  const graphPreview = useMemo(
    () => ({
      historicalMonitorRuns,
      selectedHistoricalExperiment,
      selectedHistoricalDataset,
      selectedHistoricalRunPreset,
      logRunTags: selectedRunTags,
      filteredHistoricalRunIds,
    }),
    [
      filteredHistoricalRunIds,
      historicalMonitorRuns,
      selectedRunTags,
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
