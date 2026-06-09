import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import { useQueries, useQueryClient } from "@tanstack/react-query";
import {
  fetchLogParameterStatus,
  inspectModel,
  type LogRun,
  type LogRunTags,
  type Preset,
} from "@/lib/api";
import { useLogRunsQuery, useLogTagsQuery } from "@/features/viewer/state/logs/use-log-queries";
import {
  historicalMonitorRunGroups,
  resolveRunPresetName,
} from "@/lib/historical-monitor-runs";
import {
  summarizeHistoricalParameterStatus,
  type HistoricalParameterSummaryState,
} from "@/lib/parameter-summary";
import {
  deriveDatasetSelectionState,
} from "@/features/viewer/state/graph-monitor/graph-monitor-selectors";
import { logQueryKeys, monitorQueryKeys, viewerQueryKeys } from "@/lib/query-keys";

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
  presetOptions?: Preset[];
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
  historicalParameterSummariesByRunId: Map<
    string,
    HistoricalParameterSummaryState
  >;
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
    setSelectedLogRunId((current) => (current === runId ? null : runId));
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
  presetOptions,
  syncSelectedLogRun,
  selection,
}: HistoricalRunsStateInput): HistoricalRunsState {
  const queryClient = useQueryClient();
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
    queryKey: logQueryKeys.filteredHistoricalRunTags(filteredHistoricalRunIds),
  });
  const historicalParameterSummaryGroups = useMemo(
    () =>
      historicalMonitorRunGroups(visibleHistoricalRuns).map((group) => {
        const representativeRun = group.runs[0];
        const inspectPreset = representativeRun
          ? presetOptions
            ? resolveRunPresetName(representativeRun, presetOptions) ||
              representativeRun.preset
            : ""
          : group.preset;
        const runIds = group.runs.map((run) => run.id);
        return {
          ...group,
          inspectPreset,
          runIds,
        };
      }),
    [presetOptions, visibleHistoricalRuns],
  );
  const historicalParameterSummaryQueries = useQueries({
    queries: historicalParameterSummaryGroups.map((group) => ({
      queryKey: monitorQueryKeys.historicalParameterSummary(
        group.model,
        group.inspectPreset,
        group.dataset,
        group.runIds,
      ),
      queryFn: async () => {
        const [summaryGraph, status] = await Promise.all([
          queryClient.fetchQuery({
            queryKey: viewerQueryKeys.historicalSummaryInspection(
              group.model,
              group.inspectPreset,
              group.dataset,
            ),
            queryFn: () =>
              inspectModel({
                model: group.model,
                preset: group.inspectPreset,
                dataset: group.dataset,
                overrides: {},
              }),
            retry: false,
          }),
          queryClient.fetchQuery({
            queryKey: monitorQueryKeys.historicalParameterStatus(group.runIds),
            queryFn: () => fetchLogParameterStatus({ runIds: group.runIds }),
            retry: false,
          }),
        ]);

        return summarizeHistoricalParameterStatus({
          graph: summaryGraph,
          status,
          runs: group.runs,
        });
      },
      enabled:
        group.model.length > 0 &&
        group.inspectPreset.length > 0 &&
        group.dataset.length > 0 &&
        group.runIds.length > 0,
      retry: false,
    })),
  });
  const historicalParameterSummariesByRunId = useMemo(() => {
    const summaries = new Map<string, HistoricalParameterSummaryState>();
    historicalParameterSummaryGroups.forEach((group, index) => {
      const query = historicalParameterSummaryQueries[index];
      const state: HistoricalParameterSummaryState = {
        summary: query?.data,
        isLoading: Boolean(query?.isLoading || query?.isFetching),
        isError: Boolean(query?.isError),
        error: query?.error,
      };
      for (const runId of group.cardRunIds) {
        summaries.set(runId, state);
      }
    });
    return summaries;
  }, [historicalParameterSummaryGroups, historicalParameterSummaryQueries]);

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
      historicalParameterSummariesByRunId,
      logRunsQuery,
      logRunTagsQuery,
      experimentsLoading: logRunsQuery.isLoading || modelRunTagsQuery.isLoading,
      experimentsError: logRunsQuery.error ?? modelRunTagsQuery.error,
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
