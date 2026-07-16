import { useCallback, useEffect, useMemo, useState } from "react";
import type { LogRun, LogRunFacets, LogRunTags } from "@/lib/api/logs";
import type { ModelIdentity } from "@/lib/api/model-catalog";
import {
  useInfiniteLogRunsQuery,
  useLogRunsQuery,
  useLogTagsQuery,
} from "@/features/workbench/state/logs/use-log-queries";
import {
  HISTORICAL_MONITOR_RUN_LIMIT,
  logRunHasLayerMonitorData,
  monitorEligibilityDescription,
  sortLogRunsNewestFirst,
  type HistoricalRunOption,
  type MonitorEligibility,
} from "@/lib/historical-monitor-runs";
import { logQueryKeys } from "@/lib/query-keys";

const HISTORICAL_FACET_PAGE_LIMIT = 1;
const HISTORICAL_TARGET_PAGE_LIMIT = 50;

type HistoricalBrowsePhase = "idle" | "loading" | "error" | "empty" | "ready";

export type HistoricalBrowseStatus = {
  phase: HistoricalBrowsePhase;
  message: string;
};

type HistoricalTargetBrowsingInput = {
  selectedModelType: string;
  selectedModel: string;
  selectedExperimentTask?: string;
  runsEnabled?: boolean;
  tagsEnabled?: boolean;
};

type HistoricalTargetBrowsingProjection = {
  historicalMonitorRuns: LogRun[];
  historicalExperimentOptions: HistoricalRunOption[];
  historicalDatasetOptions: HistoricalRunOption[];
  historicalPresetOptions: HistoricalRunOption[];
  historicalBrowseStatus: HistoricalBrowseStatus;
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
  logRunTagsLoading: boolean;
  selectedLogRunMonitorEligibility: MonitorEligibility | undefined;
};

type HistoricalTargetGraphFacts = {
  historicalMonitorRuns: LogRun[];
  selectedHistoricalExperiment: string;
  selectedHistoricalDataset: string;
  selectedHistoricalRunPreset: string;
  logRunTags?: LogRunTags[];
  filteredHistoricalRunIds: string[];
};

type HistoricalTargetBrowsingState = {
  browsing: HistoricalTargetBrowsingProjection;
  graphFacts: HistoricalTargetGraphFacts;
  coordination: {
    selectedRun?: LogRun;
    clearForTargetChange: () => void;
  };
};

function checkingOption(value: string, count: number): HistoricalRunOption {
  return {
    value,
    label: value,
    count,
    monitorEligibility: "checking",
    description: monitorEligibilityDescription("checking"),
  };
}

function experimentFacetOptions(facets: LogRunFacets | null | undefined) {
  return (facets?.experiments ?? []).map((facet) =>
    checkingOption(facet.experiment, facet.runCount),
  );
}

function selectedExperimentFacet(
  facets: LogRunFacets | null | undefined,
  experiment: string,
) {
  return facets?.experiments.find((facet) => facet.experiment === experiment);
}

function errorStatus(label: string): HistoricalBrowseStatus {
  return { phase: "error", message: `Unable to load ${label}.` };
}

function runMonitorEligibility(
  run: LogRun,
  tagsByRunId: ReadonlyMap<string, LogRunTags>,
): MonitorEligibility {
  const tags = tagsByRunId.get(run.id);
  if (tags) return logRunHasLayerMonitorData(tags) ? "eligible" : "ineligible";
  if (run.hasLayerMonitorData === true) return "eligible";
  if (run.hasLayerMonitorData === false) return "ineligible";
  return "checking";
}

export function useHistoricalTargetBrowsing({
  selectedModelType,
  selectedModel,
  selectedExperimentTask = "",
  runsEnabled = true,
  tagsEnabled = true,
}: HistoricalTargetBrowsingInput): HistoricalTargetBrowsingState {
  const [requestedHistoricalExperimentFilter, setExperiment] = useState("");
  const [requestedHistoricalDatasetFilter, setDataset] = useState("");
  const [requestedHistoricalPreset, setPreset] = useState("");

  const clearHistoricalSelectionForTarget = useCallback(() => {
    setExperiment("");
    setDataset("");
    setPreset("");
  }, []);

  const baseFilters = useMemo(
    () =>
      selectedModel && selectedModelType
        ? {
            models: [
              {
                modelType: selectedModelType,
                model: selectedModel,
              } satisfies ModelIdentity,
            ],
            ...(selectedExperimentTask
              ? { experimentTask: selectedExperimentTask }
              : {}),
          }
        : undefined,
    [selectedExperimentTask, selectedModel, selectedModelType],
  );
  const baseQueryEnabled = runsEnabled && Boolean(baseFilters);
  const experimentQuery = useLogRunsQuery({
    enabled: baseQueryEnabled,
    filters: baseFilters,
    pagination: { limit: HISTORICAL_FACET_PAGE_LIMIT, offset: 0 },
    projection: "summary",
    keepPreviousData: false,
  });
  const historicalExperimentOptions = useMemo(
    () => experimentFacetOptions(experimentQuery.data?.facets),
    [experimentQuery.data?.facets],
  );
  const selectedHistoricalExperimentFilter =
    requestedHistoricalExperimentFilter &&
    experimentQuery.isSuccess &&
    !experimentQuery.isFetching &&
    !historicalExperimentOptions.some(
      (option) => option.value === requestedHistoricalExperimentFilter,
    )
      ? ""
      : requestedHistoricalExperimentFilter;
  const datasetQuery = useLogRunsQuery({
    enabled: baseQueryEnabled && Boolean(selectedHistoricalExperimentFilter),
    filters: baseFilters
      ? {
          ...baseFilters,
          experiment: [selectedHistoricalExperimentFilter],
        }
      : undefined,
    pagination: { limit: HISTORICAL_FACET_PAGE_LIMIT, offset: 0 },
    projection: "summary",
    keepPreviousData: false,
  });
  const historicalDatasetOptions = useMemo(() => {
    const facet = selectedExperimentFacet(
      datasetQuery.data?.facets,
      selectedHistoricalExperimentFilter,
    );
    return (facet?.datasets ?? []).map(({ value, count }) =>
      checkingOption(value, count),
    );
  }, [
    datasetQuery.data?.facets,
    selectedHistoricalExperimentFilter,
  ]);
  const selectedHistoricalDatasetFilter =
    !selectedHistoricalExperimentFilter ||
    (requestedHistoricalDatasetFilter &&
      datasetQuery.isSuccess &&
      !datasetQuery.isFetching &&
      !historicalDatasetOptions.some(
        (option) => option.value === requestedHistoricalDatasetFilter,
      ))
      ? ""
      : requestedHistoricalDatasetFilter;
  const presetQuery = useLogRunsQuery({
    enabled:
      baseQueryEnabled &&
      Boolean(
        selectedHistoricalExperimentFilter && selectedHistoricalDatasetFilter,
      ),
    filters: baseFilters
      ? {
          ...baseFilters,
          experiment: [selectedHistoricalExperimentFilter],
          dataset: [selectedHistoricalDatasetFilter],
        }
      : undefined,
    pagination: { limit: HISTORICAL_FACET_PAGE_LIMIT, offset: 0 },
    projection: "summary",
    keepPreviousData: false,
  });
  const historicalPresetOptions = useMemo(() => {
    const facet = selectedExperimentFacet(
      presetQuery.data?.facets,
      selectedHistoricalExperimentFilter,
    );
    return (facet?.presets ?? []).map(({ value, count }) =>
      checkingOption(value, count),
    );
  }, [
    presetQuery.data?.facets,
    selectedHistoricalExperimentFilter,
  ]);
  const selectedHistoricalPreset =
    !selectedHistoricalExperimentFilter ||
    !selectedHistoricalDatasetFilter ||
    (requestedHistoricalPreset &&
      presetQuery.isSuccess &&
      !presetQuery.isFetching &&
      !historicalPresetOptions.some(
        (option) => option.value === requestedHistoricalPreset,
      ))
      ? ""
      : requestedHistoricalPreset;
  const hasCompleteCascade = Boolean(
    selectedHistoricalExperimentFilter &&
      selectedHistoricalDatasetFilter &&
      selectedHistoricalPreset,
  );
  const exactRunsQuery = useInfiniteLogRunsQuery({
    enabled: baseQueryEnabled && hasCompleteCascade,
    filters: baseFilters
      ? {
          ...baseFilters,
          experiment: [selectedHistoricalExperimentFilter],
          dataset: [selectedHistoricalDatasetFilter],
          preset: [selectedHistoricalPreset],
        }
      : undefined,
    pageSize: HISTORICAL_TARGET_PAGE_LIMIT,
    projection: "summary",
    keepPreviousData: false,
  });
  const exactRuns = useMemo(
    () => exactRunsQuery.data?.pages.flatMap((page) => page.runs) ?? [],
    [exactRunsQuery.data?.pages],
  );
  const setSelectedHistoricalExperimentFilter = useCallback(
    (experiment: string) => {
      if (experiment === selectedHistoricalExperimentFilter) return;
      setExperiment(experiment);
      setDataset("");
      setPreset("");
    },
    [selectedHistoricalExperimentFilter],
  );
  const setSelectedHistoricalDatasetFilter = useCallback(
    (dataset: string) => {
      if (dataset === selectedHistoricalDatasetFilter) return;
      setDataset(dataset);
      setPreset("");
    },
    [selectedHistoricalDatasetFilter],
  );
  const setSelectedHistoricalPresetFilter = useCallback(
    (preset: string) => {
      if (preset === selectedHistoricalPreset) return;
      setPreset(preset);
    },
    [selectedHistoricalPreset],
  );

  const exactRunIds = useMemo(() => exactRuns.map((run) => run.id), [exactRuns]);
  const logRunTagsQuery = useLogTagsQuery({
    runIds: exactRunIds,
    enabled: tagsEnabled && exactRunIds.length > 0,
    queryKey: logQueryKeys.filteredHistoricalRunTags(exactRunIds),
  });
  const selectedRunTags = useMemo(() => {
    if (!tagsEnabled || exactRunIds.length === 0) return undefined;
    const ids = new Set(exactRunIds);
    return (logRunTagsQuery.data?.runs ?? []).filter((tags) => ids.has(tags.runId));
  }, [exactRunIds, logRunTagsQuery.data?.runs, tagsEnabled]);
  const datasetSelectionState = useMemo(() => {
    const taskAware = exactRuns.some((run) => Boolean(run.experimentTask));
    const visibleHistoricalRuns = sortLogRunsNewestFirst(
      exactRuns.filter(
        (run) =>
          run.modelType === selectedModelType &&
          run.model === selectedModel &&
          run.experiment === selectedHistoricalExperimentFilter &&
          run.dataset === selectedHistoricalDatasetFilter &&
          run.preset === selectedHistoricalPreset &&
          (!selectedExperimentTask ||
            !taskAware ||
            run.experimentTask === selectedExperimentTask),
      ),
    );
    const tagsByRunId = new Map(
      (selectedRunTags ?? []).map((tags) => [tags.runId, tags]),
    );
    const selectedLogRun = hasCompleteCascade
      ? visibleHistoricalRuns[0]
      : undefined;
    const eligibleRuns = visibleHistoricalRuns.filter(
      (run) => runMonitorEligibility(run, tagsByRunId) === "eligible",
    );
    return {
      visibleHistoricalRuns,
      selectedHistoricalExperiment:
        selectedLogRun?.experiment ??
        (hasCompleteCascade ? selectedHistoricalExperimentFilter : ""),
      selectedHistoricalDataset:
        selectedLogRun?.dataset ??
        (hasCompleteCascade ? selectedHistoricalDatasetFilter : ""),
      selectedHistoricalRunPreset:
        selectedLogRun?.preset ??
        (hasCompleteCascade ? selectedHistoricalPreset : ""),
      historicalMonitorRuns: eligibleRuns.slice(
        0,
        HISTORICAL_MONITOR_RUN_LIMIT,
      ),
      filteredHistoricalRunIds: eligibleRuns.map((run) => run.id),
      selectedLogRun,
      selectedLogRunMonitorEligibility: selectedLogRun
        ? runMonitorEligibility(selectedLogRun, tagsByRunId)
        : undefined,
    };
  }, [
    exactRuns,
    hasCompleteCascade,
    selectedRunTags,
    selectedHistoricalDatasetFilter,
    selectedHistoricalExperimentFilter,
    selectedHistoricalPreset,
    selectedExperimentTask,
    selectedModel,
    selectedModelType,
  ]);
  const {
    selectedHistoricalExperiment,
    selectedHistoricalDataset,
    selectedHistoricalRunPreset,
    historicalMonitorRuns,
    filteredHistoricalRunIds,
    selectedLogRun,
    selectedLogRunMonitorEligibility,
  } = datasetSelectionState;
  const selectedLogRunId = selectedLogRun?.id ?? null;
  const resolvedTagIds = useMemo(
    () => new Set((selectedRunTags ?? []).map((tags) => tags.runId)),
    [selectedRunTags],
  );
  const loadedRunsResolved = exactRuns.every(
    (run) =>
      (run.hasLayerMonitorData !== undefined &&
        run.hasLayerMonitorData !== null) ||
      resolvedTagIds.has(run.id),
  );
  const fetchNextExactRunsPage = exactRunsQuery.fetchNextPage;
  const hasNextExactRunsPage = exactRunsQuery.hasNextPage;
  const exactRunsFetching = exactRunsQuery.isFetching;
  const exactRunsFetchingNextPage = exactRunsQuery.isFetchingNextPage;

  useEffect(() => {
    if (
      !hasCompleteCascade ||
      !tagsEnabled ||
      exactRunsFetching ||
      exactRunsFetchingNextPage ||
      logRunTagsQuery.isFetching ||
      logRunTagsQuery.isError ||
      !loadedRunsResolved ||
      historicalMonitorRuns.length >= HISTORICAL_MONITOR_RUN_LIMIT ||
      !hasNextExactRunsPage
    ) {
      return;
    }
    void fetchNextExactRunsPage();
  }, [
    exactRunsFetching,
    exactRunsFetchingNextPage,
    fetchNextExactRunsPage,
    hasCompleteCascade,
    hasNextExactRunsPage,
    historicalMonitorRuns.length,
    loadedRunsResolved,
    logRunTagsQuery.isError,
    logRunTagsQuery.isFetching,
    tagsEnabled,
  ]);

  const historicalBrowseStatus = useMemo<HistoricalBrowseStatus>(() => {
    if (!baseQueryEnabled) return { phase: "idle", message: "" };
    if (experimentQuery.isLoading || experimentQuery.isFetching)
      return { phase: "loading", message: "Loading historical experiments…" };
    if (experimentQuery.isError) return errorStatus("historical experiments");
    if (!experimentQuery.data?.facets)
      return errorStatus("historical experiment facets");
    if (historicalExperimentOptions.length === 0)
      return { phase: "empty", message: "No historical experiments match this model and task." };
    if (!selectedHistoricalExperimentFilter) return { phase: "ready", message: "" };
    if (datasetQuery.isLoading || datasetQuery.isFetching)
      return { phase: "loading", message: "Loading historical datasets…" };
    if (datasetQuery.isError) return errorStatus("historical datasets");
    if (!datasetQuery.data?.facets)
      return errorStatus("historical dataset facets");
    if (historicalDatasetOptions.length === 0)
      return { phase: "empty", message: "No historical datasets match this experiment." };
    if (!selectedHistoricalDatasetFilter) return { phase: "ready", message: "" };
    if (presetQuery.isLoading || presetQuery.isFetching)
      return { phase: "loading", message: "Loading historical presets…" };
    if (presetQuery.isError) return errorStatus("historical presets");
    if (!presetQuery.data?.facets)
      return errorStatus("historical preset facets");
    if (historicalPresetOptions.length === 0)
      return { phase: "empty", message: "No historical presets match this dataset." };
    if (!selectedHistoricalPreset) return { phase: "ready", message: "" };
    if (exactRunsQuery.isError || logRunTagsQuery.isError)
      return errorStatus("historical runs");
    if (
      exactRunsQuery.isLoading ||
      exactRunsQuery.isFetching ||
      logRunTagsQuery.isLoading ||
      logRunTagsQuery.isFetching
    ) {
      return { phase: "loading", message: "Resolving the newest monitor runs…" };
    }
    if (exactRuns.length === 0)
      return { phase: "empty", message: "No runs match the selected historical group." };
    if (!exactRunsQuery.hasNextPage && historicalMonitorRuns.length === 0)
      return { phase: "empty", message: "No monitor-eligible runs match the selected historical group." };
    return { phase: "ready", message: "" };
  }, [
    baseQueryEnabled,
    datasetQuery.data?.facets,
    datasetQuery.isError,
    datasetQuery.isFetching,
    datasetQuery.isLoading,
    exactRuns.length,
    exactRunsQuery.hasNextPage,
    exactRunsQuery.isError,
    exactRunsQuery.isFetching,
    exactRunsQuery.isLoading,
    experimentQuery.data?.facets,
    experimentQuery.isError,
    experimentQuery.isFetching,
    experimentQuery.isLoading,
    historicalDatasetOptions.length,
    historicalExperimentOptions.length,
    historicalMonitorRuns.length,
    historicalPresetOptions.length,
    logRunTagsQuery.isError,
    logRunTagsQuery.isFetching,
    logRunTagsQuery.isLoading,
    presetQuery.isError,
    presetQuery.isFetching,
    presetQuery.isLoading,
    presetQuery.data?.facets,
    selectedHistoricalDatasetFilter,
    selectedHistoricalExperimentFilter,
    selectedHistoricalPreset,
  ]);

  const browsing = useMemo(
    () => ({
      historicalMonitorRuns,
      historicalExperimentOptions,
      historicalDatasetOptions,
      historicalPresetOptions,
      historicalBrowseStatus,
      selectedHistoricalExperimentFilter,
      setSelectedHistoricalExperimentFilter,
      selectedHistoricalDatasetFilter,
      setSelectedHistoricalDatasetFilter,
      selectedHistoricalPreset,
      setSelectedHistoricalPreset: setSelectedHistoricalPresetFilter,
      selectedHistoricalExperiment,
      selectedHistoricalDataset,
      selectedHistoricalRunPreset,
      selectedLogRunId,
      selectedLogRun,
      logRunTagsLoading:
        logRunTagsQuery.isLoading ||
        logRunTagsQuery.isFetching ||
        exactRunsQuery.isFetchingNextPage,
      selectedLogRunMonitorEligibility,
    }),
    [
      exactRunsQuery.isFetchingNextPage,
      historicalBrowseStatus,
      historicalDatasetOptions,
      historicalExperimentOptions,
      historicalMonitorRuns,
      historicalPresetOptions,
      logRunTagsQuery.isFetching,
      logRunTagsQuery.isLoading,
      selectedHistoricalDataset,
      selectedHistoricalDatasetFilter,
      selectedHistoricalExperiment,
      selectedHistoricalExperimentFilter,
      selectedHistoricalPreset,
      selectedHistoricalRunPreset,
      selectedLogRun,
      selectedLogRunId,
      selectedLogRunMonitorEligibility,
      setSelectedHistoricalDatasetFilter,
      setSelectedHistoricalExperimentFilter,
      setSelectedHistoricalPresetFilter,
    ],
  );
  const graphFacts = useMemo(
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
  const coordination = useMemo(
    () => ({
      selectedRun: selectedLogRun,
      clearForTargetChange: clearHistoricalSelectionForTarget,
    }),
    [clearHistoricalSelectionForTarget, selectedLogRun],
  );

  return useMemo(
    () => ({ browsing, graphFacts, coordination }),
    [browsing, coordination, graphFacts],
  );
}
