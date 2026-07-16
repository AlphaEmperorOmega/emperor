import {
  useCallback,
  useMemo,
  useState,
} from "react";
import {
  useQueryClient,
  type QueryClient,
} from "@tanstack/react-query";
import type { LogExperimentDeleteResponse, LogRunDeleteResponse } from "@/lib/api/deletion";
import type { LogRun, LogRunArtifacts, LogRunFacets, LogRunTags } from "@/lib/api/logs";
import {
  useInfiniteLogRunsQuery,
  useLogExperimentsQuery,
  useLogRunArtifactsQuery,
  useLogTagQueries,
} from "@/features/workbench/state/logs/use-log-queries";
import { useLogQueryCache } from "@/features/workbench/state/logs/use-log-query-cache";
import { logQueryKeys } from "@/lib/query-keys";
import {
  modelIdentityKey,
  modelNameForId,
  modelTypeForId,
} from "@/lib/selection";
import {
  useLogsDeletionState,
  type LogsDeletion as LogsDeletionProjection,
  type LogsSubsetDeleteTarget,
} from "@/features/workbench/state/logs/_logs-deletion-state";
import {
  buildCountOptions,
  buildExperimentOptions,
  buildLogScalarTagOptions,
  buildModelCountOptions,
  isDefaultScalarTag,
  logRunModelKey,
  type ChecklistOption,
  type LogMetricGroupKey,
  selectedOptionsSet,
} from "@/features/workbench/state/logs/logs-selectors";

const LOG_RUN_PAGE_SIZE = 100;
const MAX_LOG_RUN_LIMIT = 2000;
const SCALAR_TAG_RUN_WINDOW_SIZE = 100;
const LOG_SELECT_ALL_TAG_LIMIT = 100;
const DEFAULT_COLLAPSED_METRIC_GROUPS = new Set<LogMetricGroupKey>(["other"]);

function emptySelection() {
  return new Set<string>();
}

function addPendingExperimentSeed(
  pendingExperiments: Set<string>,
  experiment: string,
) {
  if (pendingExperiments.has(experiment)) {
    return pendingExperiments;
  }
  const next = new Set(pendingExperiments);
  next.add(experiment);
  return next;
}

function removePendingExperimentSeed(
  pendingExperiments: Set<string>,
  experiment: string,
) {
  if (!pendingExperiments.has(experiment)) {
    return pendingExperiments;
  }
  const next = new Set(pendingExperiments);
  next.delete(experiment);
  return next;
}

function toggleSelectionValue(selection: Set<string>, value: string) {
  const next = new Set(selection);
  if (next.has(value)) {
    next.delete(value);
  } else {
    next.add(value);
  }
  return next;
}

function selectionMatchesValues(selection: Set<string>, values: string[]) {
  if (selection.size !== values.length) {
    return false;
  }
  return values.every((value) => selection.has(value));
}

type NullableSelection = Set<string> | null;

function sortedSelectionValues(values: Set<string>) {
  return Array.from(values).sort((left, right) => left.localeCompare(right));
}

function selectionKey(values: Set<string>) {
  return JSON.stringify(sortedSelectionValues(values));
}

function selectionSetOrDefault(
  selection: NullableSelection,
  defaultSelection: Set<string>,
) {
  return selection ?? defaultSelection;
}

function pruneSelectionToAvailableValues(
  selection: NullableSelection,
  availableValues: string[],
): NullableSelection {
  if (selection === null) {
    return null;
  }
  const available = new Set(availableValues);
  const next = new Set(
    Array.from(selection).filter((value) => available.has(value)),
  );
  return next.size === selection.size ? selection : next;
}

function firstAvailableSelection(availableValues: string[]) {
  return new Set(availableValues.slice(0, 1));
}

function normalizeRunFacetSelection({
  selection,
  availableValues,
  selectFirstAvailable,
}: {
  selection: NullableSelection;
  availableValues: string[];
  selectFirstAvailable: boolean;
}): NullableSelection {
  if (selectFirstAvailable) {
    return firstAvailableSelection(availableValues);
  }
  const next = pruneSelectionToAvailableValues(selection, availableValues);
  if (
    selection !== null &&
    selection.size > 0 &&
    next !== null &&
    next.size === 0 &&
    availableValues.length > 0
  ) {
    return firstAvailableSelection(availableValues);
  }
  return next;
}

function addValuesToInitializedSelection(
  selection: NullableSelection,
  values: Set<string>,
): NullableSelection {
  if (selection === null || values.size === 0) {
    return selection;
  }
  const next = new Set(selection);
  for (const value of values) {
    next.add(value);
  }
  return next;
}

function addValueToInitializedSelection(
  selection: NullableSelection,
  value: string,
): NullableSelection {
  if (selection === null || selection.has(value)) {
    return selection;
  }
  const next = new Set(selection);
  next.add(value);
  return next;
}

function removeValueFromSelection({
  selection,
  fallbackValues,
  value,
}: {
  selection: NullableSelection;
  fallbackValues: string[];
  value: string;
}) {
  const next = new Set(selection ?? fallbackValues);
  next.delete(value);
  return next;
}

function startedRunSelections({
  runs,
  startedExperiments,
}: {
  runs: LogRun[];
  startedExperiments: Set<string>;
}) {
  const startedRuns = runs.filter((run) =>
    startedExperiments.has(run.experiment),
  );
  return {
    hasStartedRuns: startedRuns.length > 0,
    experiments: new Set(startedRuns.map((run) => run.experiment)),
    datasets: new Set(startedRuns.map((run) => run.dataset)),
    models: new Set(startedRuns.map((run) => logRunModelKey(run))),
    presets: new Set(startedRuns.map((run) => run.preset)),
  };
}

type CommonRunFacetOptions = {
  datasets: ChecklistOption[];
  models: ChecklistOption[];
  presets: ChecklistOption[];
};

type ExperimentFacets = LogRunFacets["experiments"][number];

function intersectExperimentFacetValues({
  runs,
  selectedExperiments,
  valueForRun,
}: {
  runs: LogRun[];
  selectedExperiments: Set<string>;
  valueForRun: (run: LogRun) => string;
}) {
  let commonValues: Set<string> | null = null;
  for (const experiment of selectedExperiments) {
    const experimentValues = new Set(
      runs
        .filter((run) => run.experiment === experiment)
        .map((run) => valueForRun(run)),
    );
    if (commonValues === null) {
      commonValues = experimentValues;
      continue;
    }
    const previousCommon = commonValues as Set<string>;
    commonValues = new Set(
      Array.from(previousCommon).filter((value) =>
        experimentValues.has(value),
      ),
    );
  }
  return commonValues ?? new Set<string>();
}

function commonServerFacetValues<Value>(
  experiments: ExperimentFacets[],
  valuesForExperiment: (
    experiment: ExperimentFacets,
  ) => Array<{ value: string; count: number; item: Value }>,
) {
  let common: Set<string> | null = null;
  const counts = new Map<string, { count: number; item: Value }>();
  for (const experiment of experiments) {
    const values = valuesForExperiment(experiment);
    const experimentValues = new Set(values.map(({ value }) => value));
    if (common === null) {
      common = experimentValues;
    } else {
      const previousCommon = common as Set<string>;
      common = new Set(
        Array.from(previousCommon).filter((value) =>
          experimentValues.has(value),
        ),
      );
    }
    for (const { value, count, item } of values) {
      const current = counts.get(value);
      counts.set(value, { count: (current?.count ?? 0) + count, item });
    }
  }
  return Array.from(common ?? [])
    .map((value) => ({ value, ...counts.get(value)! }))
    .sort((left, right) => left.value.localeCompare(right.value));
}

function buildCommonRunFacetOptions({
  runs,
  selectedExperiments,
  facets,
}: {
  runs: LogRun[];
  selectedExperiments: Set<string>;
  facets?: LogRunFacets | null;
}): CommonRunFacetOptions {
  if (selectedExperiments.size === 0) {
    return { datasets: [], models: [], presets: [] };
  }

  const selectedExperimentFacets = facets?.experiments.filter((facet) =>
    selectedExperiments.has(facet.experiment),
  );
  if (
    selectedExperimentFacets &&
    selectedExperimentFacets.length === selectedExperiments.size
  ) {
    const datasets = commonServerFacetValues(
      selectedExperimentFacets,
      (experiment) =>
        experiment.datasets.map((item) => ({
          value: item.value,
          count: item.count,
          item,
        })),
    ).map(({ value, count }) => ({ value, label: value, count }));
    const presets = commonServerFacetValues(
      selectedExperimentFacets,
      (experiment) =>
        experiment.presets.map((item) => ({
          value: item.value,
          count: item.count,
          item,
        })),
    ).map(({ value, count }) => ({ value, label: value, count }));
    const models = commonServerFacetValues(
      selectedExperimentFacets,
      (experiment) =>
        experiment.models.map((item) => ({
          value: modelIdentityKey(item),
          count: item.count,
          item,
        })),
    ).map(({ value, count, item }) => ({
      value,
      label: `${modelNameForId(item)} · ${modelTypeForId(item)}`,
      count,
    }));
    return { datasets, models, presets };
  }

  const selectedRuns = runs.filter((run) =>
    selectedExperiments.has(run.experiment),
  );
  const commonDatasets = intersectExperimentFacetValues({
    runs,
    selectedExperiments,
    valueForRun: (run) => run.dataset,
  });
  const commonModels = intersectExperimentFacetValues({
    runs,
    selectedExperiments,
    valueForRun: logRunModelKey,
  });
  const commonPresets = intersectExperimentFacetValues({
    runs,
    selectedExperiments,
    valueForRun: (run) => run.preset,
  });
  return {
    datasets: buildCountOptions(
      selectedRuns.filter((run) => commonDatasets.has(run.dataset)),
      "dataset",
    ),
    models: buildModelCountOptions(
      selectedRuns.filter((run) => commonModels.has(logRunModelKey(run))),
    ),
    presets: buildCountOptions(
      selectedRuns.filter((run) => commonPresets.has(run.preset)),
      "preset",
    ),
  };
}

function filterVisibleLogRuns(
  runs: LogRun[],
  selections: {
    experiments: Set<string>;
    datasets: Set<string>;
    models: Set<string>;
    presets: Set<string>;
  },
) {
  return runs.filter(
    (run) =>
      selections.experiments.has(run.experiment) &&
      selections.datasets.has(run.dataset) &&
      selections.models.has(logRunModelKey(run)) &&
      selections.presets.has(run.preset),
  );
}

function experimentScalarTags(
  tagRuns: LogRunTags[],
  tagOptionValues: string[],
) {
  const scalarTags = new Set<string>();
  for (const runTags of tagRuns) {
    for (const tag of runTags.scalarTags) {
      scalarTags.add(tag);
    }
  }
  return tagOptionValues.filter((tag) => scalarTags.has(tag));
}

function appendUniqueTags(target: string[], tags: string[]) {
  const existing = new Set(target);
  for (const tag of tags) {
    if (!existing.has(tag)) {
      existing.add(tag);
      target.push(tag);
    }
  }
}

function buildExperimentScalarTagSeedSelection({
  visibleRuns,
  tagRuns,
  pendingExperiments,
  selectedTags,
  tagOptionValues,
  selectAllLimit,
}: {
  visibleRuns: LogRun[];
  tagRuns: LogRunTags[] | undefined;
  pendingExperiments: Set<string>;
  selectedTags: NullableSelection;
  tagOptionValues: string[];
  selectAllLimit: number;
}) {
  const loadedExperiments = new Set<string>();
  const tagsByRunId = new Map(
    (tagRuns ?? []).map((runTags) => [runTags.runId, runTags]),
  );
  const replacementTags: string[] = [];

  for (const experiment of pendingExperiments) {
    const experimentRuns = visibleRuns.filter(
      (run) => run.experiment === experiment,
    );
    if (experimentRuns.length === 0) {
      continue;
    }
    const loadedTagRuns = experimentRuns
      .map((run) => tagsByRunId.get(run.id))
      .filter((runTags): runTags is LogRunTags => Boolean(runTags));
    if (loadedTagRuns.length !== experimentRuns.length) {
      continue;
    }
    loadedExperiments.add(experiment);
    if (selectedTags === null) {
      continue;
    }
    const availableTags = experimentScalarTags(
      loadedTagRuns,
      tagOptionValues,
    );
    if (
      availableTags.length === 0 ||
      availableTags.some((tag) => selectedTags.has(tag))
    ) {
      continue;
    }
    const defaultTags = availableTags.filter((tag) => isDefaultScalarTag(tag));
    appendUniqueTags(
      replacementTags,
      defaultTags.length > 0
        ? defaultTags
        : availableTags.slice(0, selectAllLimit),
    );
  }

  return {
    loadedExperiments,
    selectedTags:
      replacementTags.length > 0
        ? new Set(replacementTags.slice(0, selectAllLimit))
        : selectedTags,
  };
}

function cachedLogTagData(
  queryClient: QueryClient,
  runIds: string[],
): { runs: LogRunTags[] } | undefined {
  const allowedRunIds = new Set(runIds);
  const cachedRuns = new Map<string, LogRunTags>();
  for (const [, data] of queryClient.getQueriesData<{ runs: LogRunTags[] }>({
    queryKey: logQueryKeys.tags(),
  })) {
    for (const run of data?.runs ?? []) {
      if (allowedRunIds.has(run.runId)) {
        cachedRuns.set(run.runId, run);
      }
    }
  }
  return cachedRuns.size > 0
    ? { runs: Array.from(cachedRuns.values()) }
    : undefined;
}

function nextSelectedDetailRunId(
  selectedDetailRunId: string | null,
  visibleRuns: LogRun[],
) {
  if (visibleRuns.length === 0) {
    return null;
  }
  if (
    selectedDetailRunId &&
    visibleRuns.some((run) => run.id === selectedDetailRunId)
  ) {
    return selectedDetailRunId;
  }
  return visibleRuns[0].id;
}

function pruneDeletedDetailRunId({
  selectedDetailRunId,
  deletedRunIds,
}: {
  selectedDetailRunId: string | null;
  deletedRunIds: Set<string>;
}) {
  return selectedDetailRunId && deletedRunIds.has(selectedDetailRunId)
    ? null
    : selectedDetailRunId;
}

function removeStartedExperiment(
  startedExperiments: Set<string>,
  experiment: string,
) {
  if (!startedExperiments.has(experiment)) {
    return startedExperiments;
  }
  const next = new Set(startedExperiments);
  next.delete(experiment);
  return next;
}

export type LogsBrowserFilterKey =
  | "experiments"
  | "datasets"
  | "models"
  | "presets"
  | "tags";

export type LogsBrowserFilter = {
  options: ChecklistOption[];
  selectedValues: string[];
};

export type LogsBrowser = {
  filters: Record<LogsBrowserFilterKey, LogsBrowserFilter>;
  status: {
    isScanning: boolean;
    isRefreshing: boolean;
    runsError: unknown;
    experimentsError: unknown;
    tagsError: unknown;
  };
  results: {
    hasExperiments: boolean;
    hasRuns: boolean;
  };
  pagination: {
    runs: {
      loaded: number;
      total: number;
      canLoadMore: boolean;
      isLoadingMore: boolean;
    };
    scalarTags: {
      loadedRuns: number;
      totalRuns: number;
      canLoadMore: boolean;
      isLoadingMore: boolean;
    };
  };
  actions: {
    toggleFilter: (filter: LogsBrowserFilterKey, value: string) => void;
    selectAll: (filter: LogsBrowserFilterKey) => void;
    selectNone: (filter: LogsBrowserFilterKey) => void;
    refresh: () => Promise<void>;
    loadMoreRuns: () => void;
    loadMoreScalarTags: () => void;
  };
};

/**
 * Owns all state for the logs workspace: the run/experiment/tag queries, the
 * multi-facet selection sets (experiment/dataset/model/preset/run/tag), the
 * detail-run selection, and experiment deletion. Returned to the workspace
 * panels as a single object so they stay presentational.
 */
function useLogsWorkspaceImplementation({
  enabled,
  logDeletionEnabled = true,
}: {
  enabled: boolean;
  logDeletionEnabled?: boolean;
}) {
  const queryClient = useQueryClient();
  const { refreshAfterMutation, refreshLogs } = useLogQueryCache();
  const [startedExperiments, setStartedExperiments] = useState<Set<string>>(new Set());
  const [selectedExperiments, setSelectedExperiments] = useState<Set<string> | null>(
    emptySelection,
  );
  const [selectedDatasets, setSelectedDatasets] = useState<Set<string> | null>(null);
  const [selectedModels, setSelectedModels] = useState<Set<string> | null>(null);
  const [selectedPresets, setSelectedPresets] = useState<Set<string> | null>(null);
  const [runFacetResetRequestedAt, setRunFacetResetRequestedAt] = useState<
    number | null
  >(null);
  const [scalarTagRunWindowState, setScalarTagRunWindowState] = useState({
    key: "",
    count: SCALAR_TAG_RUN_WINDOW_SIZE,
  });
  const [selectedTags, setSelectedTags] = useState<Set<string> | null>(null);
  const [retainedTagProjection, setRetainedTagProjection] = useState<{
    options: ChecklistOption[];
    selectedTags: Set<string>;
    confusionMatrixRateTags: string[];
    tagRecords: LogRunTags[];
    visibleRuns: LogRun[];
    baseSelectedTagKey: string;
  } | null>(null);
  const [pendingExperimentTagSeeds, setPendingExperimentTagSeeds] = useState<
    Set<string>
  >(new Set());
  const [pendingTagSelection, setPendingTagSelection] = useState<"all" | null>(
    null,
  );
  const [selectedDetailRunId, setSelectedDetailRunId] = useState<string | null>(null);
  const [deletedRunsAwaitingRefresh, setDeletedRunsAwaitingRefresh] = useState<{
    runIds: Set<string>;
    requestedAt: number;
  }>(() => ({ runIds: new Set(), requestedAt: 0 }));
  const [subsetExperimentReconciliation, setSubsetExperimentReconciliation] =
    useState<Map<string, number>>(new Map());
  const [collapsedMetricGroups, setCollapsedMetricGroups] = useState<
    Set<LogMetricGroupKey>
  >(new Set(DEFAULT_COLLAPSED_METRIC_GROUPS));

  const experimentsQuery = useLogExperimentsQuery({ enabled });
  const reconciledMissingExperiments = useMemo(() => {
    if (
      subsetExperimentReconciliation.size === 0 ||
      experimentsQuery.isFetching ||
      experimentsQuery.error ||
      !experimentsQuery.data
    ) {
      return new Set<string>();
    }
    const availableExperiments = new Set(
      experimentsQuery.data.experiments.map(
        (experiment) => experiment.experiment,
      ),
    );
    return new Set(
      Array.from(subsetExperimentReconciliation.entries())
        .filter(
          ([experiment, requestedAt]) =>
            experimentsQuery.dataUpdatedAt > requestedAt &&
            !availableExperiments.has(experiment),
        )
        .map(([experiment]) => experiment),
    );
  }, [
    experimentsQuery.data,
    experimentsQuery.dataUpdatedAt,
    experimentsQuery.error,
    experimentsQuery.isFetching,
    subsetExperimentReconciliation,
  ]);
  const reconciledSelectedExperiments = useMemo(() => {
    if (selectedExperiments === null || reconciledMissingExperiments.size === 0) {
      return selectedExperiments;
    }
    return new Set(
      Array.from(selectedExperiments).filter(
        (experiment) => !reconciledMissingExperiments.has(experiment),
      ),
    );
  }, [reconciledMissingExperiments, selectedExperiments]);
  const reconciledStartedExperiments = useMemo(
    () =>
      new Set(
        Array.from(startedExperiments).filter(
          (experiment) => !reconciledMissingExperiments.has(experiment),
        ),
      ),
    [reconciledMissingExperiments, startedExperiments],
  );
  const reconciledPendingExperimentTagSeeds = useMemo(
    () =>
      new Set(
        Array.from(pendingExperimentTagSeeds).filter(
          (experiment) => !reconciledMissingExperiments.has(experiment),
        ),
      ),
    [pendingExperimentTagSeeds, reconciledMissingExperiments],
  );
  const selectedExperimentQueryValues = useMemo(
    () =>
      reconciledSelectedExperiments
        ? sortedSelectionValues(reconciledSelectedExperiments)
        : [],
    [reconciledSelectedExperiments],
  );
  const hasSelectedExperimentFilters = selectedExperimentQueryValues.length > 0;
  const runFilters = useMemo(
    () =>
      hasSelectedExperimentFilters
        ? { experiment: selectedExperimentQueryValues }
        : undefined,
    [hasSelectedExperimentFilters, selectedExperimentQueryValues],
  );
  const runPagesQuery = useInfiniteLogRunsQuery({
    enabled,
    filters: runFilters,
    pageSize: LOG_RUN_PAGE_SIZE,
    keepPreviousData: false,
  });
  const paginatedRunsData = useMemo(() => {
    const pages = runPagesQuery.data?.pages;
    if (!pages?.length) {
      return undefined;
    }
    const firstPage = pages[0];
    const lastPage = pages[pages.length - 1];
    return {
      ...firstPage,
      runs: pages.flatMap((page) => page.runs),
      total: firstPage.total,
      limit: LOG_RUN_PAGE_SIZE,
      offset: 0,
      hasMore: Boolean(lastPage.hasMore),
    };
  }, [runPagesQuery.data?.pages]);
  const runsQuery = {
    ...runPagesQuery,
    data: paginatedRunsData,
  };

  const runsData = runsQuery.data?.runs;
  const runs = useMemo(() => runsData ?? [], [runsData]);
  const experimentsData = experimentsQuery.data?.experiments;
  const experiments = useMemo(() => experimentsData ?? [], [experimentsData]);

  const experimentOptions = useMemo(
    () => buildExperimentOptions(experiments),
    [experiments],
  );

  const includeStartedExperiment = useCallback((logFolder: string) => {
    setRunFacetResetRequestedAt(0);
    setStartedExperiments((previous) => {
      if (previous.has(logFolder)) {
        return previous;
      }
      const next = new Set(previous);
      next.add(logFolder);
      return next;
    });
    setSelectedExperiments((previous) => {
      return addValueToInitializedSelection(previous, logFolder);
    });
    setPendingExperimentTagSeeds((previous) =>
      addPendingExperimentSeed(previous, logFolder),
    );
  }, []);

  const resetLogSelections = useCallback(() => {
    setSelectedExperiments(emptySelection());
    setSelectedDatasets(null);
    setSelectedModels(null);
    setSelectedPresets(null);
    setRunFacetResetRequestedAt(null);
    setSelectedTags(null);
    setRetainedTagProjection(null);
    setPendingExperimentTagSeeds(new Set());
    setPendingTagSelection(null);
    setSelectedDetailRunId(null);
    setCollapsedMetricGroups(new Set(DEFAULT_COLLAPSED_METRIC_GROUPS));
  }, []);

  const startedSelections = useMemo(
    () =>
      startedRunSelections({
        runs,
        startedExperiments: reconciledStartedExperiments,
      }),
    [reconciledStartedExperiments, runs],
  );
  const selectedExperimentsWithStarted = useMemo(
    () =>
      addValuesToInitializedSelection(
        reconciledSelectedExperiments,
        startedSelections.experiments,
      ),
    [reconciledSelectedExperiments, startedSelections.experiments],
  );
  const experimentSet = useMemo(
    () => selectedExperimentsWithStarted ?? emptySelection(),
    [selectedExperimentsWithStarted],
  );
  const commonRunFacetOptions = useMemo(
    () =>
      buildCommonRunFacetOptions({
        runs,
        selectedExperiments: experimentSet,
        facets: runsQuery.data?.facets,
      }),
    [experimentSet, runs, runsQuery.data?.facets],
  );
  const datasetOptions = commonRunFacetOptions.datasets;
  const modelOptions = commonRunFacetOptions.models;
  const presetOptions = commonRunFacetOptions.presets;
  const datasetOptionValues = useMemo(
    () => datasetOptions.map((option) => option.value),
    [datasetOptions],
  );
  const modelOptionValues = useMemo(
    () => modelOptions.map((option) => option.value),
    [modelOptions],
  );
  const presetOptionValues = useMemo(
    () => presetOptions.map((option) => option.value),
    [presetOptions],
  );
  const runFacetResetReady =
    runFacetResetRequestedAt !== null &&
    Boolean(runsQuery.data) &&
    !runsQuery.error &&
    !runsQuery.isFetching &&
    !runsQuery.isPlaceholderData &&
    runsQuery.dataUpdatedAt > runFacetResetRequestedAt;
  const runFacetResetPending =
    runFacetResetRequestedAt !== null && !runFacetResetReady;
  const selectedDatasetsWithStarted = useMemo(
    () =>
      addValuesToInitializedSelection(
        selectedDatasets,
        startedSelections.datasets,
      ),
    [selectedDatasets, startedSelections.datasets],
  );
  const selectedModelsWithStarted = useMemo(
    () =>
      addValuesToInitializedSelection(selectedModels, startedSelections.models),
    [selectedModels, startedSelections.models],
  );
  const selectedPresetsWithStarted = useMemo(
    () =>
      addValuesToInitializedSelection(
        selectedPresets,
        startedSelections.presets,
      ),
    [selectedPresets, startedSelections.presets],
  );
  const normalizedDatasetSelection = useMemo(
    () =>
      normalizeRunFacetSelection({
        selection: selectedDatasetsWithStarted,
        availableValues: datasetOptionValues,
        selectFirstAvailable: runFacetResetReady,
      }),
    [
      datasetOptionValues,
      runFacetResetReady,
      selectedDatasetsWithStarted,
    ],
  );
  const normalizedModelSelection = useMemo(
    () =>
      normalizeRunFacetSelection({
        selection: selectedModelsWithStarted,
        availableValues: modelOptionValues,
        selectFirstAvailable: runFacetResetReady,
      }),
    [modelOptionValues, runFacetResetReady, selectedModelsWithStarted],
  );
  const normalizedPresetSelection = useMemo(
    () =>
      normalizeRunFacetSelection({
        selection: selectedPresetsWithStarted,
        availableValues: presetOptionValues,
        selectFirstAvailable: runFacetResetReady,
      }),
    [presetOptionValues, runFacetResetReady, selectedPresetsWithStarted],
  );
  const datasetSet = useMemo(
    () =>
      selectionSetOrDefault(
        normalizedDatasetSelection,
        new Set(datasetOptionValues),
      ),
    [datasetOptionValues, normalizedDatasetSelection],
  );
  const modelSet = useMemo(
    () =>
      selectionSetOrDefault(
        normalizedModelSelection,
        new Set(modelOptionValues),
      ),
    [modelOptionValues, normalizedModelSelection],
  );
  const presetSet = useMemo(
    () =>
      selectionSetOrDefault(
        normalizedPresetSelection,
        new Set(presetOptionValues),
      ),
    [normalizedPresetSelection, presetOptionValues],
  );
  const deletedRunIdsAwaitingRefresh = useMemo(() => {
    if (
      deletedRunsAwaitingRefresh.runIds.size === 0 ||
      !runsQuery.data ||
      runsQuery.isFetching ||
      runsQuery.isPlaceholderData ||
      runsQuery.error ||
      runsQuery.dataUpdatedAt <= deletedRunsAwaitingRefresh.requestedAt
    ) {
      return deletedRunsAwaitingRefresh.runIds;
    }
    const authoritativeRunIds = new Set(runs.map((run) => run.id));
    return new Set(
      Array.from(deletedRunsAwaitingRefresh.runIds).filter((runId) =>
        authoritativeRunIds.has(runId),
      ),
    );
  }, [
    deletedRunsAwaitingRefresh,
    runs,
    runsQuery.data,
    runsQuery.dataUpdatedAt,
    runsQuery.error,
    runsQuery.isFetching,
    runsQuery.isPlaceholderData,
  ]);
  const visibleRuns = useMemo(
    () =>
      filterVisibleLogRuns(runs, {
        experiments: experimentSet,
        datasets: datasetSet,
        models: modelSet,
        presets: presetSet,
      }).filter((run) => !deletedRunIdsAwaitingRefresh.has(run.id)),
    [
      datasetSet,
      deletedRunIdsAwaitingRefresh,
      experimentSet,
      modelSet,
      presetSet,
      runs,
    ],
  );
  const visibleRunIds = useMemo(() => visibleRuns.map((run) => run.id), [visibleRuns]);
  const scalarTagRunWindowKey = useMemo(
    () => `${runsQuery.dataUpdatedAt}\u0000${visibleRunIds.join("\u0000")}`,
    [runsQuery.dataUpdatedAt, visibleRunIds],
  );
  const scalarTagRunCount =
    scalarTagRunWindowState.key === scalarTagRunWindowKey
      ? scalarTagRunWindowState.count
      : SCALAR_TAG_RUN_WINDOW_SIZE;
  const scalarTagRunIds = useMemo(
    () => visibleRunIds.slice(0, scalarTagRunCount),
    [scalarTagRunCount, visibleRunIds],
  );
  const tagsEnabled = enabled && !runFacetResetPending;
  const scalarTagWindowInputs = useMemo(
    () =>
      Array.from(
        { length: Math.ceil(scalarTagRunIds.length / SCALAR_TAG_RUN_WINDOW_SIZE) },
        (_, windowIndex) => {
          const runIds = scalarTagRunIds.slice(
            windowIndex * SCALAR_TAG_RUN_WINDOW_SIZE,
            (windowIndex + 1) * SCALAR_TAG_RUN_WINDOW_SIZE,
          );
          return {
            enabled: tagsEnabled,
            runIds,
            queryKey: logQueryKeys.tagsForRuns(runIds),
          };
        },
      ),
    [scalarTagRunIds, tagsEnabled],
  );
  const scalarTagWindowQueries = useLogTagQueries(scalarTagWindowInputs);
  const scalarTagWindowPages = scalarTagWindowQueries
    .map((query) => query.data)
    .filter((page): page is NonNullable<typeof page> => Boolean(page));
  const scalarTagErrorQuery = scalarTagWindowQueries.find((query) => query.isError);
  const currentTagsData = useMemo(
    () =>
      scalarTagWindowPages.length > 0
        ? { runs: scalarTagWindowPages.flatMap((page) => page.runs) }
        : undefined,
    [scalarTagWindowPages],
  );
  const tagWindowsAreFetching = scalarTagWindowQueries.some(
    (query) => query.isFetching,
  );
  const tagWindowsHavePlaceholderData = scalarTagWindowQueries.some(
    (query) => query.isPlaceholderData,
  );
  const placeholderTagsData = useMemo(
    () =>
      !currentTagsData &&
      scalarTagWindowInputs.length > 0 &&
      (tagWindowsAreFetching || runFacetResetPending)
        ? cachedLogTagData(queryClient, scalarTagRunIds)
        : undefined,
    [
      currentTagsData,
      queryClient,
      runFacetResetPending,
      scalarTagRunIds,
      scalarTagWindowInputs.length,
      tagWindowsAreFetching,
    ],
  );
  const tagsData = currentTagsData ?? placeholderTagsData;
  const tagsQuery = {
    data: tagsData,
    isLoading:
      scalarTagWindowInputs.some((input) => input.enabled) &&
      !tagsData &&
      scalarTagWindowQueries.some((query) => query.isLoading),
    isFetching: tagWindowsAreFetching,
    isError: Boolean(scalarTagErrorQuery),
    error: scalarTagErrorQuery?.error ?? null,
    isPlaceholderData:
      tagWindowsHavePlaceholderData ||
      Boolean(
        !currentTagsData &&
          placeholderTagsData &&
          (tagWindowsAreFetching || runFacetResetPending),
      ),
  };

  const scalarTagOptions = useMemo(
    () => buildLogScalarTagOptions(tagsQuery.data?.runs),
    [tagsQuery.data],
  );
  const tagOptions = scalarTagOptions.tagOptions;
  const confusionMatrixRateTags = scalarTagOptions.confusionMatrixRateTags;
  const tagOptionValues = useMemo(
    () => tagOptions.map((option) => option.value),
    [tagOptions],
  );

  const defaultSelectedTags = useMemo(
    () =>
      new Set(
        tagOptions
          .map((option) => option.value)
          .filter((tag) => isDefaultScalarTag(tag)),
    ),
    [tagOptions],
  );
  const tagDataNeedsFreshOptions =
    runFacetResetPending ||
    runsQuery.isFetching ||
    Boolean(runsQuery.isPlaceholderData) ||
    tagsQuery.isFetching ||
    tagsQuery.isLoading ||
    tagsQuery.isError ||
    Boolean(tagsQuery.isPlaceholderData);
  const experimentTagSeeds = useMemo(
    () =>
      reconciledPendingExperimentTagSeeds.size === 0 ||
      visibleRuns.length === 0 ||
      tagDataNeedsFreshOptions
        ? {
            loadedExperiments: new Set<string>(),
            selectedTags,
          }
        : buildExperimentScalarTagSeedSelection({
            visibleRuns,
            tagRuns: tagsQuery.data?.runs,
            pendingExperiments: reconciledPendingExperimentTagSeeds,
            selectedTags,
            tagOptionValues,
            selectAllLimit: LOG_SELECT_ALL_TAG_LIMIT,
          }),
    [
      reconciledPendingExperimentTagSeeds,
      selectedTags,
      tagDataNeedsFreshOptions,
      tagOptionValues,
      tagsQuery.data?.runs,
      visibleRuns,
    ],
  );
  const normalizedSelectedTags = useMemo(() => {
    if (tagDataNeedsFreshOptions) {
      return experimentTagSeeds.selectedTags;
    }
    if (pendingTagSelection === "all") {
      return new Set(tagOptionValues.slice(0, LOG_SELECT_ALL_TAG_LIMIT));
    }
    return pruneSelectionToAvailableValues(
      experimentTagSeeds.selectedTags,
      tagOptionValues,
    );
  }, [
    experimentTagSeeds.selectedTags,
    pendingTagSelection,
    tagDataNeedsFreshOptions,
    tagOptionValues,
  ]);
  const selectedTagsSet = useMemo(
    () => selectionSetOrDefault(normalizedSelectedTags, defaultSelectedTags),
    [defaultSelectedTags, normalizedSelectedTags],
  );
  const tagProjectionTransitionPending =
    runFacetResetPending ||
    runsQuery.isFetching ||
    Boolean(runsQuery.isPlaceholderData) ||
    tagsQuery.isFetching ||
    tagsQuery.isLoading ||
    Boolean(tagsQuery.isPlaceholderData);
  const projectedTagOptions =
    tagProjectionTransitionPending && retainedTagProjection
      ? retainedTagProjection.options
      : tagOptions;
  const projectedSelectedTagsSet =
    tagProjectionTransitionPending && retainedTagProjection
      ? retainedTagProjection.selectedTags
      : selectedTagsSet;
  const projectedConfusionMatrixRateTags =
    tagProjectionTransitionPending && retainedTagProjection
      ? retainedTagProjection.confusionMatrixRateTags
      : confusionMatrixRateTags;
  const queriedChartTagRecords = tagsQuery.data?.runs;
  const projectedChartTagRecords = useMemo(
    () =>
      tagProjectionTransitionPending && retainedTagProjection
        ? retainedTagProjection.tagRecords
        : queriedChartTagRecords ?? [],
    [
      queriedChartTagRecords,
      retainedTagProjection,
      tagProjectionTransitionPending,
    ],
  );
  const projectedChartVisibleRuns =
    tagProjectionTransitionPending && retainedTagProjection
      ? retainedTagProjection.visibleRuns
      : visibleRuns;
  const projectedChartVisibleRunIds = useMemo(
    () => projectedChartVisibleRuns.map((run) => run.id),
    [projectedChartVisibleRuns],
  );
  const effectiveSelectedDetailRunId = nextSelectedDetailRunId(
    selectedDetailRunId,
    visibleRuns,
  );

  const selectedTagList = useMemo(
    () =>
      Array.from(
        selectedOptionsSet(projectedSelectedTagsSet, projectedTagOptions),
      ).sort((a, b) => a.localeCompare(b)),
    [projectedSelectedTagsSet, projectedTagOptions],
  );
  const preserveCurrentTagProjection = useCallback(() => {
    if (projectedTagOptions.length === 0) {
      return;
    }
    const baseSelectedTagKey = selectionKey(projectedSelectedTagsSet);
    setRetainedTagProjection((previous) => ({
      options: projectedTagOptions,
      selectedTags:
        previous?.baseSelectedTagKey === baseSelectedTagKey
          ? previous.selectedTags
          : new Set(projectedSelectedTagsSet),
      confusionMatrixRateTags: projectedConfusionMatrixRateTags,
      tagRecords: projectedChartTagRecords,
      visibleRuns: projectedChartVisibleRuns,
      baseSelectedTagKey,
    }));
  }, [
    projectedChartTagRecords,
    projectedChartVisibleRuns,
    projectedConfusionMatrixRateTags,
    projectedSelectedTagsSet,
    projectedTagOptions,
  ]);
  const setSelectedTagValues = useCallback(
    (nextSelectedTags: Set<string>) => {
      setPendingTagSelection(null);
      setPendingExperimentTagSeeds(new Set());
      setSelectedTags(nextSelectedTags);
      setRetainedTagProjection({
        options: projectedTagOptions,
        selectedTags: nextSelectedTags,
        confusionMatrixRateTags: projectedConfusionMatrixRateTags,
        tagRecords: projectedChartTagRecords,
        visibleRuns: projectedChartVisibleRuns,
        baseSelectedTagKey: selectionKey(projectedSelectedTagsSet),
      });
    },
    [
      projectedChartTagRecords,
      projectedChartVisibleRuns,
      projectedConfusionMatrixRateTags,
      projectedSelectedTagsSet,
      projectedTagOptions,
    ],
  );
  const toggleMetricGroup = useCallback((group: LogMetricGroupKey) => {
    setCollapsedMetricGroups((previous) => {
      const next = new Set(previous);
      if (next.has(group)) {
        next.delete(group);
      } else {
        next.add(group);
      }
      return next;
    });
  }, []);
  const selectedRun = visibleRuns.find(
    (run) => run.id === effectiveSelectedDetailRunId,
  );
  const onExperimentDeleted = useCallback(
    (result: LogExperimentDeleteResponse) => {
      const deletedRunIds = new Set(result.deletedRunIds);
      setDeletedRunsAwaitingRefresh((previous) => ({
        runIds: new Set([...previous.runIds, ...result.deletedRunIds]),
        requestedAt: Math.max(
          previous.requestedAt,
          runsQuery.dataUpdatedAt,
        ),
      }));
      setSelectedExperiments((previous) => {
        return removeValueFromSelection({
          selection: previous,
          fallbackValues: experimentOptions.map((option) => option.value),
          value: result.experiment,
        });
      });
      setSelectedDetailRunId((previous) =>
        pruneDeletedDetailRunId({
          selectedDetailRunId: previous,
          deletedRunIds,
        }),
      );
      setStartedExperiments((previous) => {
        return removeStartedExperiment(previous, result.experiment);
      });
      setPendingExperimentTagSeeds((previous) => {
        return removePendingExperimentSeed(previous, result.experiment);
      });
      void refreshAfterMutation({ runIds: result.deletedRunIds });
    },
    [experimentOptions, refreshAfterMutation, runsQuery.dataUpdatedAt],
  );
  const onRunsDeleted = useCallback(
    (result: LogRunDeleteResponse, target: LogsSubsetDeleteTarget) => {
      const deletedRunIds = new Set(result.deletedRunIds);
      setDeletedRunsAwaitingRefresh((previous) => ({
        runIds: new Set([...previous.runIds, ...result.deletedRunIds]),
        requestedAt: Math.max(
          previous.requestedAt,
          runsQuery.dataUpdatedAt,
        ),
      }));
      setSelectedDetailRunId((previous) =>
        pruneDeletedDetailRunId({
          selectedDetailRunId: previous,
          deletedRunIds,
        }),
      );
      setSubsetExperimentReconciliation((previous) => {
        const next = new Map(previous);
        next.set(
          target.experiment,
          Math.max(
            experimentsQuery.dataUpdatedAt,
            next.get(target.experiment) ?? 0,
          ),
        );
        return next;
      });
      void refreshAfterMutation({ runIds: result.deletedRunIds });
    },
    [
      experimentsQuery.dataUpdatedAt,
      refreshAfterMutation,
      runsQuery.dataUpdatedAt,
    ],
  );
  const deletion = useLogsDeletionState({
    active: enabled,
    enabled: logDeletionEnabled,
    selectedExperiments: experimentSet,
    onExperimentDeleted,
    onRunsDeleted,
  });
  const clearDeletionForConnectionChange = deletion.clearForConnectionChange;
  const clearForConnectionChange = useCallback(() => {
    setStartedExperiments(new Set());
    resetLogSelections();
    setScalarTagRunWindowState({
      key: "",
      count: SCALAR_TAG_RUN_WINDOW_SIZE,
    });
    setDeletedRunsAwaitingRefresh({
      runIds: new Set(),
      requestedAt: 0,
    });
    setSubsetExperimentReconciliation(new Map());
    clearDeletionForConnectionChange();
  }, [clearDeletionForConnectionChange, resetLogSelections]);

  return {
    enabled,
    logDeletionEnabled,
    runs,
    runsQuery,
    experimentsQuery,
    tagsQuery,
    datasetOptions,
    experimentOptions,
    modelOptions,
    presetOptions,
    tagOptions: projectedTagOptions,
    confusionMatrixRateTags: projectedConfusionMatrixRateTags,
    chartTagRecords: projectedChartTagRecords,
    chartVisibleRuns: projectedChartVisibleRuns,
    chartVisibleRunIds: projectedChartVisibleRunIds,
    tagProjectionTransitionPending,
    visibleRuns,
    visibleRunIds,
    selectedDatasets: datasetSet,
    selectedExperiments: experimentSet,
    selectedModels: modelSet,
    selectedPresets: presetSet,
    selectedTags: projectedSelectedTagsSet,
    selectedTagList,
    collapsedMetricGroups,
    toggleMetricGroup,
    selectedRun,
    selectedDetailRunId: effectiveSelectedDetailRunId,
    setSelectedDetailRunId,
    deletion,
    clearForConnectionChange,
    loadedRunCount: runs.length,
    totalRunCount: runsQuery.data?.total ?? runs.length,
    canLoadMoreRuns:
      enabled &&
      Boolean(runPagesQuery.hasNextPage) &&
      runs.length < MAX_LOG_RUN_LIMIT,
    isLoadingMoreRuns: runPagesQuery.isFetchingNextPage,
    loadMoreRuns: () => {
      if (
        !enabled ||
        runPagesQuery.isFetchingNextPage ||
        !runPagesQuery.hasNextPage ||
        runs.length >= MAX_LOG_RUN_LIMIT
      ) {
        return;
      }
      void runPagesQuery.fetchNextPage();
    },
    loadedScalarTagRunCount: scalarTagRunIds.length,
    totalScalarTagRunCount: visibleRunIds.length,
    canLoadMoreScalarTags:
      enabled && scalarTagRunIds.length < visibleRunIds.length,
    isLoadingMoreScalarTags: Boolean(
      scalarTagWindowQueries.some(
        (query, index) => index > 0 && query.isFetching && !query.data,
      ),
    ),
    loadMoreScalarTags: () => {
      if (
        !enabled ||
        scalarTagRunIds.length >= visibleRunIds.length ||
        tagsQuery.isFetching
      ) {
        return;
      }
      setScalarTagRunWindowState((previous) => {
        const currentCount =
          previous.key === scalarTagRunWindowKey
            ? previous.count
            : SCALAR_TAG_RUN_WINDOW_SIZE;
        return {
          key: scalarTagRunWindowKey,
          count: Math.min(
            currentCount + SCALAR_TAG_RUN_WINDOW_SIZE,
            visibleRunIds.length,
          ),
        };
      });
    },
    refreshLogLists: refreshLogs,
    includeStartedExperiment,
    toggleExperiment: (value: string) => {
      preserveCurrentTagProjection();
      const currentSelection = experimentSet;
      const isSelectingExperiment = !currentSelection.has(value);
      setRunFacetResetRequestedAt(0);
      setPendingExperimentTagSeeds((previous) => {
        return isSelectingExperiment
          ? addPendingExperimentSeed(previous, value)
          : removePendingExperimentSeed(previous, value);
      });
      if (!isSelectingExperiment) {
        setStartedExperiments((previous) =>
          removeStartedExperiment(previous, value),
        );
      }
      setSelectedExperiments(toggleSelectionValue(currentSelection, value));
    },
    toggleDataset: (value: string) => {
      preserveCurrentTagProjection();
      setRunFacetResetRequestedAt(null);
      setSelectedDatasets(toggleSelectionValue(datasetSet, value));
    },
    toggleModel: (value: string) => {
      preserveCurrentTagProjection();
      setRunFacetResetRequestedAt(null);
      setSelectedModels(toggleSelectionValue(modelSet, value));
    },
    togglePreset: (value: string) => {
      preserveCurrentTagProjection();
      setRunFacetResetRequestedAt(null);
      setSelectedPresets(toggleSelectionValue(presetSet, value));
    },
    toggleTag: (value: string) => {
      setSelectedTagValues(
        toggleSelectionValue(projectedSelectedTagsSet, value),
      );
    },
    setSelectedTagValues,
    selectAllExperiments: () => {
      preserveCurrentTagProjection();
      const experimentValues = experimentOptions.map((option) => option.value);
      if (!selectionMatchesValues(experimentSet, experimentValues)) {
        setRunFacetResetRequestedAt(0);
      }
      setPendingExperimentTagSeeds(new Set(experimentValues));
      setSelectedExperiments(new Set(experimentValues));
    },
    selectNoExperiments: () => {
      preserveCurrentTagProjection();
      if (experimentSet.size > 0) {
        setRunFacetResetRequestedAt(0);
      }
      setStartedExperiments(new Set());
      setPendingExperimentTagSeeds(new Set());
      setSelectedExperiments(new Set());
    },
    selectAllDatasets: () => {
      preserveCurrentTagProjection();
      setRunFacetResetRequestedAt(null);
      setSelectedDatasets(new Set(datasetOptionValues));
    },
    selectNoDatasets: () => {
      preserveCurrentTagProjection();
      setRunFacetResetRequestedAt(null);
      setSelectedDatasets(new Set());
    },
    selectAllModels: () => {
      preserveCurrentTagProjection();
      setRunFacetResetRequestedAt(null);
      setSelectedModels(new Set(modelOptionValues));
    },
    selectNoModels: () => {
      preserveCurrentTagProjection();
      setRunFacetResetRequestedAt(null);
      setSelectedModels(new Set());
    },
    selectAllPresets: () => {
      preserveCurrentTagProjection();
      setRunFacetResetRequestedAt(null);
      setSelectedPresets(new Set(presetOptionValues));
    },
    selectNoPresets: () => {
      preserveCurrentTagProjection();
      setRunFacetResetRequestedAt(null);
      setSelectedPresets(new Set());
    },
    selectAllTags: () => {
      setPendingExperimentTagSeeds(new Set());
      if (tagDataNeedsFreshOptions) {
        setPendingTagSelection("all");
        return;
      }
      setPendingTagSelection(null);
      setSelectedTags(
        new Set(tagOptionValues.slice(0, LOG_SELECT_ALL_TAG_LIMIT)),
      );
    },
    selectNoTags: () => {
      setPendingTagSelection(null);
      setPendingExperimentTagSeeds(new Set());
      setSelectedTags(new Set());
    },
  };
}

type LogsWorkspaceImplementation = ReturnType<
  typeof useLogsWorkspaceImplementation
>;

export type LogsChartsInput = Readonly<{
  enabled: boolean;
  visibleRuns: LogRun[];
  visibleRunIds: string[];
  runsLoading: boolean;
  hasMoreRuns: boolean;
  tagRecords: LogRunTags[];
  tagsLoading: boolean;
  tagsFetching: boolean;
  tagsRefreshing: boolean;
  tagOptions: ChecklistOption[];
  selectedTagList: string[];
  confusionMatrixRateTags: string[];
  collapsedMetricGroups: Set<LogMetricGroupKey>;
  loadedScalarTagRunCount: number;
  commands: Readonly<{
    refresh: () => void;
    openRunDetail: (runId: string) => void;
    setMetricGroupExpanded: (
      group: LogMetricGroupKey,
      expanded: boolean,
    ) => void;
    selectMetricTags: (tags: string[]) => void;
  }>;
}>;

export type LogsRunDetail = Readonly<{
  run: LogRun | undefined;
  artifacts: LogRunArtifacts | undefined;
  status: Readonly<{
    isLoading: boolean;
    error: Error | null;
  }>;
}>;

export type LogsDeletion = Pick<
  LogsDeletionProjection,
  "enabled" | "presetTargetExperiment" | "operation" | "actions"
>;

export type LogsWorkspaceState = {
  browser: LogsBrowser;
  charts: LogsChartsInput;
  detail: LogsRunDetail;
  deletion: LogsDeletion;
  commands: {
    includeStartedExperiment: (experiment: string) => void;
    clearForConnectionChange: () => void;
  };
};

function selectedBrowserValues(
  selected: Set<string>,
  options: ChecklistOption[],
) {
  return options
    .filter((option) => selected.has(option.value))
    .map((option) => option.value);
}

function logsBrowserProjection(state: LogsWorkspaceImplementation): LogsBrowser {
  const filters: LogsBrowser["filters"] = {
    experiments: {
      options: state.experimentOptions,
      selectedValues: selectedBrowserValues(
        state.selectedExperiments,
        state.experimentOptions,
      ),
    },
    datasets: {
      options: state.datasetOptions,
      selectedValues: selectedBrowserValues(
        state.selectedDatasets,
        state.datasetOptions,
      ),
    },
    models: {
      options: state.modelOptions,
      selectedValues: selectedBrowserValues(
        state.selectedModels,
        state.modelOptions,
      ),
    },
    presets: {
      options: state.presetOptions,
      selectedValues: selectedBrowserValues(
        state.selectedPresets,
        state.presetOptions,
      ),
    },
    tags: {
      options: state.tagOptions,
      selectedValues: selectedBrowserValues(
        state.selectedTags,
        state.tagOptions,
      ),
    },
  };
  const toggleActions: Record<
    LogsBrowserFilterKey,
    (value: string) => void
  > = {
    experiments: state.toggleExperiment,
    datasets: state.toggleDataset,
    models: state.toggleModel,
    presets: state.togglePreset,
    tags: state.toggleTag,
  };
  const allActions: Record<LogsBrowserFilterKey, () => void> = {
    experiments: state.selectAllExperiments,
    datasets: state.selectAllDatasets,
    models: state.selectAllModels,
    presets: state.selectAllPresets,
    tags: state.selectAllTags,
  };
  const noneActions: Record<LogsBrowserFilterKey, () => void> = {
    experiments: state.selectNoExperiments,
    datasets: state.selectNoDatasets,
    models: state.selectNoModels,
    presets: state.selectNoPresets,
    tags: state.selectNoTags,
  };
  return {
    filters,
    status: {
      isScanning: state.runsQuery.isLoading || state.experimentsQuery.isLoading,
      isRefreshing:
        state.runsQuery.isFetching || state.experimentsQuery.isFetching,
      runsError: state.runsQuery.error,
      experimentsError: state.experimentsQuery.error,
      tagsError: state.tagsQuery.error,
    },
    results: {
      hasExperiments: state.experimentOptions.length > 0,
      hasRuns: state.runs.length > 0,
    },
    pagination: {
      runs: {
        loaded: state.loadedRunCount,
        total: state.totalRunCount,
        canLoadMore: state.canLoadMoreRuns,
        isLoadingMore: state.isLoadingMoreRuns,
      },
      scalarTags: {
        loadedRuns: state.loadedScalarTagRunCount,
        totalRuns: state.totalScalarTagRunCount,
        canLoadMore: state.canLoadMoreScalarTags,
        isLoadingMore: state.isLoadingMoreScalarTags,
      },
    },
    actions: {
      toggleFilter: (filter, value) => toggleActions[filter](value),
      selectAll: (filter) => allActions[filter](),
      selectNone: (filter) => noneActions[filter](),
      refresh: state.refreshLogLists,
      loadMoreRuns: state.loadMoreRuns,
      loadMoreScalarTags: state.loadMoreScalarTags,
    },
  };
}

function logsChartProjection(
  state: LogsWorkspaceImplementation,
): LogsChartsInput {
  return {
    collapsedMetricGroups: state.collapsedMetricGroups,
    confusionMatrixRateTags: state.confusionMatrixRateTags,
    enabled: state.enabled,
    hasMoreRuns: Boolean(state.runsQuery.data?.hasMore),
    loadedScalarTagRunCount: state.loadedScalarTagRunCount,
    runsLoading: state.runsQuery.isLoading,
    selectedTagList: state.selectedTagList,
    tagOptions: state.tagOptions,
    tagRecords: state.chartTagRecords,
    tagsFetching: state.tagsQuery.isFetching,
    tagsLoading: state.tagsQuery.isLoading,
    tagsRefreshing: state.tagProjectionTransitionPending,
    visibleRunIds: state.chartVisibleRunIds,
    visibleRuns: state.chartVisibleRuns,
    commands: {
      refresh: () => {
        void state.refreshLogLists();
      },
      openRunDetail: (runId) => state.setSelectedDetailRunId(runId),
      setMetricGroupExpanded: (group, expanded) => {
        const isExpanded = !state.collapsedMetricGroups.has(group);
        if (isExpanded !== expanded) {
          state.toggleMetricGroup(group);
        }
      },
      selectMetricTags: (tags) => {
        const nextSelectedTags = new Set(tags);
        if (
          selectionKey(nextSelectedTags) ===
          selectionKey(new Set(state.selectedTagList))
        ) {
          return;
        }
        state.setSelectedTagValues(nextSelectedTags);
      },
    },
  };
}

function logsDeletionProjection(
  state: LogsWorkspaceImplementation,
): LogsDeletion {
  return {
    enabled: state.deletion.enabled,
    presetTargetExperiment: state.deletion.presetTargetExperiment,
    operation: state.deletion.operation,
    actions: state.deletion.actions,
  };
}

export function useLogsWorkspaceState(input: {
  enabled: boolean;
  logDeletionEnabled?: boolean;
}): LogsWorkspaceState {
  const state = useLogsWorkspaceImplementation(input);
  const artifactsQuery = useLogRunArtifactsQuery({
    runId: state.selectedRun?.id,
    enabled: state.enabled,
  });
  return {
    browser: logsBrowserProjection(state),
    charts: logsChartProjection(state),
    detail: {
      run: state.selectedRun,
      artifacts: artifactsQuery.data,
      status: {
        isLoading: artifactsQuery.isLoading,
        error: artifactsQuery.error,
      },
    },
    deletion: logsDeletionProjection(state),
    commands: {
      includeStartedExperiment: state.includeStartedExperiment,
      clearForConnectionChange: state.clearForConnectionChange,
    },
  };
}
