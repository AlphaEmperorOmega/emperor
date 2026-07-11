import {
  type Dispatch,
  type SetStateAction,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type LogExperimentDeleteResponse,
  type LogRunDeleteResponse,
  type LogRunTags,
} from "@/lib/api";
import {
  useInfiniteLogRunsQuery,
  useLogExperimentsQuery,
  useLogTagQueries,
  useLogRunsQuery,
} from "@/features/workbench/state/logs/use-log-queries";
import { useLogQueryCache } from "@/features/workbench/state/logs/use-log-query-cache";
import { logQueryKeys } from "@/lib/query-keys";
import {
  useLogsDeletionState,
  type LogsSubsetDeleteTarget,
} from "@/features/workbench/state/logs/_logs-deletion-state";
import {
  buildExperimentOptions,
  buildLogScalarTagOptions,
  isDefaultScalarTag,
  type LogMetricGroupKey,
  selectedOptionsSet,
  setAllValues,
  setNoValues,
} from "@/features/workbench/state/logs/logs-selectors";
import {
  addValueToInitializedSelection,
  addValuesToInitializedSelection,
  buildCommonRunFacetOptions,
  buildExperimentScalarTagSeedSelection,
  effectiveSelectionForAvailableValues,
  filterVisibleLogRuns,
  nextSelectedDetailRunId,
  normalizeRunFacetSelection,
  pruneSelectionToAvailableValues,
  pruneDeletedDetailRunId,
  removeStartedExperiment,
  removeValueFromSelection,
  selectionSetOrDefault,
  startedRunSelections,
  sortedSelectionValues,
} from "@/features/workbench/state/logs/_logs-selection-state";

const TARGET_LOG_RUN_LIMIT = 5;
const CUSTOM_LOG_RUN_LIMIT = 100;
const MAX_CUSTOM_LOG_RUN_LIMIT = 2000;
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

function toggleSetValueWithFallback(
  setValues: Dispatch<SetStateAction<Set<string> | null>>,
  fallbackValues: string[],
  value: string,
) {
  setValues((previous) => {
    const next = new Set(previous ?? fallbackValues);
    if (next.has(value)) {
      next.delete(value);
    } else {
      next.add(value);
    }
    return next;
  });
}

function selectionMatchesValues(selection: Set<string>, values: string[]) {
  if (selection.size !== values.length) {
    return false;
  }
  return values.every((value) => selection.has(value));
}

export type LogsScopeMode = "target" | "custom";

export type LogsTargetScope = {
  modelType: string;
  model: string;
  preset: string;
  datasets: string[];
};

/**
 * Owns all state for the logs workspace: the run/experiment/tag queries, the
 * multi-facet selection sets (experiment/dataset/model/preset/run/tag), the
 * detail-run selection, and experiment deletion. Returned to the workspace
 * panels as a single object so they stay presentational.
 */
export function useLogsWorkspaceImplementation({
  enabled,
  logDeletionEnabled = true,
  targetScope,
}: {
  enabled: boolean;
  logDeletionEnabled?: boolean;
  targetScope: LogsTargetScope;
}) {
  const { refreshAfterMutation, refreshLogs } = useLogQueryCache();
  const [scopeMode, setScopeMode] = useState<LogsScopeMode>("target");
  const targetScopeKey = useMemo(
    () =>
      JSON.stringify({
        modelType: targetScope.modelType,
        model: targetScope.model,
        preset: targetScope.preset,
        datasets: [...targetScope.datasets].sort(),
      }),
    [
      targetScope.datasets,
      targetScope.model,
      targetScope.modelType,
      targetScope.preset,
    ],
  );
  const [appliedTargetScopeKey, setAppliedTargetScopeKey] =
    useState(targetScopeKey);
  const [startedExperiments, setStartedExperiments] = useState<Set<string>>(new Set());
  const [selectedExperiments, setSelectedExperiments] = useState<Set<string> | null>(
    emptySelection,
  );
  const [selectedDatasets, setSelectedDatasets] = useState<Set<string> | null>(null);
  const [selectedModels, setSelectedModels] = useState<Set<string> | null>(null);
  const [selectedPresets, setSelectedPresets] = useState<Set<string> | null>(null);
  const [shouldSelectFirstRunFacets, setShouldSelectFirstRunFacets] =
    useState(false);
  const [scalarTagRunWindowState, setScalarTagRunWindowState] = useState({
    key: "",
    count: SCALAR_TAG_RUN_WINDOW_SIZE,
  });
  const [selectedTags, setSelectedTags] = useState<Set<string> | null>(null);
  const [pendingExperimentTagSeeds, setPendingExperimentTagSeeds] = useState<
    Set<string>
  >(new Set());
  const [pendingTagSelection, setPendingTagSelection] = useState<"all" | null>(
    null,
  );
  const previousTagDataRef = useRef<{ runs: LogRunTags[] } | undefined>(undefined);
  const [selectedDetailRunId, setSelectedDetailRunId] = useState<string | null>(null);
  const [deletedRunIdsAwaitingRefresh, setDeletedRunIdsAwaitingRefresh] =
    useState<Set<string>>(new Set());
  const [subsetExperimentReconciliation, setSubsetExperimentReconciliation] =
    useState<Map<string, number>>(new Map());
  const experimentsFetchGenerationRef = useRef(0);
  const previousExperimentsFetchRef = useRef({
    dataUpdatedAt: 0,
    isFetching: false,
  });
  const [collapsedMetricGroups, setCollapsedMetricGroups] = useState<
    Set<LogMetricGroupKey>
  >(new Set(DEFAULT_COLLAPSED_METRIC_GROUPS));

  const hasTargetScope = Boolean(
    targetScope.modelType &&
      targetScope.model &&
      targetScope.preset &&
      targetScope.datasets.length > 0,
  );
  const targetRunFilters = useMemo(
    () =>
      hasTargetScope
        ? {
            models: [{ modelType: targetScope.modelType, model: targetScope.model }],
            preset: [targetScope.preset],
            dataset: targetScope.datasets,
            hasEventFiles: true,
          }
        : { hasEventFiles: true },
    [
      hasTargetScope,
      targetScope.datasets,
      targetScope.model,
      targetScope.modelType,
      targetScope.preset,
    ],
  );
  const isTargetScopeMode = scopeMode === "target";
  const selectedExperimentQueryValues = useMemo(
    () => (selectedExperiments ? sortedSelectionValues(selectedExperiments) : []),
    [selectedExperiments],
  );
  const hasSelectedExperimentFilters = selectedExperimentQueryValues.length > 0;
  const customRunFilters = useMemo(
    () =>
      hasSelectedExperimentFilters
        ? { experiment: selectedExperimentQueryValues }
        : undefined,
    [hasSelectedExperimentFilters, selectedExperimentQueryValues],
  );
  const canQueryRuns = enabled && (!isTargetScopeMode || hasTargetScope);
  const targetRunsQuery = useLogRunsQuery({
    enabled: canQueryRuns && isTargetScopeMode,
    filters: targetRunFilters,
    pagination: {
      limit: TARGET_LOG_RUN_LIMIT,
      offset: 0,
    },
  });
  const customRunPagesQuery = useInfiniteLogRunsQuery({
    enabled: canQueryRuns && !isTargetScopeMode,
    filters: customRunFilters,
    pageSize: CUSTOM_LOG_RUN_LIMIT,
  });
  const customRunsData = useMemo(() => {
    const pages = customRunPagesQuery.data?.pages;
    if (!pages?.length) {
      return undefined;
    }
    const firstPage = pages[0];
    const lastPage = pages[pages.length - 1];
    return {
      ...firstPage,
      runs: pages.flatMap((page) => page.runs),
      total: firstPage.total,
      limit: CUSTOM_LOG_RUN_LIMIT,
      offset: 0,
      hasMore: Boolean(lastPage.hasMore),
    };
  }, [customRunPagesQuery.data?.pages]);
  const customRunsQuery = {
    ...customRunPagesQuery,
    data: customRunsData,
  };
  const runsQuery = isTargetScopeMode ? targetRunsQuery : customRunsQuery;
  const experimentsQuery = useLogExperimentsQuery({ enabled });

  useEffect(() => {
    const previous = previousExperimentsFetchRef.current;
    const completedSuccessfulFetch =
      Boolean(experimentsQuery.data) &&
      !experimentsQuery.isFetching &&
      !experimentsQuery.error &&
      (previous.isFetching ||
        experimentsQuery.dataUpdatedAt !== previous.dataUpdatedAt);
    if (completedSuccessfulFetch) {
      experimentsFetchGenerationRef.current += 1;
    }
    previousExperimentsFetchRef.current = {
      dataUpdatedAt: experimentsQuery.dataUpdatedAt,
      isFetching: experimentsQuery.isFetching,
    };
  }, [
    experimentsQuery.data,
    experimentsQuery.dataUpdatedAt,
    experimentsQuery.error,
    experimentsQuery.isFetching,
  ]);

  const runsData = runsQuery.data?.runs;
  const runs = useMemo(() => runsData ?? [], [runsData]);
  const experimentsData = experimentsQuery.data?.experiments;
  const experiments = useMemo(() => experimentsData ?? [], [experimentsData]);

  const experimentOptions = useMemo(
    () => buildExperimentOptions(experiments),
    [experiments],
  );

  const includeStartedExperiment = useCallback((logFolder: string) => {
    setScopeMode("custom");
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
    setShouldSelectFirstRunFacets(false);
    setSelectedTags(null);
    setPendingExperimentTagSeeds(new Set());
    setPendingTagSelection(null);
    setSelectedDetailRunId(null);
    setCollapsedMetricGroups(new Set(DEFAULT_COLLAPSED_METRIC_GROUPS));
  }, []);

  const useCurrentTargetScope = useCallback(() => {
    setScopeMode("target");
    setAppliedTargetScopeKey(targetScopeKey);
    resetLogSelections();
  }, [resetLogSelections, targetScopeKey]);

  const showAllRuns = useCallback(() => {
    setScopeMode("custom");
    resetLogSelections();
    const experimentValues = new Set([
      ...experimentOptions.map((option) => option.value),
      ...startedExperiments,
    ]);
    setSelectedExperiments(experimentValues);
    setPendingExperimentTagSeeds(new Set(experimentValues));
  }, [experimentOptions, resetLogSelections, startedExperiments]);

  const markCustomScope = useCallback(() => {
    setScopeMode("custom");
  }, []);

  useEffect(() => {
    if (scopeMode !== "target" || appliedTargetScopeKey === targetScopeKey) {
      return;
    }
    setAppliedTargetScopeKey(targetScopeKey);
    resetLogSelections();
  }, [
    appliedTargetScopeKey,
    resetLogSelections,
    scopeMode,
    targetScopeKey,
  ]);

  useEffect(() => {
    if (startedExperiments.size === 0 || runs.length === 0) {
      return;
    }
    const startedSelections = startedRunSelections({ runs, startedExperiments });
    if (!startedSelections.hasStartedRuns) {
      return;
    }
    setSelectedExperiments((previous) => {
      return addValuesToInitializedSelection(previous, startedSelections.experiments);
    });
    setSelectedDatasets((previous) => {
      return addValuesToInitializedSelection(previous, startedSelections.datasets);
    });
    setSelectedModels((previous) => {
      return addValuesToInitializedSelection(previous, startedSelections.models);
    });
    setSelectedPresets((previous) => {
      return addValuesToInitializedSelection(previous, startedSelections.presets);
    });
  }, [runs, startedExperiments]);

  const experimentSet = useMemo(
    () => selectedExperiments ?? emptySelection(),
    [selectedExperiments],
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

  const datasetSet = useMemo(
    () =>
      effectiveSelectionForAvailableValues(selectedDatasets, datasetOptionValues),
    [datasetOptionValues, selectedDatasets],
  );
  const modelSet = useMemo(
    () =>
      effectiveSelectionForAvailableValues(selectedModels, modelOptionValues),
    [modelOptionValues, selectedModels],
  );
  const presetSet = useMemo(
    () =>
      effectiveSelectionForAvailableValues(selectedPresets, presetOptionValues),
    [presetOptionValues, selectedPresets],
  );

  useEffect(() => {
    if (!runsQuery.data) {
      return;
    }
    const selectFirstAvailable = shouldSelectFirstRunFacets;
    if (
      selectFirstAvailable &&
      (runsQuery.isPlaceholderData ||
        datasetOptionValues.length === 0 ||
        modelOptionValues.length === 0 ||
        presetOptionValues.length === 0)
    ) {
      return;
    }
    setSelectedDatasets((previous) =>
      normalizeRunFacetSelection({
        selection: previous,
        availableValues: datasetOptionValues,
        selectFirstAvailable,
      }),
    );
    setSelectedModels((previous) =>
      normalizeRunFacetSelection({
        selection: previous,
        availableValues: modelOptionValues,
        selectFirstAvailable,
      }),
    );
    setSelectedPresets((previous) =>
      normalizeRunFacetSelection({
        selection: previous,
        availableValues: presetOptionValues,
        selectFirstAvailable,
      }),
    );
    if (selectFirstAvailable) {
      setShouldSelectFirstRunFacets(false);
    }
  }, [
    datasetOptionValues,
    modelOptionValues,
    presetOptionValues,
    runsQuery.data,
    runsQuery.isPlaceholderData,
    shouldSelectFirstRunFacets,
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

  useEffect(() => {
    if (
      deletedRunIdsAwaitingRefresh.size === 0 ||
      !runsQuery.data ||
      runsQuery.isFetching ||
      runsQuery.isPlaceholderData ||
      runsQuery.error
    ) {
      return;
    }
    const authoritativeRunIds = new Set(runs.map((run) => run.id));
    setDeletedRunIdsAwaitingRefresh((previous) => {
      const next = new Set(
        Array.from(previous).filter((runId) => authoritativeRunIds.has(runId)),
      );
      return next.size === previous.size ? previous : next;
    });
  }, [
    deletedRunIdsAwaitingRefresh.size,
    runs,
    runsQuery.data,
    runsQuery.error,
    runsQuery.isFetching,
    runsQuery.isPlaceholderData,
  ]);

  const visibleRunIds = useMemo(() => visibleRuns.map((run) => run.id), [visibleRuns]);
  const scalarTagRunWindowKey = useMemo(
    () => visibleRunIds.join("\u0000"),
    [visibleRunIds],
  );
  const scalarTagRunCount =
    scalarTagRunWindowState.key === scalarTagRunWindowKey
      ? scalarTagRunWindowState.count
      : SCALAR_TAG_RUN_WINDOW_SIZE;
  useEffect(() => {
    if (scalarTagRunWindowState.key === scalarTagRunWindowKey) {
      return;
    }
    setScalarTagRunWindowState({
      key: scalarTagRunWindowKey,
      count: SCALAR_TAG_RUN_WINDOW_SIZE,
    });
  }, [scalarTagRunWindowKey, scalarTagRunWindowState.key]);
  const scalarTagRunIds = useMemo(
    () => visibleRunIds.slice(0, scalarTagRunCount),
    [scalarTagRunCount, visibleRunIds],
  );
  const tagsEnabled = enabled && !shouldSelectFirstRunFacets;
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
  useEffect(() => {
    if (!currentTagsData || tagWindowsHavePlaceholderData) {
      return;
    }
    previousTagDataRef.current = currentTagsData;
  }, [currentTagsData, tagWindowsHavePlaceholderData]);
  const placeholderTagsData =
    scalarTagWindowInputs.length > 0 &&
    (tagWindowsAreFetching || shouldSelectFirstRunFacets)
      ? previousTagDataRef.current
      : undefined;
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
          (tagWindowsAreFetching || shouldSelectFirstRunFacets),
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
  const selectedTagsSet = useMemo(
    () => selectionSetOrDefault(selectedTags, defaultSelectedTags),
    [defaultSelectedTags, selectedTags],
  );
  const tagDataNeedsFreshOptions =
    tagsQuery.isFetching ||
    tagsQuery.isLoading ||
    tagsQuery.isError ||
    Boolean(tagsQuery.isPlaceholderData);

  useEffect(() => {
    if (
      pendingExperimentTagSeeds.size === 0 ||
      visibleRuns.length === 0 ||
      tagDataNeedsFreshOptions
    ) {
      return;
    }
    const tagSeeds = buildExperimentScalarTagSeedSelection({
      visibleRuns,
      tagRuns: tagsQuery.data?.runs,
      pendingExperiments: pendingExperimentTagSeeds,
      selectedTags,
      tagOptionValues,
      selectAllLimit: LOG_SELECT_ALL_TAG_LIMIT,
    });
    if (tagSeeds.loadedExperiments.size === 0) {
      return;
    }
    if (tagSeeds.selectedTags !== selectedTags) {
      setSelectedTags(tagSeeds.selectedTags);
    }
    setPendingExperimentTagSeeds((previous) => {
      const next = new Set(previous);
      for (const experiment of tagSeeds.loadedExperiments) {
        next.delete(experiment);
      }
      return next.size === previous.size ? previous : next;
    });
  }, [
    pendingExperimentTagSeeds,
    selectedTags,
    tagDataNeedsFreshOptions,
    tagOptionValues,
    tagsQuery.data?.runs,
    visibleRuns,
  ]);

  useEffect(() => {
    if (tagDataNeedsFreshOptions) {
      return;
    }
    if (pendingTagSelection === "all") {
      setSelectedTags(new Set(tagOptionValues.slice(0, LOG_SELECT_ALL_TAG_LIMIT)));
      setPendingTagSelection(null);
      return;
    }
    setSelectedTags((previous) =>
      pruneSelectionToAvailableValues(previous, tagOptionValues),
    );
  }, [pendingTagSelection, tagDataNeedsFreshOptions, tagOptionValues]);

  useEffect(() => {
    const nextDetailRunId = nextSelectedDetailRunId(selectedDetailRunId, visibleRuns);
    if (nextDetailRunId !== selectedDetailRunId) {
      setSelectedDetailRunId(nextDetailRunId);
    }
  }, [selectedDetailRunId, visibleRuns]);

  const selectedTagList = useMemo(
    () =>
      Array.from(selectedOptionsSet(selectedTagsSet, tagOptions)).sort((a, b) =>
        a.localeCompare(b),
      ),
    [selectedTagsSet, tagOptions],
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
  const selectedRun = visibleRuns.find((run) => run.id === selectedDetailRunId);
  const onExperimentDeleted = useCallback(
    (result: LogExperimentDeleteResponse) => {
      const deletedRunIds = new Set(result.deletedRunIds);
      setDeletedRunIdsAwaitingRefresh((previous) =>
        new Set([...previous, ...result.deletedRunIds]),
      );
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
    [experimentOptions, refreshAfterMutation],
  );
  const onRunsDeleted = useCallback(
    (result: LogRunDeleteResponse, target: LogsSubsetDeleteTarget) => {
      const deletedRunIds = new Set(result.deletedRunIds);
      setDeletedRunIdsAwaitingRefresh((previous) =>
        new Set([...previous, ...result.deletedRunIds]),
      );
      setSelectedDetailRunId((previous) =>
        pruneDeletedDetailRunId({
          selectedDetailRunId: previous,
          deletedRunIds,
        }),
      );
      const requiredExperimentGeneration =
        experimentsFetchGenerationRef.current + 1;
      setSubsetExperimentReconciliation((previous) => {
        const next = new Map(previous);
        next.set(
          target.experiment,
          Math.max(
            requiredExperimentGeneration,
            next.get(target.experiment) ?? 0,
          ),
        );
        return next;
      });
      void refreshAfterMutation({ runIds: result.deletedRunIds });
    },
    [refreshAfterMutation],
  );
  const deletion = useLogsDeletionState({
    active: enabled,
    enabled: logDeletionEnabled,
    runs,
    selectedExperiments: experimentSet,
    onExperimentDeleted,
    onRunsDeleted,
  });
  const clearDeletionForConnectionChange = deletion.clearForConnectionChange;
  const clearForConnectionChange = useCallback(() => {
    setScopeMode("target");
    setAppliedTargetScopeKey(targetScopeKey);
    setStartedExperiments(new Set());
    resetLogSelections();
    setScalarTagRunWindowState({
      key: "",
      count: SCALAR_TAG_RUN_WINDOW_SIZE,
    });
    previousTagDataRef.current = undefined;
    setDeletedRunIdsAwaitingRefresh(new Set());
    setSubsetExperimentReconciliation(new Map());
    experimentsFetchGenerationRef.current = 0;
    previousExperimentsFetchRef.current = {
      dataUpdatedAt: 0,
      isFetching: false,
    };
    clearDeletionForConnectionChange();
  }, [clearDeletionForConnectionChange, resetLogSelections, targetScopeKey]);

  useEffect(() => {
    if (
      subsetExperimentReconciliation.size === 0 ||
      experimentsQuery.isFetching ||
      experimentsQuery.error ||
      !experimentsQuery.data
    ) {
      return;
    }
    const currentGeneration = experimentsFetchGenerationRef.current;
    const readyReconciliations = Array.from(
      subsetExperimentReconciliation.entries(),
    ).filter(([, requiredGeneration]) => currentGeneration >= requiredGeneration);
    if (readyReconciliations.length === 0) {
      return;
    }
    const availableExperiments = new Set(
      experimentsQuery.data.experiments.map((experiment) => experiment.experiment),
    );
    for (const [experiment] of readyReconciliations) {
      if (availableExperiments.has(experiment)) {
        continue;
      }
      setSelectedExperiments((previous) =>
        removeValueFromSelection({
          selection: previous,
          fallbackValues: experimentOptions.map((option) => option.value),
          value: experiment,
        }),
      );
      setStartedExperiments((previous) =>
        removeStartedExperiment(previous, experiment),
      );
      setPendingExperimentTagSeeds((previous) =>
        removePendingExperimentSeed(previous, experiment),
      );
    }
    setSubsetExperimentReconciliation((previous) => {
      const next = new Map(previous);
      for (const [experiment, requiredGeneration] of readyReconciliations) {
        if (next.get(experiment) === requiredGeneration) {
          next.delete(experiment);
        }
      }
      return next;
    });
  }, [
    experimentOptions,
    experimentsQuery.data,
    experimentsQuery.error,
    experimentsQuery.isFetching,
    subsetExperimentReconciliation,
  ]);

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
    tagOptions,
    confusionMatrixRateTags,
    visibleRuns,
    visibleRunIds,
    selectedDatasets: datasetSet,
    selectedExperiments: experimentSet,
    selectedModels: modelSet,
    selectedPresets: presetSet,
    selectedTags: selectedTagsSet,
    selectedTagList,
    collapsedMetricGroups,
    toggleMetricGroup,
    selectedRun,
    selectedDetailRunId,
    setSelectedDetailRunId,
    deletion,
    clearForConnectionChange,
    loadedRunCount: runs.length,
    totalRunCount: runsQuery.data?.total ?? runs.length,
    canLoadMoreRuns:
      enabled &&
      !isTargetScopeMode &&
      Boolean(customRunPagesQuery.hasNextPage) &&
      runs.length < MAX_CUSTOM_LOG_RUN_LIMIT,
    isLoadingMoreRuns:
      !isTargetScopeMode && customRunPagesQuery.isFetchingNextPage,
    loadMoreRuns: () => {
      if (
        !enabled ||
        isTargetScopeMode ||
        customRunPagesQuery.isFetchingNextPage ||
        !customRunPagesQuery.hasNextPage ||
        runs.length >= MAX_CUSTOM_LOG_RUN_LIMIT
      ) {
        return;
      }
      void customRunPagesQuery.fetchNextPage();
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
    scopeMode,
    targetScope,
    useCurrentTargetScope,
    showAllRuns,
    toggleExperiment: (value: string) => {
      markCustomScope();
      const experimentFallbackValues = experimentOptions.map((option) => option.value);
      const currentSelection = selectedExperiments ?? new Set(experimentFallbackValues);
      const isSelectingExperiment = !currentSelection.has(value);
      setPendingExperimentTagSeeds((previous) => {
        return isSelectingExperiment
          ? addPendingExperimentSeed(previous, value)
          : removePendingExperimentSeed(previous, value);
      });
      setShouldSelectFirstRunFacets(true);
      toggleSetValueWithFallback(
        setSelectedExperiments,
        experimentFallbackValues,
        value,
      );
    },
    toggleDataset: (value: string) => {
      markCustomScope();
      toggleSetValueWithFallback(
        setSelectedDatasets,
        datasetOptions.map((option) => option.value),
        value,
      );
    },
    toggleModel: (value: string) => {
      markCustomScope();
      toggleSetValueWithFallback(
        setSelectedModels,
        modelOptions.map((option) => option.value),
        value,
      );
    },
    togglePreset: (value: string) => {
      markCustomScope();
      toggleSetValueWithFallback(
        setSelectedPresets,
        presetOptions.map((option) => option.value),
        value,
      );
    },
    toggleTag: (value: string) => {
      setPendingTagSelection(null);
      toggleSetValueWithFallback(
        setSelectedTags,
        Array.from(defaultSelectedTags),
        value,
      );
    },
    selectAllExperiments: () => {
      markCustomScope();
      const experimentValues = experimentOptions.map((option) => option.value);
      const currentSelection = selectedExperiments ?? new Set(experimentValues);
      if (!selectionMatchesValues(currentSelection, experimentValues)) {
        setShouldSelectFirstRunFacets(true);
      }
      setPendingExperimentTagSeeds((previous) => {
        const next = new Set(previous);
        for (const experiment of experimentValues) {
          next.add(experiment);
        }
        return next;
      });
      setAllValues(
        setSelectedExperiments,
        experimentValues,
      );
    },
    selectNoExperiments: () => {
      markCustomScope();
      const currentSelection =
        selectedExperiments ??
        new Set(experimentOptions.map((option) => option.value));
      if (currentSelection.size > 0) {
        setShouldSelectFirstRunFacets(true);
      }
      setPendingExperimentTagSeeds(new Set());
      setNoValues(setSelectedExperiments);
    },
    selectAllDatasets: () => {
      markCustomScope();
      setAllValues(setSelectedDatasets, datasetOptions.map((option) => option.value));
    },
    selectNoDatasets: () => {
      markCustomScope();
      setNoValues(setSelectedDatasets);
    },
    selectAllModels: () => {
      markCustomScope();
      setAllValues(setSelectedModels, modelOptions.map((option) => option.value));
    },
    selectNoModels: () => {
      markCustomScope();
      setNoValues(setSelectedModels);
    },
    selectAllPresets: () => {
      markCustomScope();
      setAllValues(setSelectedPresets, presetOptions.map((option) => option.value));
    },
    selectNoPresets: () => {
      markCustomScope();
      setNoValues(setSelectedPresets);
    },
    selectAllTags: () => {
      if (tagDataNeedsFreshOptions) {
        setPendingTagSelection("all");
        return;
      }
      setAllValues(
        setSelectedTags,
        tagOptionValues.slice(0, LOG_SELECT_ALL_TAG_LIMIT),
      );
    },
    selectNoTags: () => {
      setPendingTagSelection(null);
      setNoValues(setSelectedTags);
    },
  };
}

export type LogsWorkspaceImplementation = ReturnType<
  typeof useLogsWorkspaceImplementation
>;
