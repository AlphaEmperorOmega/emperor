import {
  type Dispatch,
  type SetStateAction,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { useMutation } from "@tanstack/react-query";
import {
  createLogRunDeletePlan,
  deleteLogExperiment,
  deleteLogRuns,
  type LogRunDeleteFilters,
} from "@/lib/api";
import {
  useLogExperimentsQuery,
  useLogRunsQuery,
  useLogTagsQuery,
} from "@/features/viewer/state/logs/use-log-queries";
import { useLogQueryCache } from "@/features/viewer/state/logs/use-log-query-cache";
import { logQueryKeys } from "@/lib/query-keys";
import {
  buildExperimentOptions,
  buildLogScalarTagOptions,
  isDefaultScalarTag,
  type LogMetricGroupKey,
  selectedOptionsSet,
  setAllValues,
  setNoValues,
} from "@/features/viewer/state/logs/logs-selectors";
import {
  addValueToInitializedSelection,
  addValuesToInitializedSelection,
  buildCommonRunFacetOptions,
  buildLogRunDeleteFilters,
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
} from "@/features/viewer/state/logs/logs-selection-state";

const TARGET_LOG_RUN_LIMIT = 5;
const CUSTOM_LOG_RUN_LIMIT = 500;
const LOG_SELECT_ALL_TAG_LIMIT = 100;
const DEFAULT_COLLAPSED_METRIC_GROUPS = new Set<LogMetricGroupKey>([
  "test",
  "other",
]);

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

function logDeletionDisabledError() {
  return new Error("Log deletion is disabled by backend capabilities.");
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
export function useLogsWorkspaceState({
  enabled,
  logDeletionEnabled = true,
  targetScope,
}: {
  enabled: boolean;
  logDeletionEnabled?: boolean;
  targetScope: LogsTargetScope;
}) {
  const { invalidateLogLists, refreshAfterMutation } = useLogQueryCache();
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
  const [selectedTags, setSelectedTags] = useState<Set<string> | null>(null);
  const [pendingExperimentTagSeeds, setPendingExperimentTagSeeds] = useState<
    Set<string>
  >(new Set());
  const [pendingTagSelection, setPendingTagSelection] = useState<"all" | null>(
    null,
  );
  const [selectedDetailRunId, setSelectedDetailRunId] = useState<string | null>(null);
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
  const runsQuery = useLogRunsQuery({
    enabled: canQueryRuns,
    filters: isTargetScopeMode ? targetRunFilters : customRunFilters,
    pagination: {
      limit: isTargetScopeMode ? TARGET_LOG_RUN_LIMIT : CUSTOM_LOG_RUN_LIMIT,
      offset: 0,
    },
    includeAllPages:
      !isTargetScopeMode && hasSelectedExperimentFilters ? true : undefined,
  });
  const experimentsQuery = useLogExperimentsQuery({ enabled });

  const runsData = runsQuery.data?.runs;
  const runs = useMemo(() => runsData ?? [], [runsData]);
  const experimentsData = experimentsQuery.data?.experiments;
  const experiments = useMemo(() => experimentsData ?? [], [experimentsData]);
  const experimentOptions = useMemo(
    () => buildExperimentOptions(experiments),
    [experiments],
  );

  const includeStartedExperiment = useCallback((logFolder: string) => {
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
  }, [resetLogSelections]);

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
      }),
    [experimentSet, runs],
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
    shouldSelectFirstRunFacets,
  ]);

  const visibleRuns = useMemo(
    () =>
      filterVisibleLogRuns(runs, {
        experiments: experimentSet,
        datasets: datasetSet,
        models: modelSet,
        presets: presetSet,
      }),
    [datasetSet, experimentSet, modelSet, presetSet, runs],
  );

  const visibleRunIds = useMemo(() => visibleRuns.map((run) => run.id), [visibleRuns]);
  const tagsQuery = useLogTagsQuery({
    runIds: visibleRunIds,
    enabled,
    queryKey: logQueryKeys.tagsForRuns(visibleRunIds),
  });

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
  const runDeleteFilters: LogRunDeleteFilters = useMemo(
    () => buildLogRunDeleteFilters(visibleRuns),
    [visibleRuns],
  );
  const deleteExperimentMutation = useMutation({
    mutationFn: deleteLogExperiment,
    onSuccess: (result) => {
      const deletedRunIds = new Set(result.deletedRunIds);
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
  });
  const runDeletePlanMutation = useMutation({
    mutationFn: createLogRunDeletePlan,
  });
  const deleteRunsMutation = useMutation({
    mutationFn: deleteLogRuns,
    onSuccess: (result) => {
      const deletedRunIds = new Set(result.deletedRunIds);
      setSelectedDetailRunId((previous) =>
        pruneDeletedDetailRunId({
          selectedDetailRunId: previous,
          deletedRunIds,
        }),
      );
      void refreshAfterMutation({ runIds: result.deletedRunIds });
    },
  });

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
    runDeleteFilters,
    runDeletePlan: runDeletePlanMutation.data,
    createRunDeletePlan: (filters: LogRunDeleteFilters = runDeleteFilters) => {
      if (!logDeletionEnabled) {
        return Promise.reject(logDeletionDisabledError());
      }
      return runDeletePlanMutation.mutateAsync(filters);
    },
    runDeletePlanError: runDeletePlanMutation.error,
    isPlanningRunDelete: runDeletePlanMutation.isPending,
    deleteRuns: (filters: LogRunDeleteFilters = runDeleteFilters) => {
      if (!logDeletionEnabled) {
        return Promise.reject(logDeletionDisabledError());
      }
      return deleteRunsMutation.mutateAsync(filters);
    },
    runDeleteError: deleteRunsMutation.error,
    isDeletingRunDelete: deleteRunsMutation.isPending,
    resetRunDelete: () => {
      runDeletePlanMutation.reset();
      deleteRunsMutation.reset();
    },
    deleteExperiment: (experiment: string) => {
      if (!logDeletionEnabled) {
        return Promise.reject(logDeletionDisabledError());
      }
      return deleteExperimentMutation.mutateAsync(experiment);
    },
    deleteExperimentError: deleteExperimentMutation.error,
    isDeletingExperiment: deleteExperimentMutation.isPending,
    resetDeleteExperiment: deleteExperimentMutation.reset,
    refreshLogLists: invalidateLogLists,
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

export type LogsWorkspaceState = ReturnType<typeof useLogsWorkspaceState>;
