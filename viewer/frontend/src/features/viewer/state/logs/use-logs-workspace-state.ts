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
  COMMON_SCALAR_TAGS,
  buildCountOptions,
  buildExperimentOptions,
  isDefaultScalarTag,
  type LogMetricGroupKey,
  runOption,
  selectedOptionsSet,
  setAllValues,
  setNoValues,
} from "@/features/viewer/state/logs/logs-selectors";
import {
  addValueToInitializedSelection,
  addValuesToInitializedSelection,
  buildInitialExperimentSelection,
  buildInitialRunFacetSelection,
  buildInitialRunIdSelection,
  buildLogRunDeleteFilters,
  filterVisibleLogRuns,
  nextSelectedDetailRunId,
  pruneDeletedDetailRunId,
  removeStartedExperiment,
  removeValueFromSelection,
  removeValuesFromSelection,
  startedRunSelections,
} from "@/features/viewer/state/logs/logs-selection-state";

const TARGET_LOG_RUN_LIMIT = 5;
const CUSTOM_LOG_RUN_LIMIT = 500;
const LOG_SELECT_ALL_RUN_LIMIT = 100;
const LOG_SELECT_ALL_TAG_LIMIT = 100;
const DEFAULT_COLLAPSED_METRIC_GROUPS = new Set<LogMetricGroupKey>([
  "train",
  "test",
  "other",
]);

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

export type LogsScopeMode = "target" | "custom";

export type LogsTargetScope = {
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
        model: targetScope.model,
        preset: targetScope.preset,
        datasets: [...targetScope.datasets].sort(),
      }),
    [targetScope.datasets, targetScope.model, targetScope.preset],
  );
  const [appliedTargetScopeKey, setAppliedTargetScopeKey] =
    useState(targetScopeKey);
  const [startedExperiments, setStartedExperiments] = useState<Set<string>>(new Set());
  const [selectedExperiments, setSelectedExperiments] = useState<Set<string> | null>(null);
  const [selectedDatasets, setSelectedDatasets] = useState<Set<string> | null>(null);
  const [selectedModels, setSelectedModels] = useState<Set<string> | null>(null);
  const [selectedPresets, setSelectedPresets] = useState<Set<string> | null>(null);
  const [selectedRunIds, setSelectedRunIds] = useState<Set<string> | null>(null);
  const [selectedTags, setSelectedTags] = useState<Set<string> | null>(null);
  const [selectedDetailRunId, setSelectedDetailRunId] = useState<string | null>(null);
  const [collapsedMetricGroups, setCollapsedMetricGroups] = useState<
    Set<LogMetricGroupKey>
  >(new Set(DEFAULT_COLLAPSED_METRIC_GROUPS));

  const hasTargetScope = Boolean(
    targetScope.model && targetScope.preset && targetScope.datasets.length > 0,
  );
  const targetRunFilters = useMemo(
    () =>
      hasTargetScope
        ? {
            model: [targetScope.model],
            preset: [targetScope.preset],
            dataset: targetScope.datasets,
            hasEventFiles: true,
          }
        : { hasEventFiles: true },
    [hasTargetScope, targetScope.datasets, targetScope.model, targetScope.preset],
  );
  const isTargetScopeMode = scopeMode === "target";
  const runsQuery = useLogRunsQuery({
    enabled,
    filters: isTargetScopeMode ? targetRunFilters : undefined,
    pagination: {
      limit: isTargetScopeMode ? TARGET_LOG_RUN_LIMIT : CUSTOM_LOG_RUN_LIMIT,
      offset: 0,
    },
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
    setSelectedExperiments(null);
    setSelectedDatasets(null);
    setSelectedModels(null);
    setSelectedPresets(null);
    setSelectedRunIds(null);
    setSelectedTags(null);
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
    setSelectedRunIds((previous) => {
      return addValuesToInitializedSelection(previous, startedSelections.runIds);
    });
  }, [runs, startedExperiments]);

  const experimentSet = useMemo(
    () =>
      selectedExperiments ??
      buildInitialExperimentSelection({ experimentOptions, startedExperiments }),
    [experimentOptions, selectedExperiments, startedExperiments],
  );
  const experimentRuns = useMemo(
    () => runs.filter((run) => experimentSet.has(run.experiment)),
    [experimentSet, runs],
  );
  const datasetOptions = useMemo(
    () => buildCountOptions(experimentRuns, "dataset"),
    [experimentRuns],
  );
  const modelOptions = useMemo(
    () => buildCountOptions(experimentRuns, "model"),
    [experimentRuns],
  );
  const presetOptions = useMemo(
    () => buildCountOptions(experimentRuns, "preset"),
    [experimentRuns],
  );
  const runOptions = useMemo(() => experimentRuns.map(runOption), [experimentRuns]);

  const datasetSet = useMemo(
    () => selectedDatasets ?? buildInitialRunFacetSelection(runs, "dataset"),
    [runs, selectedDatasets],
  );
  const modelSet = useMemo(
    () => selectedModels ?? buildInitialRunFacetSelection(runs, "model"),
    [runs, selectedModels],
  );
  const presetSet = useMemo(
    () => selectedPresets ?? buildInitialRunFacetSelection(runs, "preset"),
    [runs, selectedPresets],
  );
  const runIdSet = useMemo(
    () => selectedRunIds ?? buildInitialRunIdSelection(runs),
    [runs, selectedRunIds],
  );

  const visibleRuns = useMemo(
    () =>
      filterVisibleLogRuns(runs, {
        experiments: experimentSet,
        datasets: datasetSet,
        models: modelSet,
        presets: presetSet,
        runIds: runIdSet,
      }),
    [datasetSet, experimentSet, modelSet, presetSet, runIdSet, runs],
  );

  const visibleRunIds = useMemo(() => visibleRuns.map((run) => run.id), [visibleRuns]);
  const tagsQuery = useLogTagsQuery({
    runIds: visibleRunIds,
    enabled,
    queryKey: logQueryKeys.tagsForRuns(visibleRunIds),
  });

  const tagOptions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const runTags of tagsQuery.data?.runs ?? []) {
      for (const tag of runTags.scalarTags) {
        counts.set(tag, (counts.get(tag) ?? 0) + 1);
      }
    }
    return Array.from(counts, ([value, count]) => ({
      value,
      label: value,
      count,
    })).sort((a, b) => {
      const commonA = COMMON_SCALAR_TAGS.indexOf(a.value);
      const commonB = COMMON_SCALAR_TAGS.indexOf(b.value);
      if (commonA >= 0 || commonB >= 0) {
        return (commonA >= 0 ? commonA : 999) - (commonB >= 0 ? commonB : 999);
      }
      return a.value.localeCompare(b.value);
    });
  }, [tagsQuery.data]);

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
    () => selectedTags ?? defaultSelectedTags,
    [defaultSelectedTags, selectedTags],
  );

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
    () =>
      buildLogRunDeleteFilters({
        experiments: experimentSet,
        datasets: datasetSet,
        models: modelSet,
        presets: presetSet,
        runIds: runIdSet,
      }),
    [datasetSet, experimentSet, modelSet, presetSet, runIdSet],
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
      setSelectedRunIds((previous) => {
        return removeValuesFromSelection({
          selection: previous,
          fallbackValues: runs.map((run) => run.id),
          values: deletedRunIds,
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
      setSelectedRunIds((previous) => {
        return removeValuesFromSelection({
          selection: previous,
          fallbackValues: runs.map((run) => run.id),
          values: deletedRunIds,
        });
      });
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
    runOptions,
    tagOptions,
    visibleRuns,
    visibleRunIds,
    selectedDatasets: datasetSet,
    selectedExperiments: experimentSet,
    selectedModels: modelSet,
    selectedPresets: presetSet,
    selectedRunIds: runIdSet,
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
      toggleSetValueWithFallback(
        setSelectedExperiments,
        experimentOptions.map((option) => option.value),
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
    toggleRun: (value: string) => {
      markCustomScope();
      toggleSetValueWithFallback(
        setSelectedRunIds,
        runOptions.map((option) => option.value),
        value,
      );
    },
    toggleTag: (value: string) =>
      toggleSetValueWithFallback(
        setSelectedTags,
        Array.from(defaultSelectedTags),
        value,
      ),
    selectAllExperiments: () => {
      markCustomScope();
      setAllValues(
        setSelectedExperiments,
        experimentOptions.map((option) => option.value),
      );
    },
    selectNoExperiments: () => {
      markCustomScope();
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
    selectAllRuns: () => {
      markCustomScope();
      setAllValues(
        setSelectedRunIds,
        runOptions.slice(0, LOG_SELECT_ALL_RUN_LIMIT).map((option) => option.value),
      );
    },
    selectNoRuns: () => {
      markCustomScope();
      setNoValues(setSelectedRunIds);
    },
    selectAllTags: () =>
      setAllValues(
        setSelectedTags,
        tagOptions.slice(0, LOG_SELECT_ALL_TAG_LIMIT).map((option) => option.value),
      ),
    selectNoTags: () => setNoValues(setSelectedTags),
  };
}

export type LogsWorkspaceState = ReturnType<typeof useLogsWorkspaceState>;
