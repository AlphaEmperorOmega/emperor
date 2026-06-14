import { useCallback, useEffect, useMemo, useState } from "react";
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
  toggleSetValue,
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
  selectionSetOrEmpty,
  startedRunSelections,
} from "@/features/viewer/state/logs/logs-selection-state";

function logDeletionDisabledError() {
  return new Error("Log deletion is disabled by backend capabilities.");
}

/**
 * Owns all state for the logs workspace: the run/experiment/tag queries, the
 * multi-facet selection sets (experiment/dataset/model/preset/run/tag), the
 * detail-run selection, and experiment deletion. Returned to the workspace
 * panels as a single object so they stay presentational.
 */
export function useLogsWorkspaceState({
  enabled,
  logDeletionEnabled = true,
}: {
  enabled: boolean;
  logDeletionEnabled?: boolean;
}) {
  const { invalidateLogLists, refreshAfterMutation } = useLogQueryCache();
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
  >(new Set());

  const runsQuery = useLogRunsQuery({ enabled });
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

  useEffect(() => {
    if (!enabled || !experimentsQuery.isSuccess || selectedExperiments !== null) {
      return;
    }
    setSelectedExperiments(
      buildInitialExperimentSelection({ experimentOptions, startedExperiments }),
    );
  }, [
    enabled,
    experimentOptions,
    experimentsQuery.isSuccess,
    selectedExperiments,
    startedExperiments,
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

  useEffect(() => {
    if (!enabled || !runsQuery.isSuccess || selectedDatasets !== null) {
      return;
    }
    setSelectedDatasets(
      buildInitialRunFacetSelection(runs, "dataset"),
    );
  }, [enabled, runs, runsQuery.isSuccess, selectedDatasets]);

  useEffect(() => {
    if (!enabled || !runsQuery.isSuccess || selectedModels !== null) {
      return;
    }
    setSelectedModels(buildInitialRunFacetSelection(runs, "model"));
  }, [enabled, runs, runsQuery.isSuccess, selectedModels]);

  useEffect(() => {
    if (!enabled || !runsQuery.isSuccess || selectedPresets !== null) {
      return;
    }
    setSelectedPresets(
      buildInitialRunFacetSelection(runs, "preset"),
    );
  }, [enabled, runs, runsQuery.isSuccess, selectedPresets]);

  useEffect(() => {
    if (!enabled || !runsQuery.isSuccess || selectedRunIds !== null) {
      return;
    }
    setSelectedRunIds(buildInitialRunIdSelection(runs));
  }, [enabled, runs, runsQuery.isSuccess, selectedRunIds]);

  const datasetSet = useMemo(
    () => selectionSetOrEmpty(selectedDatasets),
    [selectedDatasets],
  );
  const modelSet = useMemo(
    () => selectionSetOrEmpty(selectedModels),
    [selectedModels],
  );
  const presetSet = useMemo(
    () => selectionSetOrEmpty(selectedPresets),
    [selectedPresets],
  );
  const runIdSet = useMemo(
    () => selectionSetOrEmpty(selectedRunIds),
    [selectedRunIds],
  );
  const selectedTagsSet = useMemo(
    () => selectionSetOrEmpty(selectedTags),
    [selectedTags],
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

  useEffect(() => {
    if (!enabled || !tagsQuery.isSuccess || selectedTags !== null) {
      return;
    }
    const availableTags = new Set(tagOptions.map((option) => option.value));
    setSelectedTags(
      new Set(
        Array.from(availableTags).filter((tag) => isDefaultScalarTag(tag)),
      ),
    );
  }, [enabled, selectedTags, tagOptions, tagsQuery.isSuccess]);

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
      void refreshAfterMutation();
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
      void refreshAfterMutation();
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
    toggleExperiment: (value: string) => toggleSetValue(setSelectedExperiments, value),
    toggleDataset: (value: string) => toggleSetValue(setSelectedDatasets, value),
    toggleModel: (value: string) => toggleSetValue(setSelectedModels, value),
    togglePreset: (value: string) => toggleSetValue(setSelectedPresets, value),
    toggleRun: (value: string) => toggleSetValue(setSelectedRunIds, value),
    toggleTag: (value: string) => toggleSetValue(setSelectedTags, value),
    selectAllExperiments: () =>
      setAllValues(setSelectedExperiments, experimentOptions.map((option) => option.value)),
    selectNoExperiments: () => setNoValues(setSelectedExperiments),
    selectAllDatasets: () =>
      setAllValues(setSelectedDatasets, datasetOptions.map((option) => option.value)),
    selectNoDatasets: () => setNoValues(setSelectedDatasets),
    selectAllModels: () =>
      setAllValues(setSelectedModels, modelOptions.map((option) => option.value)),
    selectNoModels: () => setNoValues(setSelectedModels),
    selectAllPresets: () =>
      setAllValues(setSelectedPresets, presetOptions.map((option) => option.value)),
    selectNoPresets: () => setNoValues(setSelectedPresets),
    selectAllRuns: () =>
      setAllValues(setSelectedRunIds, runOptions.map((option) => option.value)),
    selectNoRuns: () => setNoValues(setSelectedRunIds),
    selectAllTags: () =>
      setAllValues(setSelectedTags, tagOptions.map((option) => option.value)),
    selectNoTags: () => setNoValues(setSelectedTags),
  };
}

export type LogsWorkspaceState = ReturnType<typeof useLogsWorkspaceState>;
