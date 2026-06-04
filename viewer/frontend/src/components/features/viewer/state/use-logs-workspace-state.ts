import { useCallback, useEffect, useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
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
} from "@/hooks/use-log-queries";
import {
  COMMON_SCALAR_TAGS,
  buildCountOptions,
  buildExperimentOptions,
  runOption,
  selectedOptionsSet,
  setAllValues,
  setNoValues,
  toggleSetValue,
} from "@/lib/logs/helpers";

function sortedValues(values: Set<string>) {
  return Array.from(values).sort((a, b) => a.localeCompare(b));
}

/**
 * Owns all state for the logs workspace: the run/experiment/tag queries, the
 * multi-facet selection sets (experiment/dataset/model/preset/run/tag), the
 * detail-run selection, and experiment deletion. Returned to the workspace
 * panels as a single object so they stay presentational.
 */
export function useLogsWorkspaceState({ enabled }: { enabled: boolean }) {
  const queryClient = useQueryClient();
  const [startedExperiments, setStartedExperiments] = useState<Set<string>>(new Set());
  const [selectedExperiments, setSelectedExperiments] = useState<Set<string> | null>(null);
  const [selectedDatasets, setSelectedDatasets] = useState<Set<string> | null>(null);
  const [selectedModels, setSelectedModels] = useState<Set<string> | null>(null);
  const [selectedPresets, setSelectedPresets] = useState<Set<string> | null>(null);
  const [selectedRunIds, setSelectedRunIds] = useState<Set<string> | null>(null);
  const [selectedTags, setSelectedTags] = useState<Set<string> | null>(null);
  const [selectedDetailRunId, setSelectedDetailRunId] = useState<string | null>(null);

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
      if (previous === null || previous.has(logFolder)) {
        return previous;
      }
      const next = new Set(previous);
      next.add(logFolder);
      return next;
    });
  }, []);

  useEffect(() => {
    if (!enabled || !experimentsQuery.isSuccess || selectedExperiments !== null) {
      return;
    }
    setSelectedExperiments(
      new Set([
        ...experimentOptions.map((option) => option.value),
        ...startedExperiments,
      ]),
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
    const startedRuns = runs.filter((run) => startedExperiments.has(run.experiment));
    if (startedRuns.length === 0) {
      return;
    }
    setSelectedExperiments((previous) => {
      if (previous === null) {
        return previous;
      }
      return new Set([
        ...previous,
        ...startedRuns.map((run) => run.experiment),
      ]);
    });
    setSelectedDatasets((previous) => {
      if (previous === null) {
        return previous;
      }
      return new Set([...previous, ...startedRuns.map((run) => run.dataset)]);
    });
    setSelectedModels((previous) => {
      if (previous === null) {
        return previous;
      }
      return new Set([...previous, ...startedRuns.map((run) => run.model)]);
    });
    setSelectedPresets((previous) => {
      if (previous === null) {
        return previous;
      }
      return new Set([...previous, ...startedRuns.map((run) => run.preset)]);
    });
    setSelectedRunIds((previous) => {
      if (previous === null) {
        return previous;
      }
      return new Set([...previous, ...startedRuns.map((run) => run.id)]);
    });
  }, [runs, startedExperiments]);

  const experimentSet = useMemo(
    () =>
      selectedExperiments ??
      new Set([
        ...experimentOptions.map((option) => option.value),
        ...startedExperiments,
      ]),
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
      new Set(buildCountOptions(runs, "dataset").map((option) => option.value)),
    );
  }, [enabled, runs, runsQuery.isSuccess, selectedDatasets]);

  useEffect(() => {
    if (!enabled || !runsQuery.isSuccess || selectedModels !== null) {
      return;
    }
    setSelectedModels(new Set(buildCountOptions(runs, "model").map((option) => option.value)));
  }, [enabled, runs, runsQuery.isSuccess, selectedModels]);

  useEffect(() => {
    if (!enabled || !runsQuery.isSuccess || selectedPresets !== null) {
      return;
    }
    setSelectedPresets(
      new Set(buildCountOptions(runs, "preset").map((option) => option.value)),
    );
  }, [enabled, runs, runsQuery.isSuccess, selectedPresets]);

  useEffect(() => {
    if (!enabled || !runsQuery.isSuccess || selectedRunIds !== null) {
      return;
    }
    setSelectedRunIds(new Set(runs.map((run) => run.id)));
  }, [enabled, runs, runsQuery.isSuccess, selectedRunIds]);

  const datasetSet = useMemo(
    () => selectedDatasets ?? new Set<string>(),
    [selectedDatasets],
  );
  const modelSet = useMemo(() => selectedModels ?? new Set<string>(), [selectedModels]);
  const presetSet = useMemo(
    () => selectedPresets ?? new Set<string>(),
    [selectedPresets],
  );
  const runIdSet = useMemo(() => selectedRunIds ?? new Set<string>(), [selectedRunIds]);
  const selectedTagsSet = useMemo(
    () => selectedTags ?? new Set<string>(),
    [selectedTags],
  );

  const visibleRuns = useMemo(
    () =>
      runs.filter(
        (run) =>
          experimentSet.has(run.experiment) &&
          datasetSet.has(run.dataset) &&
          modelSet.has(run.model) &&
          presetSet.has(run.preset) &&
          runIdSet.has(run.id),
      ),
    [datasetSet, experimentSet, modelSet, presetSet, runIdSet, runs],
  );

  const visibleRunIds = useMemo(() => visibleRuns.map((run) => run.id), [visibleRuns]);
  const tagsQuery = useLogTagsQuery({
    runIds: visibleRunIds,
    enabled,
    queryKey: ["log-tags", visibleRunIds],
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
      new Set(COMMON_SCALAR_TAGS.filter((tag) => availableTags.has(tag))),
    );
  }, [enabled, selectedTags, tagOptions, tagsQuery.isSuccess]);

  useEffect(() => {
    if (visibleRuns.length === 0) {
      setSelectedDetailRunId(null);
      return;
    }
    if (!selectedDetailRunId || !visibleRuns.some((run) => run.id === selectedDetailRunId)) {
      setSelectedDetailRunId(visibleRuns[0].id);
    }
  }, [selectedDetailRunId, visibleRuns]);

  const selectedTagList = useMemo(
    () =>
      Array.from(selectedOptionsSet(selectedTagsSet, tagOptions)).sort((a, b) =>
        a.localeCompare(b),
      ),
    [selectedTagsSet, tagOptions],
  );
  const selectedRun = visibleRuns.find((run) => run.id === selectedDetailRunId);
  const runDeleteFilters: LogRunDeleteFilters = useMemo(
    () => ({
      experiments: sortedValues(experimentSet),
      datasets: sortedValues(datasetSet),
      models: sortedValues(modelSet),
      presets: sortedValues(presetSet),
      runIds: sortedValues(runIdSet),
    }),
    [datasetSet, experimentSet, modelSet, presetSet, runIdSet],
  );
  const deleteExperimentMutation = useMutation({
    mutationFn: deleteLogExperiment,
    onSuccess: (result) => {
      const deletedRunIds = new Set(result.deletedRunIds);
      setSelectedExperiments((previous) => {
        const next = new Set(
          previous ?? experimentOptions.map((option) => option.value),
        );
        next.delete(result.experiment);
        return next;
      });
      setSelectedRunIds((previous) => {
        const next = new Set(previous ?? runs.map((run) => run.id));
        for (const runId of deletedRunIds) {
          next.delete(runId);
        }
        return next;
      });
      setSelectedDetailRunId((previous) =>
        previous && deletedRunIds.has(previous) ? null : previous,
      );
      setStartedExperiments((previous) => {
        if (!previous.has(result.experiment)) {
          return previous;
        }
        const next = new Set(previous);
        next.delete(result.experiment);
        return next;
      });
      void queryClient.invalidateQueries({ queryKey: ["log-experiments"] });
      void queryClient.invalidateQueries({ queryKey: ["log-runs"] });
      queryClient.removeQueries({ queryKey: ["log-tags"] });
      queryClient.removeQueries({ queryKey: ["log-scalars"] });
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
        const next = new Set(previous ?? runs.map((run) => run.id));
        for (const runId of deletedRunIds) {
          next.delete(runId);
        }
        return next;
      });
      setSelectedDetailRunId((previous) =>
        previous && deletedRunIds.has(previous) ? null : previous,
      );
      void queryClient.invalidateQueries({ queryKey: ["log-experiments"] });
      void queryClient.invalidateQueries({ queryKey: ["log-runs"] });
      queryClient.removeQueries({ queryKey: ["log-tags"] });
      queryClient.removeQueries({ queryKey: ["log-scalars"] });
    },
  });

  return {
    enabled,
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
    selectedRun,
    selectedDetailRunId,
    setSelectedDetailRunId,
    runDeleteFilters,
    runDeletePlan: runDeletePlanMutation.data,
    createRunDeletePlan: (filters: LogRunDeleteFilters = runDeleteFilters) =>
      runDeletePlanMutation.mutateAsync(filters),
    runDeletePlanError: runDeletePlanMutation.error,
    isPlanningRunDelete: runDeletePlanMutation.isPending,
    deleteRuns: (filters: LogRunDeleteFilters = runDeleteFilters) =>
      deleteRunsMutation.mutateAsync(filters),
    runDeleteError: deleteRunsMutation.error,
    isDeletingRunDelete: deleteRunsMutation.isPending,
    resetRunDelete: () => {
      runDeletePlanMutation.reset();
      deleteRunsMutation.reset();
    },
    deleteExperiment: deleteExperimentMutation.mutateAsync,
    deleteExperimentError: deleteExperimentMutation.error,
    isDeletingExperiment: deleteExperimentMutation.isPending,
    resetDeleteExperiment: deleteExperimentMutation.reset,
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
