import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTargetConfig } from "@/features/viewer/providers/viewer-providers";
import {
  useLogRunsQuery,
  useLogScalarsQuery,
  useLogTagsQuery,
} from "@/features/viewer/state/logs/use-log-queries";
import {
  buildLogScalarTagOptions,
  isDefaultScalarTag,
  type ChecklistOption,
} from "@/features/viewer/state/logs/logs-selectors";
import {
  defaultCompareMetricTags,
  MAX_COMPARE_RUNS,
} from "@/features/viewer/components/compare-workspace/compare-run-derive";
import { sortLogRunsNewestFirst } from "@/lib/historical-monitor-runs";
import { type LogRun, type LogRunTags } from "@/lib/api";
import { logQueryKeys } from "@/lib/query-keys";

export type CompareRunsMode = "runs" | "configs";
export type CompareRunsView = "graphs" | "data";

export type CompareRunTargetEntry = {
  id: string;
  experiment: string;
  modelType: string;
  model: string;
  preset: string;
  dataset: string;
};

export type CompareRunTargetOption = ChecklistOption & {
  disabled?: boolean;
};

export type CompareRunTargetEntryData = {
  entry: CompareRunTargetEntry;
  label: string;
  fullLabel: string;
  experimentOptions: CompareRunTargetOption[];
  modelTypeOptions: CompareRunTargetOption[];
  modelOptions: CompareRunTargetOption[];
  presetOptions: CompareRunTargetOption[];
  datasetOptions: CompareRunTargetOption[];
  resolvedRun: LogRun | undefined;
  matchingRunCount: number;
  matchingScalarRunCount: number;
  scalarTagCount: number;
  status: string;
};

const EMPTY_TAGS: LogRunTags[] = [];

function runHasScalarTags(tags: LogRunTags | undefined) {
  return (tags?.scalarTags.length ?? 0) > 0;
}

function runTargetKey(
  target: Pick<
    CompareRunTargetEntry,
    "experiment" | "modelType" | "model" | "preset" | "dataset"
  >,
) {
  return [
    target.experiment,
    target.modelType,
    target.model,
    target.preset,
    target.dataset,
  ].join("\u0001");
}

function createRunTargetEntry(id: string, run: LogRun): CompareRunTargetEntry {
  return {
    id,
    experiment: run.experiment,
    modelType: run.modelType,
    model: run.model,
    preset: run.preset,
    dataset: run.dataset,
  };
}

function targetMatchesRun(entry: CompareRunTargetEntry, run: LogRun) {
  return (
    run.experiment === entry.experiment &&
    run.modelType === entry.modelType &&
    run.model === entry.model &&
    run.preset === entry.preset &&
    run.dataset === entry.dataset
  );
}

function runMatchesCurrentTarget({
  run,
  modelType,
  model,
  preset,
  datasets,
}: {
  run: LogRun;
  modelType: string;
  model: string;
  preset: string;
  datasets: string[];
}) {
  return (
    modelType.length > 0 &&
    model.length > 0 &&
    preset.length > 0 &&
    datasets.length > 0 &&
    run.modelType === modelType &&
    run.model === model &&
    run.preset === preset &&
    datasets.includes(run.dataset)
  );
}

function countOptions<TValue extends string>({
  runs,
  getValue,
  getLabel = (value) => value,
  getDetail,
  disabledValues,
}: {
  runs: LogRun[];
  getValue: (run: LogRun) => TValue;
  getLabel?: (value: TValue, run: LogRun) => string;
  getDetail?: (value: TValue, count: number, run: LogRun) => string | undefined;
  disabledValues?: ReadonlySet<string>;
}): CompareRunTargetOption[] {
  const counts = new Map<TValue, { count: number; run: LogRun }>();
  for (const run of runs) {
    const value = getValue(run);
    const current = counts.get(value);
    counts.set(value, { count: (current?.count ?? 0) + 1, run: current?.run ?? run });
  }
  return Array.from(counts, ([value, item]) => ({
    value,
    label: getLabel(value, item.run),
    detail: getDetail?.(value, item.count, item.run),
    count: item.count,
    disabled: disabledValues?.has(value) ?? false,
  }));
}

function firstEnabledValue(options: CompareRunTargetOption[]) {
  return options.find((option) => !option.disabled)?.value ?? options[0]?.value ?? "";
}

function normalizeValue(value: string, options: CompareRunTargetOption[]) {
  if (
    value &&
    options.some((option) => option.value === value && !option.disabled)
  ) {
    return value;
  }
  return firstEnabledValue(options);
}

function optionCountDescription(_value: string, count: number) {
  return `${count} ${count === 1 ? "run" : "runs"}`;
}

function buildEntryOptions({
  entry,
  otherEntryKeys,
  runs,
}: {
  entry: CompareRunTargetEntry;
  otherEntryKeys: ReadonlySet<string>;
  runs: LogRun[];
}) {
  const experimentOptions = countOptions({
    runs,
    getValue: (run) => run.experiment,
    getDetail: optionCountDescription,
  });
  const experiment = normalizeValue(entry.experiment, experimentOptions);
  const experimentRuns = runs.filter((run) => run.experiment === experiment);

  const modelTypeOptions = countOptions({
    runs: experimentRuns,
    getValue: (run) => run.modelType,
    getDetail: optionCountDescription,
  });
  const modelType = normalizeValue(entry.modelType, modelTypeOptions);
  const modelTypeRuns = experimentRuns.filter(
    (run) => run.modelType === modelType,
  );

  const modelOptions = countOptions({
    runs: modelTypeRuns,
    getValue: (run) => run.model,
    getDetail: optionCountDescription,
  });
  const model = normalizeValue(entry.model, modelOptions);
  const modelRuns = modelTypeRuns.filter((run) => run.model === model);

  const presetOptions = countOptions({
    runs: modelRuns,
    getValue: (run) => run.preset,
    getDetail: optionCountDescription,
  });
  const preset = normalizeValue(entry.preset, presetOptions);
  const presetRuns = modelRuns.filter((run) => run.preset === preset);

  const disabledDatasets = new Set<string>();
  for (const run of presetRuns) {
    const key = runTargetKey({
      experiment,
      modelType,
      model,
      preset,
      dataset: run.dataset,
    });
    if (otherEntryKeys.has(key)) {
      disabledDatasets.add(run.dataset);
    }
  }
  const datasetOptions = countOptions({
    runs: presetRuns,
    getValue: (run) => run.dataset,
    getDetail: optionCountDescription,
    disabledValues: disabledDatasets,
  });
  const dataset = normalizeValue(entry.dataset, datasetOptions);

  return {
    entry: {
      ...entry,
      experiment,
      modelType,
      model,
      preset,
      dataset,
    },
    experimentOptions,
    modelTypeOptions,
    modelOptions,
    presetOptions,
    datasetOptions,
  };
}

function applyRunTargetCascadePatch(
  entry: CompareRunTargetEntry,
  patch: Partial<CompareRunTargetEntry>,
) {
  const next = { ...entry, ...patch };

  if (patch.experiment !== undefined && patch.experiment !== entry.experiment) {
    if (patch.modelType === undefined) {
      next.modelType = "";
    }
    if (patch.model === undefined) {
      next.model = "";
    }
    if (patch.preset === undefined) {
      next.preset = "";
    }
    if (patch.dataset === undefined) {
      next.dataset = "";
    }
    return next;
  }

  if (patch.modelType !== undefined && patch.modelType !== entry.modelType) {
    if (patch.model === undefined) {
      next.model = "";
    }
    if (patch.preset === undefined) {
      next.preset = "";
    }
    if (patch.dataset === undefined) {
      next.dataset = "";
    }
    return next;
  }

  if (patch.model !== undefined && patch.model !== entry.model) {
    if (patch.preset === undefined) {
      next.preset = "";
    }
    if (patch.dataset === undefined) {
      next.dataset = "";
    }
    return next;
  }

  if (patch.preset !== undefined && patch.preset !== entry.preset) {
    if (patch.dataset === undefined) {
      next.dataset = "";
    }
  }

  return next;
}

function runTargetEntriesEqual(
  left: CompareRunTargetEntry,
  right: CompareRunTargetEntry,
) {
  return (
    left.id === right.id &&
    left.experiment === right.experiment &&
    left.modelType === right.modelType &&
    left.model === right.model &&
    left.preset === right.preset &&
    left.dataset === right.dataset
  );
}

function buildTargetLabel(entry: CompareRunTargetEntry, index: number) {
  return `Target ${index + 1} · ${entry.experiment}`;
}

function buildFullTargetLabel(entry: CompareRunTargetEntry, index: number) {
  return [
    `Target ${index + 1}`,
    entry.experiment,
    `${entry.model} · ${entry.modelType}`,
    entry.preset,
    entry.dataset,
  ].join(" · ");
}

function latestMatchingRun(runs: LogRun[], entry: CompareRunTargetEntry) {
  return runs.find((run) => targetMatchesRun(entry, run));
}

function selectedTagsPayload(
  tags: LogRunTags[] | undefined,
  selectedRunIdSet: ReadonlySet<string>,
) {
  return (tags ?? EMPTY_TAGS).filter((tag) => selectedRunIdSet.has(tag.runId));
}

export function useExperimentCompareWorkspaceState() {
  const {
    selectedModelType,
    selectedModel,
    selectedPreset,
    selectedPresetMeta,
    selectedDatasets,
  } = useTargetConfig();
  const [mode, setMode] = useState<CompareRunsMode>("runs");
  const [view, setView] = useState<CompareRunsView>("graphs");
  const [entries, setEntries] = useState<CompareRunTargetEntry[]>([]);
  const [selectedMetricTags, setSelectedMetricTags] = useState<string[]>([]);
  const nextId = useRef(1);
  const userEditedMetricsRef = useRef(false);

  const allocateId = useCallback(() => {
    const id = `run-target-${nextId.current}`;
    nextId.current += 1;
    return id;
  }, []);

  const targetPreset = selectedPresetMeta?.label ?? selectedPreset;

  const runsQuery = useLogRunsQuery({
    enabled: mode === "runs",
    filters: { hasEventFiles: true },
    includeAllPages: true,
  });
  const runs = useMemo(
    () => sortLogRunsNewestFirst(runsQuery.data?.runs ?? []),
    [runsQuery.data?.runs],
  );
  const runIds = useMemo(() => runs.map((run) => run.id), [runs]);
  const tagsQuery = useLogTagsQuery({
    enabled: mode === "runs" && runIds.length > 0,
    runIds,
    queryKey: logQueryKeys.tagsForRuns(runIds),
  });
  const tagsByRunId = useMemo(
    () => new Map((tagsQuery.data?.runs ?? []).map((tags) => [tags.runId, tags])),
    [tagsQuery.data?.runs],
  );
  const eligibleRuns = useMemo(
    () => runs.filter((run) => runHasScalarTags(tagsByRunId.get(run.id))),
    [runs, tagsByRunId],
  );
  const scalarTagCountByRunId = useMemo(
    () =>
      new Map(
        runs.map((run) => [
          run.id,
          tagsByRunId.get(run.id)?.scalarTags.length ?? 0,
        ]),
      ),
    [runs, tagsByRunId],
  );
  const seededRun = useMemo(() => {
    const currentTarget = {
      modelType: selectedModelType,
      model: selectedModel,
      preset: targetPreset,
      datasets: selectedDatasets,
    };
    return (
      eligibleRuns.find((run) =>
        runMatchesCurrentTarget({ run, ...currentTarget }),
      ) ??
      eligibleRuns[0] ??
      runs.find((run) => runMatchesCurrentTarget({ run, ...currentTarget })) ??
      runs[0]
    );
  }, [
    eligibleRuns,
    runs,
    selectedDatasets,
    selectedModel,
    selectedModelType,
    targetPreset,
  ]);

  useEffect(() => {
    if (
      entries.length > 0 ||
      runsQuery.isLoading ||
      tagsQuery.isLoading ||
      !seededRun
    ) {
      return;
    }
    setEntries([createRunTargetEntry(allocateId(), seededRun)]);
  }, [
    allocateId,
    entries.length,
    runsQuery.isLoading,
    seededRun,
    tagsQuery.isLoading,
  ]);

  useEffect(() => {
    if (entries.length === 0 || runs.length === 0) {
      return;
    }
    setEntries((current) => {
      let changed = false;
      const nextEntries = current.map((entry) => {
        const otherEntryKeys = new Set(
          current
            .filter((candidate) => candidate.id !== entry.id)
            .map(runTargetKey),
        );
        const normalized = buildEntryOptions({
          entry,
          otherEntryKeys,
          runs,
        }).entry;
        if (!runTargetEntriesEqual(entry, normalized)) {
          changed = true;
        }
        return normalized;
      });
      return changed ? nextEntries : current;
    });
  }, [entries.length, runs]);

  const usedEntryKeys = useMemo(
    () => new Set(entries.map(runTargetKey)),
    [entries],
  );
  const nextUnusedRun = useMemo(
    () =>
      eligibleRuns.find((run) => !usedEntryKeys.has(runTargetKey(run))),
    [eligibleRuns, usedEntryKeys],
  );
  const canAddEntry =
    entries.length < MAX_COMPARE_RUNS && nextUnusedRun !== undefined;

  const entryData = useMemo<CompareRunTargetEntryData[]>(
    () =>
      entries.map((entry, index) => {
        const otherEntryKeys = new Set(
          entries
            .filter((candidate) => candidate.id !== entry.id)
            .map(runTargetKey),
        );
        const options = buildEntryOptions({ entry, otherEntryKeys, runs });
        const normalizedEntry = options.entry;
        const matchingRuns = runs.filter((run) =>
          targetMatchesRun(normalizedEntry, run),
        );
        const matchingScalarRuns = matchingRuns.filter((run) =>
          runHasScalarTags(tagsByRunId.get(run.id)),
        );
        const resolvedRun = latestMatchingRun(eligibleRuns, normalizedEntry);
        const scalarTagCount = resolvedRun
          ? scalarTagCountByRunId.get(resolvedRun.id) ?? 0
          : 0;
        const status = resolvedRun
          ? `${scalarTagCount} scalar ${scalarTagCount === 1 ? "tag" : "tags"}`
          : matchingRuns.length > 0
            ? "No scalar-capable run matches these filters."
            : "No historical run matches these filters.";
        return {
          entry: normalizedEntry,
          label: buildTargetLabel(normalizedEntry, index),
          fullLabel: buildFullTargetLabel(normalizedEntry, index),
          experimentOptions: options.experimentOptions,
          modelTypeOptions: options.modelTypeOptions,
          modelOptions: options.modelOptions,
          presetOptions: options.presetOptions,
          datasetOptions: options.datasetOptions,
          resolvedRun,
          matchingRunCount: matchingRuns.length,
          matchingScalarRunCount: matchingScalarRuns.length,
          scalarTagCount,
          status,
        };
      }),
    [eligibleRuns, entries, runs, scalarTagCountByRunId, tagsByRunId],
  );

  const selectedRuns = useMemo(
    () =>
      entryData
        .map((entry) => entry.resolvedRun)
        .filter((run): run is LogRun => run !== undefined),
    [entryData],
  );
  const selectedRunIds = useMemo(
    () => selectedRuns.map((run) => run.id),
    [selectedRuns],
  );
  const selectedRunIdSet = useMemo(
    () => new Set(selectedRunIds),
    [selectedRunIds],
  );
  const selectedRunLabels = useMemo(
    () =>
      entryData
        .filter((entry) => entry.resolvedRun)
        .map((entry) => entry.label),
    [entryData],
  );
  const selectedRunFullLabels = useMemo(
    () =>
      entryData
        .filter((entry) => entry.resolvedRun)
        .map((entry) => entry.fullLabel),
    [entryData],
  );

  const metricOptions = useMemo(
    () =>
      buildLogScalarTagOptions(
        selectedTagsPayload(tagsQuery.data?.runs, selectedRunIdSet),
      ).tagOptions,
    [selectedRunIdSet, tagsQuery.data?.runs],
  );
  const metricOptionValues = useMemo(
    () => metricOptions.map((option) => option.value),
    [metricOptions],
  );
  const selectedScalarQuery = useLogScalarsQuery({
    enabled:
      mode === "runs" &&
      selectedRunIds.length > 0 &&
      selectedMetricTags.length > 0,
    runIds: selectedRunIds,
    tags: selectedMetricTags,
    queryKey: logQueryKeys.scalarsForRunsAndTags(selectedRunIds, selectedMetricTags, {
      group: "compare-runs",
    }),
    group: "compare-runs",
  });

  useEffect(() => {
    const available = new Set(metricOptionValues);
    setSelectedMetricTags((current) => {
      const next = current.filter((tag) => available.has(tag));
      return next.length === current.length ? current : next;
    });
  }, [metricOptionValues]);

  useEffect(() => {
    if (
      userEditedMetricsRef.current ||
      selectedMetricTags.length > 0 ||
      metricOptions.length === 0
    ) {
      return;
    }
    const fallbackTags = metricOptions
      .map((option) => option.value)
      .filter(isDefaultScalarTag);
    setSelectedMetricTags(defaultCompareMetricTags(metricOptions, fallbackTags));
  }, [metricOptions, selectedMetricTags.length]);

  const addEntry = useCallback(() => {
    if (!nextUnusedRun) {
      return;
    }
    setEntries((current) =>
      current.length >= MAX_COMPARE_RUNS ||
      current.some((entry) => runTargetKey(entry) === runTargetKey(nextUnusedRun))
        ? current
        : [...current, createRunTargetEntry(allocateId(), nextUnusedRun)],
    );
  }, [allocateId, nextUnusedRun]);

  const removeEntry = useCallback((id: string) => {
    setEntries((current) =>
      current.length <= 1 ? current : current.filter((entry) => entry.id !== id),
    );
  }, []);

  const resetEntries = useCallback(() => {
    userEditedMetricsRef.current = false;
    setSelectedMetricTags([]);
    setEntries([]);
  }, []);

  const updateEntry = useCallback(
    (id: string, patch: Partial<CompareRunTargetEntry>) => {
      setEntries((current) =>
        current.map((entry) => {
          if (entry.id !== id) {
            return entry;
          }
          const otherEntryKeys = new Set(
            current
              .filter((candidate) => candidate.id !== id)
              .map(runTargetKey),
          );
          return buildEntryOptions({
            entry: applyRunTargetCascadePatch(entry, patch),
            otherEntryKeys,
            runs,
          }).entry;
        }),
      );
    },
    [runs],
  );

  const setMetricTags = useCallback(
    (nextTags: string[]) => {
      userEditedMetricsRef.current = true;
      const available = new Set(metricOptionValues);
      setSelectedMetricTags(
        Array.from(new Set(nextTags)).filter((tag) => available.has(tag)),
      );
    },
    [metricOptionValues],
  );

  const selectDefaultMetrics = useCallback(() => {
    userEditedMetricsRef.current = false;
    const fallbackTags = metricOptions
      .map((option) => option.value)
      .filter(isDefaultScalarTag);
    setSelectedMetricTags(defaultCompareMetricTags(metricOptions, fallbackTags));
  }, [metricOptions]);

  const selectAllMetrics = useCallback(() => {
    userEditedMetricsRef.current = true;
    setSelectedMetricTags(metricOptionValues);
  }, [metricOptionValues]);

  const selectNoMetrics = useCallback(() => {
    userEditedMetricsRef.current = true;
    setSelectedMetricTags([]);
  }, []);

  const tagOptions = metricOptions.map<ChecklistOption>((option) => option);
  const scalarSeries = useMemo(
    () => selectedScalarQuery.data?.series ?? [],
    [selectedScalarQuery.data?.series],
  );
  const seriesByTag = useMemo(() => {
    const byTag = new Map<string, typeof scalarSeries>();
    for (const tag of selectedMetricTags) {
      byTag.set(
        tag,
        scalarSeries.filter((series) => series.tag === tag),
      );
    }
    return byTag;
  }, [scalarSeries, selectedMetricTags]);

  return {
    mode,
    setMode,
    view,
    setView,
    runsQuery,
    tagsQuery,
    scalarsQuery: selectedScalarQuery,
    runs,
    selectedRuns,
    selectedRunIds,
    selectedRunIdSet,
    selectedRunLabels,
    selectedRunFullLabels,
    eligibleRuns,
    entries,
    entryData,
    readyEntryCount: selectedRuns.length,
    addEntry,
    removeEntry,
    resetEntries,
    updateEntry,
    canAddEntry,
    canResetEntries: runs.length > 0,
    canRemoveEntry: entries.length > 1,
    maxRunCount: MAX_COMPARE_RUNS,
    scalarTagCountByRunId,
    metricOptions: tagOptions,
    selectedMetricTags,
    setMetricTags,
    selectDefaultMetrics,
    selectAllMetrics,
    selectNoMetrics,
    scalarSeries,
    seriesByTag,
    hasTruncatedSeries: scalarSeries.some((series) => Boolean(series.truncated)),
    hasMoreRuns: Boolean(runsQuery.data?.hasMore),
  };
}

export type ExperimentCompareWorkspaceState = ReturnType<
  typeof useExperimentCompareWorkspaceState
>;
