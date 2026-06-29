import { type Dispatch, type SetStateAction } from "react";
import { type LogExperiment, type LogRun, type LogScalarSeries } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import {
  modelIdentityKey,
  modelNameForId,
  modelTypeForId,
  toggleSetValue as toggleSelectionSetValue,
  uniqueValidValues,
} from "@/lib/selection";
import {
  isConfusionMatrixHeatmapTag,
  isConfusionMatrixScalarTag,
} from "@/features/viewer/state/logs/log-diagnostics";

export { formatNumber };

export const COMMON_SCALAR_TAGS = [
  "train/loss",
  "train/accuracy",
  "train/loss_epoch",
  "train/accuracy_epoch",
  "validation/loss",
  "validation/accuracy",
  "validation/loss_epoch",
  "validation/accuracy_epoch",
  "gap/accuracy",
  "gap/loss",
  "best_validation/accuracy",
  "best_validation/loss",
  "best_validation/epoch",
  "gradients/global_norm",
  "parameters/global_norm",
  "updates/update_to_weight_ratio",
  "gradients/nan_count",
  "gradients/inf_count",
  "train/confidence/mean",
  "train/calibration/ece",
  "validation/confidence/mean",
  "validation/calibration/ece",
  "test/loss",
  "test/accuracy",
];

export const DEFAULT_SCALAR_TAGS = [
  "validation/accuracy_epoch",
  "validation/loss_epoch",
  "train/loss_epoch",
  "train/accuracy_epoch",
];

export const LOG_PLOT_SELECTOR_SCALAR_TAGS = [
  "train/loss",
  "train/accuracy",
  "train/loss_epoch",
  "train/accuracy_epoch",
  "train/confidence/mean",
  "train/calibration/ece",
  "train/f1_score",
  "validation/loss",
  "validation/accuracy",
  "validation/loss_epoch",
  "validation/accuracy_epoch",
  "validation/confidence/mean",
  "validation/calibration/ece",
  "validation/f1_score",
];

export function isDefaultScalarTag(tag: string) {
  return DEFAULT_SCALAR_TAGS.includes(tag);
}

export function isLogPlotSelectorScalarTag(tag: string) {
  return LOG_PLOT_SELECTOR_SCALAR_TAGS.includes(tag);
}

export const LOG_METRIC_GROUPS = [
  { key: "train", label: "Train" },
  { key: "validation", label: "Validation" },
  { key: "test", label: "Test" },
  { key: "other", label: "Other" },
] as const;

export type LogMetricGroupKey = (typeof LOG_METRIC_GROUPS)[number]["key"];

export type ChecklistOption = {
  value: string;
  label: string;
  detail?: string;
  count?: number;
};

export type RenderableLogMetric = {
  tag: string;
  series: LogScalarSeries[];
};

export type LogMetricsByGroup = Record<LogMetricGroupKey, RenderableLogMetric[]>;
export type LogMetricTagsByGroup = Record<LogMetricGroupKey, string[]>;

type LogScalarTagRun = {
  scalarTags: string[];
};

export function buildCountOptions(
  runs: LogRun[],
  key: keyof Pick<LogRun, "experiment" | "dataset" | "model" | "preset">,
) {
  const counts = new Map<string, number>();
  for (const run of runs) {
    const value = run[key];
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return Array.from(counts, ([value, count]) => ({
    value,
    label: value,
    count,
  })).sort((a, b) => a.label.localeCompare(b.label));
}

export function logRunModelKey(run: Pick<LogRun, "modelType" | "model">) {
  return modelIdentityKey({ modelType: run.modelType, model: run.model });
}

export function buildModelCountOptions(runs: LogRun[]) {
  const counts = new Map<string, { count: number; modelType: string; model: string }>();
  for (const run of runs) {
    const value = logRunModelKey(run);
    const current = counts.get(value);
    counts.set(value, {
      count: (current?.count ?? 0) + 1,
      modelType: run.modelType,
      model: run.model,
    });
  }
  return Array.from(counts, ([value, item]) => ({
    value,
    label: `${modelNameForId(item)} · ${modelTypeForId(item)}`,
    count: item.count,
  })).sort((a, b) => a.label.localeCompare(b.label));
}

export function buildExperimentOptions(experiments: LogExperiment[]) {
  return experiments
    .map((experiment) => ({
      value: experiment.experiment,
      label: experiment.experiment,
      count: experiment.runCount,
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
}

function sortScalarTagOptions(options: ChecklistOption[]) {
  return options.sort((a, b) => {
    const commonA = COMMON_SCALAR_TAGS.indexOf(a.value);
    const commonB = COMMON_SCALAR_TAGS.indexOf(b.value);
    if (commonA >= 0 || commonB >= 0) {
      return (commonA >= 0 ? commonA : 999) - (commonB >= 0 ? commonB : 999);
    }
    return a.value.localeCompare(b.value);
  });
}

export function buildLogScalarTagOptions(tagRuns: LogScalarTagRun[] | undefined) {
  const scalarCounts = new Map<string, number>();
  const confusionMatrixRateCounts = new Map<string, number>();

  for (const runTags of tagRuns ?? []) {
    for (const tag of runTags.scalarTags) {
      if (isConfusionMatrixScalarTag(tag)) {
        if (isConfusionMatrixHeatmapTag(tag)) {
          confusionMatrixRateCounts.set(
            tag,
            (confusionMatrixRateCounts.get(tag) ?? 0) + 1,
          );
        }
        continue;
      }
      scalarCounts.set(tag, (scalarCounts.get(tag) ?? 0) + 1);
    }
  }

  const tagOptions = sortScalarTagOptions(
    Array.from(scalarCounts, ([value, count]) => ({
      value,
      label: value,
      count,
    })),
  );
  const confusionMatrixRateTags = Array.from(confusionMatrixRateCounts.keys()).sort(
    (a, b) => a.localeCompare(b),
  );

  return {
    tagOptions,
    confusionMatrixRateTags,
  };
}

export function formatRunLabel(run: LogRun) {
  return [
    run.experiment,
    run.dataset,
    `${run.model} · ${run.modelType}`,
    run.preset,
    run.timestamp ?? run.runName,
  ].join(" · ");
}

export function formatMetricValue(value: unknown) {
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (typeof value === "string" || typeof value === "boolean") {
    return String(value);
  }
  if (value === null || value === undefined) {
    return "None";
  }
  return JSON.stringify(value);
}

export function isTestMetricTag(tag: string) {
  return tag.startsWith("test/");
}

export function metricGroupForTag(tag: string): LogMetricGroupKey {
  if (tag.startsWith("train/")) {
    return "train";
  }
  if (tag.startsWith("validation/")) {
    return "validation";
  }
  if (isTestMetricTag(tag)) {
    return "test";
  }
  return "other";
}

export function groupRenderableLogMetrics({
  selectedTagList,
  seriesByTag,
}: {
  selectedTagList: string[];
  seriesByTag: Map<string, LogScalarSeries[]>;
}): LogMetricsByGroup {
  const groups: LogMetricsByGroup = {
    train: [],
    validation: [],
    test: [],
    other: [],
  };

  for (const tag of selectedTagList) {
    if (isConfusionMatrixScalarTag(tag)) {
      continue;
    }
    const series = seriesByTag.get(tag) ?? [];
    if (series.length === 0) {
      continue;
    }
    groups[metricGroupForTag(tag)].push({ tag, series });
  }

  return groups;
}

export function groupLogMetricTags(tags: string[]): LogMetricTagsByGroup {
  const groups: LogMetricTagsByGroup = {
    train: [],
    validation: [],
    test: [],
    other: [],
  };
  for (const tag of tags) {
    if (isConfusionMatrixScalarTag(tag)) {
      continue;
    }
    groups[metricGroupForTag(tag)].push(tag);
  }
  return groups;
}

export function groupLogPlotSelectorTags(tags: string[]): LogMetricTagsByGroup {
  return groupLogMetricTags(tags.filter((tag) => isLogPlotSelectorScalarTag(tag)));
}

export function toggleSetValue(
  setValues: Dispatch<SetStateAction<Set<string> | null>>,
  value: string,
) {
  setValues((previous) => {
    return toggleSelectionSetValue(previous ?? new Set<string>(), value);
  });
}

export function setAllValues(
  setValues: Dispatch<SetStateAction<Set<string> | null>>,
  values: string[],
) {
  setValues(new Set(values));
}

export function setNoValues(setValues: Dispatch<SetStateAction<Set<string> | null>>) {
  setValues(new Set());
}

export function selectedOptionsSet(selected: Set<string>, options: ChecklistOption[]) {
  return new Set(
    uniqueValidValues(
      Array.from(selected),
      options.map((option) => option.value),
    ),
  );
}
