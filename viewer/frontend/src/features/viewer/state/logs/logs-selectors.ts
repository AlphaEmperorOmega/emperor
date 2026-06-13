import { type Dispatch, type SetStateAction } from "react";
import { type LogExperiment, type LogRun, type LogScalarSeries } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import {
  toggleSetValue as toggleSelectionSetValue,
  uniqueValidValues,
} from "@/lib/selection";

export { formatNumber };

export const COMMON_SCALAR_TAGS = [
  "train/loss",
  "train/accuracy",
  "validation/loss",
  "validation/accuracy",
  "test/loss",
  "test/accuracy",
];

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

export function buildExperimentOptions(experiments: LogExperiment[]) {
  return experiments
    .map((experiment) => ({
      value: experiment.experiment,
      label: experiment.experiment,
      count: experiment.runCount,
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
}

export function runOption(run: LogRun): ChecklistOption {
  return {
    value: run.id,
    label: run.runName,
    detail: [
      run.experiment,
      run.dataset,
      run.model,
      run.preset,
      run.timestamp ?? run.version,
    ].join(" · "),
  };
}

export function formatRunLabel(run: LogRun) {
  return [
    run.experiment,
    run.dataset,
    run.model,
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
    const series = seriesByTag.get(tag) ?? [];
    if (series.length === 0) {
      continue;
    }
    groups[metricGroupForTag(tag)].push({ tag, series });
  }

  return groups;
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
