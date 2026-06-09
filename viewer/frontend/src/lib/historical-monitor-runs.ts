import { type LogRun, type LogRunTags } from "@/lib/api";

export const HISTORICAL_MONITOR_RUN_LIMIT = 5;

export type HistoricalRunOption = {
  value: string;
  label: string;
  count: number;
};

export type HistoricalExperimentRunGroup = {
  experiment: string;
  runs: LogRun[];
};

export type HistoricalMonitorRunGroup = {
  key: string;
  experiment: string;
  dataset: string;
  preset: string;
  model: string;
  runs: LogRun[];
  cardRunIds: string[];
};

function numericVersion(run: LogRun) {
  const match = /^version_(\d+)$/.exec(run.version);
  return match ? Number(match[1]) : -1;
}

export function sortLogRunsNewestFirst(runs: LogRun[]) {
  return [...runs].sort((left, right) => {
    const leftTimestamp = left.timestamp ?? "";
    const rightTimestamp = right.timestamp ?? "";
    return (
      rightTimestamp.localeCompare(leftTimestamp) ||
      (right.group ?? "").localeCompare(left.group ?? "") ||
      right.model.localeCompare(left.model) ||
      right.preset.localeCompare(left.preset) ||
      right.dataset.localeCompare(left.dataset) ||
      right.runName.localeCompare(left.runName) ||
      numericVersion(right) - numericVersion(left) ||
      right.version.localeCompare(left.version)
    );
  });
}

export function groupModelLogRunsByExperiment(
  runs: LogRun[],
): HistoricalExperimentRunGroup[] {
  const groups = new Map<string, LogRun[]>();
  for (const run of sortLogRunsNewestFirst(runs)) {
    groups.set(run.experiment, [...(groups.get(run.experiment) ?? []), run]);
  }
  return Array.from(groups, ([experiment, experimentRuns]) => ({
    experiment,
    runs: experimentRuns,
  }));
}

export function historicalExperimentOptions(runs: LogRun[]): HistoricalRunOption[] {
  return groupModelLogRunsByExperiment(runs).map((group) => ({
    value: group.experiment,
    label: group.experiment,
    count: group.runs.length,
  }));
}

export function historicalDatasetOptions(
  runs: LogRun[],
  selectedExperiment: string,
): HistoricalRunOption[] {
  const counts = new Map<string, number>();
  const experimentRuns = sortLogRunsNewestFirst(
    runs.filter((run) => run.experiment === selectedExperiment),
  );

  for (const run of experimentRuns) {
    counts.set(run.dataset, (counts.get(run.dataset) ?? 0) + 1);
  }

  return Array.from(counts, ([dataset, count]) => ({
    value: dataset,
    label: dataset,
    count,
  }));
}

export function historicalPresetOptions(runs: LogRun[]): HistoricalRunOption[] {
  const counts = new Map<string, number>();
  for (const run of sortLogRunsNewestFirst(runs)) {
    counts.set(run.preset, (counts.get(run.preset) ?? 0) + 1);
  }
  return Array.from(counts, ([preset, count]) => ({
    value: preset,
    label: preset,
    count,
  }));
}

export function resolveRunPresetName(
  run: LogRun,
  presets: Array<{ name: string; label: string }>,
) {
  const normalizedRunPreset = run.preset.toLowerCase();
  return (
    presets.find(
      (preset) =>
        preset.name === run.preset ||
        preset.label === run.preset ||
        preset.name.toLowerCase() === normalizedRunPreset ||
        preset.label.toLowerCase() === normalizedRunPreset,
    )?.name ?? ""
  );
}

export function filterHistoricalRuns(
  runs: LogRun[],
  selectedExperiment: string,
  selectedDataset: string,
  selectedPreset = "",
) {
  return sortLogRunsNewestFirst(
    runs.filter(
      (run) =>
        (!selectedExperiment || run.experiment === selectedExperiment) &&
        (!selectedDataset || run.dataset === selectedDataset) &&
        (!selectedPreset || run.preset === selectedPreset),
    ),
  );
}

export function latestHistoricalMonitorRuns(
  runs: LogRun[],
  limit = HISTORICAL_MONITOR_RUN_LIMIT,
) {
  return sortLogRunsNewestFirst(runs).slice(0, limit);
}

export function historicalMonitorRunGroupKey(
  run: Pick<LogRun, "experiment" | "dataset" | "preset">,
) {
  return `${run.experiment.length}:${run.experiment}|${run.dataset.length}:${run.dataset}|${run.preset.length}:${run.preset}`;
}

export function historicalMonitorRunGroups(
  runs: LogRun[],
  limit = HISTORICAL_MONITOR_RUN_LIMIT,
): HistoricalMonitorRunGroup[] {
  const groups = new Map<string, LogRun[]>();
  for (const run of sortLogRunsNewestFirst(runs)) {
    const key = historicalMonitorRunGroupKey(run);
    groups.set(key, [...(groups.get(key) ?? []), run]);
  }

  return Array.from(groups, ([key, groupRuns]) => {
    const firstRun = groupRuns[0];
    return {
      key,
      experiment: firstRun.experiment,
      dataset: firstRun.dataset,
      preset: firstRun.preset,
      model: firstRun.model,
      runs: latestHistoricalMonitorRuns(groupRuns, limit),
      cardRunIds: groupRuns.map((run) => run.id),
    };
  });
}

export function logRunTagsMatchNodePath(
  tags: Pick<LogRunTags, "scalarTags" | "histogramTags" | "imageTags"> | undefined,
  nodePath: string | undefined,
) {
  if (!tags || !nodePath) {
    return false;
  }
  const prefix = `${nodePath}/`;
  return [...tags.scalarTags, ...tags.histogramTags, ...tags.imageTags].some((tag) =>
    tag.startsWith(prefix),
  );
}

export function anyLogRunTagsMatchNodePath(
  tagsByRun: LogRunTags[] | undefined,
  runIds: string[],
  nodePath: string | undefined,
) {
  if (!tagsByRun || runIds.length === 0 || !nodePath) {
    return false;
  }
  const allowedRunIds = new Set(runIds);
  return tagsByRun.some(
    (runTags) =>
      allowedRunIds.has(runTags.runId) && logRunTagsMatchNodePath(runTags, nodePath),
  );
}

/**
 * Whether a run carries per-layer monitor data (as opposed to only flat
 * model-performance metrics such as `train/loss` or `epoch`). Shape-based and
 * monitor-type agnostic so it survives new monitor callbacks:
 * - any histogram or image tag implies layer monitoring (performance metrics
 *   never emit those), or
 * - a scalar tag with a node-path prefix, i.e. `node/group/stat` (>= 2 slashes).
 *   Flat performance tags have at most one slash (`train/loss`) or none (`epoch`).
 */
export function logRunHasLayerMonitorData(
  tags: Pick<LogRunTags, "scalarTags" | "histogramTags" | "imageTags"> | undefined,
) {
  if (!tags) {
    return false;
  }
  if (tags.histogramTags.length > 0 || tags.imageTags.length > 0) {
    return true;
  }
  return tags.scalarTags.some((tag) => tag.split("/").length >= 3);
}

/**
 * The set of experiments that contain at least one run with per-layer monitor
 * data. Returns an empty set when `tagsByRun` has not loaded yet so callers can
 * surface a loading state rather than briefly showing an unfiltered list.
 */
export function experimentsWithLayerMonitorData(
  runs: LogRun[],
  tagsByRun: LogRunTags[] | undefined,
) {
  const experiments = new Set<string>();
  if (!tagsByRun) {
    return experiments;
  }
  const experimentByRunId = new Map(runs.map((run) => [run.id, run.experiment]));
  for (const tags of tagsByRun) {
    if (!logRunHasLayerMonitorData(tags)) {
      continue;
    }
    const experiment = experimentByRunId.get(tags.runId);
    if (experiment !== undefined) {
      experiments.add(experiment);
    }
  }
  return experiments;
}
