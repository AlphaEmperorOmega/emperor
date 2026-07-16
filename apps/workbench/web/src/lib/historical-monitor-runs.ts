import type { LogRun, LogRunTags } from "@/lib/api/logs";
import {
  tagMatchesNodePath,
  tagsIncludeParameterMonitorData,
  tagsMatchParameterNodePath,
} from "@/lib/monitor-paths";

export const HISTORICAL_MONITOR_RUN_LIMIT = 5;

export type MonitorEligibility = "checking" | "eligible" | "ineligible";

export type HistoricalRunOption = {
  value: string;
  label: string;
  count: number;
  monitorEligibility: MonitorEligibility;
  description?: string;
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

function appendGroupedRun(
  groups: Map<string, LogRun[]>,
  key: string,
  run: LogRun,
) {
  const group = groups.get(key);
  if (group) {
    group.push(run);
  } else {
    groups.set(key, [run]);
  }
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
    appendGroupedRun(groups, run.experiment, run);
  }
  return Array.from(groups, ([experiment, experimentRuns]) => ({
    experiment,
    runs: experimentRuns,
  }));
}

export function filterLogRunsByExperimentTask(
  runs: LogRun[],
  selectedExperimentTask: string,
) {
  if (!selectedExperimentTask) {
    return runs;
  }

  const taskAwareRuns = runs.filter((run) => Boolean(run.experimentTask));
  if (taskAwareRuns.length === 0) {
    return runs;
  }

  return taskAwareRuns.filter(
    (run) => run.experimentTask === selectedExperimentTask,
  );
}

export function monitorEligibilityDescription(
  monitorEligibility: MonitorEligibility,
) {
  if (monitorEligibility === "eligible") {
    return "monitor data";
  }
  if (monitorEligibility === "ineligible") {
    return "no monitor data";
  }
  return "monitor checking";
}

export function runMonitorEligibility(run: LogRun): MonitorEligibility {
  if (run.hasLayerMonitorData === true) {
    return "eligible";
  }
  if (run.hasLayerMonitorData === false) {
    return "ineligible";
  }
  return "checking";
}

export function aggregateMonitorEligibility(
  runs: LogRun[],
  getRunEligibility: (run: LogRun) => MonitorEligibility = runMonitorEligibility,
): MonitorEligibility {
  let hasCheckingRun = false;
  for (const run of runs) {
    const eligibility = getRunEligibility(run);
    if (eligibility === "eligible") {
      return "eligible";
    }
    if (eligibility === "checking") {
      hasCheckingRun = true;
    }
  }
  return hasCheckingRun ? "checking" : "ineligible";
}

function historicalRunOption(
  value: string,
  count: number,
  runs: LogRun[],
  getRunEligibility: (run: LogRun) => MonitorEligibility,
): HistoricalRunOption {
  const monitorEligibility = aggregateMonitorEligibility(runs, getRunEligibility);
  return {
    value,
    label: value,
    count,
    monitorEligibility,
    description: monitorEligibilityDescription(monitorEligibility),
  };
}

export function historicalExperimentRunOptions(
  runs: LogRun[],
  getRunEligibility: (run: LogRun) => MonitorEligibility = runMonitorEligibility,
): HistoricalRunOption[] {
  return groupModelLogRunsByExperiment(runs).map((group) => ({
    ...historicalRunOption(
      group.experiment,
      group.runs.length,
      group.runs,
      getRunEligibility,
    ),
  }));
}

export function historicalDatasetOptions(
  runs: LogRun[],
  selectedExperiment: string,
  getRunEligibility: (run: LogRun) => MonitorEligibility = runMonitorEligibility,
): HistoricalRunOption[] {
  const runsByDataset = new Map<string, LogRun[]>();
  const experimentRuns = sortLogRunsNewestFirst(
    runs.filter((run) => run.experiment === selectedExperiment),
  );

  for (const run of experimentRuns) {
    appendGroupedRun(runsByDataset, run.dataset, run);
  }

  return Array.from(runsByDataset, ([dataset, datasetRuns]) =>
    historicalRunOption(
      dataset,
      datasetRuns.length,
      datasetRuns,
      getRunEligibility,
    ),
  );
}

export function historicalPresetOptions(
  runs: LogRun[],
  getRunEligibility: (run: LogRun) => MonitorEligibility = runMonitorEligibility,
): HistoricalRunOption[] {
  const runsByPreset = new Map<string, LogRun[]>();
  for (const run of sortLogRunsNewestFirst(runs)) {
    appendGroupedRun(runsByPreset, run.preset, run);
  }
  return Array.from(runsByPreset, ([preset, presetRuns]) =>
    historicalRunOption(
      preset,
      presetRuns.length,
      presetRuns,
      getRunEligibility,
    ),
  );
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
    appendGroupedRun(groups, key, run);
  }

  return Array.from(groups, ([key, groupRuns]) => {
    const firstRun = groupRuns[0];
    return {
      key,
      experiment: firstRun.experiment,
      dataset: firstRun.dataset,
      preset: firstRun.preset,
      model: firstRun.model,
      runs: groupRuns.slice(0, limit),
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
  return [...tags.scalarTags, ...tags.histogramTags, ...tags.imageTags].some((tag) =>
    tagMatchesNodePath(tag, nodePath),
  );
}

export function logRunTagsMatchParameterNodePath(
  tags: Pick<LogRunTags, "scalarTags" | "histogramTags" | "imageTags"> | undefined,
  nodePath: string | undefined,
) {
  return tagsMatchParameterNodePath(tags, nodePath);
}

export function anyLogRunTagsMatchParameterNodePath(
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
      allowedRunIds.has(runTags.runId) &&
      logRunTagsMatchParameterNodePath(runTags, nodePath),
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
 * Whether a run carries parameter monitor data for W/b graph state. Performance
 * metrics and non-parameter monitor media do not qualify.
 */
export function logRunHasLayerMonitorData(
  tags: Pick<LogRunTags, "scalarTags" | "histogramTags" | "imageTags"> | undefined,
) {
  return tagsIncludeParameterMonitorData(tags);
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
