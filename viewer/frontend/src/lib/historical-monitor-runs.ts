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

export function filterHistoricalRuns(
  runs: LogRun[],
  selectedExperiment: string,
  selectedDataset: string,
) {
  return sortLogRunsNewestFirst(
    runs.filter(
      (run) =>
        (!selectedExperiment || run.experiment === selectedExperiment) &&
        (!selectedDataset || run.dataset === selectedDataset),
    ),
  );
}

export function latestHistoricalMonitorRuns(
  runs: LogRun[],
  limit = HISTORICAL_MONITOR_RUN_LIMIT,
) {
  return sortLogRunsNewestFirst(runs).slice(0, limit);
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
