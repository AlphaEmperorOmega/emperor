import { performance } from "node:perf_hooks";

import { type LogRun } from "@/lib/api";
import {
  historicalMonitorRunGroupKey,
  historicalMonitorRunGroups,
  latestHistoricalMonitorRuns,
  sortLogRunsNewestFirst,
  type HistoricalMonitorRunGroup,
} from "@/lib/historical-monitor-runs";

const RUN_COUNT = 4_000;
const GROUP_LIMIT = 5;
const WARMUP_COUNT = 5;
const SAMPLE_COUNT = 30;

const runs: LogRun[] = Array.from({ length: RUN_COUNT }, (_, index) => ({
  id: `run-${index}`,
  group: "exp_a",
  experiment: "exp_a",
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  dataset: "Mnist",
  runName: `run-${index}`,
  timestamp: String(index).padStart(8, "0"),
  version: `version_${index}`,
  relativePath: `exp_a/linears/linear/baseline/Mnist/run-${index}`,
  hasResult: true,
  eventFileCount: 1,
  checkpointCount: 0,
  hasHparams: true,
  metrics: {},
}));

function repeatedSpreadBaseline(
  inputRuns: LogRun[],
  limit: number,
): HistoricalMonitorRunGroup[] {
  const groups = new Map<string, LogRun[]>();
  for (const run of sortLogRunsNewestFirst(inputRuns)) {
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

function checksum(groups: HistoricalMonitorRunGroup[]) {
  return groups.reduce(
    (total, group) => total + group.runs.length + group.cardRunIds.length,
    0,
  );
}

function percentile(values: number[], ratio: number) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.min(ordered.length - 1, Math.floor(ordered.length * ratio))];
}

function measure(run: () => HistoricalMonitorRunGroup[]) {
  for (let index = 0; index < WARMUP_COUNT; index += 1) {
    run();
  }
  const samples = Array.from({ length: SAMPLE_COUNT }, () => {
    const startedAt = performance.now();
    const result = run();
    const duration = performance.now() - startedAt;
    if (checksum(result) !== RUN_COUNT + GROUP_LIMIT) {
      throw new Error("Historical grouping benchmark produced an invalid result.");
    }
    return duration;
  });
  return {
    medianMs: percentile(samples, 0.5),
    p95Ms: percentile(samples, 0.95),
  };
}

const baseline = measure(() => repeatedSpreadBaseline(runs, GROUP_LIMIT));
const current = measure(() => historicalMonitorRunGroups(runs, GROUP_LIMIT));

console.log(
  `Historical grouping benchmark (${RUN_COUNT} runs in one group, ${SAMPLE_COUNT} samples)`,
);
console.table([
  {
    implementation: "spread + redundant-sort baseline",
    "median ms": baseline.medianMs.toFixed(3),
    "p95 ms": baseline.p95Ms.toFixed(3),
  },
  {
    implementation: "current",
    "median ms": current.medianMs.toFixed(3),
    "p95 ms": current.p95Ms.toFixed(3),
  },
]);
