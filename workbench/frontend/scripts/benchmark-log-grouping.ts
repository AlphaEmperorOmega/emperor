import { performance } from "node:perf_hooks";
import { groupLogScalarSeriesByTag } from "@/features/workbench/state/logs/_logs-chart-state";
import { type LogScalarSeries } from "@/lib/api";

const ENTRY_COUNT = 2_000;
const WARMUP_COUNT = 20;
const SAMPLE_COUNT = 100;

const seriesList: LogScalarSeries[] = Array.from(
  { length: ENTRY_COUNT },
  (_, index) => ({
    runId: `run-${index}`,
    tag: "train/loss",
    points: [{ step: index, wallTime: index, value: index }],
    sourcePointCount: 1,
    truncated: false,
  }),
);

function repeatedSpreadBaseline(entries: LogScalarSeries[]) {
  const byTag = new Map<string, LogScalarSeries[]>();
  for (const series of entries) {
    if (series.points.length === 0) {
      continue;
    }
    byTag.set(series.tag, [...(byTag.get(series.tag) ?? []), series]);
  }
  return byTag;
}

function percentile(values: number[], ratio: number) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.min(ordered.length - 1, Math.floor(ordered.length * ratio))];
}

function measure(run: () => Map<string, LogScalarSeries[]>) {
  for (let index = 0; index < WARMUP_COUNT; index += 1) {
    run();
  }
  const samples = Array.from({ length: SAMPLE_COUNT }, () => {
    const startedAt = performance.now();
    const result = run();
    const duration = performance.now() - startedAt;
    if (result.get("train/loss")?.length !== ENTRY_COUNT) {
      throw new Error("Grouping benchmark produced an incomplete result.");
    }
    return duration;
  });
  return {
    medianMs: percentile(samples, 0.5),
    p95Ms: percentile(samples, 0.95),
  };
}

const baseline = measure(() => repeatedSpreadBaseline(seriesList));
const current = measure(() => groupLogScalarSeriesByTag(seriesList));

console.log(
  `Log scalar grouping benchmark (${ENTRY_COUNT} same-tag entries, ${WARMUP_COUNT} warmups, ${SAMPLE_COUNT} samples)`,
);
console.table([
  {
    implementation: "repeated-spread baseline",
    "median ms": baseline.medianMs.toFixed(3),
    "p95 ms": baseline.p95Ms.toFixed(3),
  },
  {
    implementation: "current",
    "median ms": current.medianMs.toFixed(3),
    "p95 ms": current.p95Ms.toFixed(3),
  },
]);
