import { performance } from "node:perf_hooks";
import { gzipSync } from "node:zlib";
import { z } from "zod";
import { logScalarSeriesSchema } from "@/lib/api/logs";
import {
  buildLogScalarChunkQueryInputs,
  chunkScalarRunIdsForQueries,
  chunkScalarTagsForQueries,
} from "@/features/workbench/state/logs/_logs-scalar-query-plan";

const SYNTHETIC_POINTS_PER_SERIES = 20;
const SAMPLE_COUNT = 3;
const responseSchema = z.object({ series: z.array(logScalarSeriesSchema) });
const points = Array.from({ length: SYNTHETIC_POINTS_PER_SERIES }, (_, index) => ({
  step: index,
  wallTime: 1_720_000_000 + index,
  value: Math.sin(index / 10),
}));

type ScalarRequest = { runIds: string[]; tags: string[] };

function baselinePlan(runIds: string[], tags: string[]): ScalarRequest[] {
  const runChunks = chunkScalarRunIdsForQueries(runIds, 2);
  const tagChunks = chunkScalarTagsForQueries(tags, 6);
  return runChunks.flatMap((requestRunIds) =>
    tagChunks.map((requestTags) => ({
      runIds: requestRunIds,
      tags: requestTags,
    })),
  );
}

function currentPlan(runIds: string[], tags: string[]): ScalarRequest[] {
  return buildLogScalarChunkQueryInputs({
    enabled: true,
    group: "benchmark",
    requestedTags: new Set(tags),
    selectedTagList: tags,
    visibleRunIds: runIds,
  }).map(({ runIds: requestRunIds, tags: requestTags }) => ({
    runIds: requestRunIds,
    tags: requestTags,
  }));
}

function requestPayload({ runIds, tags }: ScalarRequest) {
  return {
    series: runIds.flatMap((runId) =>
      tags.map((tag) => ({
        runId,
        tag,
        points,
        sourcePointCount: SYNTHETIC_POINTS_PER_SERIES,
        truncated: false,
      })),
    ),
  };
}

function median(values: number[]) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.floor(ordered.length / 2)];
}

function measurePlan(requests: ScalarRequest[]) {
  const totals: number[] = [];
  const firstResponses: number[] = [];
  let transferredBytes = 0;

  for (let sample = 0; sample < SAMPLE_COUNT; sample += 1) {
    let totalDuration = 0;
    let firstResponseDuration = 0;
    let sampleBytes = 0;
    requests.forEach((request, requestIndex) => {
      const startedAt = performance.now();
      const body = JSON.stringify(requestPayload(request));
      responseSchema.parse(JSON.parse(body));
      const duration = performance.now() - startedAt;
      if (requestIndex === 0) {
        firstResponseDuration = duration;
      }
      totalDuration += duration;
      sampleBytes += Buffer.byteLength(body);
    });
    totals.push(totalDuration);
    firstResponses.push(firstResponseDuration);
    transferredBytes = sampleBytes;
  }

  return {
    requests: requests.length,
    bytes: transferredBytes,
    firstResponseMs: median(firstResponses),
    totalMs: median(totals),
  };
}

const rows = [5, 100, 500].flatMap((runCount) =>
  [6, 25, 100].map((tagCount) => {
    const runIds = Array.from({ length: runCount }, (_, index) => `run-${index}`);
    const tags = Array.from({ length: tagCount }, (_, index) => `tag/${index}`);
    const baseline = measurePlan(baselinePlan(runIds, tags));
    const current = measurePlan(currentPlan(runIds, tags));
    return {
      runs: runCount,
      tags: tagCount,
      series: runCount * tagCount,
      "requests before": baseline.requests,
      "requests after": current.requests,
      "request reduction": `${Math.round((1 - current.requests / baseline.requests) * 100)}%`,
      "bytes before": baseline.bytes,
      "bytes after": current.bytes,
      "first response before ms": baseline.firstResponseMs.toFixed(2),
      "first response after ms": current.firstResponseMs.toFixed(2),
      "total before ms": baseline.totalMs.toFixed(2),
      "total after ms": current.totalMs.toFixed(2),
    };
  }),
);

console.log(
  `Synthetic scalar transport benchmark (${SYNTHETIC_POINTS_PER_SERIES} points/series, encode + JSON parse + Zod validation, median of ${SAMPLE_COUNT})`,
);
console.table(rows);

const representativeBatch = Buffer.from(
  JSON.stringify({
    series: Array.from({ length: 10 }, (_, runIndex) =>
      Array.from({ length: 6 }, (_, tagIndex) => ({
        runId: `run-${runIndex}`,
        tag: `metric/${tagIndex}`,
        points: Array.from({ length: 500 }, (_, pointIndex) => ({
          step: pointIndex,
          wallTime: 1_720_000_000 + pointIndex,
          value: Math.sin((pointIndex + runIndex * 17 + tagIndex * 31) / 13),
        })),
        sourcePointCount: 500,
        truncated: false,
      })),
    ).flat(),
  }),
);
const compressionDurations: number[] = [];
let compressedBytes = 0;
for (let sample = 0; sample < 7; sample += 1) {
  const startedAt = performance.now();
  compressedBytes = gzipSync(representativeBatch, { level: 1 }).byteLength;
  compressionDurations.push(performance.now() - startedAt);
}
console.log(
  "Representative 10-run/6-tag/500-point gzip level 1:",
  {
    rawBytes: representativeBatch.byteLength,
    compressedBytes,
    ratio: (compressedBytes / representativeBatch.byteLength).toFixed(3),
    medianCompressionMs: median(compressionDurations).toFixed(2),
  },
);
