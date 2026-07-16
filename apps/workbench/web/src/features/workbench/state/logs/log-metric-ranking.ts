import type { LogRun, LogScalarPoint, LogScalarSeries } from "@/lib/api/logs";

export type LogMetricDirection = "higher" | "lower";
export type LogMetricPointPolicy = "best" | "latest";

export type LogMetricRankingRow = {
  runId: string;
  run: LogRun;
  tag: string;
  score: number;
  step: number;
  wallTime: number;
  visibleIndex: number;
  sourceIndex: number;
  pointIndex: number;
};

export type LogMetricDatasetRankingRow = {
  dataset: string;
  runCount: number;
  visibleIndex: number;
  best: LogMetricRankingRow | null;
};

export const DEFAULT_LOG_METRIC_DIRECTION: LogMetricDirection = "higher";
export const DEFAULT_LOG_METRIC_POINT_POLICY: LogMetricPointPolicy = "best";

const LOWER_IS_BETTER_PATTERN = /(^|[/_.-])(loss|errors?|ece)(?=$|[/_.-])/i;
const HIGHER_IS_BETTER_PATTERN =
  /(^|[/_.-])(accuracy|auc|f1|precision|recall)(?=$|[/_.-])/i;

export function inferLogMetricDirection(tag: string): LogMetricDirection {
  if (LOWER_IS_BETTER_PATTERN.test(tag)) {
    return "lower";
  }
  if (HIGHER_IS_BETTER_PATTERN.test(tag)) {
    return "higher";
  }
  return DEFAULT_LOG_METRIC_DIRECTION;
}

function scoreDelta(
  left: number,
  right: number,
  direction: LogMetricDirection,
) {
  return direction === "lower" ? left - right : right - left;
}

function pointTieDelta(left: LogScalarPoint, right: LogScalarPoint) {
  return right.wallTime - left.wallTime || right.step - left.step;
}

export function scoreLogMetricSeries({
  direction,
  pointPolicy,
  series,
}: {
  direction: LogMetricDirection;
  pointPolicy: LogMetricPointPolicy;
  series: LogScalarSeries;
}): { point: LogScalarPoint; pointIndex: number } | null {
  if (series.points.length === 0) {
    return null;
  }

  if (pointPolicy === "latest") {
    return {
      point: series.points[series.points.length - 1],
      pointIndex: series.points.length - 1,
    };
  }

  let bestPoint = series.points[0];
  let bestPointIndex = 0;

  series.points.forEach((point, pointIndex) => {
    const delta = scoreDelta(point.value, bestPoint.value, direction);
    if (
      delta < 0 ||
      (delta === 0 && pointTieDelta(point, bestPoint) < 0)
    ) {
      bestPoint = point;
      bestPointIndex = pointIndex;
    }
  });

  return { point: bestPoint, pointIndex: bestPointIndex };
}

function timestampDelta(left: LogMetricRankingRow, right: LogMetricRankingRow) {
  return right.wallTime - left.wallTime;
}

export function compareLogMetricRankingRows(
  direction: LogMetricDirection,
  left: LogMetricRankingRow,
  right: LogMetricRankingRow,
) {
  const valueDelta = scoreDelta(left.score, right.score, direction);
  if (valueDelta !== 0) {
    return valueDelta;
  }
  return (
    timestampDelta(left, right) ||
    left.visibleIndex - right.visibleIndex ||
    left.sourceIndex - right.sourceIndex
  );
}

export function buildLogMetricRankingRows({
  direction,
  pointPolicy,
  runOrder,
  runs,
  series,
  tag,
}: {
  direction: LogMetricDirection;
  pointPolicy: LogMetricPointPolicy;
  runOrder: string[];
  runs: LogRun[];
  series: LogScalarSeries[];
  tag: string | null;
}): LogMetricRankingRow[] {
  if (!tag) {
    return [];
  }

  const runsById = new Map(runs.map((run) => [run.id, run]));
  const runPositions = new Map(runOrder.map((runId, index) => [runId, index]));

  return series
    .flatMap<LogMetricRankingRow>((entry, sourceIndex) => {
      if (entry.tag !== tag) {
        return [];
      }

      const run = runsById.get(entry.runId);
      if (!run) {
        return [];
      }

      const scored = scoreLogMetricSeries({ direction, pointPolicy, series: entry });
      if (!scored) {
        return [];
      }

      return [
        {
          runId: entry.runId,
          run,
          tag,
          score: scored.point.value,
          step: scored.point.step,
          wallTime: scored.point.wallTime,
          visibleIndex: runPositions.get(entry.runId) ?? runOrder.length + sourceIndex,
          sourceIndex,
          pointIndex: scored.pointIndex,
        },
      ];
    })
    .sort((left, right) => compareLogMetricRankingRows(direction, left, right));
}

export function buildLogMetricDatasetRankingRows({
  direction,
  pointPolicy,
  runOrder,
  runs,
  series,
  tag,
}: {
  direction: LogMetricDirection;
  pointPolicy: LogMetricPointPolicy;
  runOrder: string[];
  runs: LogRun[];
  series: LogScalarSeries[];
  tag: string | null;
}): LogMetricDatasetRankingRow[] {
  const datasetsByName = new Map<string, LogMetricDatasetRankingRow>();

  runs.forEach((run, index) => {
    const datasetRow = datasetsByName.get(run.dataset);
    if (datasetRow) {
      datasetRow.runCount += 1;
      return;
    }
    datasetsByName.set(run.dataset, {
      dataset: run.dataset,
      runCount: 1,
      visibleIndex: index,
      best: null,
    });
  });

  if (!tag || datasetsByName.size === 0) {
    return Array.from(datasetsByName.values());
  }

  const candidates = buildLogMetricRankingRows({
    direction,
    pointPolicy,
    runOrder,
    runs,
    series,
    tag,
  });

  for (const candidate of candidates) {
    const datasetRow = datasetsByName.get(candidate.run.dataset);
    if (!datasetRow) {
      continue;
    }
    if (
      !datasetRow.best ||
      compareLogMetricRankingRows(direction, candidate, datasetRow.best) < 0
    ) {
      datasetRow.best = candidate;
    }
  }

  return Array.from(datasetsByName.values()).sort(
    (left, right) => left.visibleIndex - right.visibleIndex,
  );
}
