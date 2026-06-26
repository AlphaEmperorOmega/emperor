import type { LogRun, LogScalarSeries } from "@/lib/api";
import {
  compareLogMetricRankingRows,
  inferLogMetricDirection,
  scoreLogMetricSeries,
  type LogMetricDirection,
  type LogMetricRankingRow,
} from "@/features/viewer/state/logs/log-metric-ranking";
import { formatNumber } from "@/features/viewer/state/logs/logs-selectors";
import { sortLogRunsNewestFirst } from "@/lib/historical-monitor-runs";

export const MAX_COMPARE_RUNS = 8;
export const QUICK_ADD_RUN_LIMIT = 5;
export const DEFAULT_COMPARE_METRIC_LIMIT = 4;

type MetricOptionLike = {
  value: string;
};

export type CompareMetricSummaryKind =
  | "first"
  | "last"
  | "best"
  | "best-step"
  | "delta"
  | "points";

export type CompareMetricSummaryCell = {
  text: string;
  highlighted?: boolean;
};

export type CompareMetricSummaryRow = {
  key: string;
  tag: string;
  summary: CompareMetricSummaryKind;
  label: string;
  direction: LogMetricDirection;
  values: CompareMetricSummaryCell[];
};

const emptyCell: CompareMetricSummaryCell = { text: "—" };

const metricPriorityPatterns: RegExp[][] = [
  [
    /(^|[/_.-])(validation|val)(?=$|[/_.-]).*(accuracy|acc|auc|f1|precision|recall)/i,
    /(accuracy|acc|auc|f1|precision|recall).*(^|[/_.-])(validation|val)(?=$|[/_.-])/i,
  ],
  [
    /(^|[/_.-])(train|training)(?=$|[/_.-]).*(accuracy|acc|auc|f1|precision|recall)/i,
    /(accuracy|acc|auc|f1|precision|recall).*(^|[/_.-])(train|training)(?=$|[/_.-])/i,
  ],
  [
    /(^|[/_.-])(validation|val)(?=$|[/_.-]).*(loss|errors?|ece)/i,
    /(loss|errors?|ece).*(^|[/_.-])(validation|val)(?=$|[/_.-])/i,
  ],
  [
    /(^|[/_.-])(train|training)(?=$|[/_.-]).*(loss|errors?|ece)/i,
    /(loss|errors?|ece).*(^|[/_.-])(train|training)(?=$|[/_.-])/i,
  ],
];

function appendUnique(target: string[], value: string) {
  if (!target.includes(value)) {
    target.push(value);
  }
}

export function defaultCompareMetricTags(
  options: MetricOptionLike[],
  fallbackTags: string[] = [],
) {
  const optionValues = options.map((option) => option.value);
  const selected: string[] = [];

  for (const patterns of metricPriorityPatterns) {
    const match = optionValues.find((tag) =>
      patterns.some((pattern) => pattern.test(tag)),
    );
    if (match) {
      appendUnique(selected, match);
    }
  }

  for (const tag of fallbackTags) {
    if (optionValues.includes(tag)) {
      appendUnique(selected, tag);
    }
    if (selected.length >= DEFAULT_COMPARE_METRIC_LIMIT) {
      return selected.slice(0, DEFAULT_COMPARE_METRIC_LIMIT);
    }
  }

  for (const tag of optionValues) {
    appendUnique(selected, tag);
    if (selected.length >= DEFAULT_COMPARE_METRIC_LIMIT) {
      break;
    }
  }

  return selected.slice(0, DEFAULT_COMPARE_METRIC_LIMIT);
}

export function compactRunLabel(run: LogRun) {
  return `${run.runName} · ${run.version}`;
}

export function fullRunLabel(run: LogRun) {
  return [
    run.runName,
    run.version,
    run.experiment,
    run.dataset,
    run.preset,
    `${run.model} · ${run.modelType}`,
  ].join(" · ");
}

export function runMetadataLabel(run: LogRun) {
  return [
    run.experiment,
    run.dataset,
    run.preset,
    `${run.model} · ${run.modelType}`,
  ].join(" · ");
}

export function latestRunsForExperiment(runs: LogRun[], experiment: string) {
  return sortLogRunsNewestFirst(
    runs.filter((run) => run.experiment === experiment),
  );
}

function seriesKey(runId: string, tag: string) {
  return `${runId}\u0000${tag}`;
}

function formatDelta(value: number) {
  if (!Number.isFinite(value)) {
    return "—";
  }
  return `${value > 0 ? "+" : ""}${formatNumber(value)}`;
}

function scalarSeriesByRunAndTag(series: LogScalarSeries[]) {
  return new Map(series.map((entry) => [seriesKey(entry.runId, entry.tag), entry]));
}

function bestRankingRow(
  runId: string,
  tag: string,
  series: LogScalarSeries,
  direction: LogMetricDirection,
  visibleIndex: number,
): LogMetricRankingRow | null {
  const scored = scoreLogMetricSeries({ direction, pointPolicy: "best", series });
  if (!scored) {
    return null;
  }
  return {
    runId,
    run: {
      id: runId,
      group: null,
      modelType: "",
      model: "",
      preset: "",
      dataset: "",
      runName: runId,
      timestamp: null,
      version: "",
      relativePath: "",
      hasResult: false,
      eventFileCount: 0,
      checkpointCount: 0,
      hasHparams: false,
      metrics: {},
      experiment: "",
    },
    tag,
    score: scored.point.value,
    step: scored.point.step,
    wallTime: scored.point.wallTime,
    visibleIndex,
    sourceIndex: visibleIndex,
    pointIndex: scored.pointIndex,
  };
}

function bestRunIdsForTag({
  direction,
  runIds,
  seriesByKey,
  tag,
}: {
  direction: LogMetricDirection;
  runIds: string[];
  seriesByKey: Map<string, LogScalarSeries>;
  tag: string;
}) {
  const rows = runIds
    .map((runId, index) => {
      const series = seriesByKey.get(seriesKey(runId, tag));
      return series ? bestRankingRow(runId, tag, series, direction, index) : null;
    })
    .filter((row): row is LogMetricRankingRow => row !== null)
    .sort((left, right) => compareLogMetricRankingRows(direction, left, right));

  if (rows.length === 0) {
    return new Set<string>();
  }

  const best = rows[0];
  return new Set(
    rows
      .filter(
        (row) =>
          row.score === best.score &&
          row.step === best.step &&
          row.wallTime === best.wallTime,
      )
      .map((row) => row.runId),
  );
}

function valueCell(value: number | undefined, highlighted = false) {
  if (value === undefined || !Number.isFinite(value)) {
    return emptyCell;
  }
  return { text: formatNumber(value), highlighted };
}

function pointCountCell(count: number | undefined) {
  return count === undefined ? emptyCell : { text: String(count) };
}

export function buildCompareMetricSummaryRows({
  runIds,
  selectedTags,
  series,
}: {
  runIds: string[];
  selectedTags: string[];
  series: LogScalarSeries[];
}): CompareMetricSummaryRow[] {
  const seriesByKey = scalarSeriesByRunAndTag(series);
  const rows: CompareMetricSummaryRow[] = [];

  for (const tag of selectedTags) {
    const direction = inferLogMetricDirection(tag);
    const bestRunIds = bestRunIdsForTag({
      direction,
      runIds,
      seriesByKey,
      tag,
    });
    const valuesFor = (summary: CompareMetricSummaryKind) =>
      runIds.map<CompareMetricSummaryCell>((runId) => {
        const entry = seriesByKey.get(seriesKey(runId, tag));
        if (!entry || entry.points.length === 0) {
          return emptyCell;
        }
        const first = entry.points[0];
        const last = entry.points[entry.points.length - 1];
        const best = scoreLogMetricSeries({
          direction,
          pointPolicy: "best",
          series: entry,
        });

        if (summary === "first") {
          return valueCell(first.value);
        }
        if (summary === "last") {
          return valueCell(last.value);
        }
        if (summary === "best") {
          return valueCell(best?.point.value, bestRunIds.has(runId));
        }
        if (summary === "best-step") {
          return best ? { text: String(best.point.step) } : emptyCell;
        }
        if (summary === "delta") {
          return { text: formatDelta(last.value - first.value) };
        }
        return pointCountCell(entry.points.length);
      });

    const summaryRows: Array<{ summary: CompareMetricSummaryKind; label: string }> = [
      { summary: "first", label: "First" },
      { summary: "last", label: "Last" },
      { summary: "best", label: "Best" },
      { summary: "best-step", label: "Best step" },
      { summary: "delta", label: "Delta" },
      { summary: "points", label: "Points" },
    ];

    for (const row of summaryRows) {
      rows.push({
        key: `${tag}:${row.summary}`,
        tag,
        summary: row.summary,
        label: row.label,
        direction,
        values: valuesFor(row.summary),
      });
    }
  }

  return rows;
}
