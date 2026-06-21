import type {
  EChartsOption,
  LineSeriesOption,
  TooltipComponentFormatterCallbackParams,
} from "echarts";
import type { ScalarDomain } from "@/types/monitor";
import { formatNumber } from "@/lib/format";
import { applyEmaSmoothing } from "@/lib/echarts/smoothing";

export type ScalarXMode = "step" | "wallTime";
export type ScalarYScale = "linear" | "log";

export type ScalarLinePoint = { step: number; wallTime: number; value: number };

/** One overlaid run line, already resolved to a stable color + display name. */
export type ScalarLine = {
  id: string;
  name: string;
  color: string;
  points: ScalarLinePoint[];
};

export type ScalarCheckpointMarker = {
  runId: string;
  runLabel: string;
  filename: string;
  epoch: number | null;
  step: number | null;
};

export type ScalarLineOptions = {
  xMode?: ScalarXMode;
  yScale?: ScalarYScale;
  /** EMA weight in [0, 1]; 0 disables smoothing. */
  smoothing?: number;
  /** Show inside + slider dataZoom (used by the logs panel, not the dense modal). */
  dataZoom?: boolean;
  /** Fix axis extents so comparison charts share a scale. */
  domain?: Partial<ScalarDomain>;
  checkpointMarkers?: readonly ScalarCheckpointMarker[];
};

export const MAX_SCALAR_OPTION_POINTS = 100_000;
export const SCALAR_OPTION_DOWNSAMPLE_THRESHOLD = 5_000;

function escapeTooltipHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => {
    switch (character) {
      case "&":
        return "&amp;";
      case "<":
        return "&lt;";
      case ">":
        return "&gt;";
      case '"':
        return "&quot;";
      case "'":
        return "&#39;";
      default:
        return character;
    }
  });
}

function toAxisX(point: ScalarLinePoint, xMode: ScalarXMode): number {
  // TensorBoard wall time is epoch seconds; the time axis expects milliseconds.
  return xMode === "wallTime" ? point.wallTime * 1000 : point.step;
}

type AxisTooltipItem = {
  seriesName?: string;
  marker?: string;
  axisValueLabel?: string;
  axisValue?: number | string;
  value?: unknown;
};

/** Axis tooltip that de-duplicates the raw/smoothed pair sharing a run name. */
function formatAxisTooltip(params: TooltipComponentFormatterCallbackParams): string {
  const items = (Array.isArray(params) ? params : [params]) as AxisTooltipItem[];
  if (items.length === 0) {
    return "";
  }
  const header = escapeTooltipHtml(
    items[0].axisValueLabel ?? String(items[0].axisValue ?? ""),
  );
  const seen = new Set<string>();
  const rows: string[] = [];
  for (const item of items) {
    const name = item.seriesName ?? "";
    if (seen.has(name)) {
      continue;
    }
    seen.add(name);
    const raw = Array.isArray(item.value) ? item.value[1] : item.value;
    rows.push(
      `${item.marker ?? ""}${escapeTooltipHtml(name)}&nbsp;&nbsp;${formatNumber(Number(raw))}`,
    );
  }
  return [header, ...rows].join("<br/>");
}

function checkpointTooltipLabel(marker: ScalarCheckpointMarker) {
  const details = [
    marker.epoch === null ? null : `epoch ${marker.epoch}`,
    marker.step === null ? null : `step ${marker.step}`,
  ].filter(Boolean);
  return `${marker.runLabel}: ${marker.filename}${
    details.length > 0 ? ` (${details.join(", ")})` : ""
  }`;
}

function checkpointMarkLineData(markers: readonly ScalarCheckpointMarker[]) {
  const byStep = new Map<number, ScalarCheckpointMarker[]>();
  for (const marker of markers) {
    if (typeof marker.step !== "number" || !Number.isFinite(marker.step)) {
      continue;
    }
    byStep.set(marker.step, [...(byStep.get(marker.step) ?? []), marker]);
  }

  return Array.from(byStep, ([step, stepMarkers]) => ({
    name: stepMarkers.map(checkpointTooltipLabel).join("\n"),
    xAxis: step,
  })).sort((left, right) => left.xAxis - right.xAxis);
}

function formatCheckpointTooltip(params: unknown): string {
  const name =
    typeof params === "object" && params !== null && "name" in params
      ? String((params as { name?: unknown }).name ?? "")
      : "";
  return name.split("\n").map(escapeTooltipHtml).join("<br/>");
}

function downsamplePoints(
  points: readonly ScalarLinePoint[],
  limit: number,
): readonly ScalarLinePoint[] {
  if (points.length <= limit) {
    return points;
  }
  if (limit <= 1) {
    return [points[points.length - 1]];
  }
  const sampled: ScalarLinePoint[] = [];
  const lastIndex = points.length - 1;
  for (let index = 0; index < limit; index += 1) {
    const sourceIndex = Math.round((index * lastIndex) / (limit - 1));
    sampled.push(points[sourceIndex]);
  }
  sampled[sampled.length - 1] = points[lastIndex];
  return sampled;
}

function linePointLimit(lineCount: number) {
  return Math.max(
    1,
    Math.min(
      SCALAR_OPTION_DOWNSAMPLE_THRESHOLD,
      Math.floor(MAX_SCALAR_OPTION_POINTS / Math.max(1, lineCount)),
    ),
  );
}

function dataForPoints(
  points: readonly ScalarLinePoint[],
  xMode: ScalarXMode,
): Array<[number, number]> {
  return points.map((point) => [toAxisX(point, xMode), point.value]);
}

function largeSeriesOptions(pointCount: number) {
  return pointCount >= SCALAR_OPTION_DOWNSAMPLE_THRESHOLD
    ? {
        progressive: SCALAR_OPTION_DOWNSAMPLE_THRESHOLD,
        progressiveThreshold: SCALAR_OPTION_DOWNSAMPLE_THRESHOLD,
        sampling: "lttb" as const,
      }
    : {};
}

/**
 * Builds a line-chart option for one or more overlaid run series. When smoothing
 * is active each multi-point line is drawn twice: a faint raw line behind a bold
 * smoothed line (TensorBoard behavior). Single-point series render as a marker.
 */
export function buildScalarLineOption(
  lines: readonly ScalarLine[],
  options: ScalarLineOptions = {},
): EChartsOption {
  const {
    xMode = "step",
    yScale = "linear",
    smoothing = 0,
    dataZoom = false,
    domain,
    checkpointMarkers = [],
  } = options;

  const series: LineSeriesOption[] = [];
  const perLinePointLimit = linePointLimit(lines.length);
  for (const line of lines) {
    const linePoints =
      line.points.length > SCALAR_OPTION_DOWNSAMPLE_THRESHOLD
        ? downsamplePoints(line.points, perLinePointLimit)
        : line.points;
    const isSinglePoint = linePoints.length === 1;
    const rawData = dataForPoints(linePoints, xMode);
    const largeOptions = largeSeriesOptions(rawData.length);

    if (smoothing > 0 && !isSinglePoint) {
      const smoothedData = dataForPoints(
        applyEmaSmoothing(linePoints, smoothing),
        xMode,
      );
      // Smoothed pushed first so the tooltip de-dupe keeps its full-color marker.
      series.push({
        type: "line",
        name: line.name,
        data: smoothedData,
        showSymbol: false,
        lineStyle: { color: line.color, width: 2 },
        itemStyle: { color: line.color },
        z: 2,
        ...largeOptions,
      });
      series.push({
        type: "line",
        name: line.name,
        data: rawData,
        showSymbol: false,
        silent: true,
        lineStyle: { color: line.color, width: 1, opacity: 0.25 },
        itemStyle: { color: line.color },
        z: 1,
        ...largeOptions,
      });
    } else {
      series.push({
        type: "line",
        name: line.name,
        data: rawData,
        showSymbol: isSinglePoint,
        symbolSize: 6,
        lineStyle: { color: line.color, width: 2 },
        itemStyle: { color: line.color },
        ...largeOptions,
      });
    }
  }

  if (xMode === "step" && series.length > 0) {
    const markerData = checkpointMarkLineData(checkpointMarkers);
    if (markerData.length > 0) {
      series[0] = {
        ...series[0],
        markLine: {
          animation: false,
          symbol: "none",
          silent: false,
          lineStyle: {
            color: "rgba(255,255,255,0.42)",
            type: "dashed",
            width: 1,
          },
          label: { show: false },
          tooltip: {
            formatter: formatCheckpointTooltip,
          },
          data: markerData,
        },
      };
    }
  }

  return {
    animation: false,
    grid: { left: 48, right: 16, top: 16, bottom: dataZoom ? 48 : 28 },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "cross" },
      formatter: formatAxisTooltip,
    },
    xAxis: {
      type: xMode === "wallTime" ? "time" : "value",
      scale: true,
      min: xMode === "step" ? domain?.minStep : undefined,
      max: xMode === "step" ? domain?.maxStep : undefined,
    },
    yAxis: {
      type: yScale === "log" ? "log" : "value",
      scale: true,
      min: domain?.minValue,
      max: domain?.maxValue,
    },
    dataZoom: dataZoom
      ? [
          { type: "inside" },
          { type: "slider", height: 16, bottom: 8 },
        ]
      : undefined,
    series,
  };
}
