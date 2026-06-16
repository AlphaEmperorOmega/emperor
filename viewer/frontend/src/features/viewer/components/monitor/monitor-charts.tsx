import { type ReactNode } from "react";
import { Loader2 } from "lucide-react";
import Image from "next/image";
import { Badge } from "@/components/ui/badge";
import { ChartFrame } from "@/features/viewer/components/monitor/chart-frame";
import { EChart } from "@/features/viewer/components/charts/echart";
import { buildScalarLineOption } from "@/lib/echarts/scalar-options";
import { buildHistogramBarOption } from "@/lib/echarts/histogram-options";
import { multiRunLineColors } from "@/lib/charts";
import { formatNumber, formatRunDisplayName } from "@/lib/format";
import {
  type HistogramData,
  type MonitorImageData,
  type MultiRunScalarMetric,
  type ScalarDomain,
  type ScalarSeries,
} from "@/types/monitor";
import { type LogRun } from "@/lib/api";

// Single-run monitor scalars keep their original cyan accent.
const SINGLE_SCALAR_COLOR = "#7fd0ff";

// Scalar (line) charts in the modal share one group so the crosshair + tooltip
// track the same step across a node's metrics. Histograms use a value x-axis and
// are deliberately left out of the group.
const MONITOR_SCALAR_GROUP = "monitor-scalars";

function runDisplayName(run: LogRun) {
  return formatRunDisplayName({
    name: run.runName,
    id: run.id,
    startTime: run.timestamp ?? run.version,
  });
}

export function ScalarChart({ series, domain }: { series: ScalarSeries; domain?: ScalarDomain }) {
  const points = series.points;
  const values = points.map((point) => point.value);
  const localMin = values.length ? Math.min(...values) : 0;
  const localMax = values.length ? Math.max(...values) : 1;
  const latest = points.at(-1);
  const option = buildScalarLineOption(
    [{ id: series.tag, name: series.label, color: SINGLE_SCALAR_COLOR, points }],
    { domain },
  );

  return (
    <ChartFrame
      title={series.label}
      subtitle={series.tag}
      badge={latest && <Badge>step {latest.step}</Badge>}
      footer={
        <>
          {points.length === 0 ? (
            <span>0 points</span>
          ) : (
            <>
              <span>min {formatNumber(localMin)}</span>
              <span>max {formatNumber(localMax)}</span>
            </>
          )}
          {latest && <span>latest {formatNumber(latest.value)}</span>}
        </>
      }
    >
      <div className="h-24 w-full min-w-0" role="img" aria-label={`${series.tag} scalar chart`}>
        <EChart option={option} group={MONITOR_SCALAR_GROUP} />
      </div>
    </ChartFrame>
  );
}

export function HistogramChart({
  histogram,
  maxCount,
}: {
  histogram: HistogramData;
  maxCount?: number;
}) {
  const localMaxCount = Math.max(...histogram.buckets.map((bucket) => bucket.count), 1);
  const option = buildHistogramBarOption(histogram, { maxCount });

  return (
    <ChartFrame
      title={histogram.tag.split("/").slice(-2).join("/")}
      subtitle={histogram.tag}
      badge={<Badge>step {histogram.step}</Badge>}
      footer={
        <>
          <span>{histogram.buckets.length} buckets</span>
          <span>max count {formatNumber(localMaxCount)}</span>
        </>
      }
    >
      <div className="h-24 w-full min-w-0" role="img" aria-label={`${histogram.tag} histogram`}>
        <EChart option={option} />
      </div>
    </ChartFrame>
  );
}

export function MonitorImage({ image }: { image: MonitorImageData }) {
  if (image.truncated || !image.dataUrl) {
    return (
      <ChartFrame
        title={image.tag.split("/").slice(-2).join("/")}
        subtitle={image.tag}
        badge={<Badge>step {image.step}</Badge>}
      >
        <div className="grid min-h-40 place-items-center rounded-[10px] border border-line-soft bg-black/25 p-4 text-center">
          <div className="grid gap-1">
            <div className="text-sm font-semibold text-ink">Payload omitted</div>
            <div className="max-w-sm text-xs leading-5 text-ink-faint">
              {image.truncationReason ??
                "This image summary exceeded the viewer payload budget."}
            </div>
          </div>
        </div>
      </ChartFrame>
    );
  }

  return (
    <ChartFrame
      title={image.tag.split("/").slice(-2).join("/")}
      subtitle={image.tag}
      badge={<Badge>step {image.step}</Badge>}
    >
      <Image
        src={image.dataUrl}
        alt={`Monitor image for ${image.tag} at step ${image.step}`}
        width={640}
        height={320}
        sizes="(min-width: 1280px) 33vw, (min-width: 768px) 50vw, 100vw"
        unoptimized
        className="max-h-64 w-full rounded-[10px] border border-line-soft bg-black/25 object-contain"
      />
    </ChartFrame>
  );
}

export function MultiRunScalarChart({ metric }: { metric: MultiRunScalarMetric }) {
  const allPoints = metric.entries.flatMap((entry) => entry.series.points);
  const values = allPoints.map((point) => point.value);
  const minLabel = formatNumber(values.length ? Math.min(...values) : 0);
  const maxLabel = formatNumber(values.length ? Math.max(...values) : 1);
  const latestStep = allPoints.length
    ? Math.max(...allPoints.map((point) => point.step))
    : undefined;
  const option = buildScalarLineOption(
    metric.entries.map((entry, index) => ({
      id: entry.run.id,
      name: runDisplayName(entry.run),
      color: multiRunLineColors[index % multiRunLineColors.length],
      points: entry.series.points,
    })),
  );

  return (
    <ChartFrame
      title={metric.key}
      subtitle={`${metric.entries.length} / ${
        metric.entries.length + metric.missingRuns.length
      } runs`}
      badge={latestStep !== undefined && <Badge>step {latestStep}</Badge>}
      footer={
        <>
          <span>min {minLabel}</span>
          <span>max {maxLabel}</span>
        </>
      }
    >
      <div
        className="h-28 w-full min-w-0"
        role="img"
        aria-label={`${metric.key} multi-run scalar chart`}
      >
        <EChart option={option} group={MONITOR_SCALAR_GROUP} />
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-ink-dim">
        {metric.entries.map((entry, index) => (
          <span key={entry.run.id} className="inline-flex min-w-0 items-center gap-1">
            <span
              className="h-2 w-2 shrink-0 rounded-full"
              style={{
                backgroundColor: multiRunLineColors[index % multiRunLineColors.length],
              }}
              aria-hidden
            />
            <span className="max-w-44 truncate" title={runDisplayName(entry.run)}>
              {runDisplayName(entry.run)}
            </span>
          </span>
        ))}
        {metric.missingRuns.length > 0 && (
          <span className="basis-full text-ink-faint">
            Missing: {metric.missingRuns.map(runDisplayName).join(", ")}
          </span>
        )}
      </div>
    </ChartFrame>
  );
}

export function RunVisualCard({
  run,
  children,
}: {
  run: LogRun;
  children: ReactNode;
}) {
  return (
    <div className="overflow-hidden rounded-[10px] border border-line-soft bg-white/[0.018]">
      <div className="border-b border-line-soft px-3 py-2">
        <div className="truncate text-xs font-semibold text-ink" title={runDisplayName(run)}>
          {runDisplayName(run)}
        </div>
      </div>
      {children}
    </div>
  );
}

export function MonitorEmptyState({
  title,
  detail,
  busy = false,
}: {
  title: string;
  detail: string;
  busy?: boolean;
}) {
  return (
    <div className="edge grid min-h-40 place-items-center rounded-card p-5 text-center">
      <div className="grid justify-items-center gap-2">
        {busy && <Loader2 className="h-5 w-5 animate-spin text-violet" aria-hidden />}
        <div className="text-sm font-semibold text-ink">{title}</div>
        <div className="max-w-md text-sm text-ink-faint">{detail}</div>
      </div>
    </div>
  );
}

export function MissingMetricCard({
  metric,
  nodePath,
  busy = false,
}: {
  metric: string;
  nodePath: string;
  busy?: boolean;
}) {
  return (
    <div className="grid min-h-40 place-items-center p-4 text-center">
      <div className="grid justify-items-center gap-2">
        {busy && <Loader2 className="h-5 w-5 animate-spin text-violet" aria-hidden />}
        <div className="text-sm font-semibold text-ink">Metric not found</div>
        <div className="max-w-sm text-xs text-ink-faint">
          {busy
            ? `Waiting for ${metric} from ${nodePath}.`
            : `${nodePath} has no matching ${metric} tag.`}
        </div>
      </div>
    </div>
  );
}
