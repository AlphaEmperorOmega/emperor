import { type ReactNode } from "react";
import { Loader2 } from "lucide-react";
import Image from "next/image";
import { Badge } from "@/components/ui/badge";
import { ChartFrame } from "@/components/features/viewer/monitor/chart-frame";
import { buildChartPath, buildLinearScale, formatChartDomain } from "@/lib/chart-scale";
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

function runDisplayName(run: LogRun) {
  return formatRunDisplayName({
    name: run.runName,
    id: run.id,
    startTime: run.timestamp ?? run.version,
  });
}

export function ScalarChart({ series, domain }: { series: ScalarSeries; domain?: ScalarDomain }) {
  const width = 320;
  const height = 92;
  const padding = 10;
  const points = series.points;
  const values = points.map((point) => point.value);
  const localMin = values.length ? Math.min(...values) : 0;
  const localMax = values.length ? Math.max(...values) : 1;
  const scale = buildLinearScale(points, {
    width,
    height,
    padding,
    domain,
    stepDomainMode: "series",
  });
  const pointCoordinates = points.map(scale.coordinate);
  const path = buildChartPath(points, scale);
  const singlePoint = pointCoordinates[0];
  const latest = points.at(-1);
  const { minLabel, maxLabel } = formatChartDomain({
    ...scale.domain,
    minValue: localMin,
    maxValue: localMax,
  });

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
              <span>min {minLabel}</span>
              <span>max {maxLabel}</span>
            </>
          )}
          {latest && <span>latest {formatNumber(latest.value)}</span>}
        </>
      }
    >
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="h-24 w-full text-cyan"
        role="img"
      >
        <line
          x1={padding}
          y1={height - padding}
          x2={width - padding}
          y2={height - padding}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />
        <line
          x1={padding}
          y1={padding}
          x2={padding}
          y2={height - padding}
          stroke="rgba(255,255,255,0.12)"
          vectorEffect="non-scaling-stroke"
        />
        {points.length === 0 ? null : points.length === 1 ? (
          <circle
            cx={domain && singlePoint ? singlePoint.x : width / 2}
            cy={domain && singlePoint ? singlePoint.y : height / 2}
            r="3"
            fill="currentColor"
          />
        ) : (
          <polyline
            points={path}
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinejoin="round"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        )}
      </svg>
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
  const width = 320;
  const height = 92;
  const padding = 10;
  const localMaxCount = Math.max(...histogram.buckets.map((bucket) => bucket.count), 1);
  const scaleMaxCount = maxCount ?? localMaxCount;
  const barWidth = (width - padding * 2) / Math.max(histogram.buckets.length, 1);

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
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="h-24 w-full"
        role="img"
      >
        <line
          x1={padding}
          y1={height - padding}
          x2={width - padding}
          y2={height - padding}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />
        {histogram.buckets.map((bucket, index) => {
          const barHeight = (bucket.count / scaleMaxCount) * (height - padding * 2);
          return (
            <rect
              key={`${bucket.left}-${bucket.right}-${index}`}
              x={padding + index * barWidth}
              y={height - padding - barHeight}
              width={Math.max(barWidth - 1, 1)}
              height={barHeight}
              fill="#a78bfa"
              opacity="0.85"
            />
          );
        })}
      </svg>
    </ChartFrame>
  );
}

export function MonitorImage({ image }: { image: MonitorImageData }) {
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
  const width = 320;
  const height = 104;
  const padding = 10;
  const points = metric.entries.flatMap((entry) => entry.series.points);
  const scale = buildLinearScale(points, {
    width,
    height,
    padding,
  });
  const latestStep = points.length ? scale.domain.maxStep : undefined;
  const { minLabel, maxLabel } = formatChartDomain(scale.domain);

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
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="h-28 w-full"
        role="img"
        aria-label={`${metric.key} multi-run scalar chart`}
      >
        <line
          x1={padding}
          y1={height - padding}
          x2={width - padding}
          y2={height - padding}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />
        <line
          x1={padding}
          y1={padding}
          x2={padding}
          y2={height - padding}
          stroke="rgba(255,255,255,0.12)"
          vectorEffect="non-scaling-stroke"
        />
        {metric.entries.map((entry, index) => {
          const color = multiRunLineColors[index % multiRunLineColors.length];
          const coordinates = entry.series.points.map(scale.coordinate);
          const path = buildChartPath(entry.series.points, scale);
          const singlePoint = coordinates[0];

          return entry.series.points.length === 1 && singlePoint ? (
            <circle
              key={entry.run.id}
              cx={singlePoint.x}
              cy={singlePoint.y}
              r="3"
              fill={color}
            />
          ) : (
            <polyline
              key={entry.run.id}
              points={path}
              fill="none"
              stroke={color}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </svg>
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
