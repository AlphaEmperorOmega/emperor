"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Info } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { IconButton } from "@/components/ui/icon-button";
import { EChart } from "@/features/viewer/components/charts/echart";
import { SurfacePanel } from "@/features/viewer/components/shared/surface-panel";
import { TrainingMetricInfoDialog } from "@/features/viewer/components/shared/training-metric-info-dialog";
import { type LogCheckpoint, type LogRun, type LogScalarSeries } from "@/lib/api";
import { scalarSeriesColors } from "@/lib/charts";
import {
  buildScalarLineOption,
  type ScalarCheckpointMarker,
  type ScalarLine,
  type ScalarXMode,
  type ScalarYScale,
} from "@/lib/echarts/scalar-options";
import {
  formatNumber,
  formatRunLabel,
} from "@/features/viewer/state/logs/logs-selectors";
import { cn } from "@/lib/utils";

type LogScalarChartProps = {
  tag: string;
  series: LogScalarSeries[];
  runsById: Map<string, LogRun>;
  checkpointsByRunId: Map<string, LogCheckpoint[]>;
  runOrder: string[];
  onSelectRun: (runId: string) => void;
  highlightedRunId?: string | null;
  onHoverRunChange?: (runId: string | null) => void;
  xMode?: ScalarXMode;
  yScale?: ScalarYScale;
  smoothing?: number;
  group?: string;
};

export function LazyLogScalarChart(props: LogScalarChartProps) {
  const chartRef = useRef<HTMLElement | null>(null);
  const [hasEnteredView, setHasEnteredView] = useState(false);

  useEffect(() => {
    if (hasEnteredView) {
      return;
    }
    const node = chartRef.current;
    if (!node || typeof IntersectionObserver === "undefined") {
      setHasEnteredView(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          setHasEnteredView(true);
          observer.disconnect();
        }
      },
      { rootMargin: "360px 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [hasEnteredView]);

  if (!hasEnteredView) {
    return (
      <SurfacePanel
        as="section"
        padding="spacious"
        ref={chartRef}
        className="min-h-[350px]"
        aria-label={`${props.tag} scalar chart placeholder`}
      />
    );
  }

  return <LogScalarChart {...props} />;
}

export function LogScalarChart({
  tag,
  series,
  runsById,
  checkpointsByRunId,
  runOrder,
  onSelectRun,
  highlightedRunId = null,
  onHoverRunChange,
  xMode = "step",
  yScale = "linear",
  smoothing = 0,
  group,
}: LogScalarChartProps) {
  const [isInfoOpen, setIsInfoOpen] = useState(false);
  const runIndex = useMemo(
    () => new Map(runOrder.map((runId, index) => [runId, index])),
    [runOrder],
  );
  const colorFor = useCallback(
    (runId: string) =>
      scalarSeriesColors[
        Math.max(runIndex.get(runId) ?? 0, 0) % scalarSeriesColors.length
      ],
    [runIndex],
  );

  const scalarBounds = useMemo(() => {
    let minStep = Number.POSITIVE_INFINITY;
    let maxStep = Number.NEGATIVE_INFINITY;
    let minValue = Number.POSITIVE_INFINITY;
    let maxValue = Number.NEGATIVE_INFINITY;
    let pointCount = 0;
    for (const entry of series) {
      for (const point of entry.points) {
        minStep = Math.min(minStep, point.step);
        maxStep = Math.max(maxStep, point.step);
        minValue = Math.min(minValue, point.value);
        maxValue = Math.max(maxValue, point.value);
        pointCount += 1;
      }
    }
    return {
      minStep: pointCount ? minStep : 0,
      maxStep: pointCount ? maxStep : 0,
      minValue: pointCount ? minValue : 0,
      maxValue: pointCount ? maxValue : 0,
    };
  }, [series]);

  const legendEntries = useMemo(
    () =>
      series
        .map((entry) => {
          const run = runsById.get(entry.runId);
          if (!run) {
            return null;
          }
          return {
            runId: entry.runId,
            run,
            color: colorFor(entry.runId),
            latest: entry.points.at(-1),
          };
        })
        .filter((entry): entry is NonNullable<typeof entry> => entry !== null),
    [colorFor, runsById, series],
  );
  const chartHighlightedRunId = legendEntries.some(
    (entry) => entry.runId === highlightedRunId,
  )
    ? highlightedRunId
    : null;
  const rangeLabel = `${formatNumber(scalarBounds.minValue)} to ${formatNumber(
    scalarBounds.maxValue,
  )}`;

  const option = useMemo(() => {
    const lines: ScalarLine[] = series.map((entry) => {
      const run = runsById.get(entry.runId);
      return {
        id: entry.runId,
        name: run ? formatRunLabel(run) : entry.runId,
        color: colorFor(entry.runId),
        points: entry.points,
      };
    });
    const checkpointMarkers: ScalarCheckpointMarker[] = series.flatMap((entry) => {
      const run = runsById.get(entry.runId);
      const runLabel = run ? formatRunLabel(run) : entry.runId;
      return (checkpointsByRunId.get(entry.runId) ?? []).map((checkpoint) => ({
        runId: entry.runId,
        runLabel,
        filename: checkpoint.filename,
        epoch: checkpoint.epoch,
        step: checkpoint.step,
      }));
    });
    return buildScalarLineOption(lines, {
      xMode,
      yScale,
      smoothing,
      highlightedLineId: chartHighlightedRunId,
      dataZoom: true,
      checkpointMarkers,
    });
  }, [
    colorFor,
    series,
    runsById,
    checkpointsByRunId,
    xMode,
    yScale,
    smoothing,
    chartHighlightedRunId,
  ]);

  return (
    <SurfacePanel as="section" padding="spacious" className="min-w-0">
      <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-bold text-ink">{tag}</h2>
          <div className="mt-0.5 font-mono text-xs text-ink-faint">
            {series.length} lines · step {scalarBounds.minStep} to{" "}
            {scalarBounds.maxStep}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <IconButton
            label={`Explain metric ${tag}`}
            title={`Explain ${tag}`}
            size="sm"
            variant="ghost"
            aria-haspopup="dialog"
            aria-expanded={isInfoOpen ? true : undefined}
            className="h-6 w-6 rounded-[7px] border border-line-soft bg-white/[0.025] text-ink-faint hover:border-violet/40 hover:bg-violet/10 hover:text-violet focus-visible:ring-2 focus-visible:ring-focus"
            icon={<Info className="h-3.5 w-3.5" aria-hidden />}
            onClick={() => setIsInfoOpen(true)}
          />
          <Badge>{rangeLabel}</Badge>
        </div>
      </div>

      <div className="h-56 w-full min-w-0" role="img" aria-label={`${tag} scalar chart`}>
        <EChart option={option} group={group} />
      </div>

      <div className="grid max-h-48 min-h-0 gap-1.5 overflow-y-auto pr-1 sm:grid-cols-2 xl:grid-cols-3">
        {legendEntries.map((entry) => {
          const hasHighlightedRun = chartHighlightedRunId !== null;
          const isHighlightedRun = entry.runId === chartHighlightedRunId;
          return (
            <button
              key={entry.runId}
              type="button"
              className={cn(
                "grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 rounded-[9px] border border-line-soft bg-black/20 px-2 py-1.5 text-left text-xs transition hover:border-line hover:bg-white/[0.035] focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
                hasHighlightedRun && !isHighlightedRun
                  ? "opacity-30"
                  : "opacity-100",
              )}
              onClick={() => onSelectRun(entry.runId)}
              onPointerEnter={() => onHoverRunChange?.(entry.runId)}
              onPointerLeave={() => onHoverRunChange?.(null)}
              onFocus={() => onHoverRunChange?.(entry.runId)}
              onBlur={() => onHoverRunChange?.(null)}
            >
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: entry.color }}
                aria-hidden
              />
              <span className="truncate text-ink-dim">
                {formatRunLabel(entry.run)}
              </span>
              {entry.latest && (
                <span className="font-mono text-ink-faint">
                  {formatNumber(entry.latest.value)}
                </span>
              )}
            </button>
          );
        })}
      </div>
      {isInfoOpen && (
        <TrainingMetricInfoDialog
          metricKey={tag}
          valueTitle="Visible range"
          valueLabel={rangeLabel}
          onClose={() => setIsInfoOpen(false)}
        />
      )}
    </SurfacePanel>
  );
}
