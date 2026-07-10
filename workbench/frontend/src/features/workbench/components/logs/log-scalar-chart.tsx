"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Info, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { IconButton } from "@/components/ui/icon-button";
import { EChart } from "@/features/workbench/components/charts/echart";
import { ErrorPanel } from "@/features/workbench/components/error-panel";
import { SurfacePanel } from "@/components/ui/surface-panel";
import { TrainingMetricInfoDialog } from "@/features/workbench/components/shared/training-metric-info-dialog";
import { type LogCheckpoint, type LogRun, type LogScalarSeries } from "@/lib/api";
import { scalarSeriesColors } from "@/lib/charts";
import {
  buildScalarLineOption,
  type ScalarCheckpointMarker,
  type ScalarLine,
  type ScalarXMode,
  type ScalarYScale,
} from "@/lib/echarts/scalar-options";
import { formatRunLabel } from "@/features/workbench/state/logs/logs-selectors";
import { formatNumber } from "@/lib/format";
import { cn, errorMessage } from "@/lib/utils";

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
  hasRequested?: boolean;
  isLoading?: boolean;
  isError?: boolean;
  error?: unknown;
  onVisible?: (tag: string) => void;
};

type LogTrainValidationScalarChartProps = {
  suffix: string;
  trainTag: string;
  validationTag: string;
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
  hasRequested?: boolean;
  isLoading?: boolean;
  isError?: boolean;
  error?: unknown;
  onVisible?: (suffix: string) => void;
};

export type TrainValidationScalarLine = ScalarLine & {
  runId: string;
  tag: string;
  phase: "Train" | "Validation";
  latest: ScalarLine["points"][number] | undefined;
};

const LOG_SCALAR_CHART_VIEWPORT_MARGIN_PX = 360;
const UNKNOWN_RUN_ORDER = Number.MAX_SAFE_INTEGER;

type DisplayedScalarPoint = {
  point: LogScalarSeries["points"][number];
  runOrder: number;
  seriesOrder: number;
  pointOrder: number;
};

function compareDisplayedScalarPoints(
  left: DisplayedScalarPoint,
  right: DisplayedScalarPoint,
  xMode: ScalarXMode,
) {
  const leftPrimary = xMode === "wallTime" ? left.point.wallTime : left.point.step;
  const rightPrimary =
    xMode === "wallTime" ? right.point.wallTime : right.point.step;
  if (leftPrimary !== rightPrimary) {
    return leftPrimary - rightPrimary;
  }

  const leftSecondary = xMode === "wallTime" ? left.point.step : left.point.wallTime;
  const rightSecondary =
    xMode === "wallTime" ? right.point.step : right.point.wallTime;
  if (leftSecondary !== rightSecondary) {
    return leftSecondary - rightSecondary;
  }

  if (left.runOrder !== right.runOrder) {
    return left.runOrder - right.runOrder;
  }
  if (left.seriesOrder !== right.seriesOrder) {
    return left.seriesOrder - right.seriesOrder;
  }
  return left.pointOrder - right.pointOrder;
}

function summarizeDisplayedScalars(
  series: LogScalarSeries[],
  runIndex: Map<string, number>,
  xMode: ScalarXMode,
) {
  let minStep = Number.POSITIVE_INFINITY;
  let maxStep = Number.NEGATIVE_INFINITY;
  let earliestPoint: DisplayedScalarPoint | null = null;
  let latestPoint: DisplayedScalarPoint | null = null;
  let pointCount = 0;

  for (const [seriesOrder, entry] of series.entries()) {
    const runOrder = runIndex.get(entry.runId) ?? UNKNOWN_RUN_ORDER;
    for (const [pointOrder, point] of entry.points.entries()) {
      minStep = Math.min(minStep, point.step);
      maxStep = Math.max(maxStep, point.step);
      pointCount += 1;

      const displayedPoint = { point, runOrder, seriesOrder, pointOrder };
      if (
        earliestPoint === null ||
        compareDisplayedScalarPoints(displayedPoint, earliestPoint, xMode) < 0
      ) {
        earliestPoint = displayedPoint;
      }
      if (
        latestPoint === null ||
        compareDisplayedScalarPoints(displayedPoint, latestPoint, xMode) > 0
      ) {
        latestPoint = displayedPoint;
      }
    }
  }

  return {
    minStep: pointCount ? minStep : 0,
    maxStep: pointCount ? maxStep : 0,
    startValue: earliestPoint?.point.value ?? 0,
    endValue: latestPoint?.point.value ?? 0,
  };
}

function isElementNearViewport(
  node: HTMLElement,
  margin = LOG_SCALAR_CHART_VIEWPORT_MARGIN_PX,
) {
  const rect = node.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return false;
  }
  const viewportWidth =
    window.innerWidth || document.documentElement.clientWidth || 0;
  const viewportHeight =
    window.innerHeight || document.documentElement.clientHeight || 0;
  return (
    rect.bottom >= -margin &&
    rect.right >= -margin &&
    rect.top <= viewportHeight + margin &&
    rect.left <= viewportWidth + margin
  );
}

function useLazyLogChartVisibility({
  onVisible,
  visibilityKey,
}: {
  onVisible?: (value: string) => void;
  visibilityKey: string;
}) {
  const chartRef = useRef<HTMLElement | null>(null);
  const [hasEnteredView, setHasEnteredView] = useState(false);
  const markEnteredView = useCallback(() => {
    setHasEnteredView(true);
    onVisible?.(visibilityKey);
  }, [onVisible, visibilityKey]);

  useEffect(() => {
    if (hasEnteredView) {
      return;
    }
    const node = chartRef.current;
    if (!node || typeof IntersectionObserver === "undefined") {
      markEnteredView();
      return;
    }
    let cancelled = false;
    const cancelScheduledChecks: Array<() => void> = [];
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          markEnteredView();
          observer.disconnect();
        }
      },
      { rootMargin: "360px 0px" },
    );
    observer.observe(node);
    const checkInitialViewportEntry = () => {
      if (!cancelled && isElementNearViewport(node)) {
        markEnteredView();
        observer.disconnect();
      }
    };
    if (typeof window.requestAnimationFrame === "function") {
      const frameId = window.requestAnimationFrame(checkInitialViewportEntry);
      cancelScheduledChecks.push(() => window.cancelAnimationFrame?.(frameId));
    } else {
      const timeoutId = window.setTimeout(checkInitialViewportEntry, 0);
      cancelScheduledChecks.push(() => window.clearTimeout(timeoutId));
    }
    const delayedTimeoutId = window.setTimeout(checkInitialViewportEntry, 120);
    cancelScheduledChecks.push(() => window.clearTimeout(delayedTimeoutId));
    return () => {
      cancelled = true;
      observer.disconnect();
      cancelScheduledChecks.forEach((cancel) => cancel());
    };
  }, [hasEnteredView, markEnteredView]);

  return { chartRef, hasEnteredView };
}

export function LazyLogScalarChart(props: LogScalarChartProps) {
  const { chartRef, hasEnteredView } = useLazyLogChartVisibility({
    onVisible: props.onVisible,
    visibilityKey: props.tag,
  });

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

  if (props.series.length === 0) {
    if (props.isError) {
      return (
        <SurfacePanel
          as="section"
          padding="spacious"
          className="min-h-[350px]"
          aria-label={`${props.tag} scalar chart error`}
        >
          <ErrorPanel
            title={`${props.tag} scalar read failed`}
            message={errorMessage(props.error)}
          />
        </SurfacePanel>
      );
    }

    const isWaiting = props.isLoading || !props.hasRequested;
    return (
      <SurfacePanel
        as="section"
        padding="spacious"
        className="grid min-h-[350px] place-items-center"
        aria-label={`${props.tag} scalar chart loading`}
      >
        <div
          role={isWaiting ? "status" : undefined}
          className="flex items-center gap-2 text-xs text-ink-faint"
        >
          {isWaiting && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
          {isWaiting
            ? `Loading ${props.tag} scalar points`
            : `No scalar points for ${props.tag}`}
        </div>
      </SurfacePanel>
    );
  }

  return <LogScalarChart {...props} />;
}

export function LazyLogTrainValidationScalarChart(
  props: LogTrainValidationScalarChartProps,
) {
  const { chartRef, hasEnteredView } = useLazyLogChartVisibility({
    onVisible: props.onVisible,
    visibilityKey: props.suffix,
  });
  const chartLabel = `${props.suffix} train vs validation scalar chart`;

  if (!hasEnteredView) {
    return (
      <SurfacePanel
        as="section"
        padding="spacious"
        ref={chartRef}
        className="min-h-[350px]"
        aria-label={`${chartLabel} placeholder`}
      />
    );
  }

  if (props.series.length === 0) {
    if (props.isError) {
      return (
        <SurfacePanel
          as="section"
          padding="spacious"
          className="min-h-[350px]"
          aria-label={`${chartLabel} error`}
        >
          <ErrorPanel
            title={`${props.suffix} train vs validation scalar read failed`}
            message={errorMessage(props.error)}
          />
        </SurfacePanel>
      );
    }

    const isWaiting = props.isLoading || !props.hasRequested;
    return (
      <SurfacePanel
        as="section"
        padding="spacious"
        className="grid min-h-[350px] place-items-center"
        aria-label={`${chartLabel} loading`}
      >
        <div
          role={isWaiting ? "status" : undefined}
          className="flex items-center gap-2 text-xs text-ink-faint"
        >
          {isWaiting && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
          {isWaiting
            ? `Loading ${props.suffix} train vs validation scalar points`
            : `No scalar points for ${props.suffix}`}
        </div>
      </SurfacePanel>
    );
  }

  return <LogTrainValidationScalarChart {...props} />;
}

function runIndexForOrder(runOrder: readonly string[]) {
  return new Map(runOrder.map((runId, index) => [runId, index]));
}

function colorForRunIndex(runIndex: Map<string, number>, runId: string) {
  return scalarSeriesColors[
    Math.max(runIndex.get(runId) ?? 0, 0) % scalarSeriesColors.length
  ];
}

export function buildTrainValidationScalarLines({
  series,
  runsById,
  runOrder,
  trainTag,
  validationTag,
}: {
  series: readonly LogScalarSeries[];
  runsById: Map<string, LogRun>;
  runOrder: readonly string[];
  trainTag: string;
  validationTag: string;
}): TrainValidationScalarLine[] {
  const runIndex = runIndexForOrder(runOrder);
  const orderedRunIds = [...runOrder];
  const orderedRunIdSet = new Set(orderedRunIds);
  const seriesByRunAndTag = new Map<string, LogScalarSeries>();
  for (const entry of series) {
    seriesByRunAndTag.set(`${entry.runId}::${entry.tag}`, entry);
    if (!orderedRunIdSet.has(entry.runId)) {
      orderedRunIdSet.add(entry.runId);
      orderedRunIds.push(entry.runId);
      runIndex.set(entry.runId, runIndex.size);
    }
  }

  const phases = [
    { tag: trainTag, label: "Train" as const, lineStyle: { type: "solid" as const } },
    {
      tag: validationTag,
      label: "Validation" as const,
      lineStyle: { type: "dashed" as const },
    },
  ];

  return orderedRunIds.flatMap((runId) => {
    const run = runsById.get(runId);
    const runLabel = run ? formatRunLabel(run) : runId;
    return phases.flatMap(({ tag, label, lineStyle }) => {
      const entry = seriesByRunAndTag.get(`${runId}::${tag}`);
      if (!entry || entry.points.length === 0) {
        return [];
      }
      return [
        {
          id: `${runId}::${tag}`,
          runId,
          tag,
          phase: label,
          name: `${runLabel} · ${label}`,
          color: colorForRunIndex(runIndex, runId),
          lineStyle,
          points: entry.points,
          latest: entry.points.at(-1),
        },
      ];
    });
  });
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

  const scalarSummary = useMemo(
    () => summarizeDisplayedScalars(series, runIndex, xMode),
    [series, runIndex, xMode],
  );

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
  const trendLabel = `${formatNumber(scalarSummary.startValue)} to ${formatNumber(
    scalarSummary.endValue,
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
            {series.length} lines · step {scalarSummary.minStep} to{" "}
            {scalarSummary.maxStep}
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
          <Badge>{trendLabel}</Badge>
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
          valueTitle="Displayed trend"
          valueLabel={trendLabel}
          onClose={() => setIsInfoOpen(false)}
        />
      )}
    </SurfacePanel>
  );
}

export function LogTrainValidationScalarChart({
  suffix,
  trainTag,
  validationTag,
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
}: LogTrainValidationScalarChartProps) {
  const [isInfoOpen, setIsInfoOpen] = useState(false);

  const scalarSummary = useMemo(
    () => summarizeDisplayedScalars(series, runIndexForOrder(runOrder), xMode),
    [series, runOrder, xMode],
  );

  const lines = useMemo(
    () =>
      buildTrainValidationScalarLines({
        series,
        runsById,
        runOrder,
        trainTag,
        validationTag,
      }),
    [runsById, runOrder, series, trainTag, validationTag],
  );
  const highlightedLineIds = useMemo(
    () =>
      highlightedRunId
        ? [`${highlightedRunId}::${trainTag}`, `${highlightedRunId}::${validationTag}`]
        : [],
    [highlightedRunId, trainTag, validationTag],
  );
  const chartHighlightedRunId = lines.some((line) => line.runId === highlightedRunId)
    ? highlightedRunId
    : null;
  const trendLabel = `${formatNumber(scalarSummary.startValue)} to ${formatNumber(
    scalarSummary.endValue,
  )}`;

  const option = useMemo(() => {
    const checkpointRunIds = new Set(lines.map((line) => line.runId));
    const checkpointMarkers: ScalarCheckpointMarker[] = Array.from(
      checkpointRunIds,
    ).flatMap((runId) => {
      const run = runsById.get(runId);
      const runLabel = run ? formatRunLabel(run) : runId;
      return (checkpointsByRunId.get(runId) ?? []).map((checkpoint) => ({
        runId,
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
      highlightedLineIds: chartHighlightedRunId ? highlightedLineIds : [],
      dataZoom: true,
      checkpointMarkers,
    });
  }, [
    lines,
    runsById,
    checkpointsByRunId,
    xMode,
    yScale,
    smoothing,
    chartHighlightedRunId,
    highlightedLineIds,
  ]);

  return (
    <SurfacePanel as="section" padding="spacious" className="min-w-0">
      <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-bold text-ink">{suffix}</h2>
          <div className="mt-0.5 font-mono text-xs text-ink-faint">
            {lines.length} lines · step {scalarSummary.minStep} to{" "}
            {scalarSummary.maxStep}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <IconButton
            label={`Explain metric ${suffix}`}
            title={`Explain ${suffix}`}
            size="sm"
            variant="ghost"
            aria-haspopup="dialog"
            aria-expanded={isInfoOpen ? true : undefined}
            className="h-6 w-6 rounded-[7px] border border-line-soft bg-white/[0.025] text-ink-faint hover:border-violet/40 hover:bg-violet/10 hover:text-violet focus-visible:ring-2 focus-visible:ring-focus"
            icon={<Info className="h-3.5 w-3.5" aria-hidden />}
            onClick={() => setIsInfoOpen(true)}
          />
          <Badge>{trendLabel}</Badge>
        </div>
      </div>

      <div
        className="h-56 w-full min-w-0"
        role="img"
        aria-label={`${suffix} train vs validation scalar chart`}
      >
        <EChart option={option} group={group} />
      </div>

      <div className="grid max-h-48 min-h-0 gap-1.5 overflow-y-auto pr-1 sm:grid-cols-2 xl:grid-cols-3">
        {lines.map((line) => {
          const hasHighlightedRun = chartHighlightedRunId !== null;
          const isHighlightedRun = line.runId === chartHighlightedRunId;
          return (
            <button
              key={line.id}
              type="button"
              className={cn(
                "grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 rounded-[9px] border border-line-soft bg-black/20 px-2 py-1.5 text-left text-xs transition hover:border-line hover:bg-white/[0.035] focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
                hasHighlightedRun && !isHighlightedRun
                  ? "opacity-30"
                  : "opacity-100",
              )}
              onClick={() => onSelectRun(line.runId)}
              onPointerEnter={() => onHoverRunChange?.(line.runId)}
              onPointerLeave={() => onHoverRunChange?.(null)}
              onFocus={() => onHoverRunChange?.(line.runId)}
              onBlur={() => onHoverRunChange?.(null)}
            >
              <span
                className={cn(
                  "h-0 w-4 border-t-2",
                  line.phase === "Validation" && "border-dashed",
                )}
                style={{ borderColor: line.color }}
                aria-hidden
              />
              <span className="truncate text-ink-dim">{line.name}</span>
              {line.latest && (
                <span className="font-mono text-ink-faint">
                  {formatNumber(line.latest.value)}
                </span>
              )}
            </button>
          );
        })}
      </div>
      {isInfoOpen && (
        <TrainingMetricInfoDialog
          metricKey={validationTag}
          valueTitle="Displayed trend"
          valueLabel={trendLabel}
          onClose={() => setIsInfoOpen(false)}
        />
      )}
    </SurfacePanel>
  );
}
