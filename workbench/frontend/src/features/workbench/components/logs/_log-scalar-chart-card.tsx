"use client";

import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Info, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { IconButton } from "@/components/ui/icon-button";
import { SurfacePanel } from "@/components/ui/surface-panel";
import { EChart } from "@/features/workbench/components/charts/echart";
import { ErrorPanel } from "@/features/workbench/components/error-panel";
import { TrainingMetricInfoDialog } from "@/features/workbench/components/shared/training-metric-info-dialog";
import { type LogCheckpoint, type LogRun, type LogScalarSeries } from "@/lib/api";
import {
  buildScalarLineOption,
  type ScalarCheckpointMarker,
  type ScalarLine,
  type ScalarLinePoint,
  type ScalarXMode,
  type ScalarYScale,
} from "@/lib/echarts/scalar-options";
import { formatRunLabel } from "@/features/workbench/state/logs/logs-selectors";
import { formatNumber } from "@/lib/format";
import { cn, errorMessage } from "@/lib/utils";

const LOG_SCALAR_CHART_VIEWPORT_MARGIN_PX = 360;
const UNKNOWN_RUN_ORDER = Number.MAX_SAFE_INTEGER;

export type LogScalarCardLine = ScalarLine & { runId: string };

export type LogScalarLegendEntry = {
  id: string;
  runId: string;
  label: string;
  color: string;
  latest: ScalarLinePoint | undefined;
  marker:
    | { kind: "dot" }
    | { kind: "line"; style: "solid" | "dashed" };
};

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
  const leftSecondary =
    xMode === "wallTime" ? left.point.step : left.point.wallTime;
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
  series: readonly LogScalarSeries[],
  runOrder: readonly string[],
  xMode: ScalarXMode,
) {
  const runIndex = new Map(runOrder.map((runId, index) => [runId, index]));
  let minStep = Number.POSITIVE_INFINITY;
  let maxStep = Number.NEGATIVE_INFINITY;
  let earliestPoint: DisplayedScalarPoint | null = null;
  let latestPoint: DisplayedScalarPoint | null = null;
  let pointCount = 0;

  for (const [seriesOrder, entry] of series.entries()) {
    const entryRunOrder = runIndex.get(entry.runId) ?? UNKNOWN_RUN_ORDER;
    for (const [pointOrder, point] of entry.points.entries()) {
      minStep = Math.min(minStep, point.step);
      maxStep = Math.max(maxStep, point.step);
      pointCount += 1;
      const displayedPoint = {
        point,
        runOrder: entryRunOrder,
        seriesOrder,
        pointOrder,
      };
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

export function LazyLogScalarChartFrame({
  chartLabel,
  children,
  error,
  errorTitle,
  hasContent,
  hasRequested,
  isError,
  isLoading,
  loadingLabel,
  onVisible,
  visibilityKey,
}: {
  chartLabel: string;
  children: ReactNode;
  error: unknown;
  errorTitle: string;
  hasContent: boolean;
  hasRequested?: boolean;
  isError?: boolean;
  isLoading?: boolean;
  loadingLabel: string;
  onVisible?: (value: string) => void;
  visibilityKey: string;
}) {
  const { chartRef, hasEnteredView } = useLazyLogChartVisibility({
    onVisible,
    visibilityKey,
  });

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
  if (hasContent) {
    return children;
  }
  if (isError) {
    return (
      <SurfacePanel
        as="section"
        padding="spacious"
        className="min-h-[350px]"
        aria-label={`${chartLabel} error`}
      >
        <ErrorPanel title={errorTitle} message={errorMessage(error)} />
      </SurfacePanel>
    );
  }

  const isWaiting = isLoading || !hasRequested;
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
        {isWaiting ? `Loading ${loadingLabel}` : `No scalar points for ${visibilityKey}`}
      </div>
    </SurfacePanel>
  );
}

function checkpointMarkersForLines({
  checkpointsByRunId,
  lines,
  runsById,
}: {
  checkpointsByRunId: Map<string, LogCheckpoint[]>;
  lines: readonly LogScalarCardLine[];
  runsById: Map<string, LogRun>;
}) {
  return Array.from(new Set(lines.map((line) => line.runId))).flatMap((runId) => {
    const run = runsById.get(runId);
    const runLabel = run ? formatRunLabel(run) : runId;
    return (checkpointsByRunId.get(runId) ?? []).map(
      (checkpoint): ScalarCheckpointMarker => ({
        runId,
        runLabel,
        filename: checkpoint.filename,
        epoch: checkpoint.epoch,
        step: checkpoint.step,
      }),
    );
  });
}

export function LogScalarChartCard({
  chartLabel,
  checkpointsByRunId,
  group,
  highlightedRunId = null,
  infoMetricKey,
  legendEntries,
  lines,
  onHoverRunChange,
  onSelectRun,
  runOrder,
  runsById,
  smoothing = 0,
  summarySeries,
  title,
  xMode = "step",
  yScale = "linear",
}: {
  chartLabel: string;
  checkpointsByRunId: Map<string, LogCheckpoint[]>;
  group?: string;
  highlightedRunId?: string | null;
  infoMetricKey: string;
  legendEntries: readonly LogScalarLegendEntry[];
  lines: readonly LogScalarCardLine[];
  onHoverRunChange?: (runId: string | null) => void;
  onSelectRun: (runId: string) => void;
  runOrder: readonly string[];
  runsById: Map<string, LogRun>;
  smoothing?: number;
  summarySeries: readonly LogScalarSeries[];
  title: string;
  xMode?: ScalarXMode;
  yScale?: ScalarYScale;
}) {
  const [isInfoOpen, setIsInfoOpen] = useState(false);
  const scalarSummary = useMemo(
    () => summarizeDisplayedScalars(summarySeries, runOrder, xMode),
    [runOrder, summarySeries, xMode],
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
    const highlightedLineIds = chartHighlightedRunId
      ? lines
          .filter((line) => line.runId === chartHighlightedRunId)
          .map((line) => line.id)
      : [];
    return buildScalarLineOption(lines, {
      xMode,
      yScale,
      smoothing,
      highlightedLineIds,
      dataZoom: true,
      checkpointMarkers: checkpointMarkersForLines({
        checkpointsByRunId,
        lines,
        runsById,
      }),
    });
  }, [
    chartHighlightedRunId,
    checkpointsByRunId,
    lines,
    runsById,
    smoothing,
    xMode,
    yScale,
  ]);

  return (
    <SurfacePanel as="section" padding="spacious" className="min-w-0">
      <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-bold text-ink">{title}</h2>
          <div className="mt-0.5 font-mono text-xs text-ink-faint">
            {lines.length} lines · step {scalarSummary.minStep} to{" "}
            {scalarSummary.maxStep}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <IconButton
            label={`Explain metric ${title}`}
            title={`Explain ${title}`}
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

      <div className="h-56 w-full min-w-0" role="img" aria-label={chartLabel}>
        <EChart option={option} group={group} />
      </div>

      <div className="grid max-h-48 min-h-0 gap-1.5 overflow-y-auto pr-1 sm:grid-cols-2 xl:grid-cols-3">
        {legendEntries.map((entry) => {
          const hasHighlightedRun = chartHighlightedRunId !== null;
          const isHighlightedRun = entry.runId === chartHighlightedRunId;
          return (
            <button
              key={entry.id}
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
              {entry.marker.kind === "dot" ? (
                <span
                  className="h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: entry.color }}
                  aria-hidden
                />
              ) : (
                <span
                  className={cn(
                    "h-0 w-4 border-t-2",
                    entry.marker.style === "dashed" && "border-dashed",
                  )}
                  style={{ borderColor: entry.color }}
                  aria-hidden
                />
              )}
              <span className="truncate text-ink-dim">{entry.label}</span>
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
          metricKey={infoMetricKey}
          valueTitle="Displayed trend"
          valueLabel={trendLabel}
          onClose={() => setIsInfoOpen(false)}
        />
      )}
    </SurfacePanel>
  );
}
