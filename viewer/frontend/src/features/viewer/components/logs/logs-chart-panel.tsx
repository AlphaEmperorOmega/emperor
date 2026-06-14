import {
  ChevronDown,
  Columns2,
  Columns3,
  LineChart,
  Loader2,
  RefreshCw,
  RectangleHorizontal,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { ErrorPanel } from "@/features/viewer/components/error-panel";
import { ViewModeButton } from "@/features/viewer/components/view-mode-button";
import { LogConfusionMatrixHeatmaps } from "@/features/viewer/components/logs/log-confusion-matrix-heatmap";
import { LogScalarChart } from "@/features/viewer/components/logs/log-scalar-chart";
import { LogTestLeaderboardTable } from "@/features/viewer/components/logs/log-test-leaderboard-table";
import { LogValidationExamplesPanel } from "@/features/viewer/components/logs/log-validation-examples-panel";
import {
  type LogCheckpoint,
  type LogRun,
  type LogTextSummary,
} from "@/lib/api";
import { type ScalarXMode, type ScalarYScale } from "@/lib/echarts/scalar-options";
import {
  LOG_METRIC_GROUPS,
  type LogMetricsByGroup,
  type LogMetricGroupKey,
  isTestMetricTag,
} from "@/features/viewer/state/logs/logs-selectors";
import {
  type ConfusionMatrixHeatmap,
  type LogValidationExampleImage,
} from "@/features/viewer/state/logs/log-diagnostics";
import { cn, errorMessage } from "@/lib/utils";

export type ScalarChartGridMode = "full" | "two" | "three";

const SCALAR_CHART_GRID_CLASSES: Record<ScalarChartGridMode, string> = {
  full: "grid gap-4",
  two: "grid gap-4 xl:grid-cols-2",
  three: "grid gap-4 xl:grid-cols-2 2xl:grid-cols-3",
};

// Charts that share an ECharts group keep their tooltip, crosshair, and dataZoom
// in sync. All logs scalar charts belong to one group so hovering a step lights
// up every metric at once, TensorBoard-style.
const LOGS_SCALAR_GROUP = "logs-scalars";

export type LogsChartEmptyState = {
  title: string;
  detail: string;
  busy?: boolean;
};

function metricCountLabel(count: number) {
  return `${count} ${count === 1 ? "metric" : "metrics"}`;
}

function LogsMetricGroupHeader({
  group,
  metricCount,
  isCollapsed,
  controlsId,
  onToggle,
}: {
  group: (typeof LOG_METRIC_GROUPS)[number];
  metricCount: number;
  isCollapsed: boolean;
  controlsId: string;
  onToggle: (group: LogMetricGroupKey) => void;
}) {
  return (
    <button
      type="button"
      className="flex w-full min-w-0 items-center justify-between gap-3 rounded-[10px] border border-line bg-white/[0.025] px-3 py-2.5 text-left transition hover:border-white/15 hover:bg-white/[0.055] focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
      aria-expanded={!isCollapsed}
      aria-controls={controlsId}
      onClick={() => onToggle(group.key)}
    >
      <span className="flex min-w-0 items-center gap-2">
        <ChevronDown
          className={cn(
            "h-4 w-4 shrink-0 text-ink-faint transition-transform",
            isCollapsed && "-rotate-90",
          )}
          aria-hidden
        />
        <span className="truncate text-sm font-bold text-ink">{group.label}</span>
      </span>
      <Badge>{metricCountLabel(metricCount)}</Badge>
    </button>
  );
}

function ChartEmptyState({ title, detail, busy }: LogsChartEmptyState) {
  return (
    <div className="grid h-full min-h-[360px] place-items-center p-6">
      <div className="edge grid max-w-md justify-items-center gap-3 rounded-card p-6 text-center shadow-panel">
        {busy && <Loader2 className="h-5 w-5 animate-spin text-violet" aria-hidden />}
        <div className="flex h-10 w-10 items-center justify-center rounded-[10px] border border-line bg-white/[0.04] text-violet">
          <LineChart className="h-5 w-5" aria-hidden />
        </div>
        <div>
          <div className="text-sm font-semibold text-ink">{title}</div>
          <div className="mt-1 text-xs leading-5 text-ink-faint">{detail}</div>
        </div>
      </div>
    </div>
  );
}

export function LogsChartPanel({
  metricsByGroup,
  confusionHeatmaps,
  runsById,
  checkpointsByRunId,
  mediaImages,
  mediaTexts,
  runOrder,
  visibleRunCount,
  selectedTagCount,
  collapsedMetricGroups,
  onToggleMetricGroup,
  gridMode,
  onGridModeChange,
  smoothing,
  onSmoothingChange,
  xMode,
  onXModeChange,
  yScale,
  onYScaleChange,
  isFetching,
  isRefreshDisabled,
  onRefresh,
  isError,
  error,
  emptyState,
  onSelectRun,
}: {
  metricsByGroup: LogMetricsByGroup;
  confusionHeatmaps: ConfusionMatrixHeatmap[];
  runsById: Map<string, LogRun>;
  checkpointsByRunId: Map<string, LogCheckpoint[]>;
  mediaImages: LogValidationExampleImage[];
  mediaTexts: LogTextSummary[];
  runOrder: string[];
  visibleRunCount: number;
  selectedTagCount: number;
  collapsedMetricGroups: Set<LogMetricGroupKey>;
  onToggleMetricGroup: (group: LogMetricGroupKey) => void;
  gridMode: ScalarChartGridMode;
  onGridModeChange: (mode: ScalarChartGridMode) => void;
  smoothing: number;
  onSmoothingChange: (weight: number) => void;
  xMode: ScalarXMode;
  onXModeChange: (mode: ScalarXMode) => void;
  yScale: ScalarYScale;
  onYScaleChange: (scale: ScalarYScale) => void;
  isFetching: boolean;
  isRefreshDisabled: boolean;
  onRefresh: () => void;
  isError: boolean;
  error: unknown;
  emptyState: LogsChartEmptyState | null;
  onSelectRun: (runId: string) => void;
}) {
  return (
    <div className="grid min-h-0 grid-rows-[56px_minmax(0,1fr)]">
      <div className="flex min-w-0 items-center justify-between gap-3 border-b border-line bg-panel/45 px-4">
        <div className="min-w-0">
          <div className="text-sm font-bold text-ink">Historical Scalars</div>
          <div className="truncate font-mono text-xs text-ink-faint">
            {visibleRunCount} runs · {selectedTagCount} selected tags
          </div>
        </div>
        <div className="flex min-w-0 items-center justify-end gap-2 overflow-x-auto">
          <label className="flex shrink-0 items-center gap-2 text-xs text-ink-faint">
            <span>Smooth</span>
            <input
              type="range"
              min={0}
              max={0.99}
              step={0.01}
              value={smoothing}
              onChange={(event) => onSmoothingChange(Number(event.target.value))}
              aria-label="Scalar smoothing"
              className="h-1 w-24 cursor-pointer accent-violet"
            />
            <span className="w-8 font-mono tabular-nums text-ink-dim">
              {smoothing.toFixed(2)}
            </span>
          </label>
          <SegmentedControl aria-label="Scalar x axis" className="shrink-0">
            <ViewModeButton active={xMode === "step"} onClick={() => onXModeChange("step")}>
              Step
            </ViewModeButton>
            <ViewModeButton
              active={xMode === "wallTime"}
              onClick={() => onXModeChange("wallTime")}
            >
              Time
            </ViewModeButton>
          </SegmentedControl>
          <SegmentedControl aria-label="Scalar y scale" className="shrink-0">
            <ViewModeButton
              active={yScale === "linear"}
              onClick={() => onYScaleChange("linear")}
            >
              Lin
            </ViewModeButton>
            <ViewModeButton active={yScale === "log"} onClick={() => onYScaleChange("log")}>
              Log
            </ViewModeButton>
          </SegmentedControl>
          <SegmentedControl aria-label="Scalar chart layout" className="shrink-0">
            <ViewModeButton
              active={gridMode === "full"}
              onClick={() => onGridModeChange("full")}
            >
              <RectangleHorizontal className="h-3.5 w-3.5" aria-hidden />
              Full
            </ViewModeButton>
            <ViewModeButton
              active={gridMode === "two"}
              onClick={() => onGridModeChange("two")}
            >
              <Columns2 className="h-3.5 w-3.5" aria-hidden />
              2 Col
            </ViewModeButton>
            <ViewModeButton
              active={gridMode === "three"}
              onClick={() => onGridModeChange("three")}
            >
              <Columns3 className="h-3.5 w-3.5" aria-hidden />
              3 Col
            </ViewModeButton>
          </SegmentedControl>
          <Button
            variant="secondary"
            className="h-8 shrink-0 px-2"
            onClick={onRefresh}
            disabled={isRefreshDisabled}
            aria-label="Refresh scalar charts"
          >
            <RefreshCw
              className={cn("h-4 w-4", isFetching && "animate-spin")}
              aria-hidden
            />
          </Button>
        </div>
      </div>

      <div className="min-h-0 overflow-y-auto p-4">
        {isError && (
          <div className="mb-4">
            <ErrorPanel title="Scalar read failed" message={errorMessage(error)} />
          </div>
        )}

        {emptyState ? (
          <ChartEmptyState {...emptyState} />
        ) : (
          <div className="grid gap-5">
            <LogValidationExamplesPanel
              images={mediaImages}
              texts={mediaTexts}
              runsById={runsById}
            />
            <LogConfusionMatrixHeatmaps heatmaps={confusionHeatmaps} />
            {LOG_METRIC_GROUPS.map((group) => {
              const metrics = metricsByGroup[group.key];
              if (metrics.length === 0) {
                return null;
              }
              const isCollapsed = collapsedMetricGroups.has(group.key);
              const bodyId = `logs-metric-group-${group.key}`;

              return (
                <section key={group.key} className="grid gap-3">
                  <LogsMetricGroupHeader
                    group={group}
                    metricCount={metrics.length}
                    isCollapsed={isCollapsed}
                    controlsId={bodyId}
                    onToggle={onToggleMetricGroup}
                  />
                  {!isCollapsed && (
                    <div id={bodyId} className={SCALAR_CHART_GRID_CLASSES[gridMode]}>
                      {metrics.map(({ tag, series }) => {
                        if (isTestMetricTag(tag)) {
                          return (
                            <LogTestLeaderboardTable
                              key={tag}
                              tag={tag}
                              series={series}
                              runsById={runsById}
                              runOrder={runOrder}
                              onSelectRun={onSelectRun}
                            />
                          );
                        }
                        return (
                          <LogScalarChart
                            key={tag}
                            tag={tag}
                            series={series}
                            runsById={runsById}
                            checkpointsByRunId={checkpointsByRunId}
                            runOrder={runOrder}
                            onSelectRun={onSelectRun}
                            group={LOGS_SCALAR_GROUP}
                            xMode={xMode}
                            yScale={yScale}
                            smoothing={smoothing}
                          />
                        );
                      })}
                    </div>
                  )}
                </section>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
