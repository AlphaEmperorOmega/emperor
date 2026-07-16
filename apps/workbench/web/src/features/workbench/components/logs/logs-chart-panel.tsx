import {
  ChevronDown,
  Columns2,
  Columns3,
  LineChart,
  Loader2,
  RefreshCw,
  RectangleHorizontal,
} from "lucide-react";
import dynamic from "next/dynamic";
import { type ReactNode, useCallback, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { ErrorPanel } from "@/features/workbench/components/error-panel";
import { ViewModeButton } from "@/features/workbench/components/view-mode-button";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { SurfacePanel } from "@/components/ui/surface-panel";
import { LogBestRunPanel } from "@/features/workbench/components/logs/log-best-run-panel";
import { LogConfusionMatrixHeatmaps } from "@/features/workbench/components/logs/log-confusion-matrix-heatmap";
import { LogTestScoresPanel } from "@/features/workbench/components/logs/log-test-scores-panel";
import { LogValidationExamplesPanel } from "@/features/workbench/components/logs/log-validation-examples-panel";
import {
  MultiSelectDropdown,
  type MultiSelectDropdownOption,
} from "@/features/workbench/components/screen/multi-select-dropdown";
import {
  LOG_METRIC_GROUPS,
  type LogMetricGroupKey,
  type TrainValidationScalarPair,
} from "@/features/workbench/state/logs/logs-selectors";
import {
  type LogsCharts,
  type LogsChartEmptyState,
  type LogMetricChartLayoutGroupKey,
  type ScalarChartGridMode,
} from "@/features/workbench/providers/logs-workspace-provider";
import { formatDecimal } from "@/lib/format";
import { cn, errorMessage } from "@/lib/utils";

const LazyLogScalarChart = dynamic(
  () =>
    import("@/features/workbench/components/logs/log-scalar-chart").then(
      (module) => module.LazyLogScalarChart,
    ),
  { ssr: false },
);
const LazyLogTrainValidationScalarChart = dynamic(
  () =>
    import("@/features/workbench/components/logs/log-scalar-chart").then(
      (module) => module.LazyLogTrainValidationScalarChart,
    ),
  { ssr: false },
);

const SCALAR_CHART_GRID_CLASSES: Record<ScalarChartGridMode, string> = {
  full: "grid gap-4",
  two: "grid gap-4 xl:grid-cols-2",
  three: "grid gap-4 xl:grid-cols-2 2xl:grid-cols-3",
};

const SCALAR_CHART_GRID_FULL_SPAN_CLASSES: Record<ScalarChartGridMode, string> = {
  full: "",
  two: "xl:col-span-2",
  three: "xl:col-span-2 2xl:col-span-3",
};

const ACCORDION_SECTION_GRID_CLASSES: Record<ScalarChartGridMode, string> = {
  full: "grid items-start gap-5",
  two: "grid items-start gap-5 xl:grid-cols-2",
  three: "grid items-start gap-5 xl:grid-cols-2 2xl:grid-cols-3",
};

const SCALAR_CHART_LAYOUT_OPTIONS = [
  { mode: "full", label: "Full", Icon: RectangleHorizontal },
  { mode: "two", label: "2 Col", Icon: Columns2 },
  { mode: "three", label: "3 Col", Icon: Columns3 },
] as const;

// Charts that share an ECharts group keep their tooltip, crosshair, and dataZoom
// in sync. All logs scalar charts belong to one group so hovering a step lights
// up every metric at once, TensorBoard-style.
const LOGS_SCALAR_GROUP = "logs-scalars";
const LOG_METRIC_GROUP_RENDER_LIMIT = 100;
const LOG_ACCORDION_METRIC_GROUPS = ["validation", "train", "other"] as const;
const LOG_PLOT_SELECTOR_GROUPS = ["train", "validation"] as const;
const LOG_PLOT_SELECTOR_TRIGGER_CLASS_NAME =
  "!min-h-9 !px-2 !py-1.5 text-xs [&>span:first-child]:gap-0 [&>span:first-child>span:nth-child(2)]:hidden";

type LogPlotSelectorGroupKey = (typeof LOG_PLOT_SELECTOR_GROUPS)[number];

const EMPTY_HIGHLIGHTED_RUNS_BY_GROUP: Record<
  LogMetricChartLayoutGroupKey,
  string | null
> = {
  train: null,
  validation: null,
  other: null,
};

type LogsChartMetricGroup = (typeof LOG_METRIC_GROUPS)[number];
type LogsChartAccordionMetricGroup = Extract<
  LogsChartMetricGroup,
  { key: LogMetricChartLayoutGroupKey }
>;

type LogsChartSectionRenderItem =
  | {
      key: LogMetricChartLayoutGroupKey;
      kind: "metricGroup";
      group: LogsChartAccordionMetricGroup;
    }
  | { key: "validationExamples"; kind: "validationExamples" }
  | { key: "confusionMatrix"; kind: "confusionMatrix" };

const LOG_METRIC_GROUP_BY_KEY = Object.fromEntries(
  LOG_METRIC_GROUPS.map((group) => [group.key, group]),
) as Record<LogMetricGroupKey, LogsChartMetricGroup>;

function metricCountLabel(count: number) {
  return `${count} ${count === 1 ? "metric" : "metrics"}`;
}

function plotCountLabel(count: number) {
  return `${count} ${count === 1 ? "plot" : "plots"}`;
}

function mediaCountLabel(imageCount: number, textCount: number, isLoading: boolean) {
  if (isLoading) {
    return "Loading…";
  }
  const count = imageCount + textCount;
  if (count === 0) {
    return "Available";
  }
  return `${count} ${count === 1 ? "item" : "items"}`;
}

function matrixCountLabel({
  heatmapCount,
  isLoaded,
  isLoading,
}: {
  heatmapCount: number;
  isLoaded: boolean;
  isLoading: boolean;
}) {
  if (isLoading && !isLoaded) {
    return "Loading…";
  }
  if (!isLoaded) {
    return "Available";
  }
  return `${heatmapCount} ${heatmapCount === 1 ? "matrix" : "matrices"}`;
}

function ScalarChartLayoutControl({
  ariaLabel,
  compact = false,
  mode,
  onModeChange,
}: {
  ariaLabel: string;
  compact?: boolean;
  mode: ScalarChartGridMode;
  onModeChange: (mode: ScalarChartGridMode) => void;
}) {
  return (
    <SegmentedControl
      aria-label={ariaLabel}
      className={cn("shrink-0", compact && "p-0.5")}
    >
      {SCALAR_CHART_LAYOUT_OPTIONS.map(({ mode: optionMode, label, Icon }) => {
        const active = mode === optionMode;
        if (!compact) {
          return (
            <ViewModeButton
              key={optionMode}
              active={active}
              onClick={() => onModeChange(optionMode)}
            >
              <Icon className="h-3.5 w-3.5" aria-hidden />
              {label}
            </ViewModeButton>
          );
        }

        return (
          <button
            key={optionMode}
            type="button"
            role="radio"
            aria-checked={active}
            tabIndex={active ? 0 : -1}
            onClick={() => onModeChange(optionMode)}
            className={cn(
              "inline-flex h-touch min-w-touch items-center gap-1.5 rounded-control-sm px-2 text-xs font-semibold transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:h-control-sm md:min-w-0",
              active
                ? "bg-selected-control text-white shadow-control-active"
                : "text-ink-faint hover:bg-control-active hover:text-ink",
            )}
          >
            <Icon className="h-3.5 w-3.5" aria-hidden />
            {label}
          </button>
        );
      })}
    </SegmentedControl>
  );
}

function LogsAccordionHeader({
  label,
  badge,
  isCollapsed,
  controlsId,
  onToggle,
  actions,
}: {
  label: string;
  badge: string;
  isCollapsed: boolean;
  controlsId: string;
  onToggle: () => void;
  actions?: ReactNode;
}) {
  return (
    <div
      className={cn(
        "grid w-full min-w-0 gap-2 rounded-control border border-line bg-white/[0.025] p-2 transition hover:border-white/15 hover:bg-white/[0.055]",
        actions
          ? "items-center [grid-template-columns:repeat(auto-fit,minmax(min(20rem,100%),1fr))]"
          : "grid-cols-1",
      )}
    >
      <button
        type="button"
        className="flex min-h-touch min-w-0 items-center justify-between gap-3 rounded-control-md px-1 py-1 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:min-h-control"
        aria-expanded={!isCollapsed}
        aria-controls={controlsId}
        onClick={onToggle}
      >
        <span className="flex min-w-0 items-center gap-2">
          <ChevronDown
            className={cn(
              "h-4 w-4 shrink-0 text-ink-faint transition-transform",
              isCollapsed && "-rotate-90",
            )}
            aria-hidden
          />
          <span className="truncate text-sm font-bold text-ink">{label}</span>
        </span>
        <Badge className="shrink-0 whitespace-nowrap">{badge}</Badge>
      </button>
      {actions && <div className="min-w-0">{actions}</div>}
    </div>
  );
}

function LogsMetricGroupHeader({
  group,
  metricCount,
  isCollapsed,
  controlsId,
  onToggle,
  actions,
}: {
  group: LogsChartMetricGroup;
  metricCount: number;
  isCollapsed: boolean;
  controlsId: string;
  onToggle: (group: LogMetricGroupKey) => void;
  actions?: ReactNode;
}) {
  return (
    <LogsAccordionHeader
      label={group.label}
      badge={metricCountLabel(metricCount)}
      isCollapsed={isCollapsed}
      controlsId={controlsId}
      onToggle={() => onToggle(group.key)}
      actions={actions}
    />
  );
}

function isLogPlotSelectorGroup(
  group: LogMetricGroupKey,
): group is LogPlotSelectorGroupKey {
  return (LOG_PLOT_SELECTOR_GROUPS as readonly LogMetricGroupKey[]).includes(group);
}

function logPlotSelectorOptions(tags: string[]): MultiSelectDropdownOption[] {
  return tags.map((tag) => ({
    value: tag,
    label: tag,
  }));
}

function trainValidationPlotSelectorOptions(
  pairs: TrainValidationScalarPair[],
): MultiSelectDropdownOption[] {
  return pairs.map((pair) => ({
    value: pair.suffix,
    label: pair.suffix,
    description: `${pair.trainTag} + ${pair.validationTag}`,
  }));
}

function LogPlotSelectorControls({
  groupLabel,
  values,
  options,
  onChange,
}: {
  groupLabel: string;
  values: string[];
  options: MultiSelectDropdownOption[];
  onChange: (values: string[]) => void;
}) {
  return (
    <div
      role="group"
      aria-label={`${groupLabel} plot controls`}
      className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-1.5"
    >
      <MultiSelectDropdown
        label={`${groupLabel} plots`}
        values={values}
        options={options}
        onChange={onChange}
        placeholder="No plots selected"
        className="min-w-0"
        triggerClassName={LOG_PLOT_SELECTOR_TRIGGER_CLASS_NAME}
        initialVisibleCount={25}
        pageSize={25}
      />
      <Button
        variant="secondary"
        className="h-touch shrink-0 px-2 text-xs md:h-control"
        aria-label={`Select all ${groupLabel} plots`}
        onClick={() => onChange(options.map((option) => option.value))}
      >
        All
      </Button>
      <Button
        variant="ghost"
        className="h-touch shrink-0 border border-line bg-panel-2/80 px-2 text-xs md:h-control"
        aria-label={`Select no ${groupLabel} plots`}
        onClick={() => onChange([])}
      >
        None
      </Button>
    </div>
  );
}

function ChartEmptyState({ title, detail, busy }: LogsChartEmptyState) {
  return (
    <div className="grid h-full min-h-[360px] place-items-center p-6">
      <SurfacePanel
        padding="spacious"
        className="max-w-md justify-items-center text-center shadow-panel"
      >
        {busy && <Loader2 className="h-5 w-5 animate-spin text-violet" aria-hidden />}
        <div className="flex h-10 w-10 items-center justify-center rounded-control border border-line bg-white/[0.04] text-violet">
          <LineChart className="h-5 w-5" aria-hidden />
        </div>
        <div>
          <div className="text-sm font-semibold text-ink">{title}</div>
          <div className="mt-1 text-xs leading-5 text-ink-faint">{detail}</div>
        </div>
      </SurfacePanel>
    </div>
  );
}

export function LogsChartPanel({ charts }: { charts: LogsCharts }) {
  const {
    metricsByGroup,
    availableMetricTagsByGroup,
    trainValidationPairs,
    trainValidationComparisonMetrics,
    selectedTrainValidationPairSuffixes,
    confusionHeatmaps,
    runsById,
    checkpointsByRunId,
    mediaImages,
    mediaTexts,
    runOrder,
    visibleRunCount,
    selectedTagCount,
    selectedTagsByGroup,
    hasValidationExampleMedia,
    bestRun,
  } = charts.content;
  const {
    isTrainValidationComparisonCollapsed,
    trainValidationComparisonGridMode,
    isConfusionMatrixCollapsed,
    collapsedMetricGroups,
    accordionGridMode,
    metricGridModes,
    smoothing,
    xMode,
    yScale,
    isValidationExamplesCollapsed,
  } = charts.settings;
  const {
    trainValidationPairQueryStates,
    trainValidationComparisonQueryState,
    scalarQueryStates,
    scalarTagQueryStates,
    hasConfusionMatrixTags,
    isConfusionMatrixLoaded,
    isConfusionMatrixLoading,
    isConfusionMatrixError,
    confusionMatrixError,
    isTagRefreshLoading,
    isFetching,
    isRefreshDisabled,
    emptyState,
    isValidationExampleMediaLoading,
    validationExampleMediaError,
    checkpointError,
  } = charts.status;
  const {
    onMetricGroupTagSelectionChange,
    onToggleTrainValidationComparison,
    onTrainValidationComparisonGridModeChange,
    onTrainValidationPairSelectionChange,
    onTrainValidationComparisonChartVisible,
    onScalarChartVisible,
    onToggleConfusionMatrix,
    onToggleMetricGroup,
    onAccordionGridModeChange,
    onMetricGridModeChange,
    onSmoothingChange,
    onXModeChange,
    onYScaleChange,
    onRefresh,
    onToggleValidationExamples,
    onValidationExamplesVisible,
    onSelectRun,
  } = charts.actions;
  const [highlightedRunsByGroup, setHighlightedRunsByGroup] = useState<
    Record<LogMetricChartLayoutGroupKey, string | null>
  >(() => ({ ...EMPTY_HIGHLIGHTED_RUNS_BY_GROUP }));
  const [highlightedTrainValidationRunId, setHighlightedTrainValidationRunId] =
    useState<string | null>(null);
  const visibleRunIds = useMemo(() => new Set(runOrder), [runOrder]);
  const setHighlightedRunForGroup = useCallback(
    (group: LogMetricChartLayoutGroupKey, runId: string | null) => {
      setHighlightedRunsByGroup((current) =>
        current[group] === runId ? current : { ...current, [group]: runId },
      );
    },
    [],
  );

  const visibleHighlightedRunsByGroup = useMemo(() => {
    const next = { ...highlightedRunsByGroup };
    for (const group of LOG_ACCORDION_METRIC_GROUPS) {
      const highlightedRunId = next[group];
      if (highlightedRunId !== null && !visibleRunIds.has(highlightedRunId)) {
        next[group] = null;
      }
    }
    return next;
  }, [highlightedRunsByGroup, visibleRunIds]);
  const visibleHighlightedTrainValidationRunId =
    highlightedTrainValidationRunId !== null &&
    visibleRunIds.has(highlightedTrainValidationRunId)
      ? highlightedTrainValidationRunId
      : null;

  const handlePlotSelectionChange = useCallback(
    (group: LogPlotSelectorGroupKey, selectedValues: string[]) => {
      onMetricGroupTagSelectionChange(group, selectedValues);
    },
    [onMetricGroupTagSelectionChange],
  );

  const orderedLogChartSections: LogsChartSectionRenderItem[] = [
    {
      key: "validation",
      kind: "metricGroup",
      group: LOG_METRIC_GROUP_BY_KEY.validation as LogsChartAccordionMetricGroup,
    },
    {
      key: "train",
      kind: "metricGroup",
      group: LOG_METRIC_GROUP_BY_KEY.train as LogsChartAccordionMetricGroup,
    },
    { key: "validationExamples", kind: "validationExamples" },
    { key: "confusionMatrix", kind: "confusionMatrix" },
    {
      key: "other",
      kind: "metricGroup",
      group: LOG_METRIC_GROUP_BY_KEY.other as LogsChartAccordionMetricGroup,
    },
  ];

  const renderTrainValidationComparisonSection = () => {
    if (trainValidationPairs.length === 0) {
      return null;
    }
    const bodyId = "logs-train-validation-comparison";
    const options = trainValidationPlotSelectorOptions(trainValidationPairs);
    const availableSuffixes = new Set(trainValidationPairs.map((pair) => pair.suffix));
    const selectedSuffixes = selectedTrainValidationPairSuffixes.filter((suffix) =>
      availableSuffixes.has(suffix),
    );
    const hasNoSelectedPlots = selectedSuffixes.length === 0;
    const visibleMetrics = trainValidationComparisonMetrics.slice(
      0,
      LOG_METRIC_GROUP_RENDER_LIMIT,
    );
    const hiddenMetricCount = Math.max(
      0,
      trainValidationComparisonMetrics.length - visibleMetrics.length,
    );
    const fullSpanClass =
      SCALAR_CHART_GRID_FULL_SPAN_CLASSES[trainValidationComparisonGridMode];
    const headerActions = (
      <div className="flex min-w-0 flex-wrap items-center justify-end gap-2">
        <div className="min-w-[15rem] flex-1">
          <LogPlotSelectorControls
            groupLabel="Train vs Validation"
            values={selectedSuffixes}
            options={options}
            onChange={onTrainValidationPairSelectionChange}
          />
        </div>
        <ScalarChartLayoutControl
          ariaLabel="Train vs Validation chart layout"
          compact
          mode={trainValidationComparisonGridMode}
          onModeChange={onTrainValidationComparisonGridModeChange}
        />
      </div>
    );

    return (
      <section className="grid gap-3">
        <LogsAccordionHeader
          label="Train vs Validation"
          badge={plotCountLabel(trainValidationPairs.length)}
          isCollapsed={isTrainValidationComparisonCollapsed}
          controlsId={bodyId}
          onToggle={onToggleTrainValidationComparison}
          actions={headerActions}
        />
        {!isTrainValidationComparisonCollapsed && (
          <>
            <div
              id={bodyId}
              className={
                hasNoSelectedPlots
                  ? "grid gap-3"
                  : SCALAR_CHART_GRID_CLASSES[trainValidationComparisonGridMode]
              }
            >
              {hasNoSelectedPlots && (
                <InlineStatus compact>No paired plots selected</InlineStatus>
              )}
              {!hasNoSelectedPlots &&
                trainValidationComparisonQueryState.isError && (
                  <div className={fullSpanClass}>
                    <ErrorPanel
                      title="Train vs Validation scalar read failed"
                      message={errorMessage(trainValidationComparisonQueryState.error)}
                    />
                  </div>
                )}
              {!hasNoSelectedPlots &&
                visibleMetrics.map((metric) => {
                  const pairQueryState = trainValidationPairQueryStates.get(
                    metric.suffix,
                  );
                  return (
                    <LazyLogTrainValidationScalarChart
                      key={metric.suffix}
                      suffix={metric.suffix}
                      trainTag={metric.trainTag}
                      validationTag={metric.validationTag}
                      series={metric.series}
                      runsById={runsById}
                      checkpointsByRunId={checkpointsByRunId}
                      runOrder={runOrder}
                      onSelectRun={onSelectRun}
                      highlightedRunId={visibleHighlightedTrainValidationRunId}
                      onHoverRunChange={setHighlightedTrainValidationRunId}
                      group={LOGS_SCALAR_GROUP}
                      xMode={xMode}
                      yScale={yScale}
                      smoothing={smoothing}
                      hasRequested={pairQueryState?.hasRequested ?? false}
                      isLoading={
                        pairQueryState?.isInitialLoading ||
                        pairQueryState?.isFetching ||
                        false
                      }
                      isError={pairQueryState?.isError ?? false}
                      error={pairQueryState?.error}
                      onVisible={onTrainValidationComparisonChartVisible}
                    />
                  );
                })}
            </div>
            {hiddenMetricCount > 0 && (
              <div className="rounded-control border border-line-soft bg-white/[0.018] px-3 py-3 text-center text-xs text-ink-faint">
                Showing {visibleMetrics.length} of{" "}
                {trainValidationComparisonMetrics.length} paired charts. Narrow
                selected plots to inspect the rest.
              </div>
            )}
          </>
        )}
      </section>
    );
  };

  const renderMetricGroupSection = (group: LogsChartAccordionMetricGroup) => {
    const metrics = metricsByGroup[group.key];
    const selectedGroupTags = selectedTagsByGroup[group.key];
    const availableGroupTags = availableMetricTagsByGroup[group.key];
    if (
      metrics.length === 0 &&
      selectedGroupTags.length === 0 &&
      availableGroupTags.length === 0
    ) {
      return null;
    }
    const isCollapsed = collapsedMetricGroups.has(group.key);
    const queryState = scalarQueryStates[group.key];
    const bodyId = `logs-metric-group-${group.key}`;
    const metricGridMode = metricGridModes[group.key] ?? "two";
    const plotSelectorGroup = isLogPlotSelectorGroup(group.key) ? group.key : null;
    const metricCount = plotSelectorGroup
      ? availableGroupTags.length
      : selectedGroupTags.length;
    const availablePlotTagSet = plotSelectorGroup
      ? new Set(availableGroupTags)
      : null;
    const selectedPlotTags = plotSelectorGroup
      ? selectedGroupTags.filter((tag) => availablePlotTagSet?.has(tag))
      : selectedGroupTags;
    const renderedMetrics = metrics;
    const visibleMetrics = renderedMetrics.slice(0, LOG_METRIC_GROUP_RENDER_LIMIT);
    const hiddenMetricCount = Math.max(
      0,
      renderedMetrics.length - visibleMetrics.length,
    );
    const fullSpanClass = SCALAR_CHART_GRID_FULL_SPAN_CLASSES[metricGridMode];
    const plotSelectorOptions = plotSelectorGroup
      ? logPlotSelectorOptions(availableGroupTags)
      : [];
    const plotSelector = plotSelectorGroup && availableGroupTags.length > 0 ? (
      <LogPlotSelectorControls
        groupLabel={group.label}
        values={selectedPlotTags}
        options={plotSelectorOptions}
        onChange={(values) => handlePlotSelectionChange(plotSelectorGroup, values)}
      />
    ) : undefined;
    const hasNoSelectedPlots =
      plotSelectorGroup !== null &&
      availableGroupTags.length > 0 &&
      metrics.length === 0;
    const chartLayoutControl = (
      <ScalarChartLayoutControl
        ariaLabel={`${group.label} chart layout`}
        compact
        mode={metricGridMode}
        onModeChange={(mode) => onMetricGridModeChange(group.key, mode)}
      />
    );
    const metricHeaderActions = plotSelector ? (
      <div className="flex min-w-0 flex-wrap items-center justify-end gap-2">
        <div className="min-w-[12rem] flex-1">{plotSelector}</div>
        {chartLayoutControl}
      </div>
    ) : (
      <div className="flex min-w-0 justify-end">{chartLayoutControl}</div>
    );

    return (
      <section key={group.key} className="grid gap-3">
        <LogsMetricGroupHeader
          group={group}
          metricCount={metricCount}
          isCollapsed={isCollapsed}
          controlsId={bodyId}
          onToggle={onToggleMetricGroup}
          actions={metricHeaderActions}
        />
        {!isCollapsed && (
          <>
            <div
              id={bodyId}
              className={
                hasNoSelectedPlots
                  ? "grid gap-3"
                  : SCALAR_CHART_GRID_CLASSES[metricGridMode]
              }
            >
              {hasNoSelectedPlots && (
                <InlineStatus compact>No plots selected in this group</InlineStatus>
              )}
              {!hasNoSelectedPlots && queryState.isError && (
                <div className={fullSpanClass}>
                  <ErrorPanel
                    title={`${group.label} scalar read failed`}
                    message={errorMessage(queryState.error)}
                  />
                </div>
              )}
              {!hasNoSelectedPlots &&
                queryState.isInitialLoading &&
                metrics.length === 0 && (
                  <InlineStatus
                    busy
                    compact
                    role="status"
                    className={fullSpanClass}
                  >
                    Loading {group.label} scalar points…
                  </InlineStatus>
                )}
              {!hasNoSelectedPlots &&
                visibleMetrics.map(({ tag, series }) => {
                  const tagQueryState = scalarTagQueryStates.get(tag);
                  return (
                    <LazyLogScalarChart
                      key={tag}
                      tag={tag}
                      series={series}
                      runsById={runsById}
                      checkpointsByRunId={checkpointsByRunId}
                      runOrder={runOrder}
                      onSelectRun={onSelectRun}
                      highlightedRunId={visibleHighlightedRunsByGroup[group.key]}
                      onHoverRunChange={(runId) =>
                        setHighlightedRunForGroup(group.key, runId)
                      }
                      group={LOGS_SCALAR_GROUP}
                      xMode={xMode}
                      yScale={yScale}
                      smoothing={smoothing}
                      hasRequested={tagQueryState?.hasRequested ?? false}
                      isLoading={
                        tagQueryState?.isInitialLoading ||
                        tagQueryState?.isFetching ||
                        false
                      }
                      isError={tagQueryState?.isError ?? false}
                      error={tagQueryState?.error}
                      onVisible={onScalarChartVisible}
                    />
                  );
                })}
            </div>
            {hiddenMetricCount > 0 && (
              <div className="rounded-control border border-line-soft bg-white/[0.018] px-3 py-3 text-center text-xs text-ink-faint">
                Showing {visibleMetrics.length} of {renderedMetrics.length} charts in
                this group. Narrow selected tags to inspect the rest.
              </div>
            )}
          </>
        )}
      </section>
    );
  };

  const renderLogChartSection = (section: LogsChartSectionRenderItem) => {
    if (section.kind === "metricGroup") {
      return renderMetricGroupSection(section.group);
    }

    if (section.kind === "validationExamples") {
      if (!hasValidationExampleMedia) {
        return null;
      }
      return (
        <section key={section.key} className="grid gap-3">
          <LogsAccordionHeader
            label="Validation Examples"
            badge={mediaCountLabel(
              mediaImages.length,
              mediaTexts.length,
              isValidationExampleMediaLoading,
            )}
            isCollapsed={isValidationExamplesCollapsed}
            controlsId="logs-validation-examples"
            onToggle={onToggleValidationExamples}
          />
          {!isValidationExamplesCollapsed && (
            <div id="logs-validation-examples" className="grid gap-3">
              {validationExampleMediaError && (
                <ErrorPanel
                  title="Validation example read failed"
                  message={errorMessage(validationExampleMediaError)}
                />
              )}
              <LogValidationExamplesPanel
                images={mediaImages}
                texts={mediaTexts}
                runsById={runsById}
                enabled={hasValidationExampleMedia}
                isLoading={isValidationExampleMediaLoading}
                onVisible={onValidationExamplesVisible}
              />
            </div>
          )}
        </section>
      );
    }

    if (!hasConfusionMatrixTags) {
      return null;
    }
    return (
      <section key={section.key} className="grid gap-3">
        <LogsAccordionHeader
          label="Confusion Matrix"
          badge={matrixCountLabel({
            heatmapCount: confusionHeatmaps.length,
            isLoaded: isConfusionMatrixLoaded,
            isLoading: isConfusionMatrixLoading,
          })}
          isCollapsed={isConfusionMatrixCollapsed}
          controlsId="logs-confusion-matrix"
          onToggle={onToggleConfusionMatrix}
        />
        {!isConfusionMatrixCollapsed && (
          <div id="logs-confusion-matrix" className="grid gap-3">
            {isConfusionMatrixError && (
              <ErrorPanel
                title="Confusion matrix read failed"
                message={errorMessage(confusionMatrixError)}
              />
            )}
            {isConfusionMatrixLoading && confusionHeatmaps.length === 0 && (
              <InlineStatus busy compact role="status">
                Loading confusion matrix scalar points…
              </InlineStatus>
            )}
            {!isConfusionMatrixLoading &&
              !isConfusionMatrixError &&
              isConfusionMatrixLoaded &&
              confusionHeatmaps.length === 0 && (
                <InlineStatus compact>
                  No confusion matrix points for the selected runs.
                </InlineStatus>
              )}
            <LogConfusionMatrixHeatmaps heatmaps={confusionHeatmaps} />
          </div>
        )}
      </section>
    );
  };

  return (
    <div className="grid min-h-0 grid-rows-[56px_minmax(0,1fr)] bg-bg-2/40">
      <div className="flex min-w-0 items-center justify-between gap-panel overflow-hidden border-b border-line bg-panel/55 px-region shadow-divider sm:px-shell">
        <div className="min-w-0">
          <h1 className="text-sm font-bold text-ink">Historical Scalars</h1>
          <div className="truncate font-mono text-xs text-ink-faint">
            {visibleRunCount} runs · {selectedTagCount} selected tags
          </div>
        </div>
        <div className="flex min-w-0 items-center justify-start gap-2 overflow-x-auto overscroll-x-contain [scrollbar-width:none] 2xl:justify-end">
          <label className="flex shrink-0 items-center gap-2 text-xs text-ink-faint">
            <span>Smooth</span>
            <input
              name="scalar-smoothing"
              type="range"
              min={0}
              max={0.99}
              step={0.01}
              value={smoothing}
              onChange={(event) => onSmoothingChange(Number(event.target.value))}
              aria-label="Scalar smoothing"
              className="h-touch w-24 cursor-pointer rounded-full accent-violet focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg md:h-control"
            />
            <span className="w-8 font-mono tabular-nums text-ink-dim">
              {formatDecimal(smoothing, 2)}
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
          <ScalarChartLayoutControl
            ariaLabel="Scalar accordion layout"
            mode={accordionGridMode}
            onModeChange={onAccordionGridModeChange}
          />
          <Button
            variant="secondary"
            className="h-touch shrink-0 px-2 md:h-control-sm"
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

      <div className="min-h-0 overflow-y-auto p-region">
        <div className="grid gap-shell">
          <LogBestRunPanel bestRun={bestRun} onSelectRun={onSelectRun} />
          {renderTrainValidationComparisonSection()}
          <LogTestScoresPanel
            metrics={metricsByGroup.test}
            selectedTags={selectedTagsByGroup.test}
            queryState={scalarQueryStates.test}
            runsById={runsById}
            runOrder={runOrder}
            onSelectRun={onSelectRun}
          />
          {emptyState ? (
            <ChartEmptyState {...emptyState} />
          ) : (
            <>
              {isTagRefreshLoading && (
                <InlineStatus busy compact role="status">
                  Refreshing TensorBoard tags…
                </InlineStatus>
              )}
              {checkpointError && (
                <ErrorPanel
                  title="Checkpoint metadata read failed"
                  message={errorMessage(checkpointError)}
                />
              )}
              <div className={ACCORDION_SECTION_GRID_CLASSES[accordionGridMode]}>
                {orderedLogChartSections.map(renderLogChartSection)}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
