import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { RefreshCw, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { type GraphNode, type MonitorData } from "@/lib/api";
import {
  type LinearMonitorComparisonCandidateGroups,
  type LinearMonitorComparisonScope,
} from "@/lib/graph/monitor-targets";
import {
  type HistoricalMonitorRunData,
  type MonitorGroup,
  type MonitorChartsSource,
  monitorGroupOrder,
} from "@/types/monitor";
import {
  comparisonGroupCount,
  countGroups,
  groupComparisonMonitorData,
  groupMultiRunMonitorData,
  groupSingleMonitorData,
  hasHistoricalMonitorData,
  histogramMaxCountFor,
  multiRunGroupCount,
  pairMetrics,
  scalarDomainFor,
  singleGroupCount,
  tagSuffix,
} from "@/lib/monitor/grouping";
import {
  HistogramChart,
  MissingMetricCard,
  MonitorEmptyState,
  MonitorImage,
  MultiRunScalarChart,
  RunVisualCard,
  ScalarChart,
} from "@/features/viewer/components/monitor/monitor-charts";
import {
  MonitorGroupAccordion,
  useMonitorGroupAccordion,
} from "@/features/viewer/components/monitor/monitor-group-accordion";
import { SelectOnlyDropdown } from "@/features/viewer/components/screen/select-only-dropdown";
import { DialogShell } from "@/features/viewer/components/shared/dialog-shell";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { viewerStatusCopy } from "@/features/viewer/components/shared/status-copy";
import {
  emptyMonitorComparisonCandidateGroups,
  useMonitorChartsModalState,
} from "@/features/viewer/state/graph-monitor/use-monitor-charts-modal-state";
import { LabeledField } from "@/features/viewer/components/shared/labeled-field";
import { cn } from "@/lib/utils";

type MonitorChartsModalProps = {
  node: GraphNode;
  source: MonitorChartsSource;
  comparisonCandidateGroups?: LinearMonitorComparisonCandidateGroups;
  onClose: () => void;
};

type ChartGridItem = {
  key: string;
  label: string;
  render: () => ReactNode;
};

type ComparisonPairItem = {
  render: (index: number) => ReactNode;
};

const LARGE_MONITOR_COLLAPSE_THRESHOLD = 24;
const MONITOR_CHART_RENDER_LIMIT = 200;

const comparisonScopeOptions: Array<{
  value: LinearMonitorComparisonScope;
  label: string;
}> = [
  { value: "same-stack", label: "Same stack" },
  { value: "all-layers", label: "All linear layers" },
];

function chartGridItemClass(index: number) {
  return cn(
    "min-w-0 border-line-soft",
    index > 0 && "border-t",
    index === 1 && "md:border-t-0",
    index % 2 === 1 && "md:border-l",
  );
}

function comparisonPairRowClass(index: number) {
  return cn("grid md:grid-cols-2", index > 0 && "border-t border-line-soft");
}

function comparisonPairCellClass(index: number) {
  return cn(
    "min-w-0 border-line-soft",
    index > 0 && "border-t md:border-l md:border-t-0",
  );
}

function totalGroupCount(groupCounts: Record<string, number>) {
  return Object.values(groupCounts).reduce((total, count) => total + count, 0);
}

function LazyMonitorChart({
  label,
  render,
}: {
  label: string;
  render: () => ReactNode;
}) {
  const chartRef = useRef<HTMLDivElement | null>(null);
  const canObserveVisibility = typeof IntersectionObserver !== "undefined";
  const [hasEnteredView, setHasEnteredView] = useState(!canObserveVisibility);

  useEffect(() => {
    if (hasEnteredView) {
      return;
    }
    const node = chartRef.current;
    if (!node || !canObserveVisibility) {
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
      { rootMargin: "320px 0px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [canObserveVisibility, hasEnteredView]);

  return (
    <div ref={chartRef}>
      {hasEnteredView ? (
        render()
      ) : (
        <div
          className="min-h-40 rounded-[10px] border border-line-soft bg-white/[0.018]"
          aria-label={`${label} chart placeholder`}
        />
      )}
    </div>
  );
}

function ChartLimitNotice({
  hiddenCount,
  unit = "charts",
}: {
  hiddenCount: number;
  unit?: string;
}) {
  if (hiddenCount <= 0) {
    return null;
  }
  return (
    <div className="border-t border-line-soft px-3 py-3 text-center text-xs text-ink-faint">
      Showing {MONITOR_CHART_RENDER_LIMIT} of{" "}
      {MONITOR_CHART_RENDER_LIMIT + hiddenCount} {unit}. Narrow the monitor target
      or comparison to inspect the rest.
    </div>
  );
}

function HistoricalMonitorProgressRow({
  label,
  progress,
}: {
  label: string;
  progress:
    | {
        loaded: number;
        failed: number;
        total: number;
        isLoading: boolean;
      }
    | null
    | undefined;
}) {
  if (!progress?.isLoading || progress.total === 0) {
    return null;
  }
  const remaining = Math.max(0, progress.total - progress.loaded - progress.failed);
  const failedCopy = progress.failed > 0 ? `, ${progress.failed} failed` : "";
  return (
    <InlineStatus busy compact role="status">
      {label}: loaded {progress.loaded} of {progress.total} historical runs
      {failedCopy}. {remaining} remaining.
    </InlineStatus>
  );
}

function pushChartItem(items: ChartGridItem[], item: ChartGridItem) {
  items.push(item);
}

function ChartGrid({ charts }: { charts: ChartGridItem[] }) {
  return (
    <div className="grid md:grid-cols-2">
      {charts.map(({ key, label, render }, index) => (
        <div key={key} className={chartGridItemClass(index)}>
          <LazyMonitorChart label={label} render={render} />
        </div>
      ))}
    </div>
  );
}

type MonitorAccordionBand = "activations" | "weights" | "bias" | "signals";

function monitorAccordionBand(group: MonitorGroup): MonitorAccordionBand {
  if (group === "Activations") {
    return "activations";
  }
  if (group === "Weights" || group === "Weight gradients") {
    return "weights";
  }
  if (group === "Bias" || group === "Bias gradients") {
    return "bias";
  }
  return "signals";
}

function renderMonitorAccordionBands(
  items: Array<{ group: MonitorGroup; accordion: ReactNode }>,
) {
  return items.map((item, index) => {
    const previousItem = items[index - 1];
    const showDivider =
      previousItem &&
      monitorAccordionBand(previousItem.group) !== monitorAccordionBand(item.group);

    return (
      <Fragment key={item.group}>
        {showDivider && (
          <div
            className="h-px bg-line-soft"
            data-testid="monitor-accordion-band-divider"
            aria-hidden
          />
        )}
        {item.accordion}
      </Fragment>
    );
  });
}

function SingleNodeCharts({ data }: { data: MonitorData }) {
  const groups = useMemo(() => groupSingleMonitorData(data), [data]);
  const groupCounts = useMemo(() => countGroups(groups, singleGroupCount), [groups]);
  const { isGroupOpen, toggleGroup } = useMonitorGroupAccordion(groupCounts, {
    startCollapsed: totalGroupCount(groupCounts) > LARGE_MONITOR_COLLAPSE_THRESHOLD,
  });
  const accordionItems = monitorGroupOrder.flatMap((group) => {
    const chartCount = groupCounts[group];
    if (chartCount === 0) {
      return [];
    }
    const groupData = groups[group];
    const isOpen = isGroupOpen(group);
    const charts: ChartGridItem[] = [];
    for (const series of groupData.scalarSeries) {
      pushChartItem(charts, {
        key: `scalar-${series.tag}`,
        label: series.tag,
        render: () => <ScalarChart series={series} />,
      });
    }
    for (const histogram of groupData.histograms) {
      pushChartItem(charts, {
        key: `histogram-${histogram.tag}`,
        label: histogram.tag,
        render: () => <HistogramChart histogram={histogram} />,
      });
    }
    for (const image of groupData.images) {
      pushChartItem(charts, {
        key: `image-${image.tag}`,
        label: image.tag,
        render: () => <MonitorImage image={image} />,
      });
    }
    const visibleCharts = charts.slice(0, MONITOR_CHART_RENDER_LIMIT);
    const hiddenCount = Math.max(0, chartCount - visibleCharts.length);

    return [
      {
        group,
        accordion: (
          <MonitorGroupAccordion
            idPrefix="single-monitor"
            group={group}
            count={chartCount}
            countUnit="chart"
            isOpen={isOpen}
            onToggle={() => toggleGroup(group)}
          >
            <ChartGrid charts={visibleCharts} />
            <ChartLimitNotice hiddenCount={hiddenCount} />
          </MonitorGroupAccordion>
        ),
      },
    ];
  });

  return (
    <div className="grid gap-3">
      {renderMonitorAccordionBands(accordionItems)}
    </div>
  );
}

function MultiRunMonitorCharts({
  results,
  idPrefix,
}: {
  results: HistoricalMonitorRunData[];
  idPrefix: string;
}) {
  const groups = useMemo(() => groupMultiRunMonitorData(results), [results]);
  const groupCounts = useMemo(() => countGroups(groups, multiRunGroupCount), [groups]);
  const { isGroupOpen, toggleGroup } = useMonitorGroupAccordion(groupCounts, {
    startCollapsed: totalGroupCount(groupCounts) > LARGE_MONITOR_COLLAPSE_THRESHOLD,
  });
  const accordionItems = monitorGroupOrder.flatMap((group) => {
    const chartCount = groupCounts[group];
    if (chartCount === 0) {
      return [];
    }
    const groupData = groups[group];
    const isOpen = isGroupOpen(group);
    const charts: ChartGridItem[] = [];
    for (const metric of groupData.scalarMetrics) {
      pushChartItem(charts, {
        key: `scalar-${metric.key}`,
        label: metric.key,
        render: () => <MultiRunScalarChart metric={metric} />,
      });
    }
    for (const entry of groupData.histograms) {
      pushChartItem(charts, {
        key: `histogram-${entry.run.id}-${entry.item.tag}`,
        label: entry.item.tag,
        render: () => (
          <RunVisualCard run={entry.run}>
            <HistogramChart histogram={entry.item} />
          </RunVisualCard>
        ),
      });
    }
    for (const entry of groupData.images) {
      pushChartItem(charts, {
        key: `image-${entry.run.id}-${entry.item.tag}`,
        label: entry.item.tag,
        render: () => (
          <RunVisualCard run={entry.run}>
            <MonitorImage image={entry.item} />
          </RunVisualCard>
        ),
      });
    }
    const visibleCharts = charts.slice(0, MONITOR_CHART_RENDER_LIMIT);
    const hiddenCount = Math.max(0, chartCount - visibleCharts.length);

    return [
      {
        group,
        accordion: (
          <MonitorGroupAccordion
            idPrefix={idPrefix}
            group={group}
            count={chartCount}
            countUnit="chart"
            isOpen={isOpen}
            onToggle={() => toggleGroup(group)}
          >
            <ChartGrid charts={visibleCharts} />
            <ChartLimitNotice hiddenCount={hiddenCount} />
          </MonitorGroupAccordion>
        ),
      },
    ];
  });

  return (
    <div className="grid gap-3">
      {renderMonitorAccordionBands(accordionItems)}
    </div>
  );
}

function ComparisonCharts({
  primaryNode,
  comparisonNode,
  primaryData,
  comparisonData,
  comparisonLoading,
}: {
  primaryNode: GraphNode;
  comparisonNode: GraphNode;
  primaryData: MonitorData | undefined;
  comparisonData: MonitorData | undefined;
  comparisonLoading: boolean;
}) {
  const scalarPairs = useMemo(
    () =>
      pairMetrics(
        primaryData?.scalarSeries ?? [],
        comparisonData?.scalarSeries ?? [],
        (series) => series.label,
        (series) => series.label,
      ),
    [comparisonData?.scalarSeries, primaryData?.scalarSeries],
  );
  const histogramPairs = useMemo(
    () =>
      pairMetrics(
        primaryData?.histograms ?? [],
        comparisonData?.histograms ?? [],
        (histogram) => tagSuffix(histogram.tag, primaryNode.path),
        (histogram) => tagSuffix(histogram.tag, comparisonNode.path),
      ),
    [
      comparisonData?.histograms,
      comparisonNode.path,
      primaryData?.histograms,
      primaryNode.path,
    ],
  );
  const imagePairs = useMemo(
    () =>
      pairMetrics(
        primaryData?.images ?? [],
        comparisonData?.images ?? [],
        (image) => tagSuffix(image.tag, primaryNode.path),
        (image) => tagSuffix(image.tag, comparisonNode.path),
      ),
    [comparisonData?.images, comparisonNode.path, primaryData?.images, primaryNode.path],
  );
  const groups = useMemo(
    () =>
      groupComparisonMonitorData({
        scalarPairs,
        histogramPairs,
        imagePairs,
        primaryNodePath: primaryNode.path,
        comparisonNodePath: comparisonNode.path,
      }),
    [
      comparisonNode.path,
      histogramPairs,
      imagePairs,
      primaryNode.path,
      scalarPairs,
    ],
  );
  const groupCounts = useMemo(() => countGroups(groups, comparisonGroupCount), [groups]);
  const { isGroupOpen, toggleGroup } = useMonitorGroupAccordion(groupCounts, {
    startCollapsed: totalGroupCount(groupCounts) > LARGE_MONITOR_COLLAPSE_THRESHOLD,
  });
  const accordionItems = monitorGroupOrder.flatMap((group) => {
    const pairCount = groupCounts[group];
    if (pairCount === 0) {
      return [];
    }
    const groupData = groups[group];
    const isOpen = isGroupOpen(group);
    const pairItems: ComparisonPairItem[] = [];
    for (const pair of groupData.scalarPairs) {
      pairItems.push({
        render: (index) => {
          const domain = scalarDomainFor(pair.primary, pair.comparison);
          return (
            <div key={`scalar-${pair.key}`} className={comparisonPairRowClass(index)}>
              <div className={comparisonPairCellClass(0)}>
                <LazyMonitorChart
                  label={`${pair.key} primary`}
                  render={() =>
                    pair.primary ? (
                      <ScalarChart series={pair.primary} domain={domain} />
                    ) : (
                      <MissingMetricCard metric={pair.key} nodePath={primaryNode.path} />
                    )
                  }
                />
              </div>
              <div className={comparisonPairCellClass(1)}>
                <LazyMonitorChart
                  label={`${pair.key} comparison`}
                  render={() =>
                    pair.comparison ? (
                      <ScalarChart series={pair.comparison} domain={domain} />
                    ) : (
                      <MissingMetricCard
                        metric={pair.key}
                        nodePath={comparisonNode.path}
                        busy={comparisonLoading}
                      />
                    )
                  }
                />
              </div>
            </div>
          );
        },
      });
    }
    for (const pair of groupData.histogramPairs) {
      pairItems.push({
        render: (index) => {
          const maxCount = histogramMaxCountFor(pair.primary, pair.comparison);
          return (
            <div key={`histogram-${pair.key}`} className={comparisonPairRowClass(index)}>
              <div className={comparisonPairCellClass(0)}>
                <LazyMonitorChart
                  label={`${pair.key} primary`}
                  render={() =>
                    pair.primary ? (
                      <HistogramChart histogram={pair.primary} maxCount={maxCount} />
                    ) : (
                      <MissingMetricCard metric={pair.key} nodePath={primaryNode.path} />
                    )
                  }
                />
              </div>
              <div className={comparisonPairCellClass(1)}>
                <LazyMonitorChart
                  label={`${pair.key} comparison`}
                  render={() =>
                    pair.comparison ? (
                      <HistogramChart histogram={pair.comparison} maxCount={maxCount} />
                    ) : (
                      <MissingMetricCard
                        metric={pair.key}
                        nodePath={comparisonNode.path}
                        busy={comparisonLoading}
                      />
                    )
                  }
                />
              </div>
            </div>
          );
        },
      });
    }
    for (const pair of groupData.imagePairs) {
      pairItems.push({
        render: (index) => (
          <div key={`image-${pair.key}`} className={comparisonPairRowClass(index)}>
            <div className={comparisonPairCellClass(0)}>
              <LazyMonitorChart
                label={`${pair.key} primary`}
                render={() =>
                  pair.primary ? (
                    <MonitorImage image={pair.primary} />
                  ) : (
                    <MissingMetricCard metric={pair.key} nodePath={primaryNode.path} />
                  )
                }
              />
            </div>
            <div className={comparisonPairCellClass(1)}>
              <LazyMonitorChart
                label={`${pair.key} comparison`}
                render={() =>
                  pair.comparison ? (
                    <MonitorImage image={pair.comparison} />
                  ) : (
                    <MissingMetricCard
                      metric={pair.key}
                      nodePath={comparisonNode.path}
                      busy={comparisonLoading}
                    />
                  )
                }
              />
            </div>
          </div>
        ),
      });
    }
    const visiblePairs = pairItems.slice(0, MONITOR_CHART_RENDER_LIMIT);
    const hiddenCount = Math.max(0, pairCount - visiblePairs.length);

    return [
      {
        group,
        accordion: (
          <MonitorGroupAccordion
            idPrefix="comparison-monitor"
            group={group}
            count={pairCount}
            countUnit="pair"
            isOpen={isOpen}
            onToggle={() => toggleGroup(group)}
          >
            <div>
              {visiblePairs.map((item, index) => item.render(index))}
              <ChartLimitNotice hiddenCount={hiddenCount} unit="pairs" />
            </div>
          </MonitorGroupAccordion>
        ),
      },
    ];
  });

  return (
    <div className="grid gap-4">
      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-[10px] border border-line-soft bg-white/[0.018] p-3">
          <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
            Primary layer
          </div>
          <div className="mt-1 break-words font-mono text-xs text-ink">{primaryNode.path}</div>
        </div>
        <div className="rounded-[10px] border border-line-soft bg-white/[0.018] p-3">
          <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
            Comparison layer
          </div>
          <div className="mt-1 break-words font-mono text-xs text-ink">
            {comparisonNode.path}
          </div>
        </div>
      </div>

      <div className="grid gap-3">
        {renderMonitorAccordionBands(accordionItems)}
      </div>
    </div>
  );
}

function MultiRunComparisonCharts({
  primaryNode,
  comparisonNode,
  primaryResults,
  comparisonResults,
  comparisonLoading,
}: {
  primaryNode: GraphNode;
  comparisonNode: GraphNode;
  primaryResults: HistoricalMonitorRunData[];
  comparisonResults: HistoricalMonitorRunData[] | undefined;
  comparisonLoading: boolean;
}) {
  return (
    <div className="grid gap-4">
      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-[10px] border border-line-soft bg-white/[0.018] p-3">
          <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
            Primary layer
          </div>
          <div className="mt-1 break-words font-mono text-xs text-ink">{primaryNode.path}</div>
        </div>
        <div className="rounded-[10px] border border-line-soft bg-white/[0.018] p-3">
          <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
            Comparison layer
          </div>
          <div className="mt-1 break-words font-mono text-xs text-ink">
            {comparisonNode.path}
          </div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="grid min-w-0 content-start gap-3">
          <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
            Primary overlays
          </div>
          <MultiRunMonitorCharts
            results={primaryResults}
            idPrefix="historical-primary-monitor"
          />
        </div>
        <div className="grid min-w-0 content-start gap-3">
          <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
            Comparison overlays
          </div>
          {comparisonResults && hasHistoricalMonitorData(comparisonResults) ? (
            <MultiRunMonitorCharts
              results={comparisonResults}
              idPrefix="historical-comparison-monitor"
            />
          ) : (
            <MonitorEmptyState
              title={
                comparisonLoading
                  ? viewerStatusCopy.loading.comparisonData
                  : viewerStatusCopy.empty.comparisonData
              }
              detail={`No scalar, histogram, or image tags matched ${comparisonNode.path} in these historical runs.`}
              busy={comparisonLoading}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export function MonitorChartsModal({
  node,
  source,
  comparisonCandidateGroups = emptyMonitorComparisonCandidateGroups,
  onClose,
}: MonitorChartsModalProps) {
  const modalState = useMonitorChartsModalState({
    node,
    source,
    comparisonCandidateGroups,
  });
  const {
    activeJob,
    historicalRun,
    historicalRunGroup,
    historicalRuns,
    historicalExperiment,
    historicalDataset,
    historicalPreset,
    dataset,
    setDataset,
    preset,
    setPreset,
    sourceDatasets,
    sourcePresets,
    monitorCount,
    comparisonCandidates,
    hasComparisonCandidates,
    comparisonScope,
    setComparisonScope,
    comparisonPath,
    setComparisonPath,
    comparisonNode,
    query,
  } = modalState;
  const {
    data,
    comparisonData,
    historicalData,
    historicalComparisonData,
    hasData,
    isComparing,
    isFetching,
    isLoading,
    comparisonLoading,
    historicalProgress,
    historicalComparisonProgress,
    emptyMessage,
    refetch: refreshMonitorData,
  } = query;
  const comparisonScopeSelectOptions = comparisonScopeOptions.map((option) => ({
    value: option.value,
    label: option.label,
    disabled: comparisonCandidateGroups[option.value].length === 0,
  }));
  const comparisonPathOptions = [
    { value: "", label: "No comparison" },
    ...comparisonCandidates.map((candidate) => ({
      value: candidate.path,
      label: candidate.path,
    })),
  ];
  const sourcePresetOptions = sourcePresets.map((sourcePreset) => ({
    value: sourcePreset,
    label: sourcePreset,
  }));
  const sourceDatasetOptions = sourceDatasets.map((sourceDataset) => ({
    value: sourceDataset,
    label: sourceDataset,
  }));

  return (
    <DialogShell
      titleId="monitor-charts-title"
      size="fullscreen"
      onClose={onClose}
      className="grid bg-black/70 p-3 sm:p-6"
      panelClassName="grid max-h-full min-h-0 max-w-5xl grid-rows-[auto_minmax(0,1fr)] justify-self-center"
    >
        <div className="grid gap-3 border-b border-line bg-panel/85 p-4 backdrop-blur sm:grid-cols-[minmax(0,1fr)_auto]">
          <div className="min-w-0">
            <div className="flex min-w-0 flex-wrap items-center gap-2">
              <h2 id="monitor-charts-title" className="text-base font-semibold text-ink">
                Monitor charts
              </h2>
              {activeJob ? (
                <>
                  <Badge>{activeJob.status}</Badge>
                  <Badge>{activeJob.monitors.length} monitors</Badge>
                </>
              ) : (
                <>
                  <Badge>{historicalRunGroup ? "historical group" : "historical"}</Badge>
                  {historicalRunGroup ? (
                    <>
                      <Badge>{historicalExperiment}</Badge>
                      <Badge>{historicalDataset}</Badge>
                      <Badge>{historicalPreset}</Badge>
                      <Badge>{historicalRuns.length} runs</Badge>
                    </>
                  ) : historicalRun ? (
                    <>
                      <Badge>{historicalRun.experiment}</Badge>
                      <Badge>{historicalRun.eventFileCount} event files</Badge>
                    </>
                  ) : null}
                </>
              )}
            </div>
            <div className="mt-1 break-words font-mono text-xs text-ink-faint">{node.path}</div>
          </div>
          <div className="flex shrink-0 flex-wrap items-end gap-2">
            {hasComparisonCandidates && (
              <LabeledField
                label="Scope"
                id="monitor-comparison-scope"
                className="text-xs font-medium normal-case"
              >
                <SelectOnlyDropdown
                  id="monitor-comparison-scope"
                  label="Scope"
                  value={comparisonScope}
                  options={comparisonScopeSelectOptions}
                  onChange={(nextScope) =>
                    setComparisonScope(nextScope as LinearMonitorComparisonScope)
                  }
                  className="min-w-36"
                  triggerClassName="h-9 min-w-36 rounded-[10px] px-2 text-sm"
                />
              </LabeledField>
            )}
            {hasComparisonCandidates && (
              <LabeledField
                label="Compare"
                id="monitor-comparison-target"
                className="text-xs font-medium normal-case"
              >
                <SelectOnlyDropdown
                  id="monitor-comparison-target"
                  label="Compare"
                  value={comparisonPath}
                  options={comparisonPathOptions}
                  onChange={setComparisonPath}
                  className="min-w-44"
                  triggerClassName="h-9 min-w-44 rounded-[10px] px-2 text-sm"
                />
              </LabeledField>
            )}
            {activeJob && sourcePresets.length > 1 && (
              <LabeledField
                label="Preset"
                id="monitor-source-preset"
                className="text-xs font-medium normal-case"
              >
                <SelectOnlyDropdown
                  id="monitor-source-preset"
                  label="Preset"
                  value={preset}
                  options={sourcePresetOptions}
                  onChange={setPreset}
                  className="min-w-32"
                  triggerClassName="h-9 min-w-32 rounded-[10px] px-2 text-sm"
                />
              </LabeledField>
            )}
            <LabeledField
              label="Dataset"
              id="monitor-source-dataset"
              className="text-xs font-medium normal-case"
            >
              <SelectOnlyDropdown
                id="monitor-source-dataset"
                label="Dataset"
                value={dataset}
                options={sourceDatasetOptions}
                onChange={setDataset}
                className="min-w-32"
                triggerClassName="h-9 min-w-32 rounded-[10px] px-2 text-sm"
              />
            </LabeledField>
            <Button
              variant="secondary"
              onClick={refreshMonitorData}
              disabled={isFetching || monitorCount === 0}
              aria-label="Refresh monitor data"
            >
              <RefreshCw
                className={cn("h-4 w-4", isFetching && "animate-spin")}
                aria-hidden
              />
            </Button>
            <Button variant="ghost" onClick={onClose} aria-label="Close monitor charts">
              <X className="h-4 w-4" aria-hidden />
            </Button>
          </div>
        </div>

        <div className="min-h-0 overflow-y-auto bg-bg-2/80 p-4">
          {hasData && historicalRunGroup && (
            <div className="mb-3 grid gap-2">
              <HistoricalMonitorProgressRow
                label="Primary"
                progress={historicalProgress}
              />
              {isComparing && (
                <HistoricalMonitorProgressRow
                  label="Comparison"
                  progress={historicalComparisonProgress}
                />
              )}
            </div>
          )}
          {!hasData ? (
            <MonitorEmptyState
              title={emptyMessage.title}
              detail={emptyMessage.detail}
              busy={isLoading}
            />
          ) : historicalRunGroup && isComparing && comparisonNode && historicalData ? (
            <MultiRunComparisonCharts
              primaryNode={node}
              comparisonNode={comparisonNode}
              primaryResults={historicalData}
              comparisonResults={historicalComparisonData}
              comparisonLoading={comparisonLoading}
            />
          ) : historicalRunGroup && historicalData ? (
            <MultiRunMonitorCharts
              results={historicalData}
              idPrefix="historical-group-monitor"
            />
          ) : isComparing && comparisonNode ? (
            <ComparisonCharts
              primaryNode={node}
              comparisonNode={comparisonNode}
              primaryData={data}
              comparisonData={comparisonData}
              comparisonLoading={comparisonLoading}
            />
          ) : (
            data && <SingleNodeCharts data={data} />
          )}
        </div>
    </DialogShell>
  );
}
