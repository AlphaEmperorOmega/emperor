import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, RefreshCw, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  fetchLogRunMonitorData,
  fetchMonitorData,
  type GraphNode,
  type MonitorData,
} from "@/lib/api";
import {
  type LinearMonitorComparisonCandidateGroups,
  type LinearMonitorComparisonScope,
} from "@/lib/graph/monitor-targets";
import {
  type HistoricalMonitorRunData,
  type MonitorGroup,
  type MonitorQueryData,
  monitorGroupOrder,
} from "@/types/monitor";
import {
  comparisonGroupCount,
  countGroups,
  formatGroupCount,
  groupComparisonMonitorData,
  groupMultiRunMonitorData,
  groupSingleMonitorData,
  hasHistoricalMonitorData,
  hasMonitorData,
  histogramMaxCountFor,
  monitorGroupPanelId,
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
} from "@/components/features/viewer/monitor/monitor-charts";
import { cn } from "@/lib/utils";
import { type MonitorChartsSource } from "@/types/monitor";

type MonitorChartsModalProps = {
  node: GraphNode;
  source: MonitorChartsSource;
  comparisonCandidateGroups?: LinearMonitorComparisonCandidateGroups;
  onClose: () => void;
};

const runningStatuses = new Set(["running", "queued"]);
const emptyComparisonCandidateGroups: LinearMonitorComparisonCandidateGroups = {
  "same-stack": [],
  "all-layers": [],
};
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

function MonitorGroupAccordion({
  idPrefix,
  group,
  count,
  countUnit,
  isOpen,
  onToggle,
  children,
}: {
  idPrefix: string;
  group: MonitorGroup;
  count: number;
  countUnit: "chart" | "pair";
  isOpen: boolean;
  onToggle: () => void;
  children: ReactNode;
}) {
  const panelId = monitorGroupPanelId(idPrefix, group);

  return (
    <section className="edge overflow-hidden rounded-card">
      <h3>
        <button
          type="button"
          aria-expanded={isOpen}
          aria-controls={panelId}
          onClick={onToggle}
          className="flex w-full items-center justify-between gap-3 p-3 text-left transition hover:bg-white/[0.035] focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          <span className="flex min-w-0 items-center gap-2">
            <ChevronDown
              className={cn(
                "h-4 w-4 shrink-0 text-ink-faint transition-transform",
                !isOpen && "-rotate-90",
              )}
              aria-hidden
            />
            <span className="truncate text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
              {group}
            </span>
          </span>
          <Badge>{formatGroupCount(count, countUnit)}</Badge>
        </button>
      </h3>
      {isOpen && (
        <div id={panelId} className="border-t border-line-soft p-3">
          {children}
        </div>
      )}
    </section>
  );
}

function useMonitorGroupAccordion(groupCounts: Record<MonitorGroup, number>) {
  const firstNonEmptyGroup = monitorGroupOrder.find((group) => groupCounts[group] > 0);
  const [openGroups, setOpenGroups] = useState<MonitorGroup[]>(
    firstNonEmptyGroup ? [firstNonEmptyGroup] : [],
  );
  const countsKey = monitorGroupOrder
    .map((group) => `${group}:${groupCounts[group]}`)
    .join("|");
  const previousCountsKey = useRef<string | undefined>(undefined);

  useEffect(() => {
    if (previousCountsKey.current === countsKey) {
      return;
    }
    previousCountsKey.current = countsKey;
    setOpenGroups((currentGroups) => {
      const retainedGroups = currentGroups.filter((group) => groupCounts[group] > 0);
      if (retainedGroups.length > 0 || !firstNonEmptyGroup) {
        return retainedGroups;
      }
      return [firstNonEmptyGroup];
    });
  }, [countsKey, firstNonEmptyGroup, groupCounts]);

  return {
    isGroupOpen: (group: MonitorGroup) => openGroups.includes(group),
    toggleGroup: (group: MonitorGroup) =>
      setOpenGroups((currentGroups) =>
        currentGroups.includes(group)
          ? currentGroups.filter((currentGroup) => currentGroup !== group)
          : [...currentGroups, group],
      ),
  };
}

function SingleNodeCharts({ data }: { data: MonitorData }) {
  const groups = useMemo(() => groupSingleMonitorData(data), [data]);
  const groupCounts = useMemo(() => countGroups(groups, singleGroupCount), [groups]);
  const { isGroupOpen, toggleGroup } = useMonitorGroupAccordion(groupCounts);

  return (
    <div className="grid gap-3">
      {monitorGroupOrder.map((group) => {
        const chartCount = groupCounts[group];
        if (chartCount === 0) {
          return null;
        }
        const groupData = groups[group];
        const isOpen = isGroupOpen(group);
        const charts = [
          ...groupData.scalarSeries.map((series) => (
            <ScalarChart key={series.tag} series={series} />
          )),
          ...groupData.histograms.map((histogram) => (
            <HistogramChart key={histogram.tag} histogram={histogram} />
          )),
          ...groupData.images.map((image) => (
            <MonitorImage key={image.tag} image={image} />
          )),
        ];

        return (
          <MonitorGroupAccordion
            key={group}
            idPrefix="single-monitor"
            group={group}
            count={chartCount}
            countUnit="chart"
            isOpen={isOpen}
            onToggle={() => toggleGroup(group)}
          >
            <div className="grid md:grid-cols-2">
              {charts.map((chart, index) => (
                <div key={chart.key ?? index} className={chartGridItemClass(index)}>
                  {chart}
                </div>
              ))}
            </div>
          </MonitorGroupAccordion>
        );
      })}
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
  const { isGroupOpen, toggleGroup } = useMonitorGroupAccordion(groupCounts);

  return (
    <div className="grid gap-3">
      {monitorGroupOrder.map((group) => {
        const chartCount = groupCounts[group];
        if (chartCount === 0) {
          return null;
        }
        const groupData = groups[group];
        const isOpen = isGroupOpen(group);
        const charts = [
          ...groupData.scalarMetrics.map((metric) => (
            <MultiRunScalarChart key={`scalar-${metric.key}`} metric={metric} />
          )),
          ...groupData.histograms.map((entry) => (
            <RunVisualCard
              key={`histogram-${entry.run.id}-${entry.item.tag}`}
              run={entry.run}
            >
              <HistogramChart histogram={entry.item} />
            </RunVisualCard>
          )),
          ...groupData.images.map((entry) => (
            <RunVisualCard key={`image-${entry.run.id}-${entry.item.tag}`} run={entry.run}>
              <MonitorImage image={entry.item} />
            </RunVisualCard>
          )),
        ];

        return (
          <MonitorGroupAccordion
            key={group}
            idPrefix={idPrefix}
            group={group}
            count={chartCount}
            countUnit="chart"
            isOpen={isOpen}
            onToggle={() => toggleGroup(group)}
          >
            <div className="grid md:grid-cols-2">
              {charts.map((chart, index) => (
                <div key={chart.key ?? index} className={chartGridItemClass(index)}>
                  {chart}
                </div>
              ))}
            </div>
          </MonitorGroupAccordion>
        );
      })}
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
  const { isGroupOpen, toggleGroup } = useMonitorGroupAccordion(groupCounts);

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
        {monitorGroupOrder.map((group) => {
          const pairCount = groupCounts[group];
          if (pairCount === 0) {
            return null;
          }
          const groupData = groups[group];
          const isOpen = isGroupOpen(group);
          const histogramPairOffset = groupData.scalarPairs.length;
          const imagePairOffset =
            groupData.scalarPairs.length + groupData.histogramPairs.length;

          return (
            <MonitorGroupAccordion
              key={group}
              idPrefix="comparison-monitor"
              group={group}
              count={pairCount}
              countUnit="pair"
              isOpen={isOpen}
              onToggle={() => toggleGroup(group)}
            >
              <div>
                {groupData.scalarPairs.map((pair, index) => {
                  const domain = scalarDomainFor(pair.primary, pair.comparison);
                  return (
                    <div key={pair.key} className={comparisonPairRowClass(index)}>
                      <div className={comparisonPairCellClass(0)}>
                        {pair.primary ? (
                          <ScalarChart series={pair.primary} domain={domain} />
                        ) : (
                          <MissingMetricCard metric={pair.key} nodePath={primaryNode.path} />
                        )}
                      </div>
                      <div className={comparisonPairCellClass(1)}>
                        {pair.comparison ? (
                          <ScalarChart series={pair.comparison} domain={domain} />
                        ) : (
                          <MissingMetricCard
                            metric={pair.key}
                            nodePath={comparisonNode.path}
                            busy={comparisonLoading}
                          />
                        )}
                      </div>
                    </div>
                  );
                })}
                {groupData.histogramPairs.map((pair, index) => {
                  const maxCount = histogramMaxCountFor(pair.primary, pair.comparison);
                  const rowIndex = histogramPairOffset + index;
                  return (
                    <div key={pair.key} className={comparisonPairRowClass(rowIndex)}>
                      <div className={comparisonPairCellClass(0)}>
                        {pair.primary ? (
                          <HistogramChart histogram={pair.primary} maxCount={maxCount} />
                        ) : (
                          <MissingMetricCard metric={pair.key} nodePath={primaryNode.path} />
                        )}
                      </div>
                      <div className={comparisonPairCellClass(1)}>
                        {pair.comparison ? (
                          <HistogramChart histogram={pair.comparison} maxCount={maxCount} />
                        ) : (
                          <MissingMetricCard
                            metric={pair.key}
                            nodePath={comparisonNode.path}
                            busy={comparisonLoading}
                          />
                        )}
                      </div>
                    </div>
                  );
                })}
                {groupData.imagePairs.map((pair, index) => (
                  <div
                    key={pair.key}
                    className={comparisonPairRowClass(imagePairOffset + index)}
                  >
                    <div className={comparisonPairCellClass(0)}>
                      {pair.primary ? (
                        <MonitorImage image={pair.primary} />
                      ) : (
                        <MissingMetricCard metric={pair.key} nodePath={primaryNode.path} />
                      )}
                    </div>
                    <div className={comparisonPairCellClass(1)}>
                      {pair.comparison ? (
                        <MonitorImage image={pair.comparison} />
                      ) : (
                        <MissingMetricCard
                          metric={pair.key}
                          nodePath={comparisonNode.path}
                          busy={comparisonLoading}
                        />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </MonitorGroupAccordion>
          );
        })}
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
              title={comparisonLoading ? "Loading comparison data" : "No comparison data"}
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
  comparisonCandidateGroups = emptyComparisonCandidateGroups,
  onClose,
}: MonitorChartsModalProps) {
  const activeJob = source.kind === "active-job" ? source.job : undefined;
  const historicalRun = source.kind === "historical-run" ? source.run : undefined;
  const historicalRunGroup =
    source.kind === "historical-run-group" ? source : undefined;
  const historicalRuns = useMemo(
    () => historicalRunGroup?.runs ?? (historicalRun ? [historicalRun] : []),
    [historicalRun, historicalRunGroup?.runs],
  );
  const historicalRunIds = useMemo(
    () => historicalRuns.map((run) => run.id),
    [historicalRuns],
  );
  const historicalExperiment = historicalRunGroup?.experiment ?? historicalRun?.experiment ?? "";
  const historicalDataset = historicalRunGroup?.dataset ?? historicalRun?.dataset ?? "";
  const defaultDataset = activeJob?.currentDataset ?? activeJob?.datasets[0] ?? historicalDataset;
  const defaultPreset = activeJob?.currentPreset ?? activeJob?.preset ?? "";
  const defaultComparisonScope: LinearMonitorComparisonScope =
    comparisonCandidateGroups["same-stack"].length > 0 ? "same-stack" : "all-layers";
  const [dataset, setDataset] = useState(defaultDataset);
  const [preset, setPreset] = useState(defaultPreset);
  const [comparisonScope, setComparisonScope] =
    useState<LinearMonitorComparisonScope>(defaultComparisonScope);
  const [comparisonPath, setComparisonPath] = useState("");
  const isRunning = activeJob ? runningStatuses.has(activeJob.status) : false;
  const sourceDatasets = useMemo(
    () => activeJob?.datasets ?? (historicalDataset ? [historicalDataset] : []),
    [activeJob?.datasets, historicalDataset],
  );
  const sourcePresets = useMemo(
    () => activeJob?.presets ?? (activeJob?.preset ? [activeJob.preset] : []),
    [activeJob?.preset, activeJob?.presets],
  );
  const monitorCount = activeJob?.monitors.length;
  const comparisonCandidates = comparisonCandidateGroups[comparisonScope];
  const hasComparisonCandidates =
    comparisonCandidateGroups["same-stack"].length > 0 ||
    comparisonCandidateGroups["all-layers"].length > 0;
  const comparisonNode = useMemo(
    () => comparisonCandidates.find((candidate) => candidate.path === comparisonPath),
    [comparisonCandidates, comparisonPath],
  );
  const comparisonCandidatePaths = useMemo(
    () => new Set(comparisonCandidates.map((candidate) => candidate.path)),
    [comparisonCandidates],
  );

  useEffect(() => {
    if (dataset && sourceDatasets.includes(dataset)) {
      return;
    }
    setDataset(defaultDataset);
  }, [dataset, defaultDataset, sourceDatasets]);

  useEffect(() => {
    if (!activeJob) {
      return;
    }
    if (preset && sourcePresets.includes(preset)) {
      return;
    }
    setPreset(defaultPreset);
  }, [activeJob, defaultPreset, preset, sourcePresets]);

  useEffect(() => {
    setComparisonScope((currentScope) =>
      comparisonCandidateGroups[currentScope].length > 0
        ? currentScope
        : defaultComparisonScope,
    );
  }, [comparisonCandidateGroups, defaultComparisonScope]);

  useEffect(() => {
    if (!comparisonPath) {
      return;
    }
    if (comparisonCandidatePaths.has(comparisonPath)) {
      return;
    }
    setComparisonPath("");
  }, [comparisonCandidatePaths, comparisonPath]);

  const monitorQuery = useQuery<MonitorQueryData>({
    queryKey: activeJob
      ? ["monitor-data", "active-job", activeJob.id, node.path, preset, dataset]
      : historicalRunGroup
        ? ["monitor-data", "historical-run-group", historicalRunIds, node.path]
        : ["monitor-data", "historical-run", historicalRun?.id, node.path],
    queryFn: () => {
      if (activeJob) {
        return fetchMonitorData({
          jobId: activeJob.id,
          nodePath: node.path,
          preset: preset || undefined,
          dataset: dataset || undefined,
        });
      }
      if (historicalRunGroup) {
        return Promise.all(
          historicalRuns.map(async (run) => ({
            run,
            data: await fetchLogRunMonitorData({
              runId: run.id,
              nodePath: node.path,
            }),
          })),
        );
      }
      return fetchLogRunMonitorData({
        runId: historicalRun?.id ?? "",
        nodePath: node.path,
      });
    },
    enabled: activeJob
      ? activeJob.monitors.length > 0
      : historicalRunGroup
        ? historicalRuns.length > 0
        : Boolean(historicalRun),
    retry: false,
    refetchInterval: isRunning ? 1500 : false,
  });

  const comparisonQuery = useQuery<MonitorQueryData>({
    queryKey: activeJob
      ? ["monitor-data", "active-job", activeJob.id, comparisonNode?.path, preset, dataset]
      : historicalRunGroup
        ? [
            "monitor-data",
            "historical-run-group",
            historicalRunIds,
            comparisonNode?.path,
          ]
        : ["monitor-data", "historical-run", historicalRun?.id, comparisonNode?.path],
    queryFn: () => {
      if (!comparisonNode) {
        throw new Error("No comparison node selected");
      }
      if (activeJob) {
        return fetchMonitorData({
          jobId: activeJob.id,
          nodePath: comparisonNode.path,
          preset: preset || undefined,
          dataset: dataset || undefined,
        });
      }
      if (historicalRunGroup) {
        return Promise.all(
          historicalRuns.map(async (run) => ({
            run,
            data: await fetchLogRunMonitorData({
              runId: run.id,
              nodePath: comparisonNode.path,
            }),
          })),
        );
      }
      return fetchLogRunMonitorData({
        runId: historicalRun?.id ?? "",
        nodePath: comparisonNode.path,
      });
    },
    enabled:
      Boolean(comparisonNode) &&
      (activeJob
        ? activeJob.monitors.length > 0
        : historicalRunGroup
          ? historicalRuns.length > 0
          : Boolean(historicalRun)),
    retry: false,
    refetchInterval: isRunning ? 1500 : false,
  });

  const monitorQueryData = monitorQuery.data;
  const comparisonQueryData = comparisonQuery.data;
  const historicalData: HistoricalMonitorRunData[] | undefined = Array.isArray(
    monitorQueryData,
  )
    ? monitorQueryData
    : undefined;
  const historicalComparisonData: HistoricalMonitorRunData[] | undefined =
    Array.isArray(comparisonQueryData) ? comparisonQueryData : undefined;
  const data: MonitorData | undefined = Array.isArray(monitorQueryData)
    ? undefined
    : monitorQueryData;
  const comparisonData: MonitorData | undefined = Array.isArray(comparisonQueryData)
    ? undefined
    : comparisonQueryData;
  const isComparing = Boolean(comparisonNode);
  const hasData = historicalRunGroup
    ? isComparing
      ? hasHistoricalMonitorData(historicalData) ||
        hasHistoricalMonitorData(historicalComparisonData)
      : hasHistoricalMonitorData(historicalData)
    : isComparing
      ? hasMonitorData(data) || hasMonitorData(comparisonData)
      : hasMonitorData(data);
  const isFetching = monitorQuery.isFetching || (isComparing && comparisonQuery.isFetching);
  const emptyMessage = useMemo(() => {
    if (monitorCount === 0) {
      return {
        title: "No monitor selected",
        detail: "Start a training job with at least one optional monitor enabled.",
      };
    }
    if (monitorQuery.isLoading || (isComparing && comparisonQuery.isLoading)) {
      return {
        title: "Loading monitor data",
        detail: "Reading TensorBoard event files for the selected node.",
      };
    }
    if (isRunning) {
      return {
        title: "No data yet",
        detail: "The job is running, but this node has not emitted matching monitor tags yet.",
      };
    }
    if (historicalRunGroup) {
      return {
        title: "No tags for this node",
        detail:
          "No scalar, histogram, or image tags matched this node path in the latest filtered historical runs.",
      };
    }
    if (historicalRun) {
      return {
        title: "No tags for this node",
        detail: "No scalar, histogram, or image tags matched this node path in the selected historical run.",
      };
    }
    return {
      title: "No tags for this node",
      detail: "No scalar, histogram, or image tags matched this node path in the selected run.",
    };
  }, [
    comparisonQuery.isLoading,
    historicalRunGroup,
    historicalRun,
    isComparing,
    isRunning,
    monitorCount,
    monitorQuery.isLoading,
  ]);

  const refreshMonitorData = () => {
    void monitorQuery.refetch();
    if (isComparing) {
      void comparisonQuery.refetch();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 grid bg-black/70 p-3 backdrop-blur-sm sm:p-6"
      role="dialog"
      aria-modal="true"
      aria-label="Monitor charts"
    >
      <div className="edge grid max-h-full min-h-0 w-full max-w-5xl grid-rows-[auto_minmax(0,1fr)] justify-self-center overflow-hidden rounded-card shadow-[0_24px_80px_rgba(0,0,0,0.58)]">
        <div className="grid gap-3 border-b border-line bg-panel/85 p-4 backdrop-blur sm:grid-cols-[minmax(0,1fr)_auto]">
          <div className="min-w-0">
            <div className="flex min-w-0 flex-wrap items-center gap-2">
              <h2 className="text-base font-semibold text-ink">Monitor charts</h2>
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
              <label className="grid gap-1 text-xs font-medium text-ink-dim">
                Scope
                <select
                  value={comparisonScope}
                  onChange={(event) =>
                    setComparisonScope(event.target.value as LinearMonitorComparisonScope)
                  }
                  className="h-9 min-w-36 rounded-[10px] border border-line bg-black/25 px-2 text-sm text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                >
                  {comparisonScopeOptions.map((option) => (
                    <option
                      key={option.value}
                      value={option.value}
                      disabled={comparisonCandidateGroups[option.value].length === 0}
                    >
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {hasComparisonCandidates && (
              <label className="grid gap-1 text-xs font-medium text-ink-dim">
                Compare
                <select
                  value={comparisonPath}
                  onChange={(event) => setComparisonPath(event.target.value)}
                  className="h-9 min-w-44 rounded-[10px] border border-line bg-black/25 px-2 text-sm text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                >
                  <option value="">No comparison</option>
                  {comparisonCandidates.map((candidate) => (
                    <option key={candidate.path} value={candidate.path}>
                      {candidate.path}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {activeJob && sourcePresets.length > 1 && (
              <label className="grid gap-1 text-xs font-medium text-ink-dim">
                Preset
                <select
                  value={preset}
                  onChange={(event) => setPreset(event.target.value)}
                  className="h-9 min-w-32 rounded-[10px] border border-line bg-black/25 px-2 text-sm text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                >
                  {sourcePresets.map((sourcePreset) => (
                    <option key={sourcePreset} value={sourcePreset}>
                      {sourcePreset}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label className="grid gap-1 text-xs font-medium text-ink-dim">
              Dataset
              <select
                value={dataset}
                onChange={(event) => setDataset(event.target.value)}
                className="h-9 min-w-32 rounded-[10px] border border-line bg-black/25 px-2 text-sm text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                {sourceDatasets.map((sourceDataset) => (
                  <option key={sourceDataset} value={sourceDataset}>
                    {sourceDataset}
                  </option>
                ))}
              </select>
            </label>
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
          {!hasData ? (
            <MonitorEmptyState
              title={emptyMessage.title}
              detail={emptyMessage.detail}
              busy={monitorQuery.isLoading || (isComparing && comparisonQuery.isLoading)}
            />
          ) : historicalRunGroup && isComparing && comparisonNode && historicalData ? (
            <MultiRunComparisonCharts
              primaryNode={node}
              comparisonNode={comparisonNode}
              primaryResults={historicalData}
              comparisonResults={historicalComparisonData}
              comparisonLoading={comparisonQuery.isLoading || comparisonQuery.isFetching}
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
              comparisonLoading={comparisonQuery.isLoading || comparisonQuery.isFetching}
            />
          ) : (
            data && <SingleNodeCharts data={data} />
          )}
        </div>
      </div>
    </div>
  );
}
