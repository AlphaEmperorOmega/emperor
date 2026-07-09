import { type MonitorData } from "@/lib/api";
import {
  type ComparisonMonitorGroup,
  type HistogramData,
  type HistoricalMonitorRunData,
  type MetricPair,
  type MonitorGroup,
  type MonitorImageData,
  type MultiRunMonitorGroup,
  type MultiRunScalarMetric,
  type ScalarDomain,
  type ScalarSeries,
  type SingleMonitorGroup,
  monitorGroupOrder,
} from "@/types/monitor";

export function hasMonitorData(data: MonitorData | undefined) {
  return Boolean(
    (data?.scalarSeries.length ?? 0) +
      (data?.histograms.length ?? 0) +
      (data?.images.length ?? 0),
  );
}

export function hasHistoricalMonitorData(results: HistoricalMonitorRunData[] | undefined) {
  return Boolean(results?.some((result) => hasMonitorData(result.data)));
}

export function tagSuffix(tag: string, nodePath: string) {
  const prefix = `${nodePath}/`;
  return tag.startsWith(prefix) ? tag.slice(prefix.length) : tag;
}

export function createMonitorGroups<T>(factory: () => T): Record<MonitorGroup, T> {
  const groups = {} as Record<MonitorGroup, T>;
  for (const group of monitorGroupOrder) {
    groups[group] = factory();
  }
  return groups;
}

function normalizeSuffix(suffix: string) {
  return suffix.trim().replace(/^\/+/, "");
}

function suffixParts(suffix: string) {
  return normalizeSuffix(suffix).split("/").filter(Boolean);
}

function gradientGroupForSuffix(suffix: string): MonitorGroup | undefined {
  const parts = suffixParts(suffix);
  const channel = parts[0] ?? "";
  const metric = parts.at(-1) ?? "";
  const normalized = normalizeSuffix(suffix);

  if (channel === "weights" && (metric.startsWith("grad_") || metric === "update_ratio")) {
    return "Weight gradients";
  }
  if (channel === "bias" && metric.startsWith("grad_")) {
    return "Bias gradients";
  }
  if (metric.startsWith("grad_") || normalized.startsWith("grad_")) {
    return "Gradients";
  }
  return undefined;
}

export function semanticGroupForSuffix(
  suffix: string,
  dataKind: "scalar" | "histogram" | "image" = "scalar",
): MonitorGroup {
  if (dataKind !== "scalar") {
    return "Visual summaries";
  }

  const normalized = normalizeSuffix(suffix);
  const prefix = normalized.split("/")[0] ?? "";

  if (prefix === "histogram" || prefix === "heatmap") {
    return "Visual summaries";
  }
  const gradientGroup = gradientGroupForSuffix(normalized);
  if (gradientGroup) {
    return gradientGroup;
  }
  if (prefix === "input" || prefix === "output") {
    return "Activations";
  }
  if (prefix === "bias") {
    return "Bias";
  }
  if (prefix === "weights") {
    return "Weights";
  }
  if (prefix === "attention") {
    return "Attention";
  }
  if (prefix === "recurrent") {
    return "Recurrent";
  }
  if (
    prefix === "controller" ||
    prefix === "gate" ||
    prefix === "residual" ||
    prefix === "dropout" ||
    prefix === "layer_norm" ||
    prefix === "activation"
  ) {
    return "Controllers";
  }
  if (prefix === "parametric") {
    return "Parametric";
  }
  if (prefix === "router" || prefix === "mixture") {
    return "Routing";
  }
  return "Other";
}

export function groupSingleMonitorData(
  data: MonitorData,
): Record<MonitorGroup, SingleMonitorGroup> {
  const groups = createMonitorGroups<SingleMonitorGroup>(() => ({
    scalarSeries: [],
    histograms: [],
    images: [],
  }));

  for (const series of data.scalarSeries) {
    groups[semanticGroupForSuffix(series.label)].scalarSeries.push(series);
  }
  for (const histogram of data.histograms) {
    const group = semanticGroupForSuffix(
      tagSuffix(histogram.tag, data.nodePath),
      "histogram",
    );
    groups[group].histograms.push(histogram);
  }
  for (const image of data.images) {
    const group = semanticGroupForSuffix(tagSuffix(image.tag, data.nodePath), "image");
    groups[group].images.push(image);
  }

  return groups;
}

export function groupComparisonMonitorData({
  scalarPairs,
  histogramPairs,
  imagePairs,
  primaryNodePath,
  comparisonNodePath,
}: {
  scalarPairs: Array<MetricPair<ScalarSeries>>;
  histogramPairs: Array<MetricPair<HistogramData>>;
  imagePairs: Array<MetricPair<MonitorImageData>>;
  primaryNodePath: string;
  comparisonNodePath: string;
}): Record<MonitorGroup, ComparisonMonitorGroup> {
  const groups = createMonitorGroups<ComparisonMonitorGroup>(() => ({
    scalarPairs: [],
    histogramPairs: [],
    imagePairs: [],
  }));

  for (const pair of scalarPairs) {
    const suffix = pair.primary?.label ?? pair.comparison?.label ?? pair.key;
    groups[semanticGroupForSuffix(suffix)].scalarPairs.push(pair);
  }
  for (const pair of histogramPairs) {
    const suffix = pair.primary
      ? tagSuffix(pair.primary.tag, primaryNodePath)
      : pair.comparison
        ? tagSuffix(pair.comparison.tag, comparisonNodePath)
        : pair.key;
    groups[semanticGroupForSuffix(suffix, "histogram")].histogramPairs.push(pair);
  }
  for (const pair of imagePairs) {
    const suffix = pair.primary
      ? tagSuffix(pair.primary.tag, primaryNodePath)
      : pair.comparison
        ? tagSuffix(pair.comparison.tag, comparisonNodePath)
        : pair.key;
    groups[semanticGroupForSuffix(suffix, "image")].imagePairs.push(pair);
  }

  return groups;
}

export function groupMultiRunMonitorData(
  results: HistoricalMonitorRunData[],
): Record<MonitorGroup, MultiRunMonitorGroup> {
  const groups = createMonitorGroups<MultiRunMonitorGroup>(() => ({
    scalarMetrics: [],
    histograms: [],
    images: [],
  }));
  const scalarMetrics = new Map<string, MultiRunScalarMetric>();

  for (const result of results) {
    for (const series of result.data.scalarSeries) {
      const key = series.label;
      const metric = scalarMetrics.get(key) ?? {
        key,
        entries: [],
        missingRuns: [],
      };
      metric.entries.push({ run: result.run, series });
      scalarMetrics.set(key, metric);
    }

    for (const histogram of result.data.histograms) {
      const group = semanticGroupForSuffix(
        tagSuffix(histogram.tag, result.data.nodePath),
        "histogram",
      );
      groups[group].histograms.push({ run: result.run, item: histogram });
    }

    for (const image of result.data.images) {
      const group = semanticGroupForSuffix(
        tagSuffix(image.tag, result.data.nodePath),
        "image",
      );
      groups[group].images.push({ run: result.run, item: image });
    }
  }

  for (const metric of scalarMetrics.values()) {
    const runIdsWithMetric = new Set(metric.entries.map((entry) => entry.run.id));
    metric.missingRuns = results
      .map((result) => result.run)
      .filter((run) => !runIdsWithMetric.has(run.id));
    groups[semanticGroupForSuffix(metric.key)].scalarMetrics.push(metric);
  }

  return groups;
}

export function singleGroupCount(group: SingleMonitorGroup) {
  return group.scalarSeries.length + group.histograms.length + group.images.length;
}

export function comparisonGroupCount(group: ComparisonMonitorGroup) {
  return group.scalarPairs.length + group.histogramPairs.length + group.imagePairs.length;
}

export function multiRunGroupCount(group: MultiRunMonitorGroup) {
  return group.scalarMetrics.length + group.histograms.length + group.images.length;
}

export function countGroups<T>(
  groups: Record<MonitorGroup, T>,
  countForGroup: (group: T) => number,
) {
  const counts = {} as Record<MonitorGroup, number>;
  for (const group of monitorGroupOrder) {
    counts[group] = countForGroup(groups[group]);
  }
  return counts;
}

export function monitorGroupPanelId(idPrefix: string, group: MonitorGroup) {
  return `${idPrefix}-${group.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-panel`;
}

export function formatGroupCount(count: number, unit: "chart" | "pair") {
  return `${count} ${unit}${count === 1 ? "" : "s"}`;
}

export function pairMetrics<T>(
  primaryItems: T[],
  comparisonItems: T[],
  primaryKey: (item: T) => string,
  comparisonKey: (item: T) => string,
): Array<MetricPair<T>> {
  const pairs = new Map<string, MetricPair<T>>();

  for (const item of primaryItems) {
    const key = primaryKey(item);
    pairs.set(key, { key, primary: item });
  }
  for (const item of comparisonItems) {
    const key = comparisonKey(item);
    pairs.set(key, { ...(pairs.get(key) ?? { key }), comparison: item });
  }

  return [...pairs.values()];
}

export function scalarDomainFor(
  primary: ScalarSeries | undefined,
  comparison: ScalarSeries | undefined,
): ScalarDomain {
  let minStep = Infinity;
  let maxStep = -Infinity;
  let minValue = Infinity;
  let maxValue = -Infinity;
  let pointCount = 0;
  for (const series of [primary, comparison]) {
    for (const point of series?.points ?? []) {
      pointCount += 1;
      minStep = Math.min(minStep, point.step);
      maxStep = Math.max(maxStep, point.step);
      minValue = Math.min(minValue, point.value);
      maxValue = Math.max(maxValue, point.value);
    }
  }
  if (pointCount === 0) {
    return { minStep: 0, maxStep: 1, minValue: 0, maxValue: 1 };
  }

  return {
    minStep,
    maxStep,
    minValue,
    maxValue,
  };
}

export function histogramMaxCountFor(
  primary: HistogramData | undefined,
  comparison: HistogramData | undefined,
) {
  let maxCount = 1;
  for (const histogram of [primary, comparison]) {
    for (const bucket of histogram?.buckets ?? []) {
      maxCount = Math.max(maxCount, bucket.count);
    }
  }
  return maxCount;
}
