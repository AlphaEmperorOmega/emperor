import { type LogRun, type MonitorData, type TrainingJob } from "@/lib/api";

export type ActiveMonitorJob = Pick<
  TrainingJob,
  | "id"
  | "status"
  | "monitors"
  | "preset"
  | "presets"
  | "datasets"
  | "logFolder"
  | "currentPreset"
  | "currentDataset"
>;

/** Where a monitor-charts view sources its TensorBoard data from. */
export type MonitorChartsSource =
  | { kind: "active-job"; job: ActiveMonitorJob }
  | { kind: "historical-run"; run: LogRun }
  | {
      kind: "historical-run-group";
      runs: LogRun[];
      experiment: string;
      dataset: string;
      preset: string;
    };

export type ScalarSeries = MonitorData["scalarSeries"][number];
export type HistogramData = MonitorData["histograms"][number];
export type MonitorImageData = MonitorData["images"][number];

export type HistoricalMonitorRunData = {
  run: LogRun;
  data: MonitorData;
};

export type MonitorQueryData = MonitorData | HistoricalMonitorRunData[];

export type ScalarDomain = {
  minStep: number;
  maxStep: number;
  minValue: number;
  maxValue: number;
};

export type MetricPair<T> = {
  key: string;
  primary?: T;
  comparison?: T;
};

export const monitorGroupOrder = [
  "Activations",
  "Gradients",
  "Bias",
  "Weights",
  "Attention",
  "Recurrent",
  "Controllers",
  "Parametric",
  "Routing",
  "Visual summaries",
  "Other",
] as const;

export type MonitorGroup = (typeof monitorGroupOrder)[number];

export type SingleMonitorGroup = {
  scalarSeries: ScalarSeries[];
  histograms: HistogramData[];
  images: MonitorImageData[];
};

export type ComparisonMonitorGroup = {
  scalarPairs: Array<MetricPair<ScalarSeries>>;
  histogramPairs: Array<MetricPair<HistogramData>>;
  imagePairs: Array<MetricPair<MonitorImageData>>;
};

export type MultiRunScalarEntry = {
  run: LogRun;
  series: ScalarSeries;
};

export type MultiRunScalarMetric = {
  key: string;
  entries: MultiRunScalarEntry[];
  missingRuns: LogRun[];
};

export type MultiRunVisualEntry<T> = {
  run: LogRun;
  item: T;
};

export type MultiRunMonitorGroup = {
  scalarMetrics: MultiRunScalarMetric[];
  histograms: Array<MultiRunVisualEntry<HistogramData>>;
  images: Array<MultiRunVisualEntry<MonitorImageData>>;
};
