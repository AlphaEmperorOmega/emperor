import { describe, expect, it } from "vitest";
import {
  anyLogRunTagsMatchNodePath,
  experimentsWithLayerMonitorData,
  filterHistoricalRuns,
  groupModelLogRunsByExperiment,
  historicalDatasetOptions,
  historicalExperimentOptions,
  historicalPresetOptions,
  latestHistoricalMonitorRuns,
  logRunHasLayerMonitorData,
  sortLogRunsNewestFirst,
} from "@/lib/historical-monitor-runs";
import { type LogRun } from "@/lib/api";

function run(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  return {
    id: overrides.id,
    group: overrides.experiment ?? "exp_a",
    experiment: overrides.experiment ?? "exp_a",
    model: overrides.model ?? "linear",
    preset: overrides.preset ?? "BASELINE",
    dataset: overrides.dataset ?? "Mnist",
    runName: overrides.runName ?? `${overrides.id}_20260601_010203`,
    timestamp: overrides.timestamp ?? "2026-06-01 01:02:03",
    version: overrides.version ?? "version_0",
    relativePath: overrides.relativePath ?? "exp_a/linear/BASELINE/Mnist/run/version_0",
    hasResult: overrides.hasResult ?? false,
    eventFileCount: overrides.eventFileCount ?? 1,
    checkpointCount: overrides.checkpointCount ?? 0,
    hasHparams: overrides.hasHparams ?? true,
    metrics: overrides.metrics ?? {},
  };
}

describe("historical monitor run helpers", () => {
  const runs = [
    run({
      id: "old-mnist",
      experiment: "exp_a",
      dataset: "Mnist",
      timestamp: "2026-06-01 01:00:00",
    }),
    run({
      id: "new-cifar",
      experiment: "exp_b",
      dataset: "Cifar10",
      timestamp: "2026-06-02 01:00:00",
    }),
    run({
      id: "new-mnist",
      experiment: "exp_a",
      dataset: "Mnist",
      timestamp: "2026-06-03 01:00:00",
    }),
    run({
      id: "fashion",
      experiment: "exp_a",
      dataset: "FashionMnist",
      timestamp: "2026-06-02 12:00:00",
    }),
  ];

  it("sorts newest runs first", () => {
    expect(sortLogRunsNewestFirst(runs).map((item) => item.id)).toEqual([
      "new-mnist",
      "fashion",
      "new-cifar",
      "old-mnist",
    ]);
  });

  it("groups and counts experiment folders in newest-run order", () => {
    expect(groupModelLogRunsByExperiment(runs).map((group) => group.experiment)).toEqual([
      "exp_a",
      "exp_b",
    ]);
    expect(historicalExperimentOptions(runs)).toEqual([
      { value: "exp_a", label: "exp_a", count: 3 },
      { value: "exp_b", label: "exp_b", count: 1 },
    ]);
  });

  it("counts distinct presets newest-run first", () => {
    expect(
      historicalPresetOptions([
        run({ id: "a", preset: "baseline", timestamp: "2026-06-01 01:00:00" }),
        run({ id: "b", preset: "fast", timestamp: "2026-06-03 01:00:00" }),
        run({ id: "c", preset: "baseline", timestamp: "2026-06-02 01:00:00" }),
      ]),
    ).toEqual([
      { value: "fast", label: "fast", count: 1 },
      { value: "baseline", label: "baseline", count: 2 },
    ]);
  });

  it("derives dataset options from the selected experiment", () => {
    expect(historicalDatasetOptions(runs, "exp_a")).toEqual([
      { value: "Mnist", label: "Mnist", count: 2 },
      { value: "FashionMnist", label: "FashionMnist", count: 1 },
    ]);
  });

  it("filters runs by experiment and dataset", () => {
    expect(filterHistoricalRuns(runs, "exp_a", "Mnist").map((item) => item.id)).toEqual([
      "new-mnist",
      "old-mnist",
    ]);
  });

  it("caps grouped monitor runs to the latest five", () => {
    const manyRuns = Array.from({ length: 7 }, (_, index) =>
      run({
        id: `run-${index + 1}`,
        timestamp: `2026-06-0${index + 1} 01:00:00`,
      }),
    );

    expect(latestHistoricalMonitorRuns(manyRuns).map((item) => item.id)).toEqual([
      "run-7",
      "run-6",
      "run-5",
      "run-4",
      "run-3",
    ]);
  });

  it("matches node-path tags across multiple runs", () => {
    expect(
      anyLogRunTagsMatchNodePath(
        [
          {
            runId: "run-a",
            scalarTags: ["other/output/mean"],
            histogramTags: [],
            imageTags: [],
          },
          {
            runId: "run-b",
            scalarTags: [],
            histogramTags: ["main_model.0.model/histogram/usage_fraction"],
            imageTags: [],
          },
        ],
        ["run-a", "run-b"],
        "main_model.0.model",
      ),
    ).toBe(true);
  });

  it("detects per-layer monitor data by tag shape", () => {
    // Layer scalar tag: node/group/stat (>= 2 slashes).
    expect(
      logRunHasLayerMonitorData({
        scalarTags: ["main_model.0.model/weights/mean"],
        histogramTags: [],
        imageTags: [],
      }),
    ).toBe(true);
    // Histogram/image tags always imply monitoring.
    expect(
      logRunHasLayerMonitorData({
        scalarTags: [],
        histogramTags: ["main_model.0.model/histogram/usage"],
        imageTags: [],
      }),
    ).toBe(true);
    // Flat model-performance metrics do not count.
    expect(
      logRunHasLayerMonitorData({
        scalarTags: ["epoch", "train/loss", "test/accuracy"],
        histogramTags: [],
        imageTags: [],
      }),
    ).toBe(false);
    expect(logRunHasLayerMonitorData(undefined)).toBe(false);
  });

  it("returns experiments that have any run with layer monitor data", () => {
    const modelRuns = [
      run({ id: "layer-run", experiment: "exp_a" }),
      run({ id: "perf-run", experiment: "exp_b" }),
    ];
    const tags = [
      {
        runId: "layer-run",
        scalarTags: ["main_model.0.model/weights/mean"],
        histogramTags: [],
        imageTags: [],
      },
      {
        runId: "perf-run",
        scalarTags: ["epoch", "train/loss"],
        histogramTags: [],
        imageTags: [],
      },
    ];

    expect([...experimentsWithLayerMonitorData(modelRuns, tags)]).toEqual([
      "exp_a",
    ]);
    // No tags loaded yet -> empty set so callers can show a loading state.
    expect(experimentsWithLayerMonitorData(modelRuns, undefined).size).toBe(0);
  });
});
