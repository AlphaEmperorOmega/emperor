import { describe, expect, it } from "vitest";
import {
  anyLogRunTagsMatchParameterNodePath,
  anyLogRunTagsMatchNodePath,
  experimentsWithLayerMonitorData,
  filterHistoricalRuns,
  groupModelLogRunsByExperiment,
  historicalDatasetOptions,
  historicalExperimentRunOptions,
  historicalMonitorRunGroups,
  historicalPresetOptions,
  latestHistoricalMonitorRuns,
  logRunHasLayerMonitorData,
  resolveRunPresetName,
  sortLogRunsNewestFirst,
} from "@/lib/historical-monitor-runs";
import type { LogRun } from "@/lib/api/logs";

function run(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  return {
    id: overrides.id,
    group: overrides.experiment ?? "exp_a",
    experiment: overrides.experiment ?? "exp_a",
    modelType: overrides.modelType ?? "linears",
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

function option(
  value: string,
  count: number,
  monitorEligibility: "checking" | "eligible" | "ineligible" = "checking",
) {
  const description =
    monitorEligibility === "eligible"
      ? "monitor data"
      : monitorEligibility === "ineligible"
        ? "no monitor data"
        : "monitor checking";
  return {
    value,
    label: value,
    count,
    monitorEligibility,
    description,
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
    expect(historicalExperimentRunOptions(runs)).toEqual([
      option("exp_a", 3),
      option("exp_b", 1),
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
      option("fast", 1),
      option("baseline", 2),
    ]);
  });

  it("resolves a logged preset to the configured preset name", () => {
    expect(
      resolveRunPresetName(run({ id: "fast", preset: "Fast" }), [
        { name: "baseline", label: "Baseline" },
        { name: "fast", label: "Fast" },
      ]),
    ).toBe("fast");
    expect(
      resolveRunPresetName(run({ id: "missing", preset: "Unknown" }), [
        { name: "baseline", label: "Baseline" },
      ]),
    ).toBe("");
  });

  it("derives dataset options from the selected experiment", () => {
    expect(historicalDatasetOptions(runs, "exp_a")).toEqual([
      option("Mnist", 2),
      option("FashionMnist", 1),
    ]);
  });

  it("filters runs by experiment, dataset, and preset", () => {
    const mixedPresetRuns = [
      ...runs,
      run({
        id: "fast-mnist",
        experiment: "exp_a",
        dataset: "Mnist",
        preset: "fast",
        timestamp: "2026-06-04 01:00:00",
      }),
    ];

    expect(
      filterHistoricalRuns(mixedPresetRuns, "exp_a", "Mnist", "BASELINE").map(
        (item) => item.id,
      ),
    ).toEqual(["new-mnist", "old-mnist"]);
    expect(filterHistoricalRuns(mixedPresetRuns, "exp_a", "Mnist").map((item) => item.id))
      .toEqual(["fast-mnist", "new-mnist", "old-mnist"]);
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

  it("builds latest monitor groups by experiment, dataset, and preset", () => {
    const groups = historicalMonitorRunGroups([
      run({
        id: "baseline-old",
        experiment: "exp_a",
        dataset: "Mnist",
        preset: "baseline",
        timestamp: "2026-06-01 01:00:00",
      }),
      run({
        id: "baseline-new",
        experiment: "exp_a",
        dataset: "Mnist",
        preset: "baseline",
        timestamp: "2026-06-03 01:00:00",
      }),
      run({
        id: "fast",
        experiment: "exp_a",
        dataset: "Mnist",
        preset: "fast",
        timestamp: "2026-06-02 01:00:00",
      }),
    ]);

    expect(
      groups.map((group) => ({
        experiment: group.experiment,
        dataset: group.dataset,
        preset: group.preset,
        runIds: group.runs.map((item) => item.id),
        cardRunIds: group.cardRunIds,
      })),
    ).toEqual([
      {
        experiment: "exp_a",
        dataset: "Mnist",
        preset: "baseline",
        runIds: ["baseline-new", "baseline-old"],
        cardRunIds: ["baseline-new", "baseline-old"],
      },
      {
        experiment: "exp_a",
        dataset: "Mnist",
        preset: "fast",
        runIds: ["fast"],
        cardRunIds: ["fast"],
      },
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
            textTags: [],
          },
          {
            runId: "run-b",
            scalarTags: [],
            histogramTags: ["main_model.layers.0.model/histogram/usage_fraction"],
            imageTags: [],
            textTags: [],
          },
        ],
        ["run-a", "run-b"],
        "main_model.layers.0.model",
      ),
    ).toBe(true);
  });

  it("rejects retired parameter-monitor path aliases", () => {
    expect(
      anyLogRunTagsMatchParameterNodePath(
        [
          {
            runId: "run-a",
            scalarTags: ["main_model.0.model/weights/mean"],
            histogramTags: [],
            imageTags: [],
            textTags: [],
          },
        ],
        ["run-a"],
        "main_model.layers.0.model",
      ),
    ).toBe(false);
  });

  it("detects strict parameter monitor data by tag shape", () => {
    expect(
      logRunHasLayerMonitorData({
        scalarTags: ["main_model.layers.0.model/weights/mean"],
        histogramTags: [],
        imageTags: [],
      }),
    ).toBe(true);
    expect(
      logRunHasLayerMonitorData({
        scalarTags: [
          "main_model.layers.0.model/weights/spectral_norm",
          "main_model.layers.0.model/weights/grad_norm",
        ],
        histogramTags: [],
        imageTags: [],
      }),
    ).toBe(true);
    expect(
      logRunHasLayerMonitorData({
        scalarTags: [],
        histogramTags: ["main_model.layers.0.model/histogram/usage"],
        imageTags: [],
      }),
    ).toBe(false);
    expect(
      logRunHasLayerMonitorData({
        scalarTags: [
          "epoch",
          "train/loss",
          "test/accuracy",
          "parameters/global_norm",
          "gradients/global_norm",
        ],
        histogramTags: [],
        imageTags: [],
      }),
    ).toBe(false);
    expect(
      logRunHasLayerMonitorData({
        scalarTags: [
          "train/confusion_matrix/class_0/class_1",
          "validation/per_class/class_0/accuracy",
        ],
        histogramTags: [],
        imageTags: ["validation/examples/predictions"],
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
        scalarTags: ["main_model.layers.0.model/weights/mean"],
        histogramTags: [],
        imageTags: [],
        textTags: [],
      },
      {
        runId: "perf-run",
        scalarTags: ["epoch", "train/loss", "train/confusion_matrix/0/1"],
        histogramTags: [],
        imageTags: ["validation/examples/predictions"],
        textTags: [],
      },
    ];

    expect([...experimentsWithLayerMonitorData(modelRuns, tags)]).toEqual([
      "exp_a",
    ]);
    // No tags loaded yet -> empty set so callers can show a loading state.
    expect(experimentsWithLayerMonitorData(modelRuns, undefined).size).toBe(0);
  });
});
