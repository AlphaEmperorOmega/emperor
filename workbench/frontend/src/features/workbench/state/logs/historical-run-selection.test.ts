import { describe, expect, it } from "vitest";
import {
  deriveDatasetSelectionState,
} from "@/features/workbench/state/logs/historical-run-selection";
import { type LogRun, type LogRunTags } from "@/lib/api";

function run(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  return {
    id: overrides.id,
    group: overrides.group ?? overrides.experiment ?? "exp",
    experiment: overrides.experiment ?? "exp",
    modelType: overrides.modelType ?? "linears",
    model: overrides.model ?? "linear",
    preset: overrides.preset ?? "baseline",
    dataset: overrides.dataset ?? "Mnist",
    runName: overrides.runName ?? `${overrides.id}_20260601_010203`,
    timestamp: overrides.timestamp ?? "2026-06-01 01:02:03",
    version: overrides.version ?? "version_0",
    relativePath: overrides.relativePath ?? "exp/linear/baseline/Mnist/run/version_0",
    hasResult: overrides.hasResult ?? false,
    eventFileCount: overrides.eventFileCount ?? 1,
    checkpointCount: overrides.checkpointCount ?? 0,
    hasHparams: overrides.hasHparams ?? true,
    hasLayerMonitorData: overrides.hasLayerMonitorData,
    metrics: overrides.metrics ?? {},
  };
}

function tags(runId: string, scalarTags: string[]): LogRunTags {
  return {
    runId,
    scalarTags,
    histogramTags: [],
    imageTags: [],
    textTags: [],
  };
}

function option(
  value: string,
  count: number,
  monitorEligibility: "checking" | "eligible" | "ineligible",
) {
  return {
    value,
    label: value,
    count,
    monitorEligibility,
    description:
      monitorEligibility === "eligible"
        ? "monitor data"
        : monitorEligibility === "ineligible"
          ? "no monitor data"
          : "monitor checking",
  };
}

describe("deriveDatasetSelectionState", () => {
  it("shows options from run metadata before tags resolve", () => {
    const state = deriveDatasetSelectionState({
      logRuns: [run({ id: "unknown-run" })],
      modelRunTags: undefined,
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedHistoricalExperimentFilter: "exp",
      selectedHistoricalDatasetFilter: "Mnist",
      selectedHistoricalPreset: "baseline",
      selectedLogRunId: null,
    });

    expect(state.historicalExperimentOptions).toEqual([
      option("exp", 1, "checking"),
    ]);
    expect(state.historicalDatasetOptions).toEqual([
      option("Mnist", 1, "checking"),
    ]);
    expect(state.historicalPresetOptions).toEqual([
      option("baseline", 1, "checking"),
    ]);
    expect(state.visibleHistoricalRuns.map((item) => item.id)).toEqual([
      "unknown-run",
    ]);
  });

  it("derives checking, eligible, and ineligible option states", () => {
    const state = deriveDatasetSelectionState({
      logRuns: [
        run({
          id: "eligible",
          preset: "eligible",
          timestamp: "2026-06-03 01:02:03",
          hasLayerMonitorData: true,
        }),
        run({
          id: "ineligible",
          preset: "ineligible",
          timestamp: "2026-06-02 01:02:03",
          hasLayerMonitorData: false,
        }),
        run({
          id: "checking",
          preset: "checking",
          timestamp: "2026-06-01 01:02:03",
          hasLayerMonitorData: null,
        }),
      ],
      modelRunTags: [
        tags("eligible", ["main_model.0.model/weights/mean"]),
        tags("ineligible", ["train/loss"]),
      ],
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedHistoricalExperimentFilter: "exp",
      selectedHistoricalDatasetFilter: "Mnist",
      selectedHistoricalPreset: "",
      selectedLogRunId: null,
    });

    expect(state.historicalPresetOptions).toEqual([
      option("eligible", 1, "eligible"),
      option("ineligible", 1, "ineligible"),
      option("checking", 1, "checking"),
    ]);
  });

  it("resolves an ineligible selected run from metadata", () => {
    const state = deriveDatasetSelectionState({
      logRuns: [run({ id: "perf-run", hasLayerMonitorData: false })],
      modelRunTags: [tags("perf-run", ["train/loss"])],
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedHistoricalPreset: "",
      selectedLogRunId: "perf-run",
    });

    expect(state.selectedLogRun?.id).toBe("perf-run");
    expect(state.selectedLogRunMonitorEligibility).toBe("ineligible");
    expect(state.selectedHistoricalExperiment).toBe("exp");
    expect(state.selectedHistoricalDataset).toBe("Mnist");
    expect(state.selectedHistoricalRunPreset).toBe("baseline");
  });

  it("excludes ineligible runs from historical monitor inputs", () => {
    const state = deriveDatasetSelectionState({
      logRuns: [
        run({
          id: "eligible-new",
          timestamp: "2026-06-02 01:02:03",
          hasLayerMonitorData: true,
        }),
        run({
          id: "ineligible",
          timestamp: "2026-06-03 01:02:03",
          hasLayerMonitorData: false,
        }),
        run({
          id: "eligible-old",
          timestamp: "2026-06-01 01:02:03",
          hasLayerMonitorData: true,
        }),
      ],
      modelRunTags: [
        tags("eligible-new", ["main_model.0.model/weights/mean"]),
        tags("ineligible", ["train/loss"]),
        tags("eligible-old", ["main_model.0.model/bias/mean"]),
      ],
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedHistoricalExperimentFilter: "exp",
      selectedHistoricalDatasetFilter: "Mnist",
      selectedHistoricalPreset: "baseline",
      selectedLogRunId: null,
    });

    expect(state.filteredHistoricalRuns.map((item) => item.id)).toEqual([
      "ineligible",
      "eligible-new",
      "eligible-old",
    ]);
    expect(state.historicalMonitorRuns.map((item) => item.id)).toEqual([
      "eligible-new",
      "eligible-old",
    ]);
    expect(state.filteredHistoricalRunIds).toEqual([
      "eligible-new",
      "eligible-old",
    ]);
  });
});
