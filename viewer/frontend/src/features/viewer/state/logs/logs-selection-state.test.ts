import { describe, expect, it } from "vitest";
import { type LogRun } from "@/lib/api";
import {
  addValueToInitializedSelection,
  addValuesToInitializedSelection,
  buildInitialExperimentSelection,
  buildInitialRunFacetSelection,
  buildInitialRunIdSelection,
  buildLogRunDeleteFilters,
  filterVisibleLogRuns,
  nextSelectedDetailRunId,
  pruneDeletedDetailRunId,
  removeStartedExperiment,
  removeValueFromSelection,
  removeValuesFromSelection,
  startedRunSelections,
} from "@/features/viewer/state/logs/logs-selection-state";

function logRun(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  const experiment = overrides.experiment ?? "exp_a";
  const dataset = overrides.dataset ?? "Mnist";
  const model = overrides.model ?? "linear";
  const preset = overrides.preset ?? "baseline";
  return {
    id: overrides.id,
    group: overrides.group ?? null,
    experiment,
    model,
    preset,
    dataset,
    runName: overrides.runName ?? overrides.id,
    timestamp: overrides.timestamp ?? null,
    version: overrides.version ?? "version_0",
    relativePath:
      overrides.relativePath ??
      `${experiment}/${model}/${preset}/${dataset}/${overrides.id}/version_0`,
    hasResult: overrides.hasResult ?? true,
    eventFileCount: overrides.eventFileCount ?? 1,
    checkpointCount: overrides.checkpointCount ?? 0,
    hasHparams: overrides.hasHparams ?? true,
    metrics: overrides.metrics ?? {},
  };
}

function values(set: Set<string> | null) {
  return set ? Array.from(set).sort() : null;
}

const runs = [
  logRun({ id: "run-a", experiment: "exp_a", dataset: "Mnist" }),
  logRun({ id: "run-b", experiment: "exp_b", dataset: "Cifar10" }),
  logRun({ id: "run-c", experiment: "exp_b", dataset: "Mnist", preset: "wide" }),
];

describe("logs selection state", () => {
  it("builds initial selections from loaded options and started experiments", () => {
    expect(
      values(
        buildInitialExperimentSelection({
          experimentOptions: [
            { value: "exp_a", label: "exp_a" },
            { value: "exp_b", label: "exp_b" },
          ],
          startedExperiments: new Set(["fresh_run"]),
        }),
      ),
    ).toEqual(["exp_a", "exp_b", "fresh_run"]);
    expect(values(buildInitialRunFacetSelection(runs, "dataset"))).toEqual([
      "Cifar10",
      "Mnist",
    ]);
    expect(values(buildInitialRunIdSelection(runs))).toEqual(["run-a", "run-b", "run-c"]);
  });

  it("adds started run facets only after selections have initialized", () => {
    const startedSelections = startedRunSelections({
      runs,
      startedExperiments: new Set(["exp_b"]),
    });

    expect(startedSelections.hasStartedRuns).toBe(true);
    expect(values(startedSelections.datasets)).toEqual(["Cifar10", "Mnist"]);
    expect(addValuesToInitializedSelection(null, startedSelections.datasets)).toBeNull();
    expect(values(addValuesToInitializedSelection(new Set(["Mnist"]), startedSelections.datasets)))
      .toEqual(["Cifar10", "Mnist"]);
    expect(values(addValueToInitializedSelection(new Set(["exp_a"]), "exp_b")))
      .toEqual(["exp_a", "exp_b"]);
  });

  it("filters visible runs and keeps detail selection valid", () => {
    const visibleRuns = filterVisibleLogRuns(runs, {
      experiments: new Set(["exp_b"]),
      datasets: new Set(["Mnist"]),
      models: new Set(["linear"]),
      presets: new Set(["wide"]),
      runIds: new Set(["run-a", "run-b", "run-c"]),
    });

    expect(visibleRuns.map((run) => run.id)).toEqual(["run-c"]);
    expect(nextSelectedDetailRunId("run-b", visibleRuns)).toBe("run-c");
    expect(nextSelectedDetailRunId("run-c", visibleRuns)).toBe("run-c");
    expect(nextSelectedDetailRunId("run-c", [])).toBeNull();
  });

  it("prunes deleted experiments, runs, detail state, and started experiments", () => {
    const deletedRunIds = new Set(["run-a", "run-c"]);

    expect(
      values(
        removeValueFromSelection({
          selection: null,
          fallbackValues: ["exp_a", "exp_b"],
          value: "exp_a",
        }),
      ),
    ).toEqual(["exp_b"]);
    expect(
      values(
        removeValuesFromSelection({
          selection: null,
          fallbackValues: runs.map((run) => run.id),
          values: deletedRunIds,
        }),
      ),
    ).toEqual(["run-b"]);
    expect(
      pruneDeletedDetailRunId({
        selectedDetailRunId: "run-c",
        deletedRunIds,
      }),
    ).toBeNull();
    expect(values(removeStartedExperiment(new Set(["exp_a", "exp_b"]), "exp_a")))
      .toEqual(["exp_b"]);
  });

  it("builds sorted delete filters from active selections", () => {
    expect(
      buildLogRunDeleteFilters({
        experiments: new Set(["exp_b", "exp_a"]),
        datasets: new Set(["Mnist"]),
        models: new Set(["linear"]),
        presets: new Set(["wide", "baseline"]),
        runIds: new Set(["run-c", "run-a"]),
      }),
    ).toEqual({
      experiments: ["exp_a", "exp_b"],
      datasets: ["Mnist"],
      models: ["linear"],
      presets: ["baseline", "wide"],
      runIds: ["run-a", "run-c"],
    });
  });
});
