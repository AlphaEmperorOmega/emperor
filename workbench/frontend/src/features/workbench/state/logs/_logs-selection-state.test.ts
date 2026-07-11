import { describe, expect, it } from "vitest";
import { type LogRun, type LogRunTags } from "@/lib/api";
import {
  addValueToInitializedSelection,
  addValuesToInitializedSelection,
  buildCommonRunFacetOptions,
  buildExperimentScalarTagSeedSelection,
  buildInitialExperimentSelection,
  buildInitialRunFacetSelection,
  buildLogRunDeleteFilters,
  effectiveSelectionForAvailableValues,
  filterVisibleLogRuns,
  firstAvailableSelection,
  nextSelectedDetailRunId,
  normalizeRunFacetSelection,
  pruneSelectionToAvailableValues,
  pruneDeletedDetailRunId,
  removeStartedExperiment,
  removeValueFromSelection,
  removeValuesFromSelection,
  selectionSetOrDefault,
  startedRunSelections,
} from "@/features/workbench/state/logs/_logs-selection-state";

function logRun(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  const experiment = overrides.experiment ?? "exp_a";
  const dataset = overrides.dataset ?? "Mnist";
  const model = overrides.model ?? "linear";
  const preset = overrides.preset ?? "baseline";
  return {
    id: overrides.id,
    group: overrides.group ?? null,
    experiment,
    modelType: overrides.modelType ?? "linears",
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

function orderedValues(set: Set<string> | null) {
  return set ? Array.from(set) : null;
}

function logRunTags(runId: string, scalarTags: string[]): LogRunTags {
  return {
    runId,
    scalarTags,
    histogramTags: [],
    imageTags: [],
    textTags: [],
  };
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
  });

  it("adds started run facets only after selections have initialized", () => {
    const startedSelections = startedRunSelections({
      runs,
      startedExperiments: new Set(["exp_b"]),
    });

    expect(startedSelections.hasStartedRuns).toBe(true);
    expect(values(startedSelections.datasets)).toEqual(["Cifar10", "Mnist"]);
    expect(values(startedSelections.presets)).toEqual(["baseline", "wide"]);
    expect(addValuesToInitializedSelection(null, startedSelections.presets)).toBeNull();
    expect(
      values(addValuesToInitializedSelection(new Set(["baseline"]), startedSelections.presets)),
    ).toEqual(["baseline", "wide"]);
    expect(values(addValueToInitializedSelection(new Set(["exp_a"]), "exp_b")))
      .toEqual(["exp_a", "exp_b"]);
  });

  it("returns every lower facet value for one selected experiment", () => {
    const facetOptions = buildCommonRunFacetOptions({
      runs: [
        logRun({ id: "a-mnist", experiment: "exp_a", dataset: "Mnist" }),
        logRun({
          id: "a-cifar",
          experiment: "exp_a",
          dataset: "Cifar10",
          model: "wide-linear",
          preset: "WIDE",
        }),
        logRun({ id: "b-cifar", experiment: "exp_b", dataset: "Cifar10" }),
      ],
      selectedExperiments: new Set(["exp_a"]),
    });

    expect(facetOptions.datasets).toMatchObject([
      { value: "Cifar10", count: 1 },
      { value: "Mnist", count: 1 },
    ]);
    expect(facetOptions.models.map((option) => option.value)).toEqual([
      "linears/linear",
      "linears/wide-linear",
    ]);
    expect(facetOptions.presets.map((option) => option.value)).toEqual([
      "baseline",
      "WIDE",
    ]);
  });

  it("returns only lower facet values shared by every selected experiment", () => {
    const facetOptions = buildCommonRunFacetOptions({
      runs: [
        logRun({ id: "a-cifar", experiment: "exp_a", dataset: "Cifar10" }),
        logRun({
          id: "a-mnist",
          experiment: "exp_a",
          dataset: "Mnist",
          model: "wide-linear",
          preset: "WIDE",
        }),
        logRun({ id: "b-cifar", experiment: "exp_b", dataset: "Cifar10" }),
        logRun({
          id: "b-imagenet",
          experiment: "exp_b",
          dataset: "ImageNet",
          model: "conv",
          preset: "CONV",
        }),
      ],
      selectedExperiments: new Set(["exp_a", "exp_b"]),
    });

    expect(facetOptions.datasets).toEqual([
      { value: "Cifar10", label: "Cifar10", count: 2 },
    ]);
    expect(facetOptions.models.map((option) => ({
      value: option.value,
      count: option.count,
    }))).toEqual([{ value: "linears/linear", count: 2 }]);
    expect(facetOptions.presets).toEqual([
      { value: "baseline", label: "baseline", count: 2 },
    ]);
  });

  it("uses complete server facets when only the first run page is loaded", () => {
    const facetOptions = buildCommonRunFacetOptions({
      runs: [
        logRun({ id: "a-mnist", experiment: "exp_a", dataset: "Mnist" }),
      ],
      selectedExperiments: new Set(["exp_a"]),
      facets: {
        experiments: [
          {
            experiment: "exp_a",
            runCount: 105,
            datasets: [
              { value: "Mnist", count: 100 },
              { value: "ZebraSet", count: 5 },
            ],
            models: [
              { modelType: "linears", model: "linear", count: 100 },
              { modelType: "linears", model: "wide_linear", count: 5 },
            ],
            presets: [
              { value: "AAA_CONTROL", count: 100 },
              { value: "BASELINE", count: 5 },
            ],
          },
        ],
      },
    });

    expect(facetOptions.datasets).toEqual([
      { value: "Mnist", label: "Mnist", count: 100 },
      { value: "ZebraSet", label: "ZebraSet", count: 5 },
    ]);
    expect(facetOptions.models.map(({ value, count }) => ({ value, count })))
      .toEqual([
        { value: "linears/linear", count: 100 },
        { value: "linears/wide_linear", count: 5 },
      ]);
    expect(facetOptions.presets).toEqual([
      { value: "AAA_CONTROL", label: "AAA_CONTROL", count: 100 },
      { value: "BASELINE", label: "BASELINE", count: 5 },
    ]);
  });

  it("returns empty lower facets when a selected experiment has no runs", () => {
    const facetOptions = buildCommonRunFacetOptions({
      runs: [logRun({ id: "a-cifar", experiment: "exp_a", dataset: "Cifar10" })],
      selectedExperiments: new Set(["exp_a", "exp_empty"]),
    });

    expect(facetOptions.datasets).toEqual([]);
    expect(facetOptions.models).toEqual([]);
    expect(facetOptions.presets).toEqual([]);
  });

  it("prunes stale lower selections from visible-run filtering", () => {
    const selectedRuns = [
      logRun({ id: "a-cifar", experiment: "exp_a", dataset: "Cifar10" }),
      logRun({ id: "b-cifar", experiment: "exp_b", dataset: "Cifar10" }),
    ];
    const facetOptions = buildCommonRunFacetOptions({
      runs: selectedRuns,
      selectedExperiments: new Set(["exp_a", "exp_b"]),
    });

    const visibleRuns = filterVisibleLogRuns(selectedRuns, {
      experiments: new Set(["exp_a", "exp_b"]),
      datasets: effectiveSelectionForAvailableValues(
        new Set(["Mnist"]),
        facetOptions.datasets.map((option) => option.value),
      ),
      models: effectiveSelectionForAvailableValues(
        null,
        facetOptions.models.map((option) => option.value),
      ),
      presets: effectiveSelectionForAvailableValues(
        null,
        facetOptions.presets.map((option) => option.value),
      ),
    });

    expect(visibleRuns).toEqual([]);
  });

  it("replaces stale checked scalar tags for a pending experiment", () => {
    const kaggleRuns = [
      logRun({ id: "kaggle-a", experiment: "kaggle_linear_all" }),
      logRun({ id: "kaggle-b", experiment: "kaggle_linear_all" }),
    ];
    const seedSelection = buildExperimentScalarTagSeedSelection({
      visibleRuns: kaggleRuns,
      tagRuns: kaggleRuns.map((run) =>
        logRunTags(run.id, ["train/loss", "validation/accuracy"]),
      ),
      pendingExperiments: new Set(["kaggle_linear_all"]),
      selectedTags: new Set(["main_model.0.model/weights/mean"]),
      tagOptionValues: ["train/loss", "validation/accuracy"],
      selectAllLimit: 100,
    });

    expect(values(seedSelection.loadedExperiments)).toEqual(["kaggle_linear_all"]);
    expect(orderedValues(seedSelection.selectedTags)).toEqual([
      "train/loss",
      "validation/accuracy",
    ]);
  });

  it("preserves matching checked scalar tags for a pending experiment", () => {
    const selectedTags = new Set(["validation/accuracy"]);
    const seedSelection = buildExperimentScalarTagSeedSelection({
      visibleRuns: [logRun({ id: "kaggle", experiment: "kaggle_linear_all" })],
      tagRuns: [
        logRunTags("kaggle", ["train/loss", "validation/accuracy"]),
      ],
      pendingExperiments: new Set(["kaggle_linear_all"]),
      selectedTags,
      tagOptionValues: ["train/loss", "validation/accuracy"],
      selectAllLimit: 100,
    });

    expect(values(seedSelection.loadedExperiments)).toEqual(["kaggle_linear_all"]);
    expect(seedSelection.selectedTags).toBe(selectedTags);
  });

  it("leaves default scalar tag selection unset for a pending experiment", () => {
    const seedSelection = buildExperimentScalarTagSeedSelection({
      visibleRuns: [logRun({ id: "kaggle", experiment: "kaggle_linear_all" })],
      tagRuns: [
        logRunTags("kaggle", ["train/loss", "validation/accuracy"]),
      ],
      pendingExperiments: new Set(["kaggle_linear_all"]),
      selectedTags: null,
      tagOptionValues: ["train/loss", "validation/accuracy"],
      selectAllLimit: 100,
    });

    expect(values(seedSelection.loadedExperiments)).toEqual(["kaggle_linear_all"]);
    expect(seedSelection.selectedTags).toBeNull();
  });

  it("waits for visible run tag payloads before scalar tag seeding", () => {
    const selectedTags = new Set(["main_model.0.model/weights/mean"]);
    const seedSelection = buildExperimentScalarTagSeedSelection({
      visibleRuns: [
        logRun({ id: "kaggle-a", experiment: "kaggle_linear_all" }),
        logRun({ id: "kaggle-b", experiment: "kaggle_linear_all" }),
      ],
      tagRuns: [
        logRunTags("kaggle-a", ["train/loss", "validation/accuracy"]),
      ],
      pendingExperiments: new Set(["kaggle_linear_all"]),
      selectedTags,
      tagOptionValues: ["train/loss", "validation/accuracy"],
      selectAllLimit: 100,
    });

    expect(values(seedSelection.loadedExperiments)).toEqual([]);
    expect(seedSelection.selectedTags).toBe(selectedTags);
  });

  it("falls back to available scalar tags for non-standard pending experiments", () => {
    const seedSelection = buildExperimentScalarTagSeedSelection({
      visibleRuns: [logRun({ id: "custom", experiment: "custom_experiment" })],
      tagRuns: [
        logRunTags("custom", [
          "custom/first",
          "validation/confusion_matrix/true_class_0/predicted_class_0/rate",
          "custom/second",
        ]),
      ],
      pendingExperiments: new Set(["custom_experiment"]),
      selectedTags: new Set(["main_model.0.model/weights/mean"]),
      tagOptionValues: ["custom/first", "custom/second"],
      selectAllLimit: 1,
    });

    expect(values(seedSelection.loadedExperiments)).toEqual(["custom_experiment"]);
    expect(orderedValues(seedSelection.selectedTags)).toEqual(["custom/first"]);
  });

  it("filters visible runs by dataset and keeps detail selection valid", () => {
    const visibleRuns = filterVisibleLogRuns(runs, {
      experiments: new Set(["exp_b"]),
      datasets: new Set(["Mnist"]),
      models: new Set(["linears/linear"]),
      presets: new Set(["baseline", "wide"]),
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

  it("prunes manual scalar tag selections but preserves default fallback", () => {
    expect(
      values(
        pruneSelectionToAvailableValues(
          new Set(["train/loss", "validation/accuracy", "validation/kaggle_auc"]),
          ["validation/kaggle_auc", "train/kaggle_logloss"],
        ),
      ),
    ).toEqual(["validation/kaggle_auc"]);
    expect(pruneSelectionToAvailableValues(null, ["validation/kaggle_auc"]))
      .toBeNull();
    expect(
      values(
        selectionSetOrDefault(
          null,
          new Set(["train/kaggle_logloss", "validation/kaggle_auc"]),
        ),
      ),
    ).toEqual(["train/kaggle_logloss", "validation/kaggle_auc"]);
    expect(
      values(selectionSetOrDefault(new Set(["train/loss"]), new Set(["other"]))),
    ).toEqual(["train/loss"]);
  });

  it("selects only the first lower facet option after an experiment change", () => {
    expect(values(firstAvailableSelection(["Cifar10", "Mnist"]))).toEqual([
      "Cifar10",
    ]);
    expect(values(firstAvailableSelection([]))).toEqual([]);
    expect(
      values(
        normalizeRunFacetSelection({
          selection: null,
          availableValues: ["Cifar10", "Mnist"],
          selectFirstAvailable: true,
        }),
      ),
    ).toEqual(["Cifar10"]);
  });

  it("preserves manual lower facet selections until they become fully stale", () => {
    const manualSelection = new Set(["Mnist"]);

    expect(
      normalizeRunFacetSelection({
        selection: manualSelection,
        availableValues: ["Cifar10", "Mnist"],
        selectFirstAvailable: false,
      }),
    ).toBe(manualSelection);
    expect(
      values(
        normalizeRunFacetSelection({
          selection: new Set(["StaleDataset"]),
          availableValues: ["Cifar10", "Mnist"],
          selectFirstAvailable: false,
        }),
      ),
    ).toEqual(["Cifar10"]);
    expect(
      values(
        normalizeRunFacetSelection({
          selection: new Set(),
          availableValues: ["Cifar10", "Mnist"],
          selectFirstAvailable: false,
        }),
      ),
    ).toEqual([]);
  });

  it("builds sorted delete filters from active selections", () => {
    expect(
      buildLogRunDeleteFilters([runs[2], runs[0]]),
    ).toEqual({
      experiments: ["exp_a", "exp_b"],
      datasets: ["Mnist"],
      models: [{ modelType: "linears", model: "linear" }],
      presets: ["baseline", "wide"],
      runIds: ["run-a", "run-c"],
    });
  });
});
