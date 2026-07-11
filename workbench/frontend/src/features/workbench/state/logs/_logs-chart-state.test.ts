import { describe, expect, it } from "vitest";
import {
  bestRunMetricGroupForActiveScalarQuery,
  defaultLogBestRunMetricTag,
  deriveLogMetricGroupScalarQueryStates,
  deriveLogsChartEmptyState,
  groupLogScalarSeriesByTag,
  groupSelectedLogMetrics,
} from "@/features/workbench/state/logs/_logs-chart-state";
import {
  buildLogScalarChunkQueryInputs,
  buildLogScalarQueryInput,
  chunkScalarTagsForQueries,
} from "@/features/workbench/state/logs/_logs-scalar-query-plan";
import {
  buildLogScalarTagOptions,
  buildTrainValidationScalarPairs,
  defaultTrainValidationScalarPairSuffixes,
  isDefaultScalarTag,
  groupLogPlotSelectorTags,
  isLogPlotSelectorScalarTag,
} from "@/features/workbench/state/logs/logs-selectors";
import { buildConfusionMatrixHeatmaps } from "@/features/workbench/state/logs/log-diagnostics";
import { type LogRun, type LogScalarSeries } from "@/lib/api";

function scalarSeries({
  points,
  runId,
  tag,
}: Partial<LogScalarSeries> & Pick<LogScalarSeries, "runId" | "tag">): LogScalarSeries {
  return {
    runId,
    tag,
    points: points ?? [{ step: 1, wallTime: 100, value: 0.25 }],
  };
}

function logRun(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  const experiment = overrides.experiment ?? "exp_a";
  const dataset = overrides.dataset ?? "Cifar10";
  const modelType = overrides.modelType ?? "linears";
  const model = overrides.model ?? "linear";
  const preset = overrides.preset ?? "baseline";
  return {
    id: overrides.id,
    group: overrides.group ?? null,
    experiment,
    modelType,
    model,
    preset,
    dataset,
    runName: overrides.runName ?? overrides.id,
    timestamp: overrides.timestamp ?? null,
    version: overrides.version ?? "version_0",
    relativePath:
      overrides.relativePath ??
      `${experiment}/${modelType}/${model}/${preset}/${dataset}/${overrides.id}/version_0`,
    hasResult: overrides.hasResult ?? true,
    eventFileCount: overrides.eventFileCount ?? 1,
    checkpointCount: overrides.checkpointCount ?? 0,
    hasHparams: overrides.hasHparams ?? true,
    metrics: overrides.metrics ?? {},
  };
}

describe("logs chart view model", () => {
  it("builds scalar query input from visible runs and selected tags", () => {
    const runIds = ["run-1", "run-2"];
    const tags = ["train/loss", "validation/accuracy"];

    const input = buildLogScalarQueryInput({
      enabled: true,
      selectedTagList: tags,
      visibleRunIds: runIds,
    });

    expect(input).toMatchObject({
      runIds,
      tags,
      enabled: true,
    });
    expect(input.queryKey).toEqual([
      "log-scalars",
      runIds,
      tags,
      { maxPoints: 500, sampling: "tail" },
    ]);
    expect(input.queryKey[1]).not.toBe(runIds);
    expect(input.queryKey[2]).not.toBe(tags);
  });

  it("disables scalar requests without a visible run, selected tag, or enabled workspace", () => {
    expect(
      buildLogScalarQueryInput({
        enabled: true,
        selectedTagList: [],
        visibleRunIds: ["run-1"],
      }).enabled,
    ).toBe(false);
    expect(
      buildLogScalarQueryInput({
        enabled: true,
        selectedTagList: ["train/loss"],
        visibleRunIds: [],
      }).enabled,
    ).toBe(false);
    expect(
      buildLogScalarQueryInput({
        enabled: false,
        selectedTagList: ["train/loss"],
        visibleRunIds: ["run-1"],
      }).enabled,
    ).toBe(false);
  });

  it("groups non-empty scalar series by tag", () => {
    const seriesByTag = groupLogScalarSeriesByTag([
      scalarSeries({ runId: "run-1", tag: "train/loss" }),
      scalarSeries({ runId: "run-2", tag: "train/loss" }),
      scalarSeries({ runId: "run-1", tag: "validation/loss", points: [] }),
      scalarSeries({ runId: "run-1", tag: "test/accuracy" }),
    ]);

    expect(seriesByTag.get("train/loss")?.map((series) => series.runId)).toEqual([
      "run-1",
      "run-2",
    ]);
    expect(seriesByTag.has("validation/loss")).toBe(false);
    expect(seriesByTag.get("test/accuracy")?.map((series) => series.runId)).toEqual([
      "run-1",
    ]);
  });

  it("groups selected log metrics even before scalar series load", () => {
    const seriesByTag = groupLogScalarSeriesByTag([
      scalarSeries({ runId: "run-1", tag: "train/loss" }),
    ]);

    const groups = groupSelectedLogMetrics({
      selectedTagList: ["train/loss", "validation/accuracy", "custom/metric"],
      seriesByTag,
    });

    expect(groups.train).toEqual([
      { tag: "train/loss", series: seriesByTag.get("train/loss") },
    ]);
    expect(groups.validation).toEqual([
      { tag: "validation/accuracy", series: [] },
    ]);
    expect(groups.other).toEqual([{ tag: "custom/metric", series: [] }]);
  });

  it("builds stable scalar tag chunks for near-visible charts", () => {
    expect(
      chunkScalarTagsForQueries(
        ["a", "b", "a", "c", "d", "e", "f", "g"],
        3,
      ),
    ).toEqual([
      ["a", "b", "c"],
      ["d", "e", "f"],
      ["g"],
    ]);

    expect(
      buildLogScalarChunkQueryInputs({
        enabled: true,
        group: "train",
        requestedTags: new Set(["train/loss", "train/accuracy"]),
        selectedTagList: ["train/loss", "train/accuracy", "train/calibration/ece"],
        visibleRunIds: ["run-1"],
      }),
    ).toEqual([
      {
        runIds: ["run-1"],
        tags: ["train/loss", "train/accuracy"],
        enabled: true,
        group: "train",
        queryKey: [
          "log-scalars",
          ["run-1"],
          ["train/accuracy", "train/loss"],
          { group: "train", maxPoints: 500, sampling: "tail" },
        ],
      },
    ]);

    expect(
      buildLogScalarChunkQueryInputs({
        enabled: true,
        group: "train",
        requestedTags: new Set(),
        selectedTagList: ["train/loss", "train/accuracy"],
        visibleRunIds: ["run-1"],
      }),
    ).toEqual([]);
  });

  it("splits scalar chunk queries across run and tag dimensions", () => {
    const runIds = Array.from({ length: 24 }, (_, index) => `run-${index}`);
    const selectedTags = Array.from(
      { length: 7 },
      (_, index) => `train/metric-${index}`,
    );

    const inputs = buildLogScalarChunkQueryInputs({
      enabled: true,
      group: "train",
      requestedTags: new Set(selectedTags),
      selectedTagList: selectedTags,
      visibleRunIds: runIds,
    });

    expect(
      inputs.map((input) => ({
        runIds: input.runIds,
        tags: input.tags,
      })),
    ).toEqual(
      Array.from({ length: 3 }, (_, runChunkIndex) =>
        [selectedTags.slice(0, 6), selectedTags.slice(6)].map((tags) => ({
          runIds: runIds.slice(runChunkIndex * 10, (runChunkIndex + 1) * 10),
          tags,
        })),
      ).flat(),
    );
  });

  it("keeps the 100-run six-tag scalar fan-out within ten requests", () => {
    const runIds = Array.from({ length: 100 }, (_, index) => `run-${index}`);
    const tags = Array.from({ length: 6 }, (_, index) => `train/metric-${index}`);

    const inputs = buildLogScalarChunkQueryInputs({
      enabled: true,
      group: "train",
      requestedTags: new Set(tags),
      selectedTagList: tags,
      visibleRunIds: runIds,
    });

    expect(inputs.length).toBeLessThanOrEqual(10);
    expect(inputs.every((input) => input.runIds.length <= 10)).toBe(true);
  });

  it("defaults to the core epoch scalar tags without diagnostic metrics", () => {
    expect(isDefaultScalarTag("validation/accuracy_epoch")).toBe(true);
    expect(isDefaultScalarTag("validation/loss_epoch")).toBe(true);
    expect(isDefaultScalarTag("train/loss_epoch")).toBe(true);
    expect(isDefaultScalarTag("train/accuracy_epoch")).toBe(true);
    expect(isDefaultScalarTag("train/loss")).toBe(false);
    expect(isDefaultScalarTag("train/accuracy")).toBe(false);
    expect(isDefaultScalarTag("gap/accuracy")).toBe(false);
    expect(isDefaultScalarTag("gap/loss")).toBe(false);
    expect(isDefaultScalarTag("best_validation/accuracy")).toBe(false);
    expect(isDefaultScalarTag("gradients/global_norm")).toBe(false);
    expect(isDefaultScalarTag("validation/confidence/mean")).toBe(false);
    expect(isDefaultScalarTag("validation/per_class/class_3/f1")).toBe(false);
    expect(
      isDefaultScalarTag(
        "validation/confusion_matrix/true_class_3/predicted_class_5/rate",
      ),
    ).toBe(false);
    expect(
      isDefaultScalarTag(
        "validation/confusion_matrix/true_class_3/predicted_class_5/count",
      ),
    ).toBe(false);
    expect(isDefaultScalarTag("main_model.0.model/weights/mean")).toBe(false);
  });

  it("limits accordion plot selectors to compact train and validation metrics", () => {
    expect(isLogPlotSelectorScalarTag("validation/accuracy")).toBe(true);
    expect(isLogPlotSelectorScalarTag("validation/f1_score")).toBe(true);
    expect(isLogPlotSelectorScalarTag("train/f1_score")).toBe(true);
    expect(isLogPlotSelectorScalarTag("validation/per_class/class_3/f1_score"))
      .toBe(false);
    expect(isLogPlotSelectorScalarTag("validation/kaggle_auc")).toBe(false);
    expect(isLogPlotSelectorScalarTag("gap/accuracy")).toBe(false);

    expect(
      groupLogPlotSelectorTags([
        "validation/loss",
        "validation/per_class/class_3/f1_score",
        "validation/f1_score",
        "train/accuracy",
        "train/kaggle_logloss",
        "gap/accuracy",
      ]),
    ).toEqual({
      train: ["train/accuracy"],
      validation: ["validation/loss", "validation/f1_score"],
      test: [],
      other: [],
    });
  });

  it("builds complete train-validation scalar pairs by suffix", () => {
    const pairs = buildTrainValidationScalarPairs([
      { value: "train/loss_epoch" },
      { value: "validation/loss_epoch" },
      { value: "train/accuracy_epoch" },
      { value: "validation/accuracy_epoch" },
      { value: "train/only" },
      { value: "validation/other_only" },
      { value: "test/accuracy" },
    ]);

    expect(pairs).toEqual([
      {
        suffix: "loss_epoch",
        trainTag: "train/loss_epoch",
        validationTag: "validation/loss_epoch",
      },
      {
        suffix: "accuracy_epoch",
        trainTag: "train/accuracy_epoch",
        validationTag: "validation/accuracy_epoch",
      },
    ]);
  });

  it("defaults train-validation pairs only when both tags are default scalars", () => {
    const pairs = buildTrainValidationScalarPairs([
      { value: "train/loss" },
      { value: "validation/loss" },
      { value: "train/loss_epoch" },
      { value: "validation/loss_epoch" },
      { value: "train/accuracy_epoch" },
      { value: "validation/accuracy_epoch" },
      { value: "train/confidence/mean" },
      { value: "validation/confidence/mean" },
    ]);

    expect(defaultTrainValidationScalarPairSuffixes(pairs)).toEqual([
      "loss_epoch",
      "accuracy_epoch",
    ]);
  });

  it("splits confusion-matrix tags out of normal scalar tag options", () => {
    const confusionRateTag =
      "validation/confusion_matrix/true_class_0/predicted_class_1/rate";
    const confusionCountTag =
      "validation/confusion_matrix/true_class_0/predicted_class_1/count";

    const tagOptions = buildLogScalarTagOptions([
      {
        scalarTags: ["validation/accuracy", confusionRateTag, confusionCountTag],
      },
      {
        scalarTags: ["train/loss", confusionRateTag],
      },
    ]);

    expect(tagOptions.tagOptions.map((option) => option.value)).toEqual([
      "train/loss",
      "validation/accuracy",
    ]);
    expect(tagOptions.confusionMatrixRateTags).toEqual([confusionRateTag]);
  });

  it("prefers validation accuracy for the best-run default metric", () => {
    expect(
      defaultLogBestRunMetricTag([
        { value: "train/loss", label: "train/loss", count: 2 },
        { value: "test/accuracy", label: "test/accuracy", count: 2 },
        { value: "validation/accuracy", label: "validation/accuracy", count: 2 },
      ]),
    ).toBe("validation/accuracy");
    expect(
      defaultLogBestRunMetricTag([
        { value: "test/accuracy", label: "test/accuracy", count: 2 },
        { value: "train/accuracy", label: "train/accuracy", count: 2 },
      ]),
    ).toBe("test/accuracy");
    expect(defaultLogBestRunMetricTag([{ value: "custom", label: "custom" }]))
      .toBe("custom");
    expect(defaultLogBestRunMetricTag([])).toBeNull();
  });

  it("reuses an active scalar metric group for best-run rankings", () => {
    const selectedTagsByGroup = {
      train: ["train/loss"],
      validation: ["validation/accuracy"],
      test: ["test/accuracy"],
      other: ["custom/metric"],
    };

    expect(
      bestRunMetricGroupForActiveScalarQuery({
        activeGroups: {
          train: true,
          validation: true,
          test: true,
          other: false,
        },
        selectedTagsByGroup,
        tag: "validation/accuracy",
      }),
    ).toBe("validation");
    expect(
      bestRunMetricGroupForActiveScalarQuery({
        activeGroups: {
          train: true,
          validation: false,
          test: true,
          other: false,
        },
        selectedTagsByGroup,
        tag: "validation/accuracy",
      }),
    ).toBeNull();
    expect(
      bestRunMetricGroupForActiveScalarQuery({
        activeGroups: {
          train: true,
          validation: true,
          test: true,
          other: true,
        },
        selectedTagsByGroup,
        tag: "missing/metric",
      }),
    ).toBeNull();
  });

  it("builds heatmaps from confusion-matrix rates", () => {
    const confusionRateTag =
      "validation/confusion_matrix/true_class_0/predicted_class_1/rate";
    const confusionCountTag =
      "validation/confusion_matrix/true_class_0/predicted_class_1/count";
    const seriesByTag = groupLogScalarSeriesByTag([
      scalarSeries({ runId: "run-1", tag: "train/loss" }),
      scalarSeries({
        runId: "run-1",
        tag: confusionRateTag,
        points: [{ step: 2, wallTime: 200, value: 0.42 }],
      }),
      scalarSeries({
        runId: "run-1",
        tag: confusionCountTag,
        points: [{ step: 2, wallTime: 200, value: 5 }],
      }),
    ]);
    const heatmaps = buildConfusionMatrixHeatmaps({
      matrixTagList: [confusionRateTag],
      seriesByTag,
      runsById: new Map([["run-1", logRun({ id: "run-1" })]]),
      runOrder: ["run-1"],
    });

    expect(heatmaps).toHaveLength(1);
    expect(heatmaps[0]).toMatchObject({
      split: "validation",
      runId: "run-1",
      classCount: 2,
      cells: [{ trueClass: 0, predictedClass: 1, value: 0.42 }],
    });
  });

  it("derives empty states for selected tag and scalar point gaps", () => {
    expect(
      deriveLogsChartEmptyState({
        expandedSelectedTagCount: 0,
        confusionMatrixTagCount: 0,
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: false,
        selectedSeriesCount: 0,
        selectedTagCount: 0,
        tagOptionCount: 2,
        tagsLoading: false,
        tagsRefreshing: false,
        visibleRunCount: 1,
      }),
    ).toMatchObject({ title: "No scalar tags selected", busy: false });

    expect(
      deriveLogsChartEmptyState({
        expandedSelectedTagCount: 1,
        confusionMatrixTagCount: 0,
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: false,
        selectedSeriesCount: 0,
        selectedTagCount: 1,
        tagOptionCount: 2,
        tagsLoading: false,
        tagsRefreshing: false,
        visibleRunCount: 1,
      }),
    ).toBeNull();

    expect(
      deriveLogsChartEmptyState({
        expandedSelectedTagCount: 1,
        confusionMatrixTagCount: 0,
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: true,
        selectedSeriesCount: 0,
        selectedTagCount: 1,
        tagOptionCount: 2,
        tagsLoading: false,
        tagsRefreshing: true,
        visibleRunCount: 1,
      }),
    ).toMatchObject({ title: "Refreshing TensorBoard tags", busy: true });

    expect(
      deriveLogsChartEmptyState({
        expandedSelectedTagCount: 1,
        confusionMatrixTagCount: 0,
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: true,
        selectedSeriesCount: 0,
        selectedTagCount: 1,
        tagOptionCount: 2,
        tagsLoading: false,
        tagsRefreshing: false,
        visibleRunCount: 1,
      }),
    ).toBeNull();
  });

  it("allows the confusion matrix accordion when no normal scalar tags are selected", () => {
    expect(
      deriveLogsChartEmptyState({
        expandedSelectedTagCount: 0,
        confusionMatrixTagCount: 4,
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: false,
        selectedSeriesCount: 0,
        selectedTagCount: 0,
        tagOptionCount: 0,
        tagsLoading: false,
        tagsRefreshing: false,
        visibleRunCount: 1,
      }),
    ).toBeNull();
  });

  it("does not show a global scalar loading state when loaded chart data remains visible", () => {
    expect(
      deriveLogsChartEmptyState({
        expandedSelectedTagCount: 2,
        confusionMatrixTagCount: 0,
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: true,
        selectedSeriesCount: 1,
        selectedTagCount: 2,
        tagOptionCount: 2,
        tagsLoading: false,
        tagsRefreshing: false,
        visibleRunCount: 1,
      }),
    ).toBeNull();
  });

  it("does not return an empty state when selected scalar series are renderable", () => {
    expect(
      deriveLogsChartEmptyState({
        expandedSelectedTagCount: 1,
        confusionMatrixTagCount: 0,
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: false,
        selectedSeriesCount: 1,
        selectedTagCount: 1,
        tagOptionCount: 2,
        tagsLoading: false,
        tagsRefreshing: false,
        visibleRunCount: 1,
      }),
    ).toBeNull();
  });

  it("scopes scalar query loading state to active metric groups", () => {
    const hiddenError = new Error("hidden group failed");
    const states = deriveLogMetricGroupScalarQueryStates({
      train: {
        active: true,
        isInitialLoading: true,
        isFetching: true,
        isError: false,
        error: null,
      },
      validation: {
        active: true,
        isInitialLoading: false,
        isFetching: false,
        isError: false,
        error: null,
      },
      test: {
        active: false,
        isInitialLoading: true,
        isFetching: true,
        isError: true,
        error: hiddenError,
      },
      other: {
        active: false,
        isInitialLoading: false,
        isFetching: false,
        isError: false,
        error: null,
      },
    });

    expect(states.train).toMatchObject({
      isInitialLoading: true,
      isFetching: true,
      isError: false,
    });
    expect(states.validation).toMatchObject({
      isInitialLoading: false,
      isFetching: false,
      isError: false,
    });
    expect(states.test).toEqual({
      isInitialLoading: false,
      isFetching: false,
      isError: false,
      error: null,
    });
  });
});
