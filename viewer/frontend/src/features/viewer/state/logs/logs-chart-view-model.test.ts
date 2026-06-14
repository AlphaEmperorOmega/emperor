import { describe, expect, it } from "vitest";
import {
  buildLogScalarQueryInput,
  deriveLogsChartEmptyState,
  groupLogScalarSeriesByTag,
} from "@/features/viewer/state/logs/logs-chart-view-model";
import {
  isDefaultScalarTag,
  groupRenderableLogMetrics,
} from "@/features/viewer/state/logs/logs-selectors";
import { buildConfusionMatrixHeatmaps } from "@/features/viewer/state/logs/log-diagnostics";
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
  const model = overrides.model ?? "linears/linear";
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
    expect(input.queryKey).toEqual(["log-scalars", runIds, tags]);
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

  it("filters renderable metrics to selected tags with series", () => {
    const seriesByTag = groupLogScalarSeriesByTag([
      scalarSeries({ runId: "run-1", tag: "train/loss" }),
      scalarSeries({ runId: "run-1", tag: "validation/accuracy" }),
      scalarSeries({ runId: "run-1", tag: "test/accuracy" }),
    ]);

    const groups = groupRenderableLogMetrics({
      selectedTagList: ["validation/accuracy", "missing/tag", "test/accuracy"],
      seriesByTag,
    });

    expect(groups.train).toEqual([]);
    expect(groups.validation.map((metric) => metric.tag)).toEqual([
      "validation/accuracy",
    ]);
    expect(groups.test.map((metric) => metric.tag)).toEqual(["test/accuracy"]);
    expect(groups.other).toEqual([]);
  });

  it("treats classifier diagnostics as default scalar tags", () => {
    expect(isDefaultScalarTag("gap/accuracy")).toBe(true);
    expect(isDefaultScalarTag("gap/loss")).toBe(true);
    expect(isDefaultScalarTag("best_validation/accuracy")).toBe(true);
    expect(isDefaultScalarTag("gradients/global_norm")).toBe(true);
    expect(isDefaultScalarTag("validation/confidence/mean")).toBe(true);
    expect(isDefaultScalarTag("validation/per_class/class_3/f1")).toBe(true);
    expect(
      isDefaultScalarTag(
        "validation/confusion_matrix/true_class_3/predicted_class_5/rate",
      ),
    ).toBe(true);
    expect(
      isDefaultScalarTag(
        "validation/confusion_matrix/true_class_3/predicted_class_5/count",
      ),
    ).toBe(false);
    expect(isDefaultScalarTag("main_model.0.model/weights/mean")).toBe(false);
  });

  it("routes confusion-matrix rates to heatmaps and counts to scalar groups", () => {
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
    const selectedTagList = ["train/loss", confusionRateTag, confusionCountTag];

    const groups = groupRenderableLogMetrics({
      selectedTagList,
      seriesByTag,
    });
    const heatmaps = buildConfusionMatrixHeatmaps({
      selectedTagList,
      seriesByTag,
      runsById: new Map([["run-1", logRun({ id: "run-1" })]]),
      runOrder: ["run-1"],
    });

    expect(groups.train.map((metric) => metric.tag)).toEqual(["train/loss"]);
    expect(groups.validation.map((metric) => metric.tag)).toEqual([confusionCountTag]);
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
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: false,
        selectedSeriesCount: 0,
        selectedTagCount: 0,
        tagOptionCount: 2,
        tagsLoading: false,
        visibleRunCount: 1,
      }),
    ).toMatchObject({ title: "No scalar tags selected", busy: false });

    expect(
      deriveLogsChartEmptyState({
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: false,
        selectedSeriesCount: 0,
        selectedTagCount: 1,
        tagOptionCount: 2,
        tagsLoading: false,
        visibleRunCount: 1,
      }),
    ).toMatchObject({ title: "No scalar points for selection", busy: false });

    expect(
      deriveLogsChartEmptyState({
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: true,
        selectedSeriesCount: 0,
        selectedTagCount: 1,
        tagOptionCount: 2,
        tagsLoading: false,
        visibleRunCount: 1,
      }),
    ).toMatchObject({ title: "Loading scalar points", busy: true });
  });

  it("does not return an empty state when selected scalar series are renderable", () => {
    expect(
      deriveLogsChartEmptyState({
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: false,
        selectedSeriesCount: 1,
        selectedTagCount: 1,
        tagOptionCount: 2,
        tagsLoading: false,
        visibleRunCount: 1,
      }),
    ).toBeNull();
  });
});
