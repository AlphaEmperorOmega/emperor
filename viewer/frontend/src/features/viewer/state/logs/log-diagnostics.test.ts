import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildConfusionMatrixHeatmaps,
  isConfusionMatrixHeatmapTag,
  isConfusionMatrixScalarTag,
  isDefaultDiagnosticScalarTag,
  pairValidationExampleMedia,
  selectValidationExampleMediaTags,
} from "@/features/viewer/state/logs/log-diagnostics";
import { groupLogScalarSeriesByTag } from "@/features/viewer/state/logs/logs-chart-view-model";
import {
  type LogImageSummary,
  type LogRun,
  type LogScalarSeries,
  type LogTextSummary,
} from "@/lib/api";

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

function logRun(id: string): LogRun {
  return {
    id,
    group: null,
    experiment: "exp",
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    dataset: "Mnist",
    runName: id,
    timestamp: null,
    version: "version_0",
    relativePath: `exp/linear/baseline/Mnist/${id}/version_0`,
    hasResult: true,
    eventFileCount: 1,
    checkpointCount: 0,
    hasHparams: true,
    metrics: {},
  };
}

function imageSummary(overrides: Partial<LogImageSummary>): LogImageSummary {
  return {
    runId: overrides.runId ?? "run-1",
    tag: overrides.tag ?? "validation/examples/most_confident_wrong",
    step: overrides.step ?? 1,
    wallTime: overrides.wallTime ?? 100,
    mimeType: overrides.mimeType ?? "image/png",
    dataUrl: overrides.dataUrl ?? "data:image/png;base64,AAAA",
  };
}

function textSummary(overrides: Partial<LogTextSummary>): LogTextSummary {
  return {
    runId: overrides.runId ?? "run-1",
    tag: overrides.tag ?? "validation/examples/most_confident_wrong/text_summary",
    step: overrides.step ?? 1,
    wallTime: overrides.wallTime ?? 100,
    text: overrides.text ?? "true=0 predicted=1",
  };
}

describe("log diagnostics", () => {
  it("identifies default classifier diagnostic scalar tags", () => {
    const rateTag =
      "validation/confusion_matrix/true_class_3/predicted_class_5/rate";
    const countTag =
      "validation/confusion_matrix/true_class_3/predicted_class_5/count";

    expect(isConfusionMatrixHeatmapTag(rateTag)).toBe(true);
    expect(isConfusionMatrixHeatmapTag(countTag)).toBe(false);
    expect(isConfusionMatrixScalarTag(rateTag)).toBe(true);
    expect(isConfusionMatrixScalarTag(countTag)).toBe(true);
    expect(isDefaultDiagnosticScalarTag("gap/accuracy")).toBe(true);
    expect(isDefaultDiagnosticScalarTag("best_validation/loss")).toBe(true);
    expect(isDefaultDiagnosticScalarTag("gradients/global_norm")).toBe(true);
    expect(isDefaultDiagnosticScalarTag("validation/confidence/mean")).toBe(true);
    expect(isDefaultDiagnosticScalarTag("validation/calibration/ece")).toBe(true);
    expect(isDefaultDiagnosticScalarTag("validation/per_class/class_2/f1_score"))
      .toBe(true);
    expect(isDefaultDiagnosticScalarTag(rateTag)).toBe(true);
    expect(isDefaultDiagnosticScalarTag(countTag)).toBe(false);
    expect(isDefaultDiagnosticScalarTag("main_model.0.model/weights/mean")).toBe(false);
  });

  it("builds stable confusion heatmaps only from selected rate tags", () => {
    const rateTag =
      "validation/confusion_matrix/true_class_0/predicted_class_1/rate";
    const countTag =
      "validation/confusion_matrix/true_class_0/predicted_class_1/count";
    const trainRateTag =
      "train/confusion_matrix/true_class_1/predicted_class_1/rate";
    const seriesByTag = groupLogScalarSeriesByTag([
      scalarSeries({
        runId: "run-b",
        tag: rateTag,
        points: [{ step: 1, wallTime: 100, value: 0.2 }],
      }),
      scalarSeries({
        runId: "run-a",
        tag: trainRateTag,
        points: [{ step: 1, wallTime: 100, value: 0.9 }],
      }),
      scalarSeries({
        runId: "run-b",
        tag: countTag,
        points: [{ step: 1, wallTime: 100, value: 7 }],
      }),
    ]);

    const heatmaps = buildConfusionMatrixHeatmaps({
      selectedTagList: [rateTag, countTag, trainRateTag],
      seriesByTag,
      runsById: new Map([
        ["run-a", logRun("run-a")],
        ["run-b", logRun("run-b")],
      ]),
      runOrder: ["run-a", "run-b"],
    });

    expect(heatmaps.map((heatmap) => heatmap.key)).toEqual([
      "train:run-a",
      "validation:run-b",
    ]);
    expect(heatmaps[0].cells).toEqual([
      { trueClass: 1, predictedClass: 1, value: 0.9 },
    ]);
    expect(heatmaps[1]).toMatchObject({
      classCount: 2,
      cells: [{ trueClass: 0, predictedClass: 1, value: 0.2 }],
    });
  });

  it("selects validation example image and text media tags", () => {
    const mediaTags = selectValidationExampleMediaTags({
      runs: [
        {
          runId: "run-1",
          scalarTags: [],
          histogramTags: [],
          imageTags: [
            "validation/examples/most_confident_wrong",
            "train/examples/ignored",
          ],
          textTags: [
            "validation/examples/most_confident_wrong_labels/text_summary",
            "other/text_summary",
          ],
        },
        {
          runId: "run-2",
          scalarTags: [],
          histogramTags: [],
          imageTags: ["validation/examples/preview"],
          textTags: ["validation/examples/preview/text_summary"],
        },
      ],
    });

    expect(mediaTags.imageTags).toEqual([
      "validation/examples/most_confident_wrong",
      "validation/examples/preview",
    ]);
    expect(mediaTags.textTags).toEqual([
      "validation/examples/most_confident_wrong_labels/text_summary",
      "validation/examples/preview/text_summary",
    ]);
  });

  it("pairs validation images with exact, labels, and run fallback text", () => {
    const paired = pairValidationExampleMedia({
      images: [
        imageSummary({ tag: "validation/examples/exact" }),
        imageSummary({ tag: "validation/examples/labels" }),
        imageSummary({ runId: "run-2", tag: "validation/examples/fallback" }),
      ],
      texts: [
        textSummary({
          tag: "validation/examples/exact/text_summary",
          text: "exact text",
        }),
        textSummary({
          tag: "validation/examples/labels_labels/text_summary",
          text: "labels text",
        }),
        textSummary({
          runId: "run-2",
          tag: "validation/examples/other/text_summary",
          text: "fallback text",
        }),
      ],
    });

    expect(paired.images.map((image) => image.textSummary?.text)).toEqual([
      "exact text",
      "labels text",
      "fallback text",
    ]);
  });

  it("keeps LogsChartPanel free of state builders", () => {
    const source = readFileSync(
      join(
        process.cwd(),
        "src/features/viewer/components/logs/logs-chart-panel.tsx",
      ),
      "utf8",
    );

    expect(source).not.toContain("groupRenderableLogMetrics");
    expect(source).not.toContain("buildConfusionMatrixHeatmaps");
  });
});
