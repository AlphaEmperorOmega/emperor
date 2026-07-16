import type { LogImageSummary, LogRun, LogScalarSeries, LogTextSummary } from "@/lib/api/logs";

export type ConfusionMatrixSplit = "train" | "validation";

export type ConfusionMatrixCell = {
  trueClass: number;
  predictedClass: number;
  value: number;
};

export type ConfusionMatrixHeatmap = {
  key: string;
  split: ConfusionMatrixSplit;
  runId: string;
  runLabel: string;
  classCount: number;
  cells: ConfusionMatrixCell[];
};

export type LogValidationExampleImage = LogImageSummary & {
  textSummary?: LogTextSummary;
};

type LogTagsData = {
  runs: Array<{
    imageTags: string[];
    textTags?: string[];
    [key: string]: unknown;
  }>;
};

const CONFUSION_MATRIX_TAG_RE =
  /^(train|validation)\/confusion_matrix\/true_class_(\d+)\/predicted_class_(\d+)\/(count|rate)$/;

const HEALTH_DIAGNOSTIC_SCALAR_TAGS = new Set([
  "gradients/global_norm",
  "parameters/global_norm",
  "updates/update_to_weight_ratio",
  "gradients/nan_count",
  "gradients/inf_count",
]);

function parseConfusionMatrixTag(tag: string) {
  const match = CONFUSION_MATRIX_TAG_RE.exec(tag);
  if (!match) {
    return null;
  }
  return {
    split: match[1] as ConfusionMatrixSplit,
    trueClass: Number(match[2]),
    predictedClass: Number(match[3]),
    valueKind: match[4] as "count" | "rate",
  };
}

function latestValue(series: LogScalarSeries) {
  return series.points.at(-1)?.value;
}

function runLabel(run: LogRun | undefined, runId: string) {
  if (!run) {
    return runId;
  }
  return [run.runName, run.dataset, run.preset].join(" · ");
}

export function isConfusionMatrixHeatmapTag(tag: string) {
  return parseConfusionMatrixTag(tag)?.valueKind === "rate";
}

export function isConfusionMatrixScalarTag(tag: string) {
  return parseConfusionMatrixTag(tag) !== null;
}

export function isDefaultDiagnosticScalarTag(tag: string) {
  return (
    tag.startsWith("gap/") ||
    tag.startsWith("best_validation/") ||
    HEALTH_DIAGNOSTIC_SCALAR_TAGS.has(tag) ||
    /^(?:train|validation)\/(?:confidence|calibration)\//.test(tag) ||
    /^(?:train|validation)\/per_class\//.test(tag) ||
    isConfusionMatrixHeatmapTag(tag)
  );
}

export function buildConfusionMatrixHeatmaps({
  matrixTagList,
  seriesByTag,
  runsById,
  runOrder,
}: {
  matrixTagList: string[];
  seriesByTag: Map<string, LogScalarSeries[]>;
  runsById: Map<string, LogRun>;
  runOrder: string[];
}): ConfusionMatrixHeatmap[] {
  const heatmaps = new Map<string, ConfusionMatrixHeatmap>();
  const runIndex = new Map(runOrder.map((runId, index) => [runId, index]));

  for (const tag of matrixTagList) {
    const parsedTag = parseConfusionMatrixTag(tag);
    if (!parsedTag || parsedTag.valueKind !== "rate") {
      continue;
    }
    for (const series of seriesByTag.get(tag) ?? []) {
      const value = latestValue(series);
      if (value === undefined) {
        continue;
      }
      const key = `${parsedTag.split}:${series.runId}`;
      const heatmap =
        heatmaps.get(key) ??
        ({
          key,
          split: parsedTag.split,
          runId: series.runId,
          runLabel: runLabel(runsById.get(series.runId), series.runId),
          classCount: 0,
          cells: [],
        } satisfies ConfusionMatrixHeatmap);
      heatmap.classCount = Math.max(
        heatmap.classCount,
        parsedTag.trueClass + 1,
        parsedTag.predictedClass + 1,
      );
      heatmap.cells.push({
        trueClass: parsedTag.trueClass,
        predictedClass: parsedTag.predictedClass,
        value,
      });
      heatmaps.set(key, heatmap);
    }
  }

  return Array.from(heatmaps.values()).sort((left, right) => {
    const splitOrder = left.split.localeCompare(right.split);
    if (splitOrder !== 0) {
      return splitOrder;
    }
    return (runIndex.get(left.runId) ?? 9999) - (runIndex.get(right.runId) ?? 9999);
  });
}

export function selectValidationExampleMediaTags(
  tagRuns: LogTagsData | undefined,
) {
  const imageTags = new Set<string>();
  const textTags = new Set<string>();
  for (const runTags of tagRuns?.runs ?? []) {
    for (const tag of runTags.imageTags) {
      if (tag.startsWith("validation/examples/")) {
        imageTags.add(tag);
      }
    }
    for (const tag of runTags.textTags ?? []) {
      if (tag.startsWith("validation/examples/")) {
        textTags.add(tag);
      }
    }
  }
  return {
    imageTags: Array.from(imageTags).sort((a, b) => a.localeCompare(b)),
    textTags: Array.from(textTags).sort((a, b) => a.localeCompare(b)),
  };
}

export function validationExampleTextForImage(
  image: LogImageSummary,
  texts: LogTextSummary[],
) {
  const sameRunTexts = texts.filter((text) => text.runId === image.runId);
  const matchingText = sameRunTexts.find((text) =>
    [
      image.tag,
      `${image.tag}/text_summary`,
      `${image.tag}_labels/text_summary`,
    ].includes(text.tag),
  );
  return matchingText ?? sameRunTexts[0];
}

export function pairValidationExampleMedia({
  images,
  texts,
}: {
  images: LogImageSummary[];
  texts: LogTextSummary[];
}): {
  images: LogValidationExampleImage[];
  texts: LogTextSummary[];
} {
  return {
    images: images.map((image) => ({
      ...image,
      textSummary: validationExampleTextForImage(image, texts),
    })),
    texts,
  };
}
