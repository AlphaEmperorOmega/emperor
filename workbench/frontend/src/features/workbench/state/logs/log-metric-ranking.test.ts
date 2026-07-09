import { describe, expect, it } from "vitest";
import {
  buildLogMetricDatasetRankingRows,
  buildLogMetricRankingRows,
  inferLogMetricDirection,
  scoreLogMetricSeries,
  type LogMetricDirection,
  type LogMetricPointPolicy,
} from "@/features/workbench/state/logs/log-metric-ranking";
import { type LogRun, type LogScalarSeries } from "@/lib/api";

function logRun(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  const experiment = overrides.experiment ?? "experiment";
  const dataset = overrides.dataset ?? "Mnist";
  const modelType = overrides.modelType ?? "linears";
  const model = overrides.model ?? "linear";
  const preset = overrides.preset ?? "BASELINE";

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

function scalarSeries({
  points,
  runId,
  tag,
}: Partial<LogScalarSeries> &
  Pick<LogScalarSeries, "runId" | "tag">): LogScalarSeries {
  return {
    runId,
    tag,
    points: points ?? [{ step: 1, wallTime: 100, value: 0.5 }],
  };
}

function rank({
  direction = "higher",
  pointPolicy = "best",
  runs = [logRun({ id: "run-a" }), logRun({ id: "run-b" })],
  series,
  tag = "validation/accuracy",
}: {
  direction?: LogMetricDirection;
  pointPolicy?: LogMetricPointPolicy;
  runs?: LogRun[];
  series: LogScalarSeries[];
  tag?: string | null;
}) {
  return buildLogMetricRankingRows({
    direction,
    pointPolicy,
    runOrder: runs.map((run) => run.id),
    runs,
    series,
    tag,
  });
}

function datasetRank({
  direction = "higher",
  pointPolicy = "best",
  runs = [logRun({ id: "run-a" }), logRun({ id: "run-b" })],
  series,
  tag = "validation/accuracy",
}: {
  direction?: LogMetricDirection;
  pointPolicy?: LogMetricPointPolicy;
  runs?: LogRun[];
  series: LogScalarSeries[];
  tag?: string | null;
}) {
  return buildLogMetricDatasetRankingRows({
    direction,
    pointPolicy,
    runOrder: runs.map((run) => run.id),
    runs,
    series,
    tag,
  });
}

describe("log metric ranking", () => {
  it("infers lower and higher directions from metric tags", () => {
    expect(inferLogMetricDirection("validation/loss")).toBe("lower");
    expect(inferLogMetricDirection("validation/calibration/ece")).toBe("lower");
    expect(inferLogMetricDirection("test/error_rate")).toBe("lower");
    expect(inferLogMetricDirection("validation/accuracy")).toBe("higher");
    expect(inferLogMetricDirection("test/auc")).toBe("higher");
    expect(inferLogMetricDirection("validation/per_class/class_3/f1")).toBe(
      "higher",
    );
    expect(inferLogMetricDirection("gradients/global_norm")).toBe("higher");
  });

  it("ranks higher-is-better metrics by the best score", () => {
    const rows = rank({
      series: [
        scalarSeries({
          runId: "run-a",
          tag: "validation/accuracy",
          points: [
            { step: 1, wallTime: 100, value: 0.61 },
            { step: 2, wallTime: 101, value: 0.71 },
          ],
        }),
        scalarSeries({
          runId: "run-b",
          tag: "validation/accuracy",
          points: [
            { step: 1, wallTime: 100, value: 0.66 },
            { step: 2, wallTime: 101, value: 0.69 },
          ],
        }),
      ],
    });

    expect(rows.map((row) => [row.runId, row.score, row.step])).toEqual([
      ["run-a", 0.71, 2],
      ["run-b", 0.69, 2],
    ]);
  });

  it("ranks lower-is-better metrics by the best score", () => {
    const rows = rank({
      direction: "lower",
      tag: "validation/loss",
      series: [
        scalarSeries({
          runId: "run-a",
          tag: "validation/loss",
          points: [
            { step: 1, wallTime: 100, value: 0.4 },
            { step: 2, wallTime: 101, value: 0.31 },
          ],
        }),
        scalarSeries({
          runId: "run-b",
          tag: "validation/loss",
          points: [
            { step: 1, wallTime: 100, value: 0.35 },
            { step: 2, wallTime: 101, value: 0.29 },
          ],
        }),
      ],
    });

    expect(rows.map((row) => [row.runId, row.score, row.step])).toEqual([
      ["run-b", 0.29, 2],
      ["run-a", 0.31, 2],
    ]);
  });

  it("uses the final point for latest policy", () => {
    const latest = scoreLogMetricSeries({
      direction: "higher",
      pointPolicy: "latest",
      series: scalarSeries({
        runId: "run-a",
        tag: "validation/accuracy",
        points: [
          { step: 1, wallTime: 100, value: 0.95 },
          { step: 2, wallTime: 101, value: 0.74 },
        ],
      }),
    });

    expect(latest).toMatchObject({
      point: { step: 2, wallTime: 101, value: 0.74 },
      pointIndex: 1,
    });
  });

  it("builds one best row per visible dataset in current order", () => {
    const rows = datasetRank({
      runs: [
        logRun({ id: "run-a", dataset: "Mnist" }),
        logRun({ id: "run-b", dataset: "Cifar10" }),
        logRun({ id: "run-c", dataset: "Mnist" }),
      ],
      series: [
        scalarSeries({
          runId: "run-a",
          tag: "validation/accuracy",
          points: [{ step: 1, wallTime: 100, value: 0.72 }],
        }),
        scalarSeries({
          runId: "run-b",
          tag: "validation/accuracy",
          points: [{ step: 1, wallTime: 100, value: 0.61 }],
        }),
        scalarSeries({
          runId: "run-c",
          tag: "validation/accuracy",
          points: [{ step: 1, wallTime: 100, value: 0.81 }],
        }),
      ],
    });

    expect(
      rows.map((row) => [
        row.dataset,
        row.runCount,
        row.best?.runId,
        row.best?.score,
      ]),
    ).toEqual([
      ["Mnist", 2, "run-c", 0.81],
      ["Cifar10", 1, "run-b", 0.61],
    ]);
  });

  it("chooses the best score within each dataset for higher and lower metrics", () => {
    const runs = [
      logRun({ id: "mnist-a", dataset: "Mnist" }),
      logRun({ id: "mnist-b", dataset: "Mnist" }),
      logRun({ id: "cifar-a", dataset: "Cifar10" }),
      logRun({ id: "cifar-b", dataset: "Cifar10" }),
    ];

    expect(
      datasetRank({
        runs,
        direction: "higher",
        series: [
          scalarSeries({
            runId: "mnist-a",
            tag: "validation/accuracy",
            points: [{ step: 1, wallTime: 100, value: 0.7 }],
          }),
          scalarSeries({
            runId: "mnist-b",
            tag: "validation/accuracy",
            points: [{ step: 1, wallTime: 100, value: 0.9 }],
          }),
          scalarSeries({
            runId: "cifar-a",
            tag: "validation/accuracy",
            points: [{ step: 1, wallTime: 100, value: 0.6 }],
          }),
          scalarSeries({
            runId: "cifar-b",
            tag: "validation/accuracy",
            points: [{ step: 1, wallTime: 100, value: 0.65 }],
          }),
        ],
      }).map((row) => row.best?.runId),
    ).toEqual(["mnist-b", "cifar-b"]);

    expect(
      datasetRank({
        runs,
        direction: "lower",
        tag: "validation/loss",
        series: [
          scalarSeries({
            runId: "mnist-a",
            tag: "validation/loss",
            points: [{ step: 1, wallTime: 100, value: 0.4 }],
          }),
          scalarSeries({
            runId: "mnist-b",
            tag: "validation/loss",
            points: [{ step: 1, wallTime: 100, value: 0.31 }],
          }),
          scalarSeries({
            runId: "cifar-a",
            tag: "validation/loss",
            points: [{ step: 1, wallTime: 100, value: 0.2 }],
          }),
          scalarSeries({
            runId: "cifar-b",
            tag: "validation/loss",
            points: [{ step: 1, wallTime: 100, value: 0.29 }],
          }),
        ],
      }).map((row) => row.best?.runId),
    ).toEqual(["mnist-b", "cifar-a"]);
  });

  it("uses the final point per run before choosing the latest-policy dataset winner", () => {
    const rows = datasetRank({
      pointPolicy: "latest",
      runs: [
        logRun({ id: "run-a", dataset: "Mnist" }),
        logRun({ id: "run-b", dataset: "Mnist" }),
      ],
      series: [
        scalarSeries({
          runId: "run-a",
          tag: "validation/accuracy",
          points: [
            { step: 1, wallTime: 100, value: 0.95 },
            { step: 2, wallTime: 101, value: 0.74 },
          ],
        }),
        scalarSeries({
          runId: "run-b",
          tag: "validation/accuracy",
          points: [
            { step: 1, wallTime: 100, value: 0.7 },
            { step: 2, wallTime: 101, value: 0.82 },
          ],
        }),
      ],
    });

    expect(rows[0].best).toMatchObject({
      runId: "run-b",
      score: 0.82,
      step: 2,
    });
  });

  it("represents visible datasets without selected metric points as missing rows", () => {
    const rows = datasetRank({
      runs: [
        logRun({ id: "run-a", dataset: "Mnist" }),
        logRun({ id: "run-b", dataset: "Cifar10" }),
        logRun({ id: "run-c", dataset: "FashionMnist" }),
      ],
      series: [
        scalarSeries({
          runId: "run-a",
          tag: "validation/accuracy",
          points: [{ step: 1, wallTime: 100, value: 0.72 }],
        }),
        scalarSeries({ runId: "run-b", tag: "train/loss" }),
        scalarSeries({ runId: "run-c", tag: "validation/accuracy", points: [] }),
      ],
    });

    expect(
      rows.map((row) => [row.dataset, row.runCount, row.best?.runId ?? null]),
    ).toEqual([
      ["Mnist", 1, "run-a"],
      ["Cifar10", 1, null],
      ["FashionMnist", 1, null],
    ]);
  });

  it("skips missing runs, missing metric series, and empty point series", () => {
    const rows = rank({
      series: [
        scalarSeries({ runId: "run-a", tag: "train/loss" }),
        scalarSeries({ runId: "run-b", tag: "validation/accuracy", points: [] }),
        scalarSeries({ runId: "missing-run", tag: "validation/accuracy" }),
      ],
    });

    expect(rows).toEqual([]);
  });

  it("keeps dataset winner tie-breakers deterministic", () => {
    const rows = datasetRank({
      runs: [
        logRun({ id: "run-a", dataset: "Mnist" }),
        logRun({ id: "run-b", dataset: "Mnist" }),
        logRun({ id: "run-c", dataset: "Mnist" }),
      ],
      series: [
        scalarSeries({
          runId: "run-a",
          tag: "validation/accuracy",
          points: [{ step: 3, wallTime: 100, value: 0.8 }],
        }),
        scalarSeries({
          runId: "run-b",
          tag: "validation/accuracy",
          points: [{ step: 2, wallTime: 200, value: 0.8 }],
        }),
        scalarSeries({
          runId: "run-c",
          tag: "validation/accuracy",
          points: [{ step: 2, wallTime: 100, value: 0.8 }],
        }),
      ],
    });

    expect(rows.map((row) => row.best?.runId)).toEqual(["run-b"]);
  });
});
