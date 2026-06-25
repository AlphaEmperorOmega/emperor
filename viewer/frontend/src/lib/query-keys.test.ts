import { describe, expect, it } from "vitest";
import {
  LOG_EXPERIMENTS_QUERY_KEY,
  LOG_ARTIFACTS_QUERY_KEY,
  LOG_CHECKPOINTS_QUERY_KEY,
  LOG_RUNS_QUERY_KEY,
  LOG_SCALARS_QUERY_KEY,
  LOG_TAGS_QUERY_KEY,
  logQueryKeys,
  monitorQueryKeys,
  trainingQueryKeys,
  type TrainingRunPlanQueryKeyInput,
  viewerQueryKeys,
} from "@/lib/query-keys";

function trainingRunPlanInput(
  overrides: Partial<TrainingRunPlanQueryKeyInput> = {},
): TrainingRunPlanQueryKeyInput {
  return {
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    presets: ["baseline"],
    datasets: ["Mnist"],
    overrides: { hidden_size: "128" },
    logFolder: "runs",
    monitors: [],
    ...overrides,
  };
}

describe("query key factories", () => {
  it("preserves log query key shapes", () => {
    const runIds = ["run-2", "run-1"];
    const tags = ["loss", "accuracy"];

    expect(logQueryKeys.runs()).toEqual(["log-runs"]);
    expect(logQueryKeys.experiments()).toEqual(["log-experiments"]);
    expect(logQueryKeys.tags()).toEqual(["log-tags"]);
    expect(logQueryKeys.tagsForRuns(runIds)).toEqual([
      "log-tags",
      ["run-1", "run-2"],
    ]);
    expect(logQueryKeys.filteredHistoricalRunTags(runIds)).toEqual([
      "log-tags",
      "filtered-historical-runs",
      ["run-1", "run-2"],
    ]);
    expect(logQueryKeys.modelRunTags(runIds)).toEqual([
      "log-tags",
      "model-runs",
      ["run-1", "run-2"],
    ]);
    expect(logQueryKeys.scalars()).toEqual(["log-scalars"]);
    expect(logQueryKeys.scalarsForRunsAndTags(runIds, tags)).toEqual([
      "log-scalars",
      ["run-1", "run-2"],
      ["accuracy", "loss"],
      null,
    ]);
    expect(logQueryKeys.checkpoints()).toEqual(["log-checkpoints"]);
    expect(logQueryKeys.checkpointsForRuns(runIds)).toEqual([
      "log-checkpoints",
      ["run-1", "run-2"],
    ]);
    expect(logQueryKeys.artifacts()).toEqual(["log-artifacts"]);
    expect(logQueryKeys.artifactsForRun("run/1")).toEqual([
      "log-artifacts",
      "run/1",
    ]);
    expect(runIds).toEqual(["run-2", "run-1"]);
    expect(tags).toEqual(["loss", "accuracy"]);
  });

  it("normalizes set-like log query key values", () => {
    expect(logQueryKeys.tagsForRuns(["run-2", "run-1", "run-2"])).toEqual(
      logQueryKeys.tagsForRuns(["run-1", "run-2"]),
    );
    expect(
      logQueryKeys.scalarsForRunsAndTags(
        ["run-2", "run-1", "run-2"],
        ["loss", "accuracy", "loss"],
      ),
    ).toEqual(
      logQueryKeys.scalarsForRunsAndTags(
        ["run-1", "run-2"],
        ["accuracy", "loss"],
      ),
    );
  });

  it("keeps legacy log query constants on the base family keys", () => {
    expect(LOG_RUNS_QUERY_KEY).toEqual(["log-runs"]);
    expect(LOG_EXPERIMENTS_QUERY_KEY).toEqual(["log-experiments"]);
    expect(LOG_TAGS_QUERY_KEY).toEqual(["log-tags"]);
    expect(LOG_SCALARS_QUERY_KEY).toEqual(["log-scalars"]);
    expect(LOG_CHECKPOINTS_QUERY_KEY).toEqual(["log-checkpoints"]);
    expect(LOG_ARTIFACTS_QUERY_KEY).toEqual(["log-artifacts"]);
  });

  it("preserves viewer query key shapes", () => {
    expect(viewerQueryKeys.health()).toEqual(["health"]);
    expect(viewerQueryKeys.models()).toEqual(["models"]);
    expect(viewerQueryKeys.presets("linears", "linear")).toEqual([
      "presets",
      "linears",
      "linear",
    ]);
    expect(viewerQueryKeys.datasets("linears", "linear")).toEqual([
      "datasets",
      "linears",
      "linear",
    ]);
    expect(viewerQueryKeys.monitors("linears", "linear")).toEqual([
      "monitors",
      "linears",
      "linear",
    ]);
    expect(viewerQueryKeys.configSchema("linears", "linear", "baseline")).toEqual([
      "config-schema",
      "linears",
      "linear",
      "baseline",
    ]);
    expect(
      viewerQueryKeys.searchSpace("linears", "linear", "baseline", [
        "post-norm",
        "baseline",
      ]),
    ).toEqual([
      "search-space",
      "linears",
      "linear",
      "baseline",
      ["baseline", "post-norm"],
    ]);
    expect(
      viewerQueryKeys.historicalSummaryInspection("linear", "baseline", "Mnist"),
    ).toEqual(["inspect", "historical-summary", "linear", "baseline", "Mnist"]);
    expect(
      viewerQueryKeys.comparisonInspection(
        "linears",
        "linear",
        "baseline",
        "Mnist",
      ),
    ).toEqual([
      "comparison-inspection",
      "linears",
      "linear",
      "baseline",
      "Mnist",
    ]);
    expect(viewerQueryKeys.configSnapshots("linears", "linear")).toEqual([
      "config-snapshots",
      "linears",
      "linear",
    ]);
    expect(viewerQueryKeys.configSnapshotLibrary()).toEqual([
      "config-snapshot-library",
    ]);
  });

  it("preserves training query key shapes", () => {
    expect(trainingQueryKeys.job("job-1")).toEqual(["training-job", "job-1"]);
    expect(trainingQueryKeys.job(null)).toEqual(["training-job", null]);
    expect(trainingQueryKeys.jobEvents("job-1", 10, 50)).toEqual([
      "training-job-events",
      "job-1",
      10,
      50,
    ]);
    expect(trainingQueryKeys.runPlan(2, trainingRunPlanInput())).toEqual([
      "training-run-plan",
      2,
      {
        datasets: ["Mnist"],
        logFolder: "runs",
        modelType: "linears",
        model: "linear",
        monitors: [],
        overrides: { hidden_size: "128" },
        preset: "baseline",
        presets: ["baseline"],
        search: null,
        submittedRunPlan: null,
      },
    ]);
  });

  it("normalizes object-like training run-plan query key values", () => {
    const firstKey = trainingQueryKeys.runPlan(
      0,
      trainingRunPlanInput({
        overrides: { z_field: "2", a_field: "1" },
        search: {
          mode: "grid",
          values: {
            z_axis: [2],
            a_axis: ["one"],
          },
        },
      }),
    );
    const secondKey = trainingQueryKeys.runPlan(
      0,
      trainingRunPlanInput({
        overrides: { a_field: "1", z_field: "2" },
        search: {
          mode: "grid",
          values: {
            a_axis: ["one"],
            z_axis: [2],
          },
        },
      }),
    );

    expect(firstKey).toEqual(secondKey);
    expect(JSON.stringify(firstKey)).toBe(JSON.stringify(secondKey));
  });

  it("keeps training run-plan array order visible in the query key", () => {
    expect(
      trainingQueryKeys.runPlan(
        0,
        trainingRunPlanInput({
          presets: ["wide", "baseline"],
          datasets: ["Cifar10", "Mnist"],
        }),
      ),
    ).not.toEqual(
      trainingQueryKeys.runPlan(
        0,
        trainingRunPlanInput({
          presets: ["baseline", "wide"],
          datasets: ["Mnist", "Cifar10"],
        }),
      ),
    );
  });

  it("changes training run-plan query keys when monitors change", () => {
    expect(
      trainingQueryKeys.runPlan(
        0,
        trainingRunPlanInput({ monitors: ["linear"] }),
      ),
    ).not.toEqual(
      trainingQueryKeys.runPlan(
        0,
        trainingRunPlanInput({ monitors: ["linear", "halting"] }),
      ),
    );
  });

  it("preserves monitor query key shapes and undefined entries", () => {
    const runIds = ["run-2", "run-1"];

    expect(
      monitorQueryKeys.activeJob("job-1", "root.layer", "baseline", "mnist"),
    ).toEqual([
      "monitor-data",
      "active-job",
      "job-1",
      "root.layer",
      "baseline",
      "mnist",
    ]);
    expect(monitorQueryKeys.activeJob("job-1", undefined, "", "")).toEqual([
      "monitor-data",
      "active-job",
      "job-1",
      undefined,
      "",
      "",
    ]);
    expect(monitorQueryKeys.historicalRun("run-1", "root.layer")).toEqual([
      "monitor-data",
      "historical-run",
      "run-1",
      "root.layer",
    ]);
    expect(monitorQueryKeys.historicalRun(undefined, undefined)).toEqual([
      "monitor-data",
      "historical-run",
      undefined,
      undefined,
    ]);
    expect(monitorQueryKeys.historicalRunGroup(runIds, undefined)).toEqual([
      "monitor-data",
      "historical-run-group",
      ["run-1", "run-2"],
      undefined,
    ]);
    expect(monitorQueryKeys.historicalParameterStatus(runIds)).toEqual([
      "monitor-parameter-status",
      "historical-run-group",
      ["run-1", "run-2"],
    ]);
    expect(
      monitorQueryKeys.historicalParameterSummary(
        "linear",
        "baseline",
        "Mnist",
        runIds,
      ),
    ).toEqual([
      "monitor-parameter-summary",
      "historical-run-group",
      "linear",
      "baseline",
      "Mnist",
      ["run-1", "run-2"],
    ]);
    expect(runIds).toEqual(["run-2", "run-1"]);
  });

  it("normalizes set-like monitor run-id query key values", () => {
    expect(
      monitorQueryKeys.historicalRunGroup(
        ["run-2", "run-1", "run-2"],
        "root.layer",
      ),
    ).toEqual(
      monitorQueryKeys.historicalRunGroup(["run-1", "run-2"], "root.layer"),
    );
    expect(
      monitorQueryKeys.historicalParameterSummary(
        "linear",
        "baseline",
        "Mnist",
        ["run-2", "run-1", "run-2"],
      ),
    ).toEqual(
      monitorQueryKeys.historicalParameterSummary(
        "linear",
        "baseline",
        "Mnist",
        ["run-1", "run-2"],
      ),
    );
  });
});
