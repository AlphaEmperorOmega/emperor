import { describe, expect, it } from "vitest";
import {
  LOG_EXPERIMENTS_QUERY_KEY,
  LOG_RUNS_QUERY_KEY,
  LOG_SCALARS_QUERY_KEY,
  LOG_TAGS_QUERY_KEY,
  logQueryKeys,
  monitorQueryKeys,
  trainingQueryKeys,
  viewerQueryKeys,
} from "@/lib/query-keys";

describe("query key factories", () => {
  it("preserves log query key shapes", () => {
    const runIds = ["run-1", "run-2"];
    const tags = ["loss", "accuracy"];

    expect(logQueryKeys.runs()).toEqual(["log-runs"]);
    expect(logQueryKeys.experiments()).toEqual(["log-experiments"]);
    expect(logQueryKeys.tags()).toEqual(["log-tags"]);
    expect(logQueryKeys.tagsForRuns(runIds)).toEqual(["log-tags", runIds]);
    expect(logQueryKeys.filteredHistoricalRunTags(runIds)).toEqual([
      "log-tags",
      "filtered-historical-runs",
      runIds,
    ]);
    expect(logQueryKeys.scalars()).toEqual(["log-scalars"]);
    expect(logQueryKeys.scalarsForRunsAndTags(runIds, tags)).toEqual([
      "log-scalars",
      runIds,
      tags,
    ]);
    expect(logQueryKeys.tagsForRuns(runIds)[1]).toBe(runIds);
    expect(logQueryKeys.scalarsForRunsAndTags(runIds, tags)[1]).toBe(runIds);
    expect(logQueryKeys.scalarsForRunsAndTags(runIds, tags)[2]).toBe(tags);
  });

  it("keeps legacy log query constants on the base family keys", () => {
    expect(LOG_RUNS_QUERY_KEY).toEqual(["log-runs"]);
    expect(LOG_EXPERIMENTS_QUERY_KEY).toEqual(["log-experiments"]);
    expect(LOG_TAGS_QUERY_KEY).toEqual(["log-tags"]);
    expect(LOG_SCALARS_QUERY_KEY).toEqual(["log-scalars"]);
  });

  it("preserves viewer query key shapes", () => {
    expect(viewerQueryKeys.health()).toEqual(["health"]);
    expect(viewerQueryKeys.models()).toEqual(["models"]);
    expect(viewerQueryKeys.presets("linear")).toEqual(["presets", "linear"]);
    expect(viewerQueryKeys.datasets("linear")).toEqual(["datasets", "linear"]);
    expect(viewerQueryKeys.monitors("linear")).toEqual(["monitors", "linear"]);
    expect(viewerQueryKeys.configSchema("linear", "baseline")).toEqual([
      "config-schema",
      "linear",
      "baseline",
    ]);
    expect(viewerQueryKeys.searchSpace("linear", "baseline")).toEqual([
      "search-space",
      "linear",
      "baseline",
    ]);
    expect(
      viewerQueryKeys.historicalSummaryInspection("linear", "baseline", "Mnist"),
    ).toEqual(["inspect", "historical-summary", "linear", "baseline", "Mnist"]);
    expect(
      viewerQueryKeys.comparisonInspection("linear", "baseline", "Mnist"),
    ).toEqual(["comparison-inspection", "linear", "baseline", "Mnist"]);
  });

  it("preserves training query key shapes", () => {
    expect(trainingQueryKeys.job("job-1")).toEqual(["training-job", "job-1"]);
    expect(trainingQueryKeys.job(null)).toEqual(["training-job", null]);
    expect(trainingQueryKeys.runPlan(2, "input-key")).toEqual([
      "training-run-plan",
      2,
      "input-key",
    ]);
  });

  it("preserves monitor query key shapes and undefined entries", () => {
    const runIds = ["run-1", "run-2"];

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
      runIds,
      undefined,
    ]);
    expect(monitorQueryKeys.historicalRunGroup(runIds, undefined)[2]).toBe(runIds);
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
      runIds,
    ]);
  });
});
