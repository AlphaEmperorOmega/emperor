import { describe, expect, it } from "vitest";
import {
  buildLogScalarQueryInput,
  deriveLogsChartEmptyState,
  groupLogScalarSeriesByTag,
} from "@/features/viewer/state/logs/logs-chart-view-model";
import { groupRenderableLogMetrics } from "@/features/viewer/state/logs/logs-selectors";
import { type LogScalarSeries } from "@/lib/api";

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
