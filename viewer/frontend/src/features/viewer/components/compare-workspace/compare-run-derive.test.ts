import { describe, expect, it } from "vitest";
import {
  buildCompareMetricSummaryRows,
  defaultCompareMetricTags,
} from "@/features/viewer/components/compare-workspace/compare-run-derive";

describe("compare run derivation", () => {
  it("prioritizes validation/train accuracy and loss defaults", () => {
    expect(
      defaultCompareMetricTags(
        [
          { value: "parameters/global_norm" },
          { value: "train/loss" },
          { value: "validation/accuracy" },
          { value: "validation/loss" },
          { value: "train/accuracy" },
        ],
        ["parameters/global_norm"],
      ),
    ).toEqual([
      "validation/accuracy",
      "train/accuracy",
      "validation/loss",
      "train/loss",
    ]);
  });

  it("summarizes accuracy with higher best values highlighted", () => {
    const rows = buildCompareMetricSummaryRows({
      runIds: ["run-a", "run-b"],
      selectedTags: ["validation/accuracy"],
      series: [
        {
          runId: "run-a",
          tag: "validation/accuracy",
          points: [
            { step: 1, wallTime: 1, value: 0.6 },
            { step: 2, wallTime: 2, value: 0.8 },
          ],
        },
        {
          runId: "run-b",
          tag: "validation/accuracy",
          points: [
            { step: 1, wallTime: 1, value: 0.4 },
            { step: 2, wallTime: 2, value: 0.55 },
          ],
        },
      ],
    });

    expect(rows.find((row) => row.summary === "first")?.values).toEqual([
      { text: "0.6", highlighted: false },
      { text: "0.4", highlighted: false },
    ]);
    expect(rows.find((row) => row.summary === "best")?.values).toEqual([
      { text: "0.8", highlighted: true },
      { text: "0.55", highlighted: false },
    ]);
    expect(rows.find((row) => row.summary === "delta")?.values).toEqual([
      { text: "+0.2" },
      { text: "+0.15" },
    ]);
  });

  it("summarizes loss with lower best values highlighted and missing values blank", () => {
    const rows = buildCompareMetricSummaryRows({
      runIds: ["run-a", "run-b", "run-c"],
      selectedTags: ["train/loss"],
      series: [
        {
          runId: "run-a",
          tag: "train/loss",
          points: [
            { step: 1, wallTime: 1, value: 0.7 },
            { step: 2, wallTime: 2, value: 0.3 },
          ],
        },
        {
          runId: "run-b",
          tag: "train/loss",
          points: [
            { step: 1, wallTime: 1, value: 0.9 },
            { step: 2, wallTime: 2, value: 0.5 },
          ],
        },
      ],
    });

    expect(rows.find((row) => row.summary === "best")?.values).toEqual([
      { text: "0.3", highlighted: true },
      { text: "0.5", highlighted: false },
      { text: "—" },
    ]);
    expect(rows.find((row) => row.summary === "best-step")?.values).toEqual([
      { text: "2" },
      { text: "2" },
      { text: "—" },
    ]);
  });
});
