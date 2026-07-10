import { describe, expect, it } from "vitest";
import {
  datasetsForExperimentTask,
  experimentTaskOptions,
  normalizeExperimentTask,
} from "@/features/workbench/state/target/target-dataset-catalog";
import { type DatasetGroup } from "@/lib/api";

const groups: DatasetGroup[] = [
  {
    experimentTask: "classification",
    label: "Classification",
    datasets: [{ name: "Mnist", label: "Digits", inputDim: 784, outputDim: 10 }],
  },
  {
    experimentTask: "generation",
    label: "",
    datasets: [
      { name: "TinyStories", label: "Text", inputDim: 128, outputDim: 128 },
    ],
  },
];

describe("target dataset catalog", () => {
  it("normalizes task selection through current, default, and first fallbacks", () => {
    expect(normalizeExperimentTask("generation", "classification", groups))
      .toBe("generation");
    expect(normalizeExperimentTask("missing", "classification", groups))
      .toBe("classification");
    expect(normalizeExperimentTask("missing", "missing", groups))
      .toBe("classification");
  });

  it("projects task options and datasets from the same catalog seam", () => {
    expect(experimentTaskOptions(groups)).toEqual([
      { value: "classification", label: "Classification" },
      { value: "generation", label: "generation" },
    ]);
    expect(datasetsForExperimentTask(groups, "generation")).toEqual([
      { name: "TinyStories", label: "Text", inputDim: 128, outputDim: 128 },
    ]);
    expect(datasetsForExperimentTask(groups, "missing")).toEqual([]);
  });
});
