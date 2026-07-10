import { describe, expect, it } from "vitest";
import {
  previewTargetKey,
  resolvePreviewTarget,
} from "@/features/workbench/state/target/target-preview";

describe("target preview coordination", () => {
  it("resolves snapshot and historical targets without leaking catalog defaults", () => {
    expect(
      resolvePreviewTarget({
        selectedTargetMode: "snapshot",
        selectedSnapshotId: "snapshot-1",
        selectedExperimentTarget: null,
        selectedPreset: "BASELINE",
        selectedExperimentTask: "classification",
        selectedDatasets: ["Mnist"],
      }),
    ).toEqual({
      targetMode: "snapshot",
      targetId: "snapshot-1",
      preset: "BASELINE",
      experimentTask: "classification",
      dataset: "Mnist",
    });

    expect(
      resolvePreviewTarget({
        selectedTargetMode: "experiment",
        selectedSnapshotId: "",
        selectedExperimentTarget: {
          runId: "run-1",
          experiment: "experiment-1",
          preset: "GATING",
          dataset: "Cifar10",
          experimentTask: "image-classification",
        },
        selectedPreset: "BASELINE",
        selectedExperimentTask: "classification",
        selectedDatasets: ["Mnist"],
      }),
    ).toEqual({
      targetMode: "experiment",
      targetId: "run-1",
      preset: "GATING",
      experimentTask: "image-classification",
      dataset: "Cifar10",
    });
  });

  it("keys requests by target identity and normalized override content", () => {
    const input = {
      modelType: "linears",
      model: "linear",
      preset: "BASELINE",
      experimentTask: "classification",
      dataset: "Mnist",
      mode: "preset" as const,
      target: "BASELINE",
    };

    expect(previewTargetKey({ ...input, overrides: { beta: "2", alpha: "1" } }))
      .toBe(previewTargetKey({ ...input, overrides: { alpha: "1", beta: "2" } }));
    expect(previewTargetKey({ ...input, overrides: { alpha: "1" } }))
      .not.toBe(previewTargetKey({ ...input, overrides: { alpha: "2" } }));
  });
});
