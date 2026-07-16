import { describe, expect, it } from "vitest";
import {
  inspectionTargetKey,
  resolveInspectionTarget,
} from "@/features/workbench/state/target/_inspection-preview";

describe("target preview coordination", () => {
  it("resolves snapshot and historical targets without leaking catalog defaults", () => {
    expect(
      resolveInspectionTarget({
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
      resolveInspectionTarget({
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
      targetMode: "preset" as const,
      targetId: "BASELINE",
    };

    expect(inspectionTargetKey({ ...input, overrides: { beta: "2", alpha: "1" } }))
      .toBe(inspectionTargetKey({ ...input, overrides: { alpha: "1", beta: "2" } }));
    expect(inspectionTargetKey({ ...input, overrides: { alpha: "1" } }))
      .not.toBe(inspectionTargetKey({ ...input, overrides: { alpha: "2" } }));
  });

  it("distinguishes every browser-held Inspection identity field", () => {
    const base = {
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      experimentTask: "image-classification",
      dataset: "Mnist",
      targetMode: "preset" as const,
      targetId: "baseline",
      overrides: { hidden_size: "128" },
    };
    const baseKey = inspectionTargetKey(base);
    const variants = [
      { ...base, modelType: "experts" },
      { ...base, model: "linear_adaptive" },
      { ...base, preset: "fast" },
      { ...base, experimentTask: "fashion-classification" },
      { ...base, dataset: "FashionMnist" },
      { ...base, targetMode: "snapshot" as const },
      { ...base, targetId: "snapshot-baseline" },
      { ...base, overrides: { hidden_size: "256" } },
    ];

    for (const variant of variants) {
      expect(inspectionTargetKey(variant)).not.toBe(baseKey);
    }
    expect(new Set(variants.map(inspectionTargetKey)).size).toBe(
      variants.length,
    );
  });
});
