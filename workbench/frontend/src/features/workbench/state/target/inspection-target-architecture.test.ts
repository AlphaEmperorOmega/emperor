import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const orchestration = readFileSync(
  new URL("./use-model-package-inspection-state.ts", import.meta.url),
  "utf8",
);
const lifecycle = readFileSync(
  new URL("./_inspection-target-state.ts", import.meta.url),
  "utf8",
);

describe("Inspection target architecture", () => {
  it("keeps every complete-target field and semantic revision in the lifecycle Module", () => {
    for (const forbiddenSetter of [
      "setSelectedModel",
      "setSelectedModelType",
      "setSelectedPreset",
      "setSelectedExperimentTask",
      "setSelectedDatasets",
      "setPresetOverrides",
      "setInspectionTransition",
    ]) {
      expect(orchestration).not.toContain(forbiddenSetter);
    }
    for (const ownedField of [
      "modelPackage",
      "selectedPreset",
      "target",
      "experimentTask",
      "datasets",
      "runtimeDefaults",
      "restoration",
      "transition",
    ]) {
      expect(lifecycle).toContain(ownedField);
    }
  });

  it("derives Inspection request identity once from the complete lifecycle", () => {
    expect(orchestration.match(/resolveInspectionTarget\s*\(/g)).toHaveLength(1);
    expect(orchestration).toContain("requestPreview(currentInspectionRequest)");
    expect(orchestration).toContain(
      'issueInspectionPreview(currentInspectionRequest, "refresh")',
    );
  });
});
