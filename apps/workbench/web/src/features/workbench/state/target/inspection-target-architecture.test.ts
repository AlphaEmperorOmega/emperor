import ts from "typescript";
import { describe, expect, it } from "vitest";
import {
  callsNamed,
  readTypeScriptSource,
  typeLiteralPropertyNames,
  useStateSetterNames,
} from "@/test-utils/typescript-ast";

const orchestration = readTypeScriptSource(
  "src/features/workbench/state/target/use-model-package-inspection-state.ts",
);
const lifecycle = readTypeScriptSource(
  "src/features/workbench/state/target/_inspection-target-state.ts",
);

function isIdentifierArgument(
  argument: ts.Expression | undefined,
  name: string,
) {
  return argument !== undefined && ts.isIdentifier(argument) && argument.text === name;
}

describe("Inspection target architecture", () => {
  it("keeps every complete-target field and semantic revision in the lifecycle Module", () => {
    expect(useStateSetterNames(orchestration)).not.toEqual(
      expect.arrayContaining([
        "setSelectedModel",
        "setSelectedModelType",
        "setSelectedPreset",
        "setSelectedExperimentTask",
        "setSelectedDatasets",
        "setPresetOverrides",
        "setInspectionTransition",
      ]),
    );
    expect(
      typeLiteralPropertyNames(lifecycle, "InspectionTargetLifecycleState"),
    ).toEqual(
      expect.arrayContaining([
        "modelPackage",
        "selectedPreset",
        "target",
        "experimentTask",
        "datasets",
        "runtimeDefaults",
        "restoration",
        "transition",
      ]),
    );
  });

  it("derives Inspection request identity once from the complete lifecycle", () => {
    expect(callsNamed(orchestration, "resolveInspectionTarget")).toHaveLength(1);
    expect(
      callsNamed(orchestration, "requestPreview").some((call) =>
        isIdentifierArgument(call.arguments[0], "currentInspectionRequest"),
      ),
    ).toBe(true);
    expect(
      callsNamed(orchestration, "issueInspectionPreview").some(
        (call) =>
          isIdentifierArgument(
            call.arguments[0],
            "currentInspectionRequest",
          ) &&
          call.arguments[1] !== undefined &&
          ts.isStringLiteralLike(call.arguments[1]) &&
          call.arguments[1].text === "refresh",
      ),
    ).toBe(true);
  });
});
