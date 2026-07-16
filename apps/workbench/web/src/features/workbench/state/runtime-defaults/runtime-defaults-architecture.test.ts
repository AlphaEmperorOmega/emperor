import { describe, expect, it } from "vitest";
import {
  identifierCount,
  readTypeScriptSource,
  staticImportSources,
} from "@/test-utils/typescript-ast";

const runtimeDefaultsInterface =
  "@/features/workbench/state/runtime-defaults/runtime-defaults";
const runtimeDefaultsPresentation =
  "@/features/workbench/state/runtime-defaults/runtime-defaults-presentation";

const genericConfig = readTypeScriptSource("src/lib/config.ts");
const runtimeDefaults = readTypeScriptSource(
  "src/features/workbench/state/runtime-defaults/runtime-defaults.ts",
);
const presentation = readTypeScriptSource(
  "src/features/workbench/state/runtime-defaults/runtime-defaults-presentation.ts",
);
const fullConfigDialog = readTypeScriptSource(
  "src/features/workbench/components/config/full-config-dialog.tsx",
);
const lifecycleOwners = [
  "src/features/workbench/state/config-snapshots/use-config-snapshot-editor.ts",
  "src/features/workbench/state/model-package/model-package-selection.ts",
  "src/features/workbench/state/target/use-model-package-inspection-state.ts",
  "src/features/workbench/state/training/use-training-draft-state.ts",
  "src/features/workbench/state/training/use-training-plan-state.ts",
].map(readTypeScriptSource);

describe("Runtime Defaults architecture", () => {
  it("keeps canonical editing and preset projection out of generic config", () => {
    expect(identifierCount(genericConfig, "runtimeDefaultsEditor")).toBe(0);
    expect(identifierCount(genericConfig, "effectivePresetOverrides")).toBe(0);
    expect(
      identifierCount(genericConfig, "inactivePresetOwnedOverrideKeys"),
    ).toBe(0);
    expect(staticImportSources(runtimeDefaults)).toContain("@/lib/config");
    expect(staticImportSources(genericConfig)).not.toContain(
      runtimeDefaultsInterface,
    );
  });

  it("routes lifecycle owners through the Runtime Defaults interface", () => {
    for (const owner of lifecycleOwners) {
      expect(staticImportSources(owner)).toContain(runtimeDefaultsInterface);
    }
  });

  it("keeps schema presentation with its domain owner", () => {
    expect(staticImportSources(presentation)).toContain(
      runtimeDefaultsInterface,
    );
    expect(staticImportSources(fullConfigDialog)).toContain(
      runtimeDefaultsPresentation,
    );
  });
});
