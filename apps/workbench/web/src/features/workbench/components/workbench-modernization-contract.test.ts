import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  dynamicImportSources,
  hasJsxExpressionIdentifier,
  identifierCount,
  jsxElementNames,
  readTypeScriptSource,
  staticImportSources,
  variableDeclaration,
} from "@/test-utils/typescript-ast";

const app = readTypeScriptSource(
  "src/features/workbench/components/workbench-app.tsx",
);
const workspaces = readTypeScriptSource(
  "src/features/workbench/components/workbench-workspaces.tsx",
);
const screen = readTypeScriptSource(
  "src/features/workbench/components/workbench-screen.tsx",
);
const modelWorkspace = readTypeScriptSource(
  "src/features/workbench/components/workbench-model-workspace.tsx",
);
const providers = readTypeScriptSource(
  "src/features/workbench/providers/workbench-providers.tsx",
);
const graphNodeView = readTypeScriptSource(
  "src/features/workbench/components/graph/graph-node-view.tsx",
);
const nextConfig = readTypeScriptSource("next.config.ts");

const removedProductionModules = [
  "src/features/workbench/components/graph/graph-node-diagrams.ts",
  "src/features/workbench/state/full-config/runtime-defaults-schema-presentation.ts",
  "src/features/workbench/state/training/config-snapshot-run-plan.ts",
  "src/features/workbench/state/training/use-training-job-lifecycle.ts",
  "src/lib/api.ts",
  "src/lib/training/summary.ts",
] as const;

describe("Next 16 workspace modernization contract", () => {
  it("keeps the measured-failing View Transition experiment disabled", () => {
    expect(identifierCount(nextConfig, "viewTransition")).toBe(0);
    expect(identifierCount(screen, "ViewTransition")).toBe(0);
  });

  it("keeps Training execution outside Activity and Logs state inside it", () => {
    const logs = variableDeclaration(workspaces, "logs");
    const logsActivity = variableDeclaration(workspaces, "logsActivity");
    const trainingActivity = variableDeclaration(
      workspaces,
      "trainingActivity",
    );
    const trainingBoundary = variableDeclaration(
      workspaces,
      "trainingBoundary",
    );

    expect(jsxElementNames(logs?.initializer)).toEqual(
      expect.arrayContaining([
        "DeferredLogsWorkspaceProvider",
        "WorkbenchThreeRegionLayout",
      ]),
    );
    expect(jsxElementNames(logsActivity?.initializer)).toContain("Activity");
    expect(hasJsxExpressionIdentifier(logsActivity?.initializer, "logs")).toBe(
      true,
    );
    expect(jsxElementNames(trainingActivity?.initializer)).toContain(
      "Activity",
    );
    expect(
      hasJsxExpressionIdentifier(trainingActivity?.initializer, "training"),
    ).toBe(true);
    expect(jsxElementNames(trainingBoundary?.initializer)).toContain(
      "DeferredTrainingExecutionProvider",
    );
    expect(jsxElementNames(trainingBoundary?.initializer)).not.toContain(
      "Activity",
    );
    expect(
      hasJsxExpressionIdentifier(trainingBoundary?.initializer, "activity"),
    ).toBe(true);

    const appImports = [
      ...staticImportSources(app),
      ...dynamicImportSources(app),
    ];
    expect(appImports).not.toContain(
      "@/features/workbench/providers/logs-workspace-provider",
    );
    expect(appImports).not.toContain(
      "@/features/workbench/providers/training-execution-provider",
    );
  });

  it("keeps deferred workspace code behind explicit dynamic import seams", () => {
    expect(dynamicImportSources(workspaces)).toEqual(
      expect.arrayContaining([
        "@/features/workbench/providers/training-execution-provider",
        "@/features/workbench/providers/logs-workspace-provider",
        "@/features/workbench/components/training-panel",
        "@/features/workbench/components/workbench-model-workspace",
      ]),
    );
    expect(staticImportSources(modelWorkspace)).toEqual(
      expect.arrayContaining([
        "@/features/workbench/components/screen/preview-panel",
        "@/features/workbench/components/screen/node-details-panel",
        "@/features/workbench/components/screen/preview-toolbar",
        "@/features/workbench/components/workbench-model-sidebar",
      ]),
    );
    expect(dynamicImportSources(screen)).toContain(
      "@/features/workbench/components/workbench-overlays",
    );
    expect([
      ...staticImportSources(providers),
      ...dynamicImportSources(providers),
    ]).not.toContain(
      "@/features/workbench/providers/training-execution-context",
    );
  });

  it("keeps deleted compatibility Modules absent and imports graph diagrams directly", () => {
    for (const relativePath of removedProductionModules) {
      expect(existsSync(resolve(process.cwd(), relativePath))).toBe(false);
    }

    const graphImports = staticImportSources(graphNodeView);
    expect(graphImports).toEqual(
      expect.arrayContaining([
        "@/features/workbench/components/graph/graph-node-cluster-diagram",
        "@/features/workbench/components/graph/graph-node-expert-diagram",
        "@/features/workbench/components/graph/graph-node-stack-diagram",
      ]),
    );
    expect(graphImports).not.toContain(
      "@/features/workbench/components/graph/graph-node-diagrams",
    );
  });
});
