import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const appSource = readFileSync(
  resolve(process.cwd(), "src/features/workbench/components/workbench-app.tsx"),
  "utf8",
);
const workspacesSource = readFileSync(
  resolve(
    process.cwd(),
    "src/features/workbench/components/workbench-workspaces.tsx",
  ),
  "utf8",
);
const screenSource = readFileSync(
  resolve(process.cwd(), "src/features/workbench/components/workbench-screen.tsx"),
  "utf8",
);
const modelWorkspaceSource = readFileSync(
  resolve(
    process.cwd(),
    "src/features/workbench/components/workbench-model-workspace.tsx",
  ),
  "utf8",
);
const providersSource = readFileSync(
  resolve(
    process.cwd(),
    "src/features/workbench/providers/workbench-providers.tsx",
  ),
  "utf8",
);
const nextConfigSource = readFileSync(
  resolve(process.cwd(), "next.config.ts"),
  "utf8",
);

describe("Next 16 workspace modernization contract", () => {
  it("keeps the measured-failing View Transition experiment disabled", () => {
    expect(nextConfigSource).not.toContain("viewTransition");
    expect(screenSource).not.toContain("ViewTransition");
  });

  it("keeps Training execution outside Activity and Logs state inside it", () => {
    expect(workspacesSource).toMatch(
      /<DeferredTrainingExecutionProvider[\s\S]*\{activity\}[\s\S]*<\/DeferredTrainingExecutionProvider>/,
    );
    expect(workspacesSource).toContain("<Activity");
    expect(workspacesSource).toMatch(
      /const logs = \([\s\S]*<DeferredLogsWorkspaceProvider[\s\S]*<WorkbenchThreeRegionLayout/,
    );
    expect(appSource).not.toContain("DeferredLogsWorkspaceProvider");
    expect(appSource).not.toContain("DeferredTrainingExecutionProvider");
  });

  it("keeps deferred workspace code behind explicit dynamic import seams", () => {
    expect(workspacesSource).toMatch(
      /import\(\s*"@\/features\/workbench\/providers\/training-execution-provider"/,
    );
    expect(workspacesSource).toContain(
      'import("@/features/workbench/providers/logs-workspace-provider")',
    );
    expect(workspacesSource).toContain(
      'import("@/features/workbench/components/training-panel")',
    );
    expect(workspacesSource).toContain(
      'import("@/features/workbench/components/workbench-model-workspace")',
    );
    expect(modelWorkspaceSource).toContain(
      'from "@/features/workbench/components/screen/preview-panel"',
    );
    expect(modelWorkspaceSource).toContain(
      'from "@/features/workbench/components/screen/node-details-panel"',
    );
    expect(modelWorkspaceSource).toContain(
      'from "@/features/workbench/components/screen/preview-toolbar"',
    );
    expect(modelWorkspaceSource).toContain(
      'from "@/features/workbench/components/workbench-model-sidebar"',
    );
    expect(screenSource).toContain(
      'import("@/features/workbench/components/workbench-overlays")',
    );
    expect(providersSource).not.toContain("training-execution-context");
  });
});
