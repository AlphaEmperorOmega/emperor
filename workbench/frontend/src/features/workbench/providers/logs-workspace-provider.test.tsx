import { render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  includeStartedExperiment: vi.fn(),
}));

vi.mock("@/features/workbench/providers/workbench-providers", () => ({
  useTargetCatalog: () => ({
    capabilities: { logDeletionEnabled: true },
  }),
  useModelTargetConfig: () => ({
    selectedModelType: "linears",
    selectedModel: "linear",
    selectedPreset: "baseline",
    selectedPresetMeta: undefined,
    selectedDatasets: ["Mnist"],
  }),
  useActiveTrainingJob: () => ({ activeTrainingJob: undefined }),
}));

vi.mock("@/features/workbench/state/logs/use-logs-workspace-state", () => ({
  useLogsWorkspaceState: () => ({
    includeStartedExperiment: mocks.includeStartedExperiment,
  }),
}));

import { LogsWorkspaceProvider } from "@/features/workbench/providers/logs-workspace-provider";

beforeEach(() => {
  mocks.includeStartedExperiment.mockReset();
});

describe("LogsWorkspaceProvider", () => {
  it("seeds every Training folder observed before the provider mounts", async () => {
    render(
      <LogsWorkspaceProvider
        enabled
        startedExperiments={["first_run", "second_run"]}
      >
        <span>Logs child</span>
      </LogsWorkspaceProvider>,
    );

    await waitFor(() => {
      expect(mocks.includeStartedExperiment.mock.calls).toEqual([
        ["first_run"],
        ["second_run"],
      ]);
    });
  });
});
