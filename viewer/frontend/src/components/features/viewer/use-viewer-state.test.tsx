import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  fetchHealth: vi.fn(),
  fetchModels: vi.fn(),
  fetchPresets: vi.fn(),
  fetchDatasets: vi.fn(),
  fetchMonitors: vi.fn(),
  fetchConfigSchema: vi.fn(),
  fetchSearchSpace: vi.fn(),
  fetchLogRuns: vi.fn(),
  fetchLogTags: vi.fn(),
  inspectModel: vi.fn(),
}));
vi.mock("@/lib/api", () => mocks);

import { useViewerState } from "@/components/features/viewer/use-viewer-state";

function renderViewerState() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return renderHook(() => useViewerState(), {
    wrapper: ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children),
  });
}

beforeEach(() => {
  mocks.fetchHealth.mockReset().mockResolvedValue({ status: "ok" });
  mocks.fetchModels.mockReset().mockResolvedValue({ models: ["linear"] });
  mocks.fetchPresets.mockReset().mockResolvedValue({
    model: "linear",
    presets: [
      { name: "baseline", label: "Baseline", description: "" },
      { name: "fast", label: "Fast", description: "" },
    ],
  });
  mocks.fetchDatasets.mockReset().mockResolvedValue({
    model: "linear",
    datasets: [
      { name: "Mnist", label: "MNIST", inputDim: 784, outputDim: 10 },
    ],
  });
  mocks.fetchMonitors.mockReset().mockResolvedValue({
    model: "linear",
    monitors: [],
  });
  mocks.fetchConfigSchema.mockReset().mockResolvedValue({
    model: "linear",
    fields: [],
  });
  mocks.fetchSearchSpace.mockReset().mockResolvedValue({
    model: "linear",
    preset: "baseline",
    axes: [],
  });
  mocks.fetchLogRuns.mockReset().mockResolvedValue({ runs: [] });
  mocks.fetchLogTags.mockReset().mockResolvedValue({ runs: [] });
  mocks.inspectModel.mockReset().mockResolvedValue({
    model: "linear",
    preset: "baseline",
    parameterCount: 0,
    nodes: [],
    edges: [],
  });
});

describe("useViewerState", () => {
  it("settles the auto-selected training preset without an update loop", async () => {
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    });
  });
});
