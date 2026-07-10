import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  useGraphPreviewController,
  useGraphPreviewOrchestration,
} from "@/features/workbench/state/graph-monitor/use-graph-preview-orchestration";
import { type InspectResponse } from "@/lib/api";

type GraphPreviewControllerState = ReturnType<typeof useGraphPreviewController>;

function renderOrchestration(
  input: Parameters<typeof useGraphPreviewOrchestration>[0],
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return renderHook(() => useGraphPreviewOrchestration(input), {
    wrapper: ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children),
  });
}

function graph(
  model: string,
  preset = "baseline",
  modelType = "neuron",
): InspectResponse {
  return {
    modelType,
    model,
    preset,
    parameterCount: 0,
    parameterSizeBytes: 0,
    nodes: [
      {
        id: "__root__",
        label: "Model",
        typeName: "Model",
        path: "model",
        graphRole: "architecture",
        parameterCount: 0,
        parameterSizeBytes: 0,
        details: {},
        config: null,
      },
    ],
    edges: [],
  };
}

function controller(
  overrides: Partial<GraphPreviewControllerState>,
): GraphPreviewControllerState {
  return {
    graph: undefined,
    previewRequest: null,
    clearPreview: vi.fn(),
    requestPreview: vi.fn(),
    previewInspection: {
      isBuilding: false,
      isError: false,
      error: null,
    },
    resetGraphSelectionAndExpansion: vi.fn(),
    resetGraphExpansion: vi.fn(),
    bindGraphResetHandlers: vi.fn(),
    ...overrides,
  };
}

const baseInput = {
  activeTrainingJob: undefined,
  historicalMonitorRuns: [],
  selectedHistoricalExperiment: "",
  selectedHistoricalDataset: "",
  selectedHistoricalPreset: "",
  logRunTags: [],
  filteredHistoricalRunIds: [],
  targetModelType: "neuron",
  targetModel: "linear",
  targetPreset: "baseline",
  targetDatasets: ["Mnist"],
  targetMode: "preset" as const,
  targetId: "baseline",
};

describe("useGraphPreviewOrchestration", () => {
  it("does not expose a graph whose identity no longer matches the target", () => {
    const { result } = renderOrchestration({
      ...baseInput,
      controller: controller({
        graph: graph("linear", "baseline", "experts"),
      }),
    });

    expect(result.current.graph.graph).toBeUndefined();
    expect(result.current.graph.nodes).toEqual([]);
  });

  it("exposes the graph when its identity matches the target", async () => {
    const matchingGraph = graph("linear");
    const { result } = renderOrchestration({
      ...baseInput,
      controller: controller({
        graph: matchingGraph,
        previewRequest: {
          modelType: "neuron",
          model: "linear",
          preset: "baseline",
          dataset: "Mnist",
          overrides: {},
          targetMode: "preset",
          targetId: "baseline",
        },
      }),
    });

    expect(result.current.graph.graph).toBe(matchingGraph);
    // Layout loads asynchronously (dagre is lazily imported); wait for nodes.
    await waitFor(() => expect(result.current.graph.nodes).not.toEqual([]));
  });

  it("exposes an experiment graph when the backend returns a canonical preset name", async () => {
    const matchingGraph = graph("linear", "baseline", "linears");
    const { result } = renderOrchestration({
      ...baseInput,
      targetModelType: "linears",
      targetModel: "linear",
      targetPreset: "BASELINE",
      targetMode: "experiment",
      targetId: "run-1",
      controller: controller({
        graph: matchingGraph,
        previewRequest: {
          modelType: "linears",
          model: "linear",
          preset: "BASELINE",
          dataset: "Mnist",
          overrides: {},
          targetMode: "experiment",
          targetId: "run-1",
          logRunId: "run-1",
        },
      }),
    });

    expect(result.current.graph.graph).toBe(matchingGraph);
    await waitFor(() => expect(result.current.graph.nodes).not.toEqual([]));
  });

  it("does not expose a graph when the preview request belongs to another dataset", () => {
    const { result } = renderOrchestration({
      ...baseInput,
      controller: controller({
        graph: graph("linear"),
        previewRequest: {
          modelType: "neuron",
          model: "linear",
          preset: "baseline",
          dataset: "Cifar10",
          overrides: {},
          targetMode: "preset",
          targetId: "baseline",
        },
      }),
    });

    expect(result.current.graph.graph).toBeUndefined();
    expect(result.current.graph.nodes).toEqual([]);
  });

  it("does not expose a graph when the preview request belongs to another experiment target", () => {
    const { result } = renderOrchestration({
      ...baseInput,
      targetMode: "experiment",
      targetId: "run-new",
      controller: controller({
        graph: graph("linear"),
        previewRequest: {
          modelType: "neuron",
          model: "linear",
          preset: "baseline",
          dataset: "Mnist",
          overrides: {},
          targetMode: "experiment",
          targetId: "run-old",
        },
      }),
    });

    expect(result.current.graph.graph).toBeUndefined();
    expect(result.current.graph.nodes).toEqual([]);
  });
});
