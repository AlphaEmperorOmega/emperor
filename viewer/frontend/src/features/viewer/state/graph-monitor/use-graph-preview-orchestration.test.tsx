import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  useGraphPreviewController,
  useGraphPreviewOrchestration,
} from "@/features/viewer/state/graph-monitor/use-graph-preview-orchestration";
import { type InspectResponse, type OperationGraphResponse } from "@/lib/api";

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

function graph(model: string, preset = "baseline"): InspectResponse {
  return {
    modelType: "linears",
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

function operationGraph(
  model: string,
  preset = "baseline",
): OperationGraphResponse {
  return {
    modelType: "linears",
    model,
    preset,
    source: "torch-export",
    status: "ok",
    nodes: [],
    edges: [],
    warnings: [],
  };
}

function controller(
  overrides: Partial<GraphPreviewControllerState>,
): GraphPreviewControllerState {
  return {
    graph: undefined,
    operationGraph: undefined,
    previewRequest: null,
    previewRequestKey: null,
    operationGraphRequestKey: null,
    operationGraphInFlightRequestKey: null,
    operationGraphFailedRequestKey: null,
    clearPreview: vi.fn(),
    requestPreview: vi.fn(),
    requestOperationGraph: vi.fn(),
    resetOperationGraphFailure: vi.fn(),
    previewInspection: {
      isBuilding: false,
      isError: false,
      error: null,
    },
    operationInspection: {
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
  targetModel: "neuron/neuron_linear",
  targetPreset: "baseline",
  targetDatasets: ["Mnist"],
};

describe("useGraphPreviewOrchestration", () => {
  it("does not expose a graph whose identity no longer matches the target", () => {
    const { result } = renderOrchestration({
      ...baseInput,
      controller: controller({
        graph: graph("experts/experts_linear"),
        operationGraph: operationGraph("experts/experts_linear"),
      }),
    });

    expect(result.current.graph.graph).toBeUndefined();
    expect(result.current.graph.operationGraph).toBeUndefined();
    expect(result.current.graph.nodes).toEqual([]);
    expect(result.current.graph.operationNodes).toEqual([]);
  });

  it("exposes the graph when its identity matches the target", async () => {
    const matchingGraph = graph("neuron/neuron_linear");
    const { result } = renderOrchestration({
      ...baseInput,
      controller: controller({
        graph: matchingGraph,
      }),
    });

    expect(result.current.graph.graph).toBe(matchingGraph);
    // Layout loads asynchronously (dagre is lazily imported); wait for nodes.
    await waitFor(() => expect(result.current.graph.nodes).not.toEqual([]));
  });
});
