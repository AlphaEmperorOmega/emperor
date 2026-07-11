import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  useGraphPreviewOrchestration,
} from "@/features/workbench/state/graph-monitor/use-graph-preview-orchestration";
import { type InspectResponse } from "@/lib/api";

type Input = Parameters<typeof useGraphPreviewOrchestration>[0];

function renderOrchestration(input: Input) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return renderHook(
    ({ value }: { value: Input }) => useGraphPreviewOrchestration(value),
    {
      initialProps: { value: input },
      wrapper: ({ children }: { children: ReactNode }) =>
        createElement(QueryClientProvider, { client }, children),
    },
  );
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
      {
        id: "__child__",
        label: "Child",
        typeName: "Layer",
        path: "model.child",
        graphRole: "architecture",
        parameterCount: 0,
        parameterSizeBytes: 0,
        details: {},
        config: null,
      },
      {
        id: "__leaf__",
        label: "Leaf",
        typeName: "Layer",
        path: "model.child.leaf",
        graphRole: "architecture",
        parameterCount: 0,
        parameterSizeBytes: 0,
        details: {},
        config: null,
      },
    ],
    edges: [
      { id: "root-child", source: "__root__", target: "__child__" },
      { id: "child-leaf", source: "__child__", target: "__leaf__" },
    ],
  };
}

function inspection(
  response: InspectResponse | undefined,
  revision = 0,
  cause: "target-changed" | "inspection-refreshed" = "target-changed",
): Input["inspection"] {
  return {
    graph: response,
    status: { isBuilding: false, isError: false, error: null },
    transition: { revision, cause },
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
  targetPreset: "baseline",
  targetDatasets: ["Mnist"],
};

describe("useGraphPreviewOrchestration", () => {
  it.each(["target-changed", "inspection-refreshed"] as const)(
    "applies each %s Inspection transition once without handler binding",
    async (cause) => {
      const matchingGraph = graph("linear");
      const initialInput: Input = {
        ...baseInput,
        inspection: inspection(matchingGraph),
      };
      const { result, rerender } = renderOrchestration(initialInput);

      await waitFor(() => expect(result.current.graph.nodes).not.toEqual([]));
      act(() => result.current.graph.revealGraphNode("__leaf__"));
      expect(result.current.graph.selectedNodeId).toBe("__leaf__");
      expect(result.current.graph.expandedGraphNodeIds.size).toBeGreaterThan(0);

      rerender({
        value: {
          ...initialInput,
          inspection: inspection(matchingGraph, 1, cause),
        },
      });
      await waitFor(() => {
        expect(result.current.graph.selectedNodeId).toBeNull();
        expect(result.current.graph.expandedGraphNodeIds.size).toBe(0);
      });

      act(() => result.current.graph.revealGraphNode("__leaf__"));
      rerender({
        value: {
          ...initialInput,
          inspection: inspection(matchingGraph, 1, cause),
        },
      });
      expect(result.current.graph.selectedNodeId).toBe("__leaf__");
      expect(result.current.graph.expandedGraphNodeIds.size).toBeGreaterThan(0);
    },
  );

  it("initializes from an advanced transition revision before graph arrival", async () => {
    const matchingGraph = graph("linear");
    const initialInput: Input = {
      ...baseInput,
      inspection: inspection(undefined, 7),
    };
    const { result, rerender } = renderOrchestration(initialInput);

    expect(result.current.graph.nodes).toEqual([]);
    rerender({
      value: {
        ...initialInput,
        inspection: inspection(matchingGraph, 7),
      },
    });
    await waitFor(() => expect(result.current.graph.nodes).not.toEqual([]));
    act(() => result.current.graph.revealGraphNode("__leaf__"));
    expect(result.current.graph.selectedNodeId).toBe("__leaf__");

    rerender({
      value: {
        ...initialInput,
        inspection: inspection(matchingGraph, 8),
      },
    });
    await waitFor(() => {
      expect(result.current.graph.selectedNodeId).toBeNull();
      expect(result.current.graph.expandedGraphNodeIds.size).toBe(0);
    });
  });

  it("renders the graph supplied by the Inspection projection", async () => {
    const matchingGraph = graph("linear");
    const { result } = renderOrchestration({
      ...baseInput,
      inspection: inspection(matchingGraph),
    });

    expect(result.current.graph.graph).toBe(matchingGraph);
    await waitFor(() => expect(result.current.graph.nodes).not.toEqual([]));
  });
});
