import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useGraphViewState } from "@/features/viewer/state/graph-monitor/use-graph-view-state";
import { type GraphNode, type InspectResponse } from "@/lib/api";

function node(id: string): GraphNode {
  return {
    id,
    label: id,
    typeName: "Layer",
    path: id,
    graphRole: "architecture",
    parameterCount: 0,
    parameterSizeBytes: 0,
    details: {},
    config: null,
  };
}

function roleNode(id: string, graphRole: GraphNode["graphRole"]): GraphNode {
  return {
    ...node(id),
    graphRole,
  };
}

// Root n0 with child n1; both architecture so they survive basic-mode filtering
// and opened-scope expansion (root reveals its children).
const graph: InspectResponse = {
  model: "m",
  preset: "p",
  parameterCount: 0,
  parameterSizeBytes: 0,
  nodes: [node("n0"), node("n1")],
  edges: [{ id: "n0-n1", source: "n0", target: "n1" }],
};

describe("useGraphViewState selection", () => {
  it("defaults to graph visualization and can switch to parameters", () => {
    const { result } = renderHook(() => useGraphViewState(graph));

    expect(result.current.previewVisualizationMode).toBe("graph");

    act(() => result.current.setPreviewVisualizationMode("parameters"));

    expect(result.current.previewVisualizationMode).toBe("parameters");
  });

  it("applies selection without relayout (non-selected nodes keep their reference)", async () => {
    const { result } = renderHook(() => useGraphViewState(graph));

    // Layout loads asynchronously (dagre is lazily imported); wait for the
    // structural pass before asserting on laid-out nodes.
    await waitFor(() =>
      expect(result.current.nodes.length).toBeGreaterThanOrEqual(2),
    );

    const nodesBefore = result.current.nodes;
    const edgesBefore = result.current.edges;

    const n0Before = nodesBefore.find((candidate) => candidate.id === "n0")!;

    act(() => result.current.setSelectedNodeId("n1"));

    const nodesAfter = result.current.nodes;
    const n0After = nodesAfter.find((candidate) => candidate.id === "n0")!;
    const n1After = nodesAfter.find((candidate) => candidate.id === "n1")!;

    // The unaffected node keeps its exact object reference — proof that dagre
    // did not re-run (a relayout would allocate all-new node objects).
    expect(n0After).toBe(n0Before);
    expect(n0After.selected).toBe(false);
    // Only the selected node gets a new object with the flag flipped.
    expect(n1After.selected).toBe(true);
    // Edges are produced by the structural pass and stay untouched.
    expect(result.current.edges).toBe(edgesBefore);
  });

  it("preserves parameter focus across preview tab switches", () => {
    const { result } = renderHook(() => useGraphViewState(graph));

    act(() => {
      result.current.setParameterFocusNodeId("n1");
      result.current.setPreviewVisualizationMode("parameters");
    });

    expect(result.current.parameterFocusNodeId).toBe("n1");

    act(() => result.current.setPreviewVisualizationMode("graph"));

    expect(result.current.parameterFocusNodeId).toBe("n1");
  });

  it("resets parameter focus when the inspected graph changes", async () => {
    const nextGraph: InspectResponse = {
      ...graph,
      preset: "next",
      nodes: [node("next-root")],
      edges: [],
    };
    const { result, rerender } = renderHook(
      ({ inputGraph }: { inputGraph: InspectResponse }) => useGraphViewState(inputGraph),
      { initialProps: { inputGraph: graph } },
    );

    act(() => result.current.setParameterFocusNodeId("n1"));
    expect(result.current.parameterFocusNodeId).toBe("n1");

    rerender({ inputGraph: nextGraph });

    await waitFor(() => {
      expect(result.current.parameterFocusNodeId).toBeNull();
    });
  });

  it("falls parameter focus back to a visible ancestor on detail changes", async () => {
    const fullGraph: InspectResponse = {
      model: "m",
      preset: "p",
      parameterCount: 0,
      parameterSizeBytes: 0,
      nodes: [
        roleNode("model", "architecture"),
        roleNode("internal", "internal"),
        roleNode("leaf", "architecture"),
      ],
      edges: [
        { id: "model-internal", source: "model", target: "internal" },
        { id: "internal-leaf", source: "internal", target: "leaf" },
      ],
    };
    const { result } = renderHook(() => useGraphViewState(fullGraph));

    act(() => result.current.setGraphDetailMode("full"));
    act(() => result.current.setParameterFocusNodeId("internal"));
    expect(result.current.parameterFocusNodeId).toBe("internal");

    act(() => result.current.setGraphDetailMode("basic"));

    await waitFor(() => {
      expect(result.current.parameterFocusNodeId).toBe("model");
    });
  });
});
