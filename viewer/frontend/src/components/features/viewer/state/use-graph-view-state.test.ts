import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useGraphViewState } from "@/components/features/viewer/state/use-graph-view-state";
import { type GraphNode, type InspectResponse } from "@/lib/api";

function node(id: string): GraphNode {
  return {
    id,
    label: id,
    typeName: "Layer",
    path: id,
    graphRole: "architecture",
    parameterCount: 0,
    details: {},
    config: null,
  };
}

// Root n0 with child n1; both architecture so they survive basic-mode filtering
// and opened-scope expansion (root reveals its children).
const graph: InspectResponse = {
  model: "m",
  preset: "p",
  parameterCount: 0,
  nodes: [node("n0"), node("n1")],
  edges: [{ id: "n0-n1", source: "n0", target: "n1" }],
};

describe("useGraphViewState selection", () => {
  it("applies selection without relayout (non-selected nodes keep their reference)", () => {
    const { result } = renderHook(() => useGraphViewState(graph));

    const nodesBefore = result.current.nodes;
    const edgesBefore = result.current.edges;
    expect(nodesBefore.length).toBeGreaterThanOrEqual(2);

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
});
