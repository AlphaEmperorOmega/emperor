import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useGraphViewState } from "@/features/workbench/state/graph-monitor/use-graph-view-state";
import type { GraphNode, InspectResponse } from "@/lib/api/inspection";

function node(id: string, overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id,
    label: overrides.label ?? id,
    typeName: overrides.typeName ?? "Layer",
    path: overrides.path ?? id,
    graphRole: overrides.graphRole ?? "architecture",
    parameterCount: overrides.parameterCount ?? 0,
    parameterSizeBytes: overrides.parameterSizeBytes ?? 0,
    details: overrides.details ?? {},
    config: overrides.config ?? null,
  };
}

// Root n0 with child n1; both architecture so they survive basic-mode filtering
// and opened-scope expansion (root reveals its children).
const graph: InspectResponse = {
  modelType: "linears",
  model: "m",
  preset: "p",
  parameterCount: 0,
  parameterSizeBytes: 0,
  nodes: [node("n0"), node("n1")],
  edges: [{ id: "n0-n1", source: "n0", target: "n1" }],
};

describe("useGraphViewState selection", () => {
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

  it("maps parameter activity through a resolver that is independent from monitor availability", async () => {
    const root = node("root", {
      typeName: "LayerStack",
      path: "main_model",
    });
    const layer = node("layer", {
      typeName: "Layer",
      path: "main_model.0",
      config: {
        typeName: "LayerConfig",
        fields: [
          { key: "input_dim", value: 16 },
          { key: "output_dim", value: 16 },
        ],
      },
    });
    const linear = node("linear", {
      typeName: "LinearLayer",
      path: "main_model.0.model",
      details: {
        dims: "32 -> 32",
        weightShape: "32 x 32",
      },
    });
    const monitorGraph: InspectResponse = {
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      parameterCount: 0,
      parameterSizeBytes: 0,
      nodes: [root, layer, linear],
      edges: [
        { id: "root-layer", source: root.id, target: layer.id },
        { id: "layer-linear", source: layer.id, target: linear.id },
      ],
    };
    const activity = {
      targetPath: "main_model.0.model",
      weights: {
        status: "updated" as const,
        source: "historical" as const,
        sourceLabel: "1 historical run",
        observedPoints: 2,
      },
    };
    const { result } = renderHook(() =>
      useGraphViewState(monitorGraph, {
        canOpenMonitor: () => false,
        resolveMonitorTarget: () => undefined,
        resolveParameterActivityTarget: (candidate) =>
          candidate.id === "layer" || candidate.id === "linear"
            ? linear
            : undefined,
        parameterActivityByNodePath: new Map([
          ["main_model.0.model", activity],
        ]),
      }),
    );

    await waitFor(() =>
      expect(result.current.nodes.map((candidate) => candidate.id)).toContain(
        "layer",
      ),
    );

    const layerNode = result.current.nodes.find(
      (candidate) => candidate.id === "layer",
    );
    const rootNode = result.current.nodes.find(
      (candidate) => candidate.id === "root",
    );
    expect(layerNode?.data.canOpenMonitor).toBe(false);
    expect(layerNode?.data.parameterActivity).toBe(activity);
    expect(layerNode?.data.details).not.toHaveProperty("weightShape");
    expect(rootNode?.data.childSummaries).toEqual([
      expect.objectContaining({
        label: "Layer 0",
        nestedLabel: "LinearLayer",
        dims: "16 -> 16",
        kind: "child",
        sourceNodeId: "layer",
        parameterActivity: activity,
      }),
    ]);
    expect(layerNode?.data.childSummaries).toEqual([
      expect.objectContaining({
        label: "LinearLayer",
        kind: "child",
        sourceNodeId: "linear",
        parameterActivity: activity,
      }),
    ]);

    act(() => result.current.setGraphScope("entire"));

    await waitFor(() =>
      expect(result.current.nodes.map((candidate) => candidate.id)).toContain(
        "linear",
      ),
    );

    const linearNode = result.current.nodes.find(
      (candidate) => candidate.id === "linear",
    );
    expect(linearNode?.data.details).toMatchObject({
      dims: "32 -> 32",
      weightShape: "32 x 32",
    });
    expect(linearNode?.data.parameterActivity).toBe(activity);
  });
});
