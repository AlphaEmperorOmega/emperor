import { describe, expect, it } from "vitest";
import type { GraphNode, InspectResponse } from "@/lib/api";
import {
  buildParameterFocusData,
  fallbackParameterFocusNodeId,
} from "@/lib/echarts/parameter-treemap-options";
import { filterGraphByDetail } from "@/lib/graph/filtering";

type GraphRole = GraphNode["graphRole"];

function node(id: string, overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id,
    label: overrides.label ?? id,
    typeName: overrides.typeName ?? `${id}Type`,
    path: overrides.path ?? `model.${id}`,
    graphRole: (overrides.graphRole ?? "architecture") as GraphRole,
    parameterCount: overrides.parameterCount ?? 0,
    parameterSizeBytes: overrides.parameterSizeBytes ?? (overrides.parameterCount ?? 0) * 4,
    details: overrides.details ?? {},
    config: overrides.config ?? null,
  };
}

function graph(
  nodes: GraphNode[],
  edges: Array<[string, string]>,
  overrides: Partial<InspectResponse> = {},
): InspectResponse {
  return {
    model: "linear",
    preset: "baseline",
    parameterCount:
      overrides.parameterCount ??
      nodes.reduce((total, candidate) => total + Math.max(candidate.parameterCount, 0), 0),
    parameterSizeBytes:
      overrides.parameterSizeBytes ??
      nodes.reduce((total, candidate) => total + Math.max(candidate.parameterSizeBytes, 0), 0),
    nodes,
    edges: edges.map(([source, target], index) => ({
      id: `${source}-${target}-${index}`,
      source,
      target,
    })),
  };
}

function parameterChildIds(data: ReturnType<typeof buildParameterFocusData>) {
  return data.immediateChildren
    .filter((child) => child.hasParameters)
    .map((child) => child.id);
}

describe("buildParameterFocusData", () => {
  it("returns empty focused data for missing or empty graphs", () => {
    const emptyData = {
      focusedNode: null,
      focusNodeId: null,
      ancestors: [],
      immediateChildren: [],
      zeroParameterChildren: [],
      totalParameterCount: 0,
      focusedParameterCount: 0,
      hasParameters: false,
    };

    expect(buildParameterFocusData(undefined)).toEqual(emptyData);
    expect(buildParameterFocusData(graph([], []))).toEqual(emptyData);
  });

  it("opens at the root overview with immediate child summaries", () => {
    const data = buildParameterFocusData(
      graph(
        [
          node("model", {
            typeName: "Model",
            path: "model",
            parameterCount: 100,
          }),
          node("layer", {
            typeName: "Layer",
            path: "main_model.0",
            parameterCount: 60,
          }),
          node("empty", {
            typeName: "Sequential",
            path: "main_model.empty",
            parameterCount: 0,
          }),
          node("head", {
            typeName: "LinearLayer",
            path: "head",
            parameterCount: 40,
          }),
        ],
        [
          ["model", "layer"],
          ["model", "empty"],
          ["model", "head"],
        ],
        { parameterCount: 100 },
      ),
    );

    expect(data.focusNodeId).toBeNull();
    expect(data.focusedNode).toMatchObject({
      id: "model",
      label: "model: Model",
      parameterCount: 100,
      childCount: 3,
    });
    expect(parameterChildIds(data)).toEqual(["layer", "head"]);
    expect(data.immediateChildren.map((child) => child.id)).toEqual([
      "layer",
      "empty",
      "head",
    ]);
    expect(data.zeroParameterChildren.map((child) => child.id)).toEqual(["empty"]);
  });

  it("focuses a child into its subtree and derives breadcrumbs from edges", () => {
    const data = buildParameterFocusData(
      graph(
        [
          node("model", { typeName: "Model", path: "model", parameterCount: 100 }),
          node("block", {
            typeName: "Layer",
            path: "main_model.0",
            parameterCount: 80,
          }),
          node("linear", {
            typeName: "LinearLayer",
            path: "main_model.0.model",
            parameterCount: 50,
          }),
          node("norm", {
            typeName: "LayerNorm",
            path: "main_model.0.norm",
            graphRole: "internal",
            parameterCount: 0,
          }),
          node("projection", {
            typeName: "LinearLayer",
            path: "main_model.0.projection",
            parameterCount: 20,
          }),
        ],
        [
          ["model", "block"],
          ["block", "linear"],
          ["block", "norm"],
          ["block", "projection"],
        ],
        { parameterCount: 100 },
      ),
      "block",
    );

    expect(data.focusNodeId).toBe("block");
    expect(data.focusedNode).toMatchObject({
      id: "block",
      label: "0: Layer",
      path: "main_model.0",
    });
    expect(data.focusedParameterCount).toBe(80);
    expect(data.ancestors.map((ancestor) => ancestor.id)).toEqual(["model"]);
    expect(data.immediateChildren.map((child) => child.id)).toEqual([
      "linear",
      "norm",
      "projection",
    ]);
    expect(parameterChildIds(data)).toEqual(["linear", "projection"]);
    expect(data.zeroParameterChildren.map((child) => child.id)).toEqual(["norm"]);
  });

  it("makes repeated sibling labels unique by adding parent path segments", () => {
    const data = buildParameterFocusData(
      graph(
        [
          node("parent", { typeName: "Container", path: "main_model", parameterCount: 20 }),
          node("layer0", {
            typeName: "LinearLayer",
            path: "main_model.0.model",
            parameterCount: 8,
          }),
          node("layer1", {
            typeName: "LinearLayer",
            path: "main_model.1.model",
            parameterCount: 7,
          }),
        ],
        [
          ["parent", "layer0"],
          ["parent", "layer1"],
        ],
        { parameterCount: 20 },
      ),
    );

    expect(data.immediateChildren.map((child) => child.label)).toEqual([
      "0.model: LinearLayer",
      "1.model: LinearLayer",
    ]);
  });

  it("keeps zero-parameter immediate children separate from parameter-bearing children", () => {
    const data = buildParameterFocusData(
      graph(
        [
          node("parent", { parameterCount: 10 }),
          node("empty", { parameterCount: 0 }),
          node("kept", { parameterCount: 4 }),
        ],
        [
          ["parent", "empty"],
          ["parent", "kept"],
        ],
        { parameterCount: 10 },
      ),
    );

    expect(data.immediateChildren.map((child) => child.id)).toEqual(["empty", "kept"]);
    expect(parameterChildIds(data)).toEqual(["kept"]);
    expect(data.zeroParameterChildren.map((child) => child.id)).toEqual(["empty"]);
  });

  it("terminates cycles while deriving focused children and ancestors", () => {
    const data = buildParameterFocusData(
      graph(
        [
          node("a", { parameterCount: 10 }),
          node("b", { parameterCount: 5 }),
        ],
        [
          ["a", "b"],
          ["b", "a"],
        ],
        { parameterCount: 10 },
      ),
      "a",
    );

    expect(data.ancestors).toEqual([]);
    expect(data.immediateChildren.map((child) => child.id)).toEqual(["b"]);
  });

  it("uses caller-filtered detail graphs for simple/basic and full visibility", () => {
    const fullGraph = graph(
      [
        node("model", { typeName: "Model", path: "model", parameterCount: 12 }),
        node("architecture", {
          typeName: "Layer",
          graphRole: "architecture",
          parameterCount: 8,
        }),
        node("internal", {
          typeName: "LayerNorm",
          graphRole: "internal",
          parameterCount: 2,
        }),
        node("runtime", {
          typeName: "Metric",
          graphRole: "runtime",
          parameterCount: 2,
        }),
      ],
      [
        ["model", "architecture"],
        ["model", "internal"],
        ["model", "runtime"],
      ],
      { parameterCount: 12 },
    );

    const basicData = buildParameterFocusData(filterGraphByDetail(fullGraph, "basic"));
    const simpleData = buildParameterFocusData(filterGraphByDetail(fullGraph, "simple"));
    const fullData = buildParameterFocusData(filterGraphByDetail(fullGraph, "full"));

    expect(parameterChildIds(basicData)).not.toContain("internal");
    expect(parameterChildIds(simpleData)).not.toContain("runtime");
    expect(parameterChildIds(fullData)).toEqual([
      "architecture",
      "internal",
      "runtime",
    ]);
  });

  it("resolves focus to the nearest visible ancestor when detail filtering hides it", () => {
    const fullGraph = graph(
      [
        node("model", { graphRole: "architecture" }),
        node("internal", { graphRole: "internal" }),
        node("leaf", { graphRole: "architecture" }),
      ],
      [
        ["model", "internal"],
        ["internal", "leaf"],
      ],
    );
    const basicGraph = filterGraphByDetail(fullGraph, "basic");

    expect(
      fallbackParameterFocusNodeId("internal", basicGraph, fullGraph),
    ).toBe("model");
    expect(
      fallbackParameterFocusNodeId("missing", basicGraph, fullGraph),
    ).toBeNull();
  });
});
