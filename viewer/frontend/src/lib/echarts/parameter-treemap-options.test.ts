import { describe, expect, it } from "vitest";
import type { GraphNode, InspectResponse } from "@/lib/api";
import {
  buildParameterTreemapData,
  buildParameterTreemapOption,
  fallbackParameterTreemapFocusNodeId,
  type ParameterTreemapItem,
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

function series(option: ReturnType<typeof buildParameterTreemapOption>) {
  return (option.series as Array<{ data?: ParameterTreemapItem[] }>)[0];
}

function firstChild(item: ParameterTreemapItem, nodeId: string) {
  return item.children?.find((child) => child.nodeId === nodeId);
}

describe("buildParameterTreemapData", () => {
  it("returns empty focused data for missing or empty graphs", () => {
    const emptyData = {
      chartRoots: [],
      focusedNode: null,
      focusNodeId: null,
      ancestors: [],
      immediateChildren: [],
      zeroParameterChildren: [],
      totalParameterCount: 0,
      focusedParameterCount: 0,
      hasParameters: false,
      hasChartParameters: false,
    };

    expect(buildParameterTreemapData(undefined)).toEqual(emptyData);
    expect(buildParameterTreemapData(graph([], []))).toEqual(emptyData);
  });

  it("opens at the root overview with immediate parameter-bearing chart roots", () => {
    const data = buildParameterTreemapData(
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
    expect(data.chartRoots.map((root) => root.nodeId)).toEqual(["layer", "head"]);
    expect(data.immediateChildren.map((child) => child.id)).toEqual([
      "layer",
      "empty",
      "head",
    ]);
    expect(data.zeroParameterChildren.map((child) => child.id)).toEqual(["empty"]);
  });

  it("focuses a child into its subtree and derives breadcrumbs from edges", () => {
    const data = buildParameterTreemapData(
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
    expect(data.ancestors.map((ancestor) => ancestor.id)).toEqual(["model"]);
    expect(data.chartRoots.map((root) => root.nodeId)).toEqual([
      "linear",
      "projection",
      "block",
    ]);
    expect(data.chartRoots.at(-1)).toMatchObject({
      id: "block::__direct_params",
      nodeId: "block",
      isDirectParameterBucket: true,
      canDrill: false,
      parameterCount: 10,
    });
    expect(data.zeroParameterChildren.map((child) => child.id)).toEqual(["norm"]);
  });

  it("makes repeated sibling labels unique by adding parent path segments", () => {
    const data = buildParameterTreemapData(
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

    expect(data.chartRoots.map((child) => child.name)).toEqual([
      "0.model: LinearLayer",
      "1.model: LinearLayer",
      "Direct parameters",
    ]);
  });

  it("keeps zero-parameter immediate children in inspector data, not chart roots", () => {
    const data = buildParameterTreemapData(
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

    expect(data.chartRoots.map((root) => root.nodeId)).toEqual(["kept", "parent"]);
    expect(data.zeroParameterChildren.map((child) => child.id)).toEqual(["empty"]);
    expect(data.chartRoots.some((root) => root.nodeId === "empty")).toBe(false);
  });

  it("normalizes layout values when shared child totals exceed parent totals", () => {
    const data = buildParameterTreemapData(
      graph(
        [
          node("parent", { parameterCount: 100 }),
          node("left", { parameterCount: 80 }),
          node("right", { parameterCount: 70 }),
        ],
        [
          ["parent", "left"],
          ["parent", "right"],
        ],
        { parameterCount: 100 },
      ),
    );

    const left = data.chartRoots.find((root) => root.nodeId === "left");
    const right = data.chartRoots.find((root) => root.nodeId === "right");
    expect(left?.parameterCount).toBe(80);
    expect(right?.parameterCount).toBe(70);
    expect(left?.value).toBeCloseTo(53.333, 3);
    expect(right?.value).toBeCloseTo(46.667, 3);
    expect(data.chartRoots.find((root) => root.id === "parent::__direct_params"))
      .toBeUndefined();
  });

  it("does not infer direct buckets when shared child totals hide direct ownership", () => {
    const data = buildParameterTreemapData(
      graph(
        [
          node("parent", { parameterCount: 100 }),
          node("sharedLeft", { parameterCount: 80 }),
          node("sharedRight", { parameterCount: 80 }),
        ],
        [
          ["parent", "sharedLeft"],
          ["parent", "sharedRight"],
        ],
        { parameterCount: 100 },
      ),
    );

    expect(data.chartRoots.find((root) => root.id === "parent::__direct_params"))
      .toBeUndefined();
    expect(data.chartRoots.find((root) => root.nodeId === "sharedLeft")?.value).toBe(50);
    expect(data.chartRoots.find((root) => root.nodeId === "sharedRight")?.value).toBe(50);
  });

  it("adds direct-parameter buckets that select their owner and do not drill", () => {
    const data = buildParameterTreemapData(
      graph(
        [
          node("parent", { parameterCount: 100, parameterSizeBytes: 400 }),
          node("child", { parameterCount: 30, parameterSizeBytes: 120 }),
        ],
        [["parent", "child"]],
        { parameterCount: 100, parameterSizeBytes: 400 },
      ),
    );

    const directBucket = data.chartRoots.find((root) => root.id === "parent::__direct_params");
    expect(directBucket).toMatchObject({
      name: "Direct parameters",
      nodeId: "parent",
      path: "model.parent",
      typeName: "DirectParameters",
      graphRole: "architecture",
      parameterCount: 70,
      parameterSizeBytes: 280,
      value: 70,
      canDrill: false,
      isDirectParameterBucket: true,
    });
  });

  it("terminates cycles in focused subtree rendering", () => {
    const data = buildParameterTreemapData(
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

    expect(data.chartRoots.map((root) => root.nodeId)).toEqual(["b", "a"]);
    expect(firstChild(data.chartRoots[0], "a")).toBeUndefined();
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

    const basicData = buildParameterTreemapData(filterGraphByDetail(fullGraph, "basic"));
    const simpleData = buildParameterTreemapData(filterGraphByDetail(fullGraph, "simple"));
    const fullData = buildParameterTreemapData(filterGraphByDetail(fullGraph, "full"));

    expect(basicData.chartRoots.map((child) => child.nodeId)).not.toContain("internal");
    expect(simpleData.chartRoots.map((child) => child.nodeId)).not.toContain("runtime");
    expect(fullData.chartRoots.map((child) => child.nodeId)).toEqual([
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
      fallbackParameterTreemapFocusNodeId("internal", basicGraph, fullGraph),
    ).toBe("model");
    expect(
      fallbackParameterTreemapFocusNodeId("missing", basicGraph, fullGraph),
    ).toBeNull();
  });
});

describe("buildParameterTreemapOption", () => {
  it("builds one fixed non-interactive treemap series on a transparent background", () => {
    const data = buildParameterTreemapData(
      graph([node("root", { parameterCount: 5 })], [], { parameterCount: 5 }),
    );
    const option = buildParameterTreemapOption(data);
    const treemapSeries = series(option);

    expect(option.animation).toBe(false);
    expect(option.backgroundColor).toBe("transparent");
    expect(treemapSeries).toMatchObject({
      type: "treemap",
      animation: false,
      roam: false,
      nodeClick: false,
      breadcrumb: { show: false },
    });
    expect(treemapSeries.data).toHaveLength(1);
  });

  it("uses restrained treemap labels, group headers, and dark gutters", () => {
    const data = buildParameterTreemapData(
      graph(
        [
          node("root", { parameterCount: 10 }),
          node("group", { parameterCount: 6 }),
          node("leaf", { parameterCount: 4 }),
        ],
        [
          ["root", "group"],
          ["group", "leaf"],
        ],
        { parameterCount: 10 },
      ),
    );
    const option = buildParameterTreemapOption(data);
    const treemapSeries = series(option) as {
      left?: number;
      top?: number;
      emphasis?: { label?: { show?: boolean } };
      itemStyle?: { gapWidth?: number; borderWidth?: number };
      levels?: Array<{ upperLabel?: { show?: boolean; height?: number } }>;
      data?: ParameterTreemapItem[];
    };

    expect(treemapSeries.left).toBe(10);
    expect(treemapSeries.top).toBe(10);
    expect(treemapSeries.emphasis?.label?.show).toBe(false);
    expect(treemapSeries.itemStyle).toMatchObject({
      gapWidth: 4,
      borderWidth: 2,
    });
    expect(treemapSeries.levels?.[1].upperLabel).toMatchObject({
      show: true,
      height: 18,
    });
    expect(treemapSeries.data?.[0].label?.show).toBe(false);
    expect(treemapSeries.data?.[0].children?.[0].label?.show).toBe(true);
  });

  it("applies selected-node border and highlight styling", () => {
    const data = buildParameterTreemapData(
      graph([node("root", { parameterCount: 5 })], [], { parameterCount: 5 }),
    );
    const option = buildParameterTreemapOption(data, { selectedNodeId: "root" });

    expect(series(option).data?.[0].itemStyle).toMatchObject({
      borderColor: "#f8fafc",
      borderWidth: 3,
      shadowBlur: 14,
    });
  });

  it("formats tooltips with identity, counts, memory, and focus/model shares", () => {
    const data = buildParameterTreemapData(
      graph(
        [node("root", { typeName: "Root", parameterCount: 1500 })],
        [],
        { parameterCount: 1500 },
      ),
    );
    const option = buildParameterTreemapOption(data);
    const formatter = option.tooltip as unknown as {
      formatter: (params: { data: ParameterTreemapItem }) => string;
    };
    const tooltip = formatter.formatter({ data: series(option).data?.[0] as ParameterTreemapItem });

    expect(tooltip).toContain("<strong>root: Root</strong>");
    expect(tooltip).toContain("Path: model.root");
    expect(tooltip).toContain("Type: Root");
    expect(tooltip).toContain("Exact params: 1,500");
    expect(tooltip).toContain("Compact params: 1.5K");
    expect(tooltip).toContain("Memory:");
    expect(tooltip).toContain("Share of focus: 100%");
    expect(tooltip).toContain("Share of model: 100%");
  });

  it("ignores partial ECharts tooltip payloads without crashing", () => {
    const option = buildParameterTreemapOption(buildParameterTreemapData(undefined));
    const formatter = option.tooltip as unknown as {
      formatter: (params: { data: unknown }) => string;
    };

    expect(() => formatter.formatter({ data: { value: 1 } })).not.toThrow();
    expect(formatter.formatter({ data: { value: 1 } })).toBe("");
  });
});
