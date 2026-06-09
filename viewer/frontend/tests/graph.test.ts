import { describe, expect, it } from "vitest";
import {
  configDetailText,
  detailText,
  formatCompactCount,
  formatExactCount,
  nodeDimsText,
  nodeBadges,
  nodeDetailEntries,
  parameterShapeEntries,
  nodeSubtitle,
  nodeTitle,
  simpleGraphParamText,
  structureNodeLabel,
} from "@/lib/graph/formatting";
import {
  ancestorNodeIds,
  buildGraphNavigation,
  buildHierarchy,
  expandableSubtreeNodeIds,
} from "@/lib/graph/navigation";
import { buildChildSummaries } from "@/lib/graph/child-summaries";
import { buildExpertDiagrams } from "@/lib/graph/expert-diagrams";
import { buildStackDiagrams } from "@/lib/graph/stack-diagrams";
import { buildClusterDiagrams } from "@/lib/graph/cluster-diagrams";
import { buildTerminalReachGrid } from "@/lib/graph/terminal-reach";
import { buildGraphLocationSummaries } from "@/lib/graph/locations";
import { filterGraphByDetail, filterGraphByExpansion } from "@/lib/graph/filtering";
import { layoutGraph } from "@/lib/graph/layout";
import {
  buildMonitorComparisonCandidateGroups,
  buildLinearMonitorComparisonCandidateGroups,
  buildLinearMonitorComparisonCandidates,
  createMonitorTargetResolver,
  resolveMonitorTarget,
  resolveLinearMonitorTarget,
} from "@/lib/graph";
import { SIMPLE_NODE_HEIGHT } from "@/lib/graph/constants";
import { inspectResponseSchema, type GraphNode, type InspectResponse } from "@/lib/api";

type GraphRole = GraphNode["graphRole"];

function node(
  id: string,
  overrides: Partial<GraphNode> = {},
): GraphNode {
  return {
    id,
    label: overrides.label ?? id,
    typeName: overrides.typeName ?? id,
    path: overrides.path ?? id,
    graphRole: (overrides.graphRole ?? "architecture") as GraphRole,
    parameterCount: overrides.parameterCount ?? 0,
    details: overrides.details ?? {},
    config: overrides.config ?? null,
  };
}

function graph(
  nodes: GraphNode[],
  edges: Array<[string, string]>,
): InspectResponse {
  return {
    model: "linear",
    preset: "baseline",
    parameterCount: nodes[0]?.parameterCount ?? 0,
    nodes,
    edges: edges.map(([source, target]) => ({
      id: `${source}-${target}`,
      source,
      target,
    })),
  };
}

describe("detailText", () => {
  it("renders nullish as empty, objects as JSON, scalars as strings", () => {
    expect(detailText(null)).toBe("");
    expect(detailText(undefined)).toBe("");
    expect(detailText({ a: 1 })).toBe('{"a":1}');
    expect(detailText(4)).toBe("4");
    expect(detailText("x")).toBe("x");
  });

  it("renders config null values as None", () => {
    expect(configDetailText(null)).toBe("None");
    expect(configDetailText("LinearLayerConfig")).toBe("LinearLayerConfig");
  });
});

describe("nodeDetailEntries", () => {
  it("prefers config fields over raw preview metadata", () => {
    expect(
      nodeDetailEntries(
        {
          weightShape: "3 x 4",
          biasShape: "3",
          dims: "4 -> 3",
          activation: "GELU",
        },
        {
          typeName: "LinearLayerConfig",
          fields: [
            { key: "input_dim", value: 4 },
            { key: "output_dim", value: 3 },
            { key: "bias_flag", value: false },
          ],
        },
      ),
    ).toEqual([
      { key: "input_dim", value: 4, source: "config" },
      { key: "output_dim", value: 3, source: "config" },
      { key: "bias_flag", value: false, source: "config" },
    ]);
  });

  it("filters preview-only raw details when no config is available", () => {
    expect(
      nodeDetailEntries({
        weightShape: "3 x 4",
        biasShape: "3",
        dims: "4 -> 3",
        inputDim: 4,
        hiddenDim: 5,
        outputDim: 3,
        activation: "GELU",
      }),
    ).toEqual([{ key: "activation", value: "GELU", source: "details" }]);
  });
});

describe("parameter count formatting", () => {
  it("formats compact and exact parameter counts", () => {
    expect(formatCompactCount(65800)).toBe("65.8K");
    expect(formatCompactCount(1000000)).toBe("1M");
    expect(formatExactCount(65800)).toBe("65,800");
  });
});

describe("simple graph inline metric formatting", () => {
  it("formats compact parameter text and omits missing or zero counts", () => {
    expect(simpleGraphParamText(33024)).toBe("33K params");
    expect(simpleGraphParamText(0)).toBeUndefined();
    expect(simpleGraphParamText(undefined)).toBeUndefined();
  });

  it("prefers explicit dims text over input/output fallback fields", () => {
    expect(nodeDimsText({ dims: "64 -> 32", inputDim: 128, outputDim: 128 }))
      .toBe("64 -> 32");
  });

  it("falls back to inputDim and outputDim when dims text is absent", () => {
    expect(nodeDimsText({ inputDim: 128, outputDim: 64 })).toBe("128 -> 64");
  });

  it("falls back to config input_dim and output_dim fields when preview dims are absent", () => {
    expect(
      nodeDimsText(
        {},
        {
          typeName: "LinearLayerConfig",
          fields: [
            { key: "input_dim", value: 128 },
            { key: "output_dim", value: 64 },
          ],
        },
      ),
    ).toBe("128 -> 64");
  });

  it("omits dimension text when fallback values are missing or zero", () => {
    expect(nodeDimsText({ inputDim: 128 })).toBeUndefined();
    expect(nodeDimsText({ inputDim: 0, outputDim: 64 })).toBeUndefined();
    expect(nodeDimsText({ dims: "  " })).toBeUndefined();
    expect(nodeDimsText({ dims: "0 -> 64" })).toBeUndefined();
    expect(
      nodeDimsText(
        {},
        {
          typeName: "LinearLayerConfig",
          fields: [
            { key: "input_dim", value: 128 },
            { key: "output_dim", value: 0 },
          ],
        },
      ),
    ).toBeUndefined();
    expect(nodeDimsText(undefined)).toBeUndefined();
  });
});

describe("inspectResponseSchema", () => {
  it("requires top-level and per-node parameter counts", () => {
    const parsed = inspectResponseSchema.parse(
      graph(
        [
          node("model", { parameterCount: 15 }),
          node("linear", { path: "linear", parameterCount: 15 }),
        ],
        [["model", "linear"]],
      ),
    );

    expect(parsed.parameterCount).toBe(15);
    expect(parsed.nodes[0].parameterCount).toBe(15);
  });
});

describe("nodeBadges", () => {
  it("emits badges for set fields and skips DISABLED / zero / off", () => {
    expect(
      nodeBadges({
        dims: "128 -> 128",
        activation: "GELU",
        layerNorm: "DISABLED",
        dropout: 0,
        gate: true,
        recurrent: { maxSteps: 4 },
      }),
    ).toEqual([
      ["dims", "128 -> 128"],
      ["act", "GELU"],
      ["gate", "on"],
      ["steps", 4],
    ]);
  });
});

describe("parameterShapeEntries", () => {
  it("formats direct parameter shape details for graph cards", () => {
    expect(
      parameterShapeEntries({
        weightShape: "128 x 128",
        biasShape: "128",
      }),
    ).toEqual([
      { key: "weightShape", label: "W", shape: "128 x 128" },
      { key: "biasShape", label: "b", shape: "128" },
    ]);
  });

  it("ignores missing or empty shape details", () => {
    expect(parameterShapeEntries({ weightShape: "", dims: "128 -> 128" })).toEqual([]);
  });
});

describe("nodeTitle / nodeSubtitle", () => {
  it("humanizes semantic container types with a named segment", () => {
    const semantic = node("x", { typeName: "Sequential", path: "model.gate_model" });
    expect(nodeTitle(semantic)).toBe("Gate Model");
    expect(nodeSubtitle(semantic)).toBe("Sequential · model.gate_model");
  });

  it("falls back to typeName for numeric or non-semantic segments", () => {
    const indexed = node("x", { typeName: "Sequential", path: "main_model.0" });
    expect(nodeTitle(indexed)).toBe("Sequential");
    expect(nodeSubtitle(indexed)).toBe("main_model.0");
  });
});

describe("structureNodeLabel", () => {
  it("uses the final path segment with the node type", () => {
    expect(
      structureNodeLabel(
        node("main_model.0.model", {
          typeName: "LinearLayer",
          path: "main_model.0.model",
        }),
      ),
    ).toBe("model: LinearLayer");
  });
});

describe("buildGraphNavigation", () => {
  it("maps children, parents, and roots", () => {
    const nav = buildGraphNavigation(
      graph([node("a"), node("b"), node("c")], [["a", "b"], ["a", "c"]]),
    );
    expect(nav.childrenById.get("a")).toEqual(["b", "c"]);
    expect(nav.parentById.get("b")).toBe("a");
    expect(nav.parentById.get("c")).toBe("a");
    expect([...nav.rootIds]).toEqual(["a"]);
  });

  it("falls back to the first node when no root exists", () => {
    const nav = buildGraphNavigation(graph([node("a"), node("b")], [["a", "b"], ["b", "a"]]));
    expect([...nav.rootIds]).toEqual(["a"]);
  });
});

describe("resolveLinearMonitorTarget", () => {
  const g = graph(
    [
      node("model", { typeName: "Model", path: "model" }),
      node("main_model.0", { typeName: "Layer", path: "main_model.0" }),
      node("main_model.0.model", {
        typeName: "LinearLayer",
        path: "main_model.0.model",
      }),
      node("main_model.0.processor", {
        typeName: "Processor",
        path: "main_model.0.processor",
        graphRole: "internal",
      }),
      node("main_model.0.processor.projection", {
        typeName: "LinearLayer",
        path: "main_model.0.processor.projection",
      }),
      node("main_model.1", {
        typeName: "AdaptiveLinearLayer",
        path: "main_model.1",
      }),
      node("main_model.2", { typeName: "Layer", path: "main_model.2" }),
      node("main_model.2.model", { typeName: "Sequential", path: "main_model.2.model" }),
      node("runtime_linear", {
        typeName: "LinearLayer",
        path: "runtime_linear",
        graphRole: "runtime",
      }),
    ],
    [
      ["model", "main_model.0"],
      ["main_model.0", "main_model.0.model"],
      ["main_model.0", "main_model.0.processor"],
      ["main_model.0.processor", "main_model.0.processor.projection"],
      ["model", "main_model.1"],
      ["model", "main_model.2"],
      ["main_model.2", "main_model.2.model"],
      ["model", "runtime_linear"],
    ],
  );
  const byId = new Map(g.nodes.map((graphNode) => [graphNode.id, graphNode]));

  it("resolves a Layer wrapper to its direct linear .model child", () => {
    expect(resolveLinearMonitorTarget(g, byId.get("main_model.0"))?.path)
      .toBe("main_model.0.model");
  });

  it("resolves direct LinearLayer and AdaptiveLinearLayer nodes to themselves", () => {
    expect(resolveLinearMonitorTarget(g, byId.get("main_model.0.model"))?.path)
      .toBe("main_model.0.model");
    expect(resolveLinearMonitorTarget(g, byId.get("main_model.1"))?.path)
      .toBe("main_model.1");
  });

  it("rejects non-linear, root, runtime, and nested projection nodes", () => {
    const rootLinearGraph = graph(
      [node("linear", { typeName: "LinearLayer", path: "linear" })],
      [],
    );

    expect(resolveLinearMonitorTarget(g, byId.get("model"))).toBeUndefined();
    expect(resolveLinearMonitorTarget(g, byId.get("main_model.2"))).toBeUndefined();
    expect(resolveLinearMonitorTarget(g, byId.get("runtime_linear"))).toBeUndefined();
    expect(resolveLinearMonitorTarget(g, byId.get("main_model.0.processor"))).toBeUndefined();
    expect(resolveLinearMonitorTarget(rootLinearGraph, rootLinearGraph.nodes[0]))
      .toBeUndefined();
  });
});

describe("resolveMonitorTarget", () => {
  const g = graph(
    [
      node("model", { typeName: "Model", path: "model" }),
      node("attention.0", {
        typeName: "SelfAttention",
        path: "attention.0",
      }),
      node("attention.1", {
        typeName: "IndependentAttention",
        path: "attention.1",
      }),
      node("recurrent.0", {
        typeName: "RecurrentLayer",
        path: "recurrent.0",
      }),
      node("main_model.0", { typeName: "Layer", path: "main_model.0" }),
      node("main_model.0.model", {
        typeName: "ParametricLayer",
        path: "main_model.0.model",
      }),
      node("main_model.1", { typeName: "Layer", path: "main_model.1" }),
      node("main_model.1.model", {
        typeName: "LinearLayer",
        path: "main_model.1.model",
      }),
    ],
    [
      ["model", "attention.0"],
      ["model", "attention.1"],
      ["model", "recurrent.0"],
      ["model", "main_model.0"],
      ["main_model.0", "main_model.0.model"],
      ["model", "main_model.1"],
      ["main_model.1", "main_model.1.model"],
    ],
  );
  const byId = new Map(g.nodes.map((graphNode) => [graphNode.id, graphNode]));

  it("resolves non-linear monitor target families", () => {
    expect(resolveMonitorTarget(g, byId.get("attention.0"))).toMatchObject({
      monitorName: "attention",
      node: byId.get("attention.0"),
    });
    expect(resolveMonitorTarget(g, byId.get("recurrent.0"))).toMatchObject({
      monitorName: "recurrent-layer",
      node: byId.get("recurrent.0"),
    });
    expect(resolveMonitorTarget(g, byId.get("main_model.0"))).toMatchObject({
      monitorName: "parametric",
      node: byId.get("main_model.0.model"),
    });
  });

  it("selects the first available monitor target for multi-monitor Layer nodes", () => {
    const linearOnlyResolver = createMonitorTargetResolver(
      g,
      (target) => target.monitorName === "linear",
    );
    const controllerOnlyResolver = createMonitorTargetResolver(
      g,
      (target) => target.monitorName === "layer-controller",
    );

    expect(linearOnlyResolver(byId.get("main_model.1"))).toMatchObject({
      monitorName: "linear",
      node: byId.get("main_model.1.model"),
    });
    expect(controllerOnlyResolver(byId.get("main_model.1"))).toMatchObject({
      monitorName: "layer-controller",
      node: byId.get("main_model.1"),
    });
  });

  it("builds same-kind comparison candidates for attention targets", () => {
    const groups = buildMonitorComparisonCandidateGroups(
      g,
      byId.get("attention.0"),
      "attention",
    );

    expect(groups["all-layers"].map((candidate) => candidate.path)).toEqual([
      "attention.1",
    ]);
  });
});

describe("buildLinearMonitorComparisonCandidates", () => {
  it("finds resolved linear targets for numeric Layer wrappers in the same stack", () => {
    const g = graph(
      [
        node("model", { typeName: "Model", path: "model" }),
        node("main_model.0", { typeName: "Layer", path: "main_model.0" }),
        node("main_model.0.model", {
          typeName: "LinearLayer",
          path: "main_model.0.model",
        }),
        node("main_model.1", { typeName: "Layer", path: "main_model.1" }),
        node("main_model.1.model", {
          typeName: "LinearLayer",
          path: "main_model.1.model",
        }),
        node("main_model.2", { typeName: "Layer", path: "main_model.2" }),
        node("main_model.2.model", { typeName: "Sequential", path: "main_model.2.model" }),
        node("main_model.helper", {
          typeName: "LinearLayer",
          path: "main_model.helper",
        }),
      ],
      [
        ["model", "main_model.0"],
        ["main_model.0", "main_model.0.model"],
        ["model", "main_model.1"],
        ["main_model.1", "main_model.1.model"],
        ["model", "main_model.2"],
        ["main_model.2", "main_model.2.model"],
        ["model", "main_model.helper"],
      ],
    );
    const byId = new Map(g.nodes.map((graphNode) => [graphNode.id, graphNode]));

    expect(
      buildLinearMonitorComparisonCandidates(g, byId.get("main_model.0.model")).map(
        (candidate) => candidate.path,
      ),
    ).toEqual(["main_model.1.model"]);
  });

  it("finds direct linear numeric siblings in index order", () => {
    const g = graph(
      [
        node("stack", { typeName: "Sequential", path: "stack" }),
        node("stack.10", { typeName: "LinearLayer", path: "stack.10" }),
        node("stack.2", { typeName: "AdaptiveLinearLayer", path: "stack.2" }),
        node("stack.1", { typeName: "LinearLayer", path: "stack.1" }),
      ],
      [
        ["stack", "stack.10"],
        ["stack", "stack.2"],
        ["stack", "stack.1"],
      ],
    );
    const byId = new Map(g.nodes.map((graphNode) => [graphNode.id, graphNode]));

    expect(
      buildLinearMonitorComparisonCandidates(g, byId.get("stack.10")).map(
        (candidate) => candidate.path,
      ),
    ).toEqual(["stack.1", "stack.2"]);
  });

  it("excludes the primary target, non-linear siblings, runtime/internal nodes, non-numeric siblings, and nested projections", () => {
    const g = graph(
      [
        node("stack", { typeName: "Sequential", path: "stack" }),
        node("stack.0", { typeName: "LinearLayer", path: "stack.0" }),
        node("stack.0.projection", {
          typeName: "LinearLayer",
          path: "stack.0.projection",
        }),
        node("stack.1", { typeName: "Layer", path: "stack.1" }),
        node("stack.1.model", { typeName: "Sequential", path: "stack.1.model" }),
        node("stack.2", {
          typeName: "LinearLayer",
          path: "stack.2",
          graphRole: "runtime",
        }),
        node("stack.3", {
          typeName: "LinearLayer",
          path: "stack.3",
          graphRole: "internal",
        }),
        node("stack.helper", { typeName: "LinearLayer", path: "stack.helper" }),
      ],
      [
        ["stack", "stack.0"],
        ["stack.0", "stack.0.projection"],
        ["stack", "stack.1"],
        ["stack.1", "stack.1.model"],
        ["stack", "stack.2"],
        ["stack", "stack.3"],
        ["stack", "stack.helper"],
      ],
    );
    const byId = new Map(g.nodes.map((graphNode) => [graphNode.id, graphNode]));

    expect(buildLinearMonitorComparisonCandidates(g, byId.get("stack.0"))).toEqual([]);
    expect(buildLinearMonitorComparisonCandidates(g, byId.get("stack.0.projection"))).toEqual([]);
  });
});

describe("buildLinearMonitorComparisonCandidateGroups", () => {
  const g = graph(
    [
      node("model", { typeName: "Model", path: "model" }),
      node("input_model", { typeName: "Layer", path: "input_model" }),
      node("input_model.model", {
        typeName: "LinearLayer",
        path: "input_model.model",
      }),
      node("input_model.processor", {
        typeName: "Processor",
        path: "input_model.processor",
        graphRole: "internal",
      }),
      node("input_model.processor.projection", {
        typeName: "LinearLayer",
        path: "input_model.processor.projection",
      }),
      node("main_model", { typeName: "Sequential", path: "main_model" }),
      node("main_model.0", { typeName: "Layer", path: "main_model.0" }),
      node("main_model.0.model", {
        typeName: "LinearLayer",
        path: "main_model.0.model",
      }),
      node("main_model.1", { typeName: "Layer", path: "main_model.1" }),
      node("main_model.1.model", {
        typeName: "AdaptiveLinearLayer",
        path: "main_model.1.model",
      }),
      node("output_model", { typeName: "Layer", path: "output_model" }),
      node("output_model.model", {
        typeName: "LinearLayer",
        path: "output_model.model",
      }),
      node("runtime_linear", {
        typeName: "LinearLayer",
        path: "runtime_linear",
        graphRole: "runtime",
      }),
      node("internal_linear", {
        typeName: "LinearLayer",
        path: "internal_linear",
        graphRole: "internal",
      }),
      node("root_linear", { typeName: "LinearLayer", path: "root_linear" }),
      node("main_model.activation", {
        typeName: "ReLU",
        path: "main_model.activation",
      }),
    ],
    [
      ["model", "input_model"],
      ["input_model", "input_model.model"],
      ["input_model", "input_model.processor"],
      ["input_model.processor", "input_model.processor.projection"],
      ["model", "main_model"],
      ["main_model", "main_model.0"],
      ["main_model.0", "main_model.0.model"],
      ["main_model", "main_model.1"],
      ["main_model.1", "main_model.1.model"],
      ["model", "output_model"],
      ["output_model", "output_model.model"],
      ["model", "runtime_linear"],
      ["model", "internal_linear"],
      ["main_model", "main_model.activation"],
    ],
  );
  const byId = new Map(g.nodes.map((graphNode) => [graphNode.id, graphNode]));

  it("keeps same-stack scope limited to numeric sibling layers", () => {
    const groups = buildLinearMonitorComparisonCandidateGroups(
      g,
      byId.get("main_model.0.model"),
    );

    expect(groups["same-stack"].map((candidate) => candidate.path)).toEqual([
      "main_model.1.model",
    ]);
  });

  it("lists all resolved linear layer targets in graph order", () => {
    const groups = buildLinearMonitorComparisonCandidateGroups(
      g,
      byId.get("main_model.0.model"),
    );

    expect(groups["all-layers"].map((candidate) => candidate.path)).toEqual([
      "input_model.model",
      "main_model.1.model",
      "output_model.model",
    ]);
  });

  it("excludes the primary target and duplicate wrapper/model targets", () => {
    const groups = buildLinearMonitorComparisonCandidateGroups(
      g,
      byId.get("input_model"),
    );
    const allLayerPaths = groups["all-layers"].map((candidate) => candidate.path);

    expect(allLayerPaths).toEqual([
      "main_model.0.model",
      "main_model.1.model",
      "output_model.model",
    ]);
    expect(allLayerPaths).not.toContain("input_model.model");
    expect(new Set(allLayerPaths).size).toBe(allLayerPaths.length);
  });
});

describe("ancestorNodeIds", () => {
  it("returns ancestors from root to parent", () => {
    const nav = buildGraphNavigation(
      graph([node("root"), node("block"), node("leaf")], [["root", "block"], ["block", "leaf"]]),
    );

    expect(ancestorNodeIds("leaf", nav)).toEqual(["root", "block"]);
  });

  it("returns an empty list for a missing node", () => {
    const nav = buildGraphNavigation(graph([node("root")], []));

    expect(ancestorNodeIds("missing", nav)).toEqual([]);
  });

  it("handles cyclic graphs that use the fallback root", () => {
    const nav = buildGraphNavigation(graph([node("a"), node("b")], [["a", "b"], ["b", "a"]]));

    expect(ancestorNodeIds("b", nav)).toEqual(["a"]);
    expect(ancestorNodeIds("a", nav)).toEqual([]);
  });
});

describe("expandableSubtreeNodeIds", () => {
  it("returns the clicked node and expandable descendants, excluding leaves", () => {
    const nav = buildGraphNavigation(
      graph(
        [node("root"), node("branch"), node("leaf"), node("deep"), node("deep.leaf")],
        [
          ["root", "branch"],
          ["branch", "leaf"],
          ["branch", "deep"],
          ["deep", "deep.leaf"],
        ],
      ),
    );

    expect(expandableSubtreeNodeIds("branch", nav)).toEqual(["branch", "deep"]);
  });

  it("returns an empty list for leaves and missing nodes", () => {
    const nav = buildGraphNavigation(graph([node("root"), node("leaf")], [["root", "leaf"]]));

    expect(expandableSubtreeNodeIds("leaf", nav)).toEqual([]);
    expect(expandableSubtreeNodeIds("missing", nav)).toEqual([]);
  });

  it("guards against cycles while collecting expandable nodes", () => {
    const nav = buildGraphNavigation(
      graph([node("a"), node("b"), node("c")], [["a", "b"], ["b", "c"], ["c", "b"]]),
    );

    expect(expandableSubtreeNodeIds("b", nav)).toEqual(["b", "c"]);
  });
});

describe("buildHierarchy", () => {
  it("nests children under roots", () => {
    const roots = buildHierarchy(graph([node("a"), node("b")], [["a", "b"]]));
    expect(roots).toHaveLength(1);
    expect(roots[0].node.id).toBe("a");
    expect(roots[0].children.map((child) => child.node.id)).toEqual(["b"]);
  });
});

describe("buildChildSummaries", () => {
  it("keeps repeated layer children separate and includes their child type", () => {
    const g = graph(
      [
        node("a"),
        node("main_model.0", {
          typeName: "Layer",
          path: "main_model.0",
          details: { dims: "128 -> 128" },
        }),
        node("main_model.1", {
          typeName: "Layer",
          path: "main_model.1",
          details: { dims: "128 -> 10" },
        }),
        node("main_model.0.model", { typeName: "LinearLayer", path: "main_model.0.model" }),
        node("main_model.1.model", { typeName: "LinearLayer", path: "main_model.1.model" }),
      ],
      [
        ["a", "main_model.0"],
        ["a", "main_model.1"],
        ["main_model.0", "main_model.0.model"],
        ["main_model.1", "main_model.1.model"],
      ],
    );
    const summaries = buildChildSummaries(g, buildGraphNavigation(g));
    expect(summaries.get("a")).toEqual([
      {
        label: "Layer 0",
        nestedLabel: "LinearLayer",
        dims: "128 -> 128",
        kind: "child",
        stackKind: "layer",
      },
      {
        label: "Layer 1",
        nestedLabel: "LinearLayer",
        dims: "128 -> 10",
        kind: "child",
        stackKind: "layer",
      },
    ]);
  });

  it("uses the parent Layer dims on its primary inner-model summary", () => {
    const g = graph(
      [
        node("main_model.0", {
          typeName: "Layer",
          path: "main_model.0",
          details: { dims: "512 -> 10" },
        }),
        node("main_model.0.model", {
          typeName: "LinearLayer",
          path: "main_model.0.model",
        }),
        node("main_model.0.gate_model", {
          typeName: "Sequential",
          path: "main_model.0.gate_model",
        }),
      ],
      [
        ["main_model.0", "main_model.0.model"],
        ["main_model.0", "main_model.0.gate_model"],
      ],
    );

    expect(buildChildSummaries(g, buildGraphNavigation(g)).get("main_model.0")).toEqual([
      { label: "LinearLayer", dims: "512 -> 10", kind: "child" },
      { label: "Gate", kind: "child" },
    ]);
  });

  it("uses shared dimension fallbacks on child summary rows", () => {
    const g = graph(
      [
        node("block"),
        node("block.linear", {
          typeName: "LinearLayer",
          path: "block.linear",
          details: { inputDim: 128, outputDim: 64 },
        }),
        node("block.0", {
          typeName: "Layer",
          path: "block.0",
          config: {
            typeName: "LayerConfig",
            fields: [
              { key: "input_dim", value: 64 },
              { key: "output_dim", value: 32 },
            ],
          },
        }),
        node("block.0.model", {
          typeName: "LinearLayer",
          path: "block.0.model",
        }),
        node("layer_parent", {
          typeName: "Layer",
          path: "layer_parent",
          config: {
            typeName: "LayerConfig",
            fields: [
              { key: "input_dim", value: 256 },
              { key: "output_dim", value: 10 },
            ],
          },
        }),
        node("layer_parent.model", {
          typeName: "LinearLayer",
          path: "layer_parent.model",
          details: { inputDim: 999, outputDim: 999 },
        }),
      ],
      [
        ["block", "block.linear"],
        ["block", "block.0"],
        ["block.0", "block.0.model"],
        ["layer_parent", "layer_parent.model"],
      ],
    );
    const summaries = buildChildSummaries(g, buildGraphNavigation(g));

    expect(summaries.get("block")).toEqual([
      { label: "LinearLayer", dims: "128 -> 64", kind: "child" },
      {
        label: "Layer 0",
        nestedLabel: "LinearLayer",
        dims: "64 -> 32",
        kind: "child",
        stackKind: "layer",
      },
    ]);
    expect(summaries.get("layer_parent")).toEqual([
      { label: "LinearLayer", dims: "256 -> 10", kind: "child" },
    ]);
  });

  it("collapses long layer stacks with an ellipsis and total layer count", () => {
    const layerCount = 9;
    const layerNodes = Array.from({ length: layerCount }, (_, index) => [
      node(`main_model.${index}`, {
        typeName: "Layer",
        path: `main_model.${index}`,
        details: { dims: `${index + 1} -> ${index + 2}` },
      }),
      node(`main_model.${index}.model`, {
        typeName: "LinearLayer",
        path: `main_model.${index}.model`,
      }),
    ]).flat();
    const layerEdges = Array.from({ length: layerCount }, (_, index) => [
      ["model", `main_model.${index}`] as [string, string],
      [`main_model.${index}`, `main_model.${index}.model`] as [string, string],
    ]).flat();
    const g = graph([node("model"), ...layerNodes], layerEdges);

    expect(buildChildSummaries(g, buildGraphNavigation(g)).get("model")).toEqual([
      {
        label: "Layer 0",
        nestedLabel: "LinearLayer",
        dims: "1 -> 2",
        kind: "child",
        stackKind: "layer",
      },
      {
        label: "Layer 1",
        nestedLabel: "LinearLayer",
        dims: "2 -> 3",
        kind: "child",
        stackKind: "layer",
      },
      {
        label: "Layer 2",
        nestedLabel: "LinearLayer",
        dims: "3 -> 4",
        kind: "child",
        stackKind: "layer",
      },
      {
        label: "Layer 3",
        nestedLabel: "LinearLayer",
        dims: "4 -> 5",
        kind: "child",
        stackKind: "layer",
      },
      {
        label: "Layer 4",
        nestedLabel: "LinearLayer",
        dims: "5 -> 6",
        kind: "child",
        stackKind: "layer",
      },
      { label: "...", kind: "overflow", title: "4 more layers" },
      { label: "9 layers", kind: "child", title: "9 layers total" },
    ]);
  });

  it("adds gate/halting mechanism rows from flags without duplicating real children", () => {
    const flagged = graph(
      [node("ctrl", { details: { gate: true, halting: true } })],
      [],
    );
    expect(buildChildSummaries(flagged, buildGraphNavigation(flagged)).get("ctrl")).toEqual([
      { label: "Gate", kind: "mechanism" },
      { label: "Halting mechanism", kind: "mechanism" },
    ]);

    const withChild = graph(
      [
        node("ctrl", { details: { gate: true } }),
        node("ctrl.gate_model", { typeName: "Sequential", path: "ctrl.gate_model" }),
      ],
      [["ctrl", "ctrl.gate_model"]],
    );
    const summaries = withChild;
    const built = buildChildSummaries(summaries, buildGraphNavigation(summaries));
    expect(built.get("ctrl")).toEqual([{ label: "Gate", kind: "child" }]);
  });
});

describe("buildExpertDiagrams", () => {
  it("derives sampler and expert cells for MixtureOfExperts", () => {
    const g = graph(
      [
        node("moe", {
          typeName: "MixtureOfExperts",
          path: "moe",
          details: { topK: 2, numExperts: 4, routingMode: "LAYER" },
        }),
        node("moe.sampler", { typeName: "SamplerModel", path: "moe.sampler" }),
        node("moe.expert_modules", {
          typeName: "ModuleList",
          path: "moe.expert_modules",
        }),
        ...Array.from({ length: 4 }, (_, index) =>
          node(`moe.expert_modules.${index}`, {
            typeName: "Sequential",
            path: `moe.expert_modules.${index}`,
          }),
        ),
      ],
      [
        ["moe", "moe.sampler"],
        ["moe", "moe.expert_modules"],
        ...Array.from({ length: 4 }, (_, index) => [
          "moe.expert_modules",
          `moe.expert_modules.${index}`,
        ] as [string, string]),
      ],
    );
    const diagram = buildExpertDiagrams(g, buildGraphNavigation(g)).get("moe");

    expect(diagram?.samplerLabel).toBe("Sampler");
    expect(diagram?.samplerTitle).toBe("moe.sampler");
    expect(diagram?.totalExperts).toBe(4);
    expect(diagram?.cells.map((cell) => cell.label)).toEqual(["E0", "E1", "E2", "E3"]);
  });

  it("derives a shared-sampler diagram for MixtureOfExpertsModel", () => {
    const g = graph(
      [
        node("model", {
          typeName: "MixtureOfExpertsModel",
          path: "model",
          details: { topK: 2, numExperts: 6, routingMode: "SHARED" },
        }),
        node("model.shared_sampler", {
          typeName: "SamplerModel",
          path: "model.shared_sampler",
        }),
        node("model.expert_stack", {
          typeName: "Sequential",
          path: "model.expert_stack",
        }),
        node("model.expert_stack.0", {
          typeName: "MixtureOfExpertsLayer",
          path: "model.expert_stack.0",
        }),
        node("model.expert_stack.1", {
          typeName: "MixtureOfExpertsLayer",
          path: "model.expert_stack.1",
        }),
      ],
      [
        ["model", "model.shared_sampler"],
        ["model", "model.expert_stack"],
        ["model.expert_stack", "model.expert_stack.0"],
        ["model.expert_stack", "model.expert_stack.1"],
      ],
    );
    const diagram = buildExpertDiagrams(g, buildGraphNavigation(g)).get("model");

    expect(diagram?.samplerLabel).toBe("Shared sampler");
    expect(diagram?.totalExperts).toBe(6);
    expect(diagram?.layerCount).toBe(2);
    expect(diagram?.cells.map((cell) => cell.label)).toEqual([
      "E0",
      "E1",
      "E2",
      "E3",
      "E4",
      "E5",
    ]);
  });

  it("falls back to child summaries when the sampler is not present", () => {
    const g = graph(
      [
        node("moe", {
          typeName: "MixtureOfExperts",
          path: "moe",
          details: { numExperts: 2 },
        }),
        node("moe.expert_modules", {
          typeName: "ModuleList",
          path: "moe.expert_modules",
        }),
      ],
      [["moe", "moe.expert_modules"]],
    );

    expect(buildExpertDiagrams(g, buildGraphNavigation(g)).get("moe")).toBeUndefined();
  });

  it("collapses more than seven experts to five cells, ellipsis, and total", () => {
    const expertCount = 16;
    const g = graph(
      [
        node("moe", {
          typeName: "MixtureOfExperts",
          path: "moe",
          details: { numExperts: expertCount },
        }),
        node("moe.sampler", { typeName: "SamplerModel", path: "moe.sampler" }),
        node("moe.expert_modules", {
          typeName: "ModuleList",
          path: "moe.expert_modules",
        }),
        ...Array.from({ length: expertCount }, (_, index) =>
          node(`moe.expert_modules.${index}`, {
            typeName: "Sequential",
            path: `moe.expert_modules.${index}`,
          }),
        ),
      ],
      [
        ["moe", "moe.sampler"],
        ["moe", "moe.expert_modules"],
        ...Array.from({ length: expertCount }, (_, index) => [
          "moe.expert_modules",
          `moe.expert_modules.${index}`,
        ] as [string, string]),
      ],
    );
    const diagram = buildExpertDiagrams(g, buildGraphNavigation(g)).get("moe");

    expect(diagram?.hasOverflow).toBe(true);
    expect(diagram?.cells.map((cell) => cell.label)).toEqual([
      "E0",
      "E1",
      "E2",
      "E3",
      "E4",
      "...",
      "16 experts",
    ]);
  });
});

describe("buildStackDiagrams", () => {
  it("derives compact layer cells for Sequential layer stacks", () => {
    const g = graph(
      [
        node("main_model", {
          typeName: "Sequential",
          path: "main_model",
          details: { numLayers: 3 },
        }),
        ...Array.from({ length: 3 }, (_, index) => [
          node(`main_model.${index}`, {
            typeName: "Layer",
            path: `main_model.${index}`,
            details: { dims: "256 -> 256" },
          }),
          node(`main_model.${index}.model`, {
            typeName: "LinearLayer",
            path: `main_model.${index}.model`,
            details: { dims: "999 -> 999" },
          }),
        ]).flat(),
      ],
      Array.from({ length: 3 }, (_, index) => [
        ["main_model", `main_model.${index}`] as [string, string],
        [`main_model.${index}`, `main_model.${index}.model`] as [string, string],
      ]).flat(),
    );

    const diagram = buildStackDiagrams(g, buildGraphNavigation(g)).get("main_model");

    expect(diagram?.totalLayers).toBe(3);
    expect(diagram?.hasOverflow).toBe(false);
    expect(diagram?.dims).toBe("256 -> 256");
    expect(diagram?.cells.map((cell) => cell.label)).toEqual([
      "Layer 0 · LinearLayer",
      "Layer 1 · LinearLayer",
      "Layer 2 · LinearLayer",
    ]);
    expect(diagram?.cells[0]).toEqual({
      label: "Layer 0 · LinearLayer",
      title: "Layer 0 · LinearLayer · 256 -> 256",
      dims: "256 -> 256",
      kind: "layer",
      layerIndex: 0,
    });
  });

  it("uses stack container config dimensions before deriving from child layers", () => {
    const g = graph(
      [
        node("main_model", {
          typeName: "Sequential",
          path: "main_model",
          config: {
            typeName: "LayerStackConfig",
            fields: [
              { key: "input_dim", value: 128 },
              { key: "output_dim", value: 64 },
            ],
          },
        }),
        node("main_model.0", {
          typeName: "Layer",
          path: "main_model.0",
          details: { inputDim: 999, outputDim: 999 },
        }),
        node("main_model.1", {
          typeName: "Layer",
          path: "main_model.1",
          details: { inputDim: 999, outputDim: 999 },
        }),
      ],
      [
        ["main_model", "main_model.0"],
        ["main_model", "main_model.1"],
      ],
    );

    const diagram = buildStackDiagrams(g, buildGraphNavigation(g)).get("main_model");

    expect(diagram?.dims).toBe("128 -> 64");
  });

  it("derives stack dimensions from first-layer input to last-layer output", () => {
    const g = graph(
      [
        node("main_model", {
          typeName: "Sequential",
          path: "main_model",
        }),
        node("main_model.0", {
          typeName: "Layer",
          path: "main_model.0",
          details: { inputDim: 128, outputDim: 256 },
        }),
        node("main_model.1", {
          typeName: "Layer",
          path: "main_model.1",
          details: { inputDim: 256, outputDim: 10 },
        }),
      ],
      [
        ["main_model", "main_model.0"],
        ["main_model", "main_model.1"],
      ],
    );

    const diagram = buildStackDiagrams(g, buildGraphNavigation(g)).get("main_model");

    expect(diagram?.dims).toBe("128 -> 10");
  });

  it("falls back to primary layer content dimensions when wrapper layer dims are absent", () => {
    const g = graph(
      [
        node("main_model", {
          typeName: "Sequential",
          path: "main_model",
          details: { numLayers: 1 },
        }),
        node("main_model.0", {
          typeName: "Layer",
          path: "main_model.0",
        }),
        node("main_model.0.model", {
          typeName: "LinearLayer",
          path: "main_model.0.model",
          details: { inputDim: 64, outputDim: 32 },
        }),
      ],
      [
        ["main_model", "main_model.0"],
        ["main_model.0", "main_model.0.model"],
      ],
    );

    const diagram = buildStackDiagrams(g, buildGraphNavigation(g)).get("main_model");

    expect(diagram?.cells[0]).toMatchObject({
      label: "Layer 0 · LinearLayer",
      title: "Layer 0 · LinearLayer · 64 -> 32",
      dims: "64 -> 32",
      kind: "layer",
      layerIndex: 0,
    });
    expect(diagram?.dims).toBe("64 -> 32");
  });

  it("derives MixtureOfExpertsModel depth from its direct expert_stack child", () => {
    const g = graph(
      [
        node("model", {
          typeName: "MixtureOfExpertsModel",
          path: "model",
          details: { topK: 2, numExperts: 4 },
        }),
        node("model.shared_sampler", {
          typeName: "SamplerModel",
          path: "model.shared_sampler",
        }),
        node("model.expert_stack", {
          typeName: "Sequential",
          path: "model.expert_stack",
        }),
        ...Array.from({ length: 2 }, (_, index) => [
          node(`model.expert_stack.${index}`, {
            typeName: "MixtureOfExpertsLayer",
            path: `model.expert_stack.${index}`,
            details: { dims: "128 -> 128" },
          }),
          node(`model.expert_stack.${index}.model`, {
            typeName: "MixtureOfExperts",
            path: `model.expert_stack.${index}.model`,
          }),
        ]).flat(),
      ],
      [
        ["model", "model.shared_sampler"],
        ["model", "model.expert_stack"],
        ...Array.from({ length: 2 }, (_, index) => [
          ["model.expert_stack", `model.expert_stack.${index}`] as [string, string],
          [
            `model.expert_stack.${index}`,
            `model.expert_stack.${index}.model`,
          ] as [string, string],
        ]).flat(),
      ],
    );

    const diagram = buildStackDiagrams(g, buildGraphNavigation(g)).get("model");

    expect(diagram?.totalLayers).toBe(2);
    expect(diagram?.cells.map((cell) => cell.label)).toEqual([
      "Layer 0 · MixtureOfExperts",
      "Layer 1 · MixtureOfExperts",
    ]);
    expect(diagram?.cells[0].title).toBe("Layer 0 · MixtureOfExperts · 128 -> 128");
  });

  it("collapses more than seven layers to five cells, ellipsis, and total", () => {
    const layerCount = 16;
    const g = graph(
      [
        node("main_model", { typeName: "ModuleList", path: "main_model" }),
        ...Array.from({ length: layerCount }, (_, index) =>
          node(`main_model.${index}`, {
            typeName: "LinearLayer",
            path: `main_model.${index}`,
            details: { dims: `${index + 1} -> ${index + 2}` },
          }),
        ),
      ],
      Array.from({ length: layerCount }, (_, index) => [
        "main_model",
        `main_model.${index}`,
      ] as [string, string]),
    );

    const diagram = buildStackDiagrams(g, buildGraphNavigation(g)).get("main_model");

    expect(diagram?.hasOverflow).toBe(true);
    expect(diagram?.cells.map((cell) => cell.label)).toEqual([
      "Layer 0 · LinearLayer",
      "Layer 1 · LinearLayer",
      "Layer 2 · LinearLayer",
      "Layer 3 · LinearLayer",
      "Layer 4 · LinearLayer",
      "...",
      "16 layers",
    ]);
    expect(diagram?.cells[5].title).toBe("11 more layers");
    expect(diagram?.cells[6].title).toBe("16 layers total");
    expect(diagram?.dims).toBe("1 -> 17");
    expect(diagram?.cells[0].dims).toBe("1 -> 2");
    expect(diagram?.cells[5].dims).toBeUndefined();
    expect(diagram?.cells[6].dims).toBeUndefined();
  });

  it("does not derive stack diagrams for expert_modules lists", () => {
    const g = graph(
      [
        node("moe.expert_modules", {
          typeName: "ModuleList",
          path: "moe.expert_modules",
        }),
        node("moe.expert_modules.0", {
          typeName: "Layer",
          path: "moe.expert_modules.0",
        }),
        node("moe.expert_modules.1", {
          typeName: "Layer",
          path: "moe.expert_modules.1",
        }),
      ],
      [
        ["moe.expert_modules", "moe.expert_modules.0"],
        ["moe.expert_modules", "moe.expert_modules.1"],
      ],
    );

    expect(
      buildStackDiagrams(g, buildGraphNavigation(g)).get("moe.expert_modules"),
    ).toBeUndefined();
  });

  it("leaves MixtureOfExperts on the expert-routing diagram path", () => {
    const g = graph(
      [
        node("moe", {
          typeName: "MixtureOfExperts",
          path: "moe",
          details: { numExperts: 2 },
        }),
        node("moe.sampler", { typeName: "SamplerModel", path: "moe.sampler" }),
        node("moe.expert_modules", {
          typeName: "ModuleList",
          path: "moe.expert_modules",
        }),
        node("moe.expert_modules.0", {
          typeName: "Sequential",
          path: "moe.expert_modules.0",
        }),
        node("moe.expert_modules.1", {
          typeName: "Sequential",
          path: "moe.expert_modules.1",
        }),
      ],
      [
        ["moe", "moe.sampler"],
        ["moe", "moe.expert_modules"],
        ["moe.expert_modules", "moe.expert_modules.0"],
        ["moe.expert_modules", "moe.expert_modules.1"],
      ],
    );
    const nav = buildGraphNavigation(g);

    expect(buildStackDiagrams(g, nav).get("moe")).toBeUndefined();
    expect(buildExpertDiagrams(g, nav).get("moe")?.cells.map((cell) => cell.label)).toEqual([
      "E0",
      "E1",
    ]);
  });
});

describe("buildClusterDiagrams", () => {
  it("builds an x-by-y grid per z-plane and marks instantiated coordinates", () => {
    const g = graph(
      [
        node("neuron_cluster", {
          typeName: "NeuronCluster",
          path: "neuron_cluster",
          details: {
            cluster: {
              capacity: [3, 2, 1],
              initial: [2, 2, 1],
              instantiated: 3,
              coordinates: [
                [1, 1, 1],
                [2, 1, 1],
                [1, 2, 1],
              ],
              maxSteps: 1,
              growthThreshold: null,
            },
          },
        }),
      ],
      [],
    );
    const diagram = buildClusterDiagrams(g).get("neuron_cluster");

    expect(diagram?.columns).toBe(3);
    expect(diagram?.rows).toBe(2);
    expect(diagram?.planes).toHaveLength(1);
    expect(diagram?.instantiated).toBe(3);
    expect(diagram?.capacityTotal).toBe(6);
    expect(diagram?.maxSteps).toBe(1);

    const plane = diagram?.planes[0];
    expect(plane?.z).toBe(1);
    expect(plane?.cells).toHaveLength(6);
    const filledCoords = plane?.cells
      .filter((cell) => cell.filled)
      .map((cell) => [cell.x, cell.y]);
    expect(filledCoords).toEqual([
      [1, 1],
      [2, 1],
      [1, 2],
    ]);
  });

  it("flags overflow when capacity exceeds the display limits", () => {
    const g = graph(
      [
        node("neuron_cluster", {
          typeName: "NeuronCluster",
          path: "neuron_cluster",
          details: {
            cluster: {
              capacity: [12, 1, 6],
              initial: [1, 1, 1],
              instantiated: 1,
              coordinates: [[1, 1, 1]],
              maxSteps: 4,
              growthThreshold: 50,
            },
          },
        }),
      ],
      [],
    );
    const diagram = buildClusterDiagrams(g).get("neuron_cluster");

    expect(diagram?.columns).toBe(8);
    expect(diagram?.planes).toHaveLength(4);
    expect(diagram?.hasColumnOverflow).toBe(true);
    expect(diagram?.hasPlaneOverflow).toBe(true);
    expect(diagram?.hasRowOverflow).toBe(false);
    expect(diagram?.growthThreshold).toBe(50);
  });

  it("ignores nodes without cluster details", () => {
    const g = graph([node("main_model", { typeName: "Sequential" })], []);
    expect(buildClusterDiagrams(g).get("main_model")).toBeUndefined();
  });
});

describe("buildTerminalReachGrid", () => {
  it("classifies self, reachable, and out-of-reach cells across z-planes", () => {
    const grid = buildTerminalReachGrid({
      terminalReach: {
        position: [2, 2, 1],
        connections: [
          [2, 2, 1],
          [3, 2, 1],
          [2, 2, 2],
        ],
        total: 3,
      },
    });

    expect(grid?.minX).toBe(2);
    expect(grid?.minY).toBe(2);
    expect(grid?.columns).toBe(2);
    expect(grid?.rows).toBe(1);
    expect(grid?.total).toBe(3);
    expect(grid?.planes.map((plane) => plane.z)).toEqual([1, 2]);

    const planeOne = grid?.planes.find((plane) => plane.z === 1);
    const kindByCoord = new Map(
      planeOne?.cells.map((cell) => [`${cell.x},${cell.y}`, cell.kind]),
    );
    expect(kindByCoord.get("2,2")).toBe("self");
    expect(kindByCoord.get("3,2")).toBe("reach");

    const planeTwo = grid?.planes.find((plane) => plane.z === 2);
    const planeTwoKinds = new Map(
      planeTwo?.cells.map((cell) => [`${cell.x},${cell.y}`, cell.kind]),
    );
    expect(planeTwoKinds.get("2,2")).toBe("reach");
    expect(planeTwoKinds.get("3,2")).toBe("empty");
  });

  it("returns undefined when there is no terminal reach detail", () => {
    expect(buildTerminalReachGrid({})).toBeUndefined();
  });
});

describe("buildGraphLocationSummaries", () => {
  it("returns neuron cluster coordinates with instantiated and capacity counts", () => {
    const g = graph(
      [
        node("neuron_cluster", {
          label: "Cluster",
          typeName: "NeuronCluster",
          path: "neuron_cluster",
          details: {
            cluster: {
              capacity: [3, 2, 1],
              instantiated: 2,
              coordinates: [
                [1, 1, 1],
                [2, 1, 1],
              ],
            },
          },
        }),
      ],
      [],
    );

    expect(buildGraphLocationSummaries(g)).toEqual([
      {
        kind: "cluster",
        nodeId: "neuron_cluster",
        nodePath: "neuron_cluster",
        nodeLabel: "Cluster",
        nodeType: "NeuronCluster",
        coordinates: [
          [1, 1, 1],
          [2, 1, 1],
        ],
        instantiated: 2,
        capacityTotal: 6,
        hasOverflow: false,
      },
    ]);
  });

  it("returns terminal reach position and reachable coordinates", () => {
    const g = graph(
      [
        node("terminal", {
          label: "Terminal",
          typeName: "Terminal",
          path: "cluster.terminal",
          details: {
            terminalReach: {
              position: [2, 2, 1],
              connections: [
                [2, 2, 1],
                [3, 2, 1],
              ],
              total: 5,
            },
          },
        }),
      ],
      [],
    );

    expect(buildGraphLocationSummaries(g)).toEqual([
      {
        kind: "terminalReach",
        nodeId: "terminal",
        nodePath: "cluster.terminal",
        nodeLabel: "Terminal",
        nodeType: "Terminal",
        position: [2, 2, 1],
        connections: [
          [2, 2, 1],
          [3, 2, 1],
        ],
        total: 5,
        hasOverflow: true,
      },
    ]);
  });

  it("ignores nodes without valid location details", () => {
    const g = graph(
      [
        node("plain", { typeName: "LinearLayer", details: {} }),
        node("cluster", {
          typeName: "NeuronCluster",
          details: {
            cluster: {
              capacity: [2, 2, 1],
              coordinates: [["x", 1, 1]],
            },
          },
        }),
        node("terminal", {
          typeName: "Terminal",
          details: {
            terminalReach: {
              position: [1, 2],
              connections: [[1, 2, 3]],
            },
          },
        }),
      ],
      [],
    );

    expect(buildGraphLocationSummaries(g)).toEqual([]);
  });

  it("handles empty and invalid coordinate arrays safely", () => {
    const g = graph(
      [
        node("empty_cluster", {
          typeName: "NeuronCluster",
          details: {
            cluster: {
              capacity: [2, 2, 1],
              coordinates: [],
            },
          },
        }),
        node("terminal", {
          typeName: "Terminal",
          details: {
            terminalReach: {
              position: [4, 5, 6],
              connections: [[5, 5, 6], [Number.POSITIVE_INFINITY, 5, 6], ["bad"]],
            },
          },
        }),
      ],
      [],
    );

    expect(buildGraphLocationSummaries(g)).toEqual([
      {
        kind: "terminalReach",
        nodeId: "terminal",
        nodePath: "terminal",
        nodeLabel: "terminal",
        nodeType: "Terminal",
        position: [4, 5, 6],
        connections: [[5, 5, 6]],
        total: 1,
        hasOverflow: false,
      },
    ]);
  });
});

describe("filterGraphByDetail", () => {
  it("keeps architecture nodes and reparents edges through hidden ancestors", () => {
    const g = graph(
      [
        node("model"),
        node("proc", { graphRole: "internal" }),
        node("proc.linear", { path: "proc.linear" }),
      ],
      [["model", "proc"], ["proc", "proc.linear"]],
    );
    const filtered = filterGraphByDetail(g, "basic");
    expect(filtered?.nodes.map((n) => n.id)).toEqual(["model", "proc.linear"]);
    expect(filtered?.edges).toEqual([{ id: "model-proc.linear", source: "model", target: "proc.linear" }]);
  });

  it("uses the basic architecture-filtered topology in simple mode", () => {
    const g = graph(
      [
        node("model"),
        node("proc", { graphRole: "internal" }),
        node("proc.linear", { path: "proc.linear" }),
      ],
      [["model", "proc"], ["proc", "proc.linear"]],
    );

    expect(filterGraphByDetail(g, "simple")).toEqual(filterGraphByDetail(g, "basic"));
  });

  it("returns the graph untouched in full mode", () => {
    const g = graph([node("model"), node("x", { graphRole: "internal" })], [["model", "x"]]);
    expect(filterGraphByDetail(g, "full")).toBe(g);
  });
});

describe("filterGraphByExpansion", () => {
  it("shows only roots and expanded subtrees in opened scope", () => {
    const g = graph([node("a"), node("b"), node("b.c", { path: "b.c" })], [["a", "b"], ["b", "b.c"]]);
    const nav = buildGraphNavigation(g);
    const collapsed = filterGraphByExpansion(g, nav, new Set(), "opened");
    expect(collapsed?.nodes.map((n) => n.id)).toEqual(["a", "b"]);

    const expanded = filterGraphByExpansion(g, nav, new Set(["b"]), "opened");
    expect(expanded?.nodes.map((n) => n.id)).toEqual(["a", "b", "b.c"]);
  });

  it("returns the graph untouched in entire scope", () => {
    const g = graph([node("a"), node("b")], [["a", "b"]]);
    expect(filterGraphByExpansion(g, buildGraphNavigation(g), new Set(), "entire")).toBe(g);
  });
});

describe("layoutGraph", () => {
  it("produces positioned react-flow nodes and edges", () => {
    const g = graph(
      [node("a", { parameterCount: 1234 }), node("b", { typeName: "Layer" })],
      [["a", "b"]],
    );
    const nav = buildGraphNavigation(g);
    const { nodes, edges } = layoutGraph(g, {
      graphDetailMode: "basic",
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      expandedGraphNodeIds: new Set(),
      expandedDetailNodeIds: new Set(),
      enableExpansion: true,
      selectedNodeId: "a",
      onActivateNode: () => {},
      onToggleExpansion: () => {},
      onToggleDetails: () => {},
    });

    expect(nodes.map((n) => n.id)).toEqual(["a", "b"]);
    expect(nodes[0].selected).toBe(true);
    expect(nodes[0].data.parameterCount).toBe(1234);
    expect(nodes[0].data.height).toBe(126);
    expect(nodes[1].data.label).toBe("Layer");
    expect(typeof nodes[0].position.x).toBe("number");
    expect(edges.map((e) => e.id)).toEqual(["a-b"]);
  });

  it("keeps reserving a dedicated badge row in full mode", () => {
    const g = graph(
      [node("a", { parameterCount: 1234 }), node("b", { typeName: "Layer" })],
      [["a", "b"]],
    );
    const nav = buildGraphNavigation(g);
    const { nodes } = layoutGraph(g, {
      graphDetailMode: "full",
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      expandedGraphNodeIds: new Set(),
      expandedDetailNodeIds: new Set(),
      enableExpansion: true,
      selectedNodeId: "a",
      onActivateNode: () => {},
      onToggleExpansion: () => {},
      onToggleDetails: () => {},
    });

    expect(nodes[0].data.height).toBe(154);
    expect(nodes[0].style?.height).toBe(154);
  });

  it("does not reserve expanded detail space for preview-only dimensions", () => {
    const g = graph(
      [
        node("a", {
          details: {
            dims: "128 -> 128",
          },
        }),
        node("b", { typeName: "LinearLayer" }),
      ],
      [["a", "b"]],
    );
    const nav = buildGraphNavigation(g);
    const { nodes } = layoutGraph(g, {
      graphDetailMode: "basic",
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      expandedGraphNodeIds: new Set(),
      expandedDetailNodeIds: new Set(),
      enableExpansion: true,
      selectedNodeId: "a",
      onActivateNode: () => {},
      onToggleExpansion: () => {},
      onToggleDetails: () => {},
    });

    expect(nodes[0].data.height).toBe(126);
  });

  it("reserves card height for direct weight and bias shapes", () => {
    const g = graph(
      [
        node("a", {
          details: {
            weightShape: "3 x 4",
            biasShape: "3",
          },
        }),
        node("b", { typeName: "LinearLayer" }),
      ],
      [["a", "b"]],
    );
    const nav = buildGraphNavigation(g);
    const { nodes } = layoutGraph(g, {
      graphDetailMode: "basic",
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      expandedGraphNodeIds: new Set(),
      expandedDetailNodeIds: new Set(),
      enableExpansion: true,
      selectedNodeId: "a",
      onActivateNode: () => {},
      onToggleExpansion: () => {},
      onToggleDetails: () => {},
    });

    expect(nodes[0].data.height).toBe(162);
  });

  it("sizes expanded detail accordions without adding an extra header gap", () => {
    const g = graph(
      [
        node("a", {
          details: {
            dims: "128 -> 128",
            activation: "GELU",
            dropout: 0.1,
          },
        }),
        node("b", { typeName: "LinearLayer" }),
      ],
      [["a", "b"]],
    );
    const nav = buildGraphNavigation(g);
    const { nodes } = layoutGraph(g, {
      graphDetailMode: "basic",
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      expandedGraphNodeIds: new Set(),
      expandedDetailNodeIds: new Set(["a"]),
      enableExpansion: true,
      selectedNodeId: "a",
      onActivateNode: () => {},
      onToggleExpansion: () => {},
      onToggleDetails: () => {},
    });

    expect(nodes[0].data.height).toBe(263);
  });

  it("sizes expanded detail accordions from config row counts", () => {
    const g = graph(
      [
        node("a", {
          details: {
            dims: "128 -> 128",
            weightShape: "128 x 128",
          },
          config: {
            typeName: "LinearLayerConfig",
            fields: [
              { key: "input_dim", value: 128 },
              { key: "output_dim", value: 128 },
              { key: "bias_flag", value: false },
            ],
          },
        }),
        node("b", { typeName: "LinearLayer" }),
      ],
      [["a", "b"]],
    );
    const nav = buildGraphNavigation(g);
    const collapsed = layoutGraph(g, {
      graphDetailMode: "basic",
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      expandedGraphNodeIds: new Set(),
      expandedDetailNodeIds: new Set(),
      enableExpansion: true,
      selectedNodeId: "a",
      onActivateNode: () => {},
      onToggleExpansion: () => {},
      onToggleDetails: () => {},
    }).nodes[0];
    const expanded = layoutGraph(g, {
      graphDetailMode: "basic",
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      expandedGraphNodeIds: new Set(),
      expandedDetailNodeIds: new Set(["a"]),
      enableExpansion: true,
      selectedNodeId: "a",
      onActivateNode: () => {},
      onToggleExpansion: () => {},
      onToggleDetails: () => {},
    }).nodes[0];

    expect(collapsed.data.height).toBe(223);
    expect(expanded.data.height - collapsed.data.height).toBe(112);
  });

  it("reserves diagram height and keeps detail expansion math stable", () => {
    const g = graph(
      [
        node("moe", {
          typeName: "MixtureOfExperts",
          path: "moe",
          details: { topK: 2, numExperts: 4 },
        }),
        node("moe.sampler", { typeName: "SamplerModel", path: "moe.sampler" }),
        node("moe.expert_modules", {
          typeName: "ModuleList",
          path: "moe.expert_modules",
        }),
        ...Array.from({ length: 4 }, (_, index) =>
          node(`moe.expert_modules.${index}`, {
            typeName: "Sequential",
            path: `moe.expert_modules.${index}`,
          }),
        ),
      ],
      [
        ["moe", "moe.sampler"],
        ["moe", "moe.expert_modules"],
        ...Array.from({ length: 4 }, (_, index) => [
          "moe.expert_modules",
          `moe.expert_modules.${index}`,
        ] as [string, string]),
      ],
    );
    const nav = buildGraphNavigation(g);
    const expertDiagramsById = buildExpertDiagrams(g, nav);
    const baseOptions = {
      graphDetailMode: "basic" as const,
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      expertDiagramsById,
      expandedGraphNodeIds: new Set<string>(),
      enableExpansion: true,
      selectedNodeId: "moe",
      onActivateNode: () => {},
      onToggleExpansion: () => {},
      onToggleDetails: () => {},
    };

    const collapsed = layoutGraph(g, {
      ...baseOptions,
      expandedDetailNodeIds: new Set<string>(),
    }).nodes[0];
    const expanded = layoutGraph(g, {
      ...baseOptions,
      expandedDetailNodeIds: new Set(["moe"]),
    }).nodes[0];

    expect(collapsed.data.expertDiagram?.cells.map((cell) => cell.label)).toEqual([
      "E0",
      "E1",
      "E2",
      "E3",
    ]);
    expect(collapsed.data.height).toBeGreaterThan(148);
    expect(expanded.data.height - collapsed.data.height).toBe(76);
  });

  it("reserves stack diagram height and keeps detail expansion math stable", () => {
    const g = graph(
      [
        node("main_model", {
          typeName: "Sequential",
          path: "main_model",
          details: { numLayers: 3, dims: "128 -> 128" },
        }),
        ...Array.from({ length: 3 }, (_, index) =>
          node(`main_model.${index}`, {
            typeName: "Layer",
            path: `main_model.${index}`,
            details: { dims: "128 -> 128" },
          }),
        ),
      ],
      Array.from({ length: 3 }, (_, index) => [
        "main_model",
        `main_model.${index}`,
      ] as [string, string]),
    );
    const nav = buildGraphNavigation(g);
    const stackDiagramsById = buildStackDiagrams(g, nav);
    const baseOptions = {
      graphDetailMode: "basic" as const,
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      stackDiagramsById,
      expandedGraphNodeIds: new Set<string>(),
      enableExpansion: true,
      selectedNodeId: "main_model",
      onActivateNode: () => {},
      onToggleExpansion: () => {},
      onToggleDetails: () => {},
    };

    const collapsed = layoutGraph(g, {
      ...baseOptions,
      expandedDetailNodeIds: new Set<string>(),
    }).nodes[0];
    const expanded = layoutGraph(g, {
      ...baseOptions,
      expandedDetailNodeIds: new Set(["main_model"]),
    }).nodes[0];

    expect(collapsed.data.stackDiagram?.cells.map((cell) => cell.label)).toEqual([
      "Layer 0 · Layer",
      "Layer 1 · Layer",
      "Layer 2 · Layer",
    ]);
    expect(collapsed.data.height).toBe(263);
    expect(expanded.data.height - collapsed.data.height).toBe(40);
  });

  it("attaches cluster diagrams to graph nodes and reserves map height", () => {
    const g = graph(
      [
        node("neuron_cluster", {
          typeName: "NeuronCluster",
          path: "neuron_cluster",
          details: {
            cluster: {
              capacity: [2, 2, 1],
              instantiated: 2,
              coordinates: [
                [1, 1, 1],
                [2, 1, 1],
              ],
            },
          },
        }),
      ],
      [],
    );
    const nav = buildGraphNavigation(g);
    const { nodes } = layoutGraph(g, {
      graphDetailMode: "basic",
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      clusterDiagramsById: buildClusterDiagrams(g),
      expandedGraphNodeIds: new Set<string>(),
      expandedDetailNodeIds: new Set<string>(),
      enableExpansion: true,
      selectedNodeId: "neuron_cluster",
      onActivateNode: () => {},
      onToggleExpansion: () => {},
      onToggleDetails: () => {},
    });

    expect(nodes[0].data.clusterDiagram?.instantiated).toBe(2);
    expect(nodes[0].data.clusterDiagram?.capacityTotal).toBe(4);
    expect(nodes[0].data.clusterDiagram?.planes[0]?.cells.filter((cell) => cell.filled))
      .toHaveLength(2);
    expect(nodes[0].data.height).toBe(186);
  });

  it("reserves large cluster map height before config options", () => {
    const activeCoordinates = Array.from({ length: 64 }, (_, index) => [
      (index % 8) + 1,
      Math.floor(index / 8) + 1,
      1,
    ]);
    const g = graph(
      [
        node("model", {
          typeName: "Sequential",
          path: "model",
        }),
        node("model.cluster", {
          typeName: "NeuronCluster",
          path: "model.cluster",
          parameterCount: 12345,
          details: {
            cluster: {
              capacity: [10, 10, 1],
              instantiated: 64,
              coordinates: activeCoordinates,
            },
          },
          config: {
            typeName: "NeuronClusterConfig",
            fields: [
              { key: "capacity", value: [10, 10, 1] },
              { key: "growth_threshold", value: 0.5 },
            ],
          },
        }),
        node("model.cluster.terminal_a", {
          typeName: "TerminalNeuron",
          path: "model.cluster.terminal_a",
        }),
        node("model.cluster.terminal_b", {
          typeName: "TerminalNeuron",
          path: "model.cluster.terminal_b",
        }),
      ],
      [
        ["model", "model.cluster"],
        ["model.cluster", "model.cluster.terminal_a"],
        ["model.cluster", "model.cluster.terminal_b"],
      ],
    );
    const nav = buildGraphNavigation(g);
    const { nodes } = layoutGraph(g, {
      graphDetailMode: "basic",
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      clusterDiagramsById: buildClusterDiagrams(g),
      expandedGraphNodeIds: new Set<string>(),
      expandedDetailNodeIds: new Set<string>(),
      enableExpansion: true,
      selectedNodeId: "model.cluster",
      onActivateNode: () => {},
      onToggleExpansion: () => {},
      onToggleDetails: () => {},
    });

    const clusterNode = nodes.find((node) => node.id === "model.cluster");
    expect(clusterNode?.data.canToggleExpansion).toBe(true);
    expect(clusterNode?.data.parameterCount).toBe(12345);
    expect(clusterNode?.data.childCount).toBe(2);
    expect(clusterNode?.data.clusterDiagram?.rows).toBe(8);
    expect(clusterNode?.data.clusterDiagram?.columns).toBe(8);
    expect(clusterNode?.data.config?.fields).toHaveLength(2);
    expect(clusterNode?.data.height).toBe(395);
    expect(clusterNode?.style?.height).toBe(395);
  });

  it("reserves taller stack diagram height for dense layer previews", () => {
    const g = graph(
      [
        node("main_model", {
          typeName: "Sequential",
          path: "main_model",
          details: { numLayers: 5, dims: "128 -> 128" },
        }),
        ...Array.from({ length: 5 }, (_, index) =>
          node(`main_model.${index}`, {
            typeName: "Layer",
            path: `main_model.${index}`,
            details: { dims: "128 -> 128" },
          }),
        ),
      ],
      Array.from({ length: 5 }, (_, index) => [
        "main_model",
        `main_model.${index}`,
      ] as [string, string]),
    );
    const nav = buildGraphNavigation(g);
    const { nodes } = layoutGraph(g, {
      graphDetailMode: "basic",
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      stackDiagramsById: buildStackDiagrams(g, nav),
      expandedGraphNodeIds: new Set<string>(),
      expandedDetailNodeIds: new Set<string>(),
      enableExpansion: true,
      selectedNodeId: "main_model",
      onActivateNode: () => {},
      onToggleExpansion: () => {},
      onToggleDetails: () => {},
    });

    expect(nodes[0].data.stackDiagram?.cells).toHaveLength(5);
    expect(nodes[0].data.height).toBe(311);
  });

  it("uses compact fixed card dimensions in simple mode", () => {
    const g = graph(
      [
        node("a", {
          parameterCount: 1234,
          details: {
            activation: "GELU",
            weightShape: "3 x 4",
          },
        }),
        node("b", { typeName: "Layer" }),
      ],
      [["a", "b"]],
    );
    const nav = buildGraphNavigation(g);
    const { nodes } = layoutGraph(g, {
      graphDetailMode: "simple",
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      expandedGraphNodeIds: new Set(),
      expandedDetailNodeIds: new Set(["a"]),
      enableExpansion: true,
      selectedNodeId: "a",
      onActivateNode: () => {},
      onToggleExpansion: () => {},
      onToggleDetails: () => {},
    });

    expect(nodes[0].data.graphDetailMode).toBe("simple");
    expect(nodes[0].data.height).toBe(SIMPLE_NODE_HEIGHT);
    expect(nodes[0].style?.height).toBe(SIMPLE_NODE_HEIGHT);
  });

  it("returns empty layout for undefined graph", () => {
    const nav = buildGraphNavigation(undefined);
    expect(
      layoutGraph(undefined, {
        graphDetailMode: "basic",
        navigation: nav,
        childSummariesById: new Map(),
        expandedGraphNodeIds: new Set(),
        expandedDetailNodeIds: new Set(),
        enableExpansion: true,
        selectedNodeId: null,
        onActivateNode: () => {},
        onToggleExpansion: () => {},
        onToggleDetails: () => {},
      }),
    ).toEqual({ nodes: [], edges: [] });
  });
});
