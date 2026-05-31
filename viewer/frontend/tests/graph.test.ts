import { describe, expect, it } from "vitest";
import {
  buildChildSummaries,
  buildExpertDiagrams,
  buildGraphNavigation,
  buildHierarchy,
  buildStackDiagrams,
  detailText,
  filterGraphByDetail,
  filterGraphByExpansion,
  layoutGraph,
  formatCompactCount,
  formatExactCount,
  nodeBadges,
  nodeSubtitle,
  nodeTitle,
} from "@/lib/graph";
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
});

describe("parameter count formatting", () => {
  it("formats compact and exact parameter counts", () => {
    expect(formatCompactCount(65800)).toBe("65.8K");
    expect(formatCompactCount(1000000)).toBe("1M");
    expect(formatExactCount(65800)).toBe("65,800");
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

describe("buildGraphNavigation", () => {
  it("maps children and identifies roots", () => {
    const nav = buildGraphNavigation(
      graph([node("a"), node("b"), node("c")], [["a", "b"], ["a", "c"]]),
    );
    expect(nav.childrenById.get("a")).toEqual(["b", "c"]);
    expect([...nav.rootIds]).toEqual(["a"]);
  });

  it("falls back to the first node when no root exists", () => {
    const nav = buildGraphNavigation(graph([node("a"), node("b")], [["a", "b"], ["b", "a"]]));
    expect([...nav.rootIds]).toEqual(["a"]);
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
        node("main_model.0", { typeName: "Layer", path: "main_model.0" }),
        node("main_model.1", { typeName: "Layer", path: "main_model.1" }),
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
      { label: "Layer 0", nestedLabel: "LinearLayer", kind: "child", stackKind: "layer" },
      { label: "Layer 1", nestedLabel: "LinearLayer", kind: "child", stackKind: "layer" },
    ]);
  });

  it("collapses long layer stacks with an ellipsis and total layer count", () => {
    const layerCount = 9;
    const layerNodes = Array.from({ length: layerCount }, (_, index) => [
      node(`main_model.${index}`, {
        typeName: "Layer",
        path: `main_model.${index}`,
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
      { label: "Layer 0", nestedLabel: "LinearLayer", kind: "child", stackKind: "layer" },
      { label: "Layer 1", nestedLabel: "LinearLayer", kind: "child", stackKind: "layer" },
      { label: "Layer 2", nestedLabel: "LinearLayer", kind: "child", stackKind: "layer" },
      { label: "Layer 3", nestedLabel: "LinearLayer", kind: "child", stackKind: "layer" },
      { label: "Layer 4", nestedLabel: "LinearLayer", kind: "child", stackKind: "layer" },
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
    expect(diagram?.cells.map((cell) => cell.label)).toEqual([
      "Layer 0 · LinearLayer",
      "Layer 1 · LinearLayer",
      "Layer 2 · LinearLayer",
    ]);
    expect(diagram?.cells[0]).toEqual({
      label: "Layer 0 · LinearLayer",
      title: "Layer 0 · LinearLayer · 256 -> 256",
      kind: "layer",
      layerIndex: 0,
    });
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
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      expandedGraphNodeIds: new Set(),
      expandedDetailNodeIds: new Set(),
      enableExpansion: true,
      selectedNodeId: "a",
      onActivateNode: () => {},
      onToggleDetails: () => {},
    });

    expect(nodes.map((n) => n.id)).toEqual(["a", "b"]);
    expect(nodes[0].selected).toBe(true);
    expect(nodes[0].data.parameterCount).toBe(1234);
    expect(nodes[0].data.height).toBe(116);
    expect(nodes[1].data.label).toBe("Layer");
    expect(typeof nodes[0].position.x).toBe("number");
    expect(edges.map((e) => e.id)).toEqual(["a-b"]);
  });

  it("reserves bottom padding for collapsed metadata cards", () => {
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
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      expandedGraphNodeIds: new Set(),
      expandedDetailNodeIds: new Set(),
      enableExpansion: true,
      selectedNodeId: "a",
      onActivateNode: () => {},
      onToggleDetails: () => {},
    });

    expect(nodes[0].data.height).toBe(148);
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
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      expandedGraphNodeIds: new Set(),
      expandedDetailNodeIds: new Set(["a"]),
      enableExpansion: true,
      selectedNodeId: "a",
      onActivateNode: () => {},
      onToggleDetails: () => {},
    });

    expect(nodes[0].data.height).toBe(230);
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
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      expertDiagramsById,
      expandedGraphNodeIds: new Set<string>(),
      enableExpansion: true,
      selectedNodeId: "moe",
      onActivateNode: () => {},
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
    expect(expanded.data.height - collapsed.data.height).toBe(56);
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
      navigation: nav,
      childSummariesById: buildChildSummaries(g, nav),
      stackDiagramsById,
      expandedGraphNodeIds: new Set<string>(),
      enableExpansion: true,
      selectedNodeId: "main_model",
      onActivateNode: () => {},
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
    expect(collapsed.data.height).toBe(194);
    expect(expanded.data.height - collapsed.data.height).toBe(56);
  });

  it("returns empty layout for undefined graph", () => {
    const nav = buildGraphNavigation(undefined);
    expect(
      layoutGraph(undefined, {
        navigation: nav,
        childSummariesById: new Map(),
        expandedGraphNodeIds: new Set(),
        expandedDetailNodeIds: new Set(),
        enableExpansion: true,
        selectedNodeId: null,
        onActivateNode: () => {},
        onToggleDetails: () => {},
      }),
    ).toEqual({ nodes: [], edges: [] });
  });
});
