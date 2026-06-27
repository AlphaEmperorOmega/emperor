import { useEffect } from "react";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import {
  ReactFlow,
  useStoreApi,
  type Node as ReactFlowNode,
} from "@xyflow/react";
import { describe, expect, it, vi } from "vitest";
import { GraphPreviewPanel } from "@/features/viewer/components/screen/graph-canvas";
import { ParameterActivityMinimapDialog } from "@/features/viewer/components/graph/parameter-activity-minimap-dialog";
import { layoutGraph } from "@/lib/graph/layout";
import {
  buildChildSummaries,
  buildGraphNavigation,
  type GraphParameterActivity,
} from "@/lib/graph";
import { type GraphNode, type InspectResponse } from "@/lib/api";
import { type MonitorChartsSource } from "@/types/monitor";

vi.mock("@/features/viewer/components/monitor/monitor-charts-modal", () => ({
  MonitorChartsModal: ({
    node,
  }: {
    node: { path: string };
  }) => (
    <section role="dialog" aria-label="Monitor charts">
      Monitor charts for {node.path}
    </section>
  ),
}));

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

type ReactFlowStoreApi = ReturnType<typeof useStoreApi>;

function OuterFlowStoreProbe({
  onStore,
}: {
  onStore: (store: ReactFlowStoreApi) => void;
}) {
  const store = useStoreApi();

  useEffect(() => {
    onStore(store);
  }, [onStore, store]);

  return null;
}

describe("GraphPreviewPanel", () => {
  it("renders real React Flow graph nodes with child parameter activity rows", () => {
    const root = node("root", { typeName: "LayerStack", path: "main_model" });
    const layer = node("layer", {
      typeName: "Layer",
      path: "main_model.0",
      details: { dims: "128 -> 64" },
    });
    const linear = node("linear", {
      typeName: "LinearLayer",
      path: "main_model.0.model",
    });
    const graph: InspectResponse = {
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
    const activity: GraphParameterActivity = {
      targetPath: "main_model.0.model",
      weights: {
        status: "updated",
        source: "historical",
        sourceLabel: "1 historical run",
        observedPoints: 2,
      },
      bias: {
        status: "unchanged",
        source: "historical",
        sourceLabel: "1 historical run",
        observedPoints: 2,
      },
    };
    const navigation = buildGraphNavigation(graph);
    const { nodes, edges } = layoutGraph(graph, {
      graphDetailMode: "basic",
      navigation,
      childSummariesById: buildChildSummaries(graph, navigation),
      childSummarySourceNodesById: new Map(graph.nodes.map((candidate) => [
        candidate.id,
        candidate,
      ])),
      expandedGraphNodeIds: new Set(["layer"]),
      expandedDetailNodeIds: new Set(),
      enableExpansion: true,
      selectedNodeId: null,
      parameterActivityForNode: (candidate) =>
        candidate.id === "layer" || candidate.id === "linear"
          ? activity
          : undefined,
      onActivateNode: () => {},
      onToggleExpansion: () => {},
      onToggleDetails: () => {},
    });

    render(
      <div style={{ width: 900, height: 600 }}>
        <GraphPreviewPanel
          graph={graph}
          nodes={nodes}
          edges={edges}
          selectedNodeId={null}
          hasSelectedClusterNode={false}
          cluster3dNodeId={null}
          onSelectNode={() => {}}
          onRevealNode={() => {}}
          onOpenCluster3d={() => {}}
        />
      </div>,
    );

    const childSummaries = screen.getByTestId("child-summaries-root");
    const activityTrigger = within(childSummaries).getByTestId(
      "graph-parameter-indicators",
    );
    const activityRow = activityTrigger.closest(
      '[data-testid^="child-summary-"]',
    );

    expect(activityTrigger).toHaveAttribute("role", "button");
    expect(activityTrigger).toHaveAttribute(
      "aria-label",
      expect.stringMatching(
        /parameter activity: weights updated, bias unchanged/i,
      ),
    );
    expect(activityTrigger).toHaveAttribute(
      "data-testid",
      "graph-parameter-indicators",
    );
    expect(childSummaries).toContainElement(activityTrigger);
    expect(activityRow).toBeInstanceOf(HTMLElement);
    expect(activityRow).not.toHaveAttribute("role", "button");
  });

  it("keeps the parameter activity minimap React Flow store isolated from the outer graph", async () => {
    const outerNodes: ReactFlowNode[] = [
      { id: "outer-main", position: { x: 0, y: 0 }, data: {} },
    ];
    let outerStore: ReactFlowStoreApi | undefined;
    const minimapNode = node("minimap-linear", {
      typeName: "LinearLayer",
      path: "main_model.linear",
    });
    const graph: InspectResponse = {
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      parameterCount: 0,
      parameterSizeBytes: 0,
      nodes: [minimapNode],
      edges: [],
    };
    const activity: GraphParameterActivity = {
      targetPath: minimapNode.path,
      weights: {
        status: "updated",
        source: "historical",
        sourceLabel: "1 historical run",
        observedPoints: 2,
      },
    };
    const source: MonitorChartsSource = {
      kind: "active-job",
      job: {
        id: "job-1",
        status: "running",
        monitors: [],
        preset: "baseline",
        presets: ["baseline"],
        datasets: ["Mnist"],
        logFolder: "/tmp/emperor-viewer",
        currentPreset: "baseline",
        currentDataset: "Mnist",
      },
    };

    render(
      <div style={{ width: 900, height: 600 }}>
        <ReactFlow
          nodes={outerNodes}
          edges={[]}
          minZoom={0.5}
          maxZoom={1.5}
        >
          <OuterFlowStoreProbe
            onStore={(store) => {
              outerStore = store;
            }}
          />
          <ParameterActivityMinimapDialog
            graph={graph}
            activityByNodePath={new Map([[minimapNode.path, activity]])}
            source={source}
            onClose={() => {}}
          />
        </ReactFlow>
      </div>,
    );

    await screen.findByTestId("parameter-activity-minimap-node-minimap-linear");

    await waitFor(() => {
      expect(outerStore).toBeDefined();
      expect(
        outerStore?.getState().nodes.map((flowNode) => flowNode.id),
      ).toEqual(["outer-main"]);
      expect(outerStore?.getState().minZoom).toBe(0.5);
      expect(outerStore?.getState().maxZoom).toBe(1.5);
    });
  });

  it("opens the parameter activity minimap by default with compact branch cards", async () => {
    const root = node("minimap-root", {
      typeName: "LayerStack",
      path: "main_model",
    });
    const branch = node("minimap-branch", {
      typeName: "Layer",
      path: "main_model.0",
    });
    const linear = node("minimap-linear", {
      typeName: "LinearLayer",
      path: "main_model.0.model",
    });
    const graph: InspectResponse = {
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      parameterCount: 0,
      parameterSizeBytes: 0,
      nodes: [root, branch, linear],
      edges: [
        { id: "root-branch", source: root.id, target: branch.id },
        { id: "branch-linear", source: branch.id, target: linear.id },
      ],
    };
    const activity: GraphParameterActivity = {
      targetPath: linear.path,
      weights: {
        status: "updated",
        source: "historical",
        sourceLabel: "1 historical run",
        observedPoints: 2,
      },
      bias: {
        status: "updated",
        source: "historical",
        sourceLabel: "1 historical run",
        observedPoints: 2,
      },
    };
    const source: MonitorChartsSource = {
      kind: "active-job",
      job: {
        id: "job-1",
        status: "running",
        monitors: [],
        preset: "baseline",
        presets: ["baseline"],
        datasets: ["Mnist"],
        logFolder: "/tmp/emperor-viewer",
        currentPreset: "baseline",
        currentDataset: "Mnist",
      },
    };

    render(
      <div style={{ width: 900, height: 600 }}>
        <ParameterActivityMinimapDialog
          graph={graph}
          activityByNodePath={new Map([[linear.path, activity]])}
          source={source}
          onClose={() => {}}
        />
      </div>,
    );

    await screen.findByTestId("parameter-activity-minimap-node-minimap-linear");

    const branchNode = screen.getByTestId("parameter-activity-minimap-node-minimap-branch");

    expect(screen.getByTestId("rf__node-minimap-branch")).toHaveStyle({
      width: "42px",
      height: "42px",
    });
    expect(screen.getByTestId("rf__node-minimap-linear")).toHaveStyle({
      width: "104px",
      height: "42px",
    });

    const collapseBranchButton = branchNode.querySelector<HTMLButtonElement>(
      'button[aria-label="Collapse main_model.0"]',
    );
    expect(collapseBranchButton).toBeInstanceOf(HTMLButtonElement);
    fireEvent.click(collapseBranchButton as HTMLButtonElement);
    expect(
      screen.queryByTestId("parameter-activity-minimap-node-minimap-linear"),
    ).not.toBeInTheDocument();

    const expandBranchButton = branchNode.querySelector<HTMLButtonElement>(
      'button[aria-label="Expand main_model.0"]',
    );
    expect(expandBranchButton).toBeInstanceOf(HTMLButtonElement);
    fireEvent.click(expandBranchButton as HTMLButtonElement);
    const expandedLinearNode = await screen.findByTestId(
      "parameter-activity-minimap-node-minimap-linear",
    );

    const monitorButton = expandedLinearNode.querySelector<HTMLButtonElement>(
      'button[aria-label="Open monitor charts for main_model.0.model"]',
    );
    expect(monitorButton).toBeInstanceOf(HTMLButtonElement);
    fireEvent.click(monitorButton as HTMLButtonElement);
    expect(await screen.findByRole("dialog", {
      name: /monitor charts/i,
    })).toHaveTextContent("main_model.0.model");
  });
});
