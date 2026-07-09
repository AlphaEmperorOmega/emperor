import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { GraphLocationsCard } from "@/features/workbench/components/graph/graph-locations-card";
import { type GraphNode, type InspectResponse } from "@/lib/api";

function graphNode(id: string, overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id,
    label: overrides.label ?? id,
    typeName: overrides.typeName ?? id,
    path: overrides.path ?? id,
    graphRole: overrides.graphRole ?? "architecture",
    parameterCount: overrides.parameterCount ?? 0,
    parameterSizeBytes: overrides.parameterSizeBytes ?? (overrides.parameterCount ?? 0) * 4,
    details: overrides.details ?? {},
    config: overrides.config ?? null,
  };
}

function locationGraph(): InspectResponse {
  return {
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    parameterCount: 0,
    parameterSizeBytes: 0,
    nodes: [
      graphNode("neuron_cluster", {
        label: "Cluster",
        typeName: "NeuronCluster",
        path: "neuron_cluster",
        details: {
          cluster: {
            capacity: [3, 2, 1],
            instantiated: 3,
            coordinates: [
              [1, 1, 1],
              [2, 1, 1],
              [1, 2, 1],
            ],
          },
        },
      }),
      graphNode("terminal", {
        label: "Terminal",
        typeName: "Terminal",
        path: "neuron_cluster.terminal",
        details: {
          terminalReach: {
            position: [2, 2, 1],
            connections: [
              [2, 2, 1],
              [3, 2, 1],
            ],
            total: 2,
          },
        },
      }),
    ],
    edges: [],
  };
}

describe("GraphLocationsCard", () => {
  it("renders a collapsed icon button by default for a selected cluster", () => {
    render(
      <GraphLocationsCard
        graph={locationGraph()}
        selectedNodeId="neuron_cluster"
        onRevealNode={() => {}}
      />,
    );

    expect(
      screen.getByRole("button", { name: /show cluster locations/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("group", { name: /neuron_cluster locations/i }),
    ).not.toBeInTheDocument();
  });

  it("opens the selected cluster locations card when the icon is clicked", async () => {
    const user = userEvent.setup();
    render(
      <GraphLocationsCard
        graph={locationGraph()}
        selectedNodeId="neuron_cluster"
        onRevealNode={() => {}}
      />,
    );

    await user.click(screen.getByRole("button", { name: /show cluster locations/i }));

    const cluster = screen.getByRole("group", { name: /neuron_cluster locations/i });
    const kindBadge = within(cluster)
      .getAllByText("Cluster")
      .find((element) => element.className.includes("uppercase"));
    if (!kindBadge) {
      throw new Error("Expected cluster kind badge");
    }
    expect(within(cluster).getByText("3 instantiated")).toBeInTheDocument();
    expect(within(cluster).getByText("6 capacity")).toBeInTheDocument();
    expect(kindBadge).toHaveClass(
      "shrink-0",
      "rounded-[7px]",
      "border-line-soft",
      "bg-black/20",
      "px-1.5",
      "py-1",
      "text-[10px]",
      "leading-none",
    );
    expect(
      within(cluster).getByRole("button", {
        name: /reveal neuron_cluster coordinate \(1, 1, 1\)/i,
      }),
    ).toBeInTheDocument();
    expect(within(cluster).getByText("(2, 1, 1)")).toBeInTheDocument();
    expect(
      screen.queryByRole("group", {
        name: /neuron_cluster\.terminal locations/i,
      }),
    ).not.toBeInTheDocument();
  });

  it("collapses the selected cluster locations card from the header control", async () => {
    const user = userEvent.setup();
    render(
      <GraphLocationsCard
        graph={locationGraph()}
        selectedNodeId="neuron_cluster"
        onRevealNode={() => {}}
      />,
    );

    await user.click(screen.getByRole("button", { name: /show cluster locations/i }));
    await user.click(screen.getByRole("button", { name: /hide cluster locations/i }));

    expect(
      screen.getByRole("button", { name: /show cluster locations/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("group", { name: /neuron_cluster locations/i }),
    ).not.toBeInTheDocument();
  });

  it("collapses when the selected cluster node changes", async () => {
    const user = userEvent.setup();
    const graph = locationGraph();
    const graphWithTwoClusters: InspectResponse = {
      ...graph,
      nodes: [
        ...graph.nodes,
        graphNode("other_cluster", {
          label: "Other Cluster",
          typeName: "NeuronCluster",
          path: "other_cluster",
          details: {
            cluster: {
              capacity: [2, 1, 1],
              instantiated: 1,
              coordinates: [[1, 1, 1]],
            },
          },
        }),
      ],
    };
    const { rerender } = render(
      <GraphLocationsCard
        graph={graphWithTwoClusters}
        selectedNodeId="neuron_cluster"
        onRevealNode={() => {}}
      />,
    );

    await user.click(screen.getByRole("button", { name: /show cluster locations/i }));
    expect(
      screen.getByRole("group", { name: /neuron_cluster locations/i }),
    ).toBeInTheDocument();

    rerender(
      <GraphLocationsCard
        graph={graphWithTwoClusters}
        selectedNodeId="other_cluster"
        onRevealNode={() => {}}
      />,
    );

    expect(
      screen.getByRole("button", { name: /show cluster locations/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("group", { name: /other_cluster locations/i }),
    ).not.toBeInTheDocument();
  });

  it("renders nothing for a selected non-cluster node", () => {
    const { container } = render(
      <GraphLocationsCard
        graph={locationGraph()}
        selectedNodeId="terminal"
        onRevealNode={() => {}}
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing without a selected node", () => {
    const { container } = render(
      <GraphLocationsCard
        graph={locationGraph()}
        selectedNodeId={null}
        onRevealNode={() => {}}
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing for terminal-reach-only data", () => {
    const { container } = render(
      <GraphLocationsCard
        graph={{
          modelType: "linears",
          model: "linear",
          preset: "baseline",
          parameterCount: 0,
          parameterSizeBytes: 0,
          nodes: [
            graphNode("terminal", {
              label: "Terminal",
              typeName: "Terminal",
              path: "terminal",
              details: {
                terminalReach: {
                  position: [2, 2, 1],
                  connections: [[3, 2, 1]],
                  total: 1,
                },
              },
            }),
          ],
          edges: [],
        }}
        selectedNodeId="terminal"
        onRevealNode={() => {}}
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("reveals the owning node when a coordinate is clicked", async () => {
    const user = userEvent.setup();
    const onRevealNode = vi.fn();
    render(
      <GraphLocationsCard
        graph={locationGraph()}
        selectedNodeId="neuron_cluster"
        onRevealNode={onRevealNode}
      />,
    );

    await user.click(screen.getByRole("button", { name: /show cluster locations/i }));
    await user.click(
      screen.getByRole("button", {
        name: /reveal neuron_cluster coordinate \(1, 2, 1\)/i,
      }),
    );

    expect(onRevealNode).toHaveBeenCalledWith("neuron_cluster");
  });
});
