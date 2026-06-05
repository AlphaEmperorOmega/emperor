import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { GraphLocationsCard } from "@/components/features/viewer/graph-locations-card";
import { type GraphNode, type InspectResponse } from "@/lib/api";

function graphNode(id: string, overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id,
    label: overrides.label ?? id,
    typeName: overrides.typeName ?? id,
    path: overrides.path ?? id,
    graphRole: overrides.graphRole ?? "architecture",
    parameterCount: overrides.parameterCount ?? 0,
    details: overrides.details ?? {},
    config: overrides.config ?? null,
  };
}

function locationGraph(): InspectResponse {
  return {
    model: "linear",
    preset: "baseline",
    parameterCount: 0,
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
  it("renders cluster coordinates and counts", () => {
    render(
      <GraphLocationsCard
        graph={locationGraph()}
        selectedNodeId={null}
        onRevealNode={() => {}}
      />,
    );

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
  });

  it("renders terminal reach position and reachable coordinates", () => {
    render(
      <GraphLocationsCard
        graph={locationGraph()}
        selectedNodeId={null}
        onRevealNode={() => {}}
      />,
    );

    const terminal = screen.getByRole("group", {
      name: /neuron_cluster\.terminal locations/i,
    });
    expect(within(terminal).getByText("2 connections")).toBeInTheDocument();
    expect(
      within(terminal).getByRole("button", {
        name: /terminal position \(2, 2, 1\)/i,
      }),
    ).toBeInTheDocument();
    expect(
      within(terminal).getByRole("button", {
        name: /reachable coordinate \(3, 2, 1\)/i,
      }),
    ).toBeInTheDocument();
  });

  it("shows an empty state when no locations exist", () => {
    render(
      <GraphLocationsCard
        graph={{
          model: "linear",
          preset: "baseline",
          parameterCount: 0,
          nodes: [graphNode("main_model", { typeName: "Sequential" })],
          edges: [],
        }}
        selectedNodeId={null}
        onRevealNode={() => {}}
      />,
    );

    expect(screen.getByText("No analysed locations found.")).toBeInTheDocument();
  });

  it("reveals the owning node when a coordinate is clicked", async () => {
    const user = userEvent.setup();
    const onRevealNode = vi.fn();
    render(
      <GraphLocationsCard
        graph={locationGraph()}
        selectedNodeId={null}
        onRevealNode={onRevealNode}
      />,
    );

    await user.click(
      screen.getByRole("button", {
        name: /reveal neuron_cluster coordinate \(1, 2, 1\)/i,
      }),
    );

    expect(onRevealNode).toHaveBeenCalledWith("neuron_cluster");
  });
});
