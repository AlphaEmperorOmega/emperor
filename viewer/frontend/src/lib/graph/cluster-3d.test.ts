import { describe, expect, it } from "vitest";
import { buildCluster3DSceneModel } from "@/lib/graph/cluster-3d";
import {
  type GraphNode,
  type InspectResponse,
  type TrainingJob,
  type TrainingProgressEvent,
} from "@/lib/api";

type GraphRole = GraphNode["graphRole"];

function node(id: string, overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id,
    label: overrides.label ?? id,
    typeName: overrides.typeName ?? id,
    path: overrides.path ?? id,
    graphRole: (overrides.graphRole ?? "architecture") as GraphRole,
    parameterCount: overrides.parameterCount ?? 0,
    parameterSizeBytes:
      overrides.parameterSizeBytes ?? (overrides.parameterCount ?? 0) * 4,
    details: overrides.details ?? {},
    config: overrides.config ?? null,
  };
}

function graph(nodes: GraphNode[], edges: Array<[string, string]>): InspectResponse {
  return {
    model: "linear",
    preset: "baseline",
    parameterCount: nodes[0]?.parameterCount ?? 0,
    parameterSizeBytes: nodes[0]?.parameterSizeBytes ?? 0,
    nodes,
    edges: edges.map(([source, target]) => ({
      id: `${source}-${target}`,
      source,
      target,
    })),
  };
}

function trainingJob(events: TrainingProgressEvent[]): TrainingJob {
  return {
    id: "job-1",
    status: "running",
    model: "neuron_linear",
    preset: "baseline",
    datasets: ["Mnist"],
    overrides: {},
    monitors: [],
    logFolder: "test_model",
    createdAt: "",
    updatedAt: "",
    exitCode: null,
    pid: 1,
    currentDataset: "Mnist",
    epoch: 0,
    step: 0,
    metrics: {},
    logDir: null,
    events,
    eventCount: events.length,
    eventCounts: {},
    eventsTruncated: false,
    clusterGrowth: [],
    logTail: [],
    resultLinks: [],
  };
}

function clusterGraph() {
  return graph(
    [
      node("model.cluster", {
        label: "Cluster",
        typeName: "NeuronCluster",
        path: "model.cluster",
        details: {
          cluster: {
            capacity: [5, 4, 2],
            initial: [2, 2, 1],
            initialStart: [2, 2, 1],
            instantiated: 2,
            coordinates: [
              [2, 2, 1],
              [4, 2, 1],
            ],
            maxSteps: 3,
            growthThreshold: 8,
          },
        },
      }),
    ],
    [],
  );
}

describe("buildCluster3DSceneModel", () => {
  it("returns null for missing, non-cluster, or malformed selected nodes", () => {
    expect(
      buildCluster3DSceneModel({
        graph: undefined,
        selectedNodeId: "model.cluster",
      }),
    ).toBeNull();
    expect(
      buildCluster3DSceneModel({
        graph: graph([node("linear", { typeName: "Linear" })], []),
        selectedNodeId: "linear",
      }),
    ).toBeNull();
    expect(
      buildCluster3DSceneModel({
        graph: graph([
          node("bad", {
            typeName: "NeuronCluster",
            details: { cluster: { capacity: [0, 2, 1] } },
          }),
        ], []),
        selectedNodeId: "bad",
      }),
    ).toBeNull();
  });

  it("derives capacity, initial start, active coordinates, and counts", () => {
    const scene = buildCluster3DSceneModel({
      graph: clusterGraph(),
      selectedNodeId: "model.cluster",
    });

    expect(scene).toMatchObject({
      clusterNodeId: "model.cluster",
      clusterNodePath: "model.cluster",
      clusterNodeLabel: "Cluster",
      capacity: [5, 4, 2],
      initial: [2, 2, 1],
      initialStart: [2, 2, 1],
      capacityTotal: 40,
      instantiated: 2,
      initialCount: 1,
      grownCount: 1,
      recentAddedCount: 0,
      maxSteps: 3,
      growthThreshold: 8,
      renderGhostCells: true,
    });
    expect(scene?.activeCells.map((cell) => [cell.coordinate, cell.category]))
      .toEqual([
        [[2, 2, 1], "initial"],
        [[4, 2, 1], "grown"],
      ]);
  });

  it("classifies recent additions before initial and grown coordinates", () => {
    const scene = buildCluster3DSceneModel({
      graph: clusterGraph(),
      selectedNodeId: "model.cluster",
      activeTrainingJob: trainingJob([
        {
          type: "neuron_added",
          node: "model.cluster",
          coord: [3, 2, 1],
          count: 3,
          capacity: [5, 4, 2],
          step: 12,
        },
      ]),
    });

    expect(scene?.activeCells.map((cell) => [cell.coordinate, cell.category]))
      .toEqual([
        [[2, 2, 1], "initial"],
        [[3, 2, 1], "recentAdded"],
        [[4, 2, 1], "grown"],
      ]);
    expect(scene?.recentAddedCount).toBe(1);
    expect(scene?.overlayOnlyCount).toBe(1);
    expect(scene?.activeCells[1]?.source).toBe("growth-overlay");
  });

  it("ignores growth additions for other clusters", () => {
    const scene = buildCluster3DSceneModel({
      graph: clusterGraph(),
      selectedNodeId: "model.cluster",
      activeTrainingJob: trainingJob([
        {
          type: "neuron_added",
          node: "other.cluster",
          coord: [3, 2, 1],
          count: 3,
          capacity: [5, 4, 2],
          step: 12,
        },
      ]),
    });

    expect(scene?.activeCells).toHaveLength(2);
    expect(scene?.recentAddedCount).toBe(0);
  });

  it("renders overlay-only additions without a descendant graph-node match", () => {
    const scene = buildCluster3DSceneModel({
      graph: clusterGraph(),
      selectedNodeId: "model.cluster",
      activeTrainingJob: trainingJob([
        {
          type: "neuron_added",
          node: "model.cluster",
          coord: [5, 4, 2],
          count: 3,
          capacity: [5, 4, 2],
          step: 20,
        },
      ]),
    });
    const overlayCell = scene?.activeCells.find(
      (cell) => cell.key === "5,4,2",
    );

    expect(overlayCell).toMatchObject({
      category: "recentAdded",
      source: "growth-overlay",
      isOverlayOnly: true,
      nodeMatch: null,
    });
  });

  it("maps coordinates through descendant reach metadata and prefers neurons", () => {
    const scene = buildCluster3DSceneModel({
      graph: graph(
        [
          node("model.cluster", {
            typeName: "NeuronCluster",
            path: "model.cluster",
            details: {
              cluster: {
                capacity: [3, 1, 1],
                initial: [1, 1, 1],
                initialStart: [1, 1, 1],
                instantiated: 2,
                coordinates: [
                  [1, 1, 1],
                  [2, 1, 1],
                ],
              },
            },
          }),
          node("model.cluster.neuron_1_1_1.terminal", {
            typeName: "Terminal",
            path: "model.cluster.neuron_1_1_1.terminal",
            details: {
              terminalReach: {
                position: [1, 1, 1],
                connections: [
                  [1, 1, 1],
                  [2, 1, 1],
                  [3, 1, 1],
                ],
              },
            },
          }),
          node("model.cluster.neuron_1_1_1", {
            typeName: "Neuron",
            path: "model.cluster.neuron_1_1_1",
            details: {
              terminalReach: {
                position: [1, 1, 1],
                connections: [
                  [1, 1, 1],
                  [2, 1, 1],
                ],
              },
            },
          }),
        ],
        [
          ["model.cluster", "model.cluster.neuron_1_1_1"],
          ["model.cluster.neuron_1_1_1", "model.cluster.neuron_1_1_1.terminal"],
        ],
      ),
      selectedNodeId: "model.cluster",
    });
    const cell = scene?.activeCells.find((candidate) => candidate.key === "1,1,1");

    expect(cell?.nodeMatch).toEqual({
      nodeId: "model.cluster.neuron_1_1_1",
      nodePath: "model.cluster.neuron_1_1_1",
      nodeType: "Neuron",
    });
    expect(cell?.reach).toMatchObject({
      position: [1, 1, 1],
      connections: [
        [1, 1, 1],
        [2, 1, 1],
      ],
      activeConnectionTotal: 1,
      emptyConnectionTotal: 0,
      outOfBoundsTotal: 0,
    });
  });

  it("uses path suffix matching only as a fallback", () => {
    const scene = buildCluster3DSceneModel({
      graph: graph(
        [
          node("model.cluster", {
            typeName: "NeuronCluster",
            details: {
              cluster: {
                capacity: [2, 1, 1],
                coordinates: [[2, 1, 1]],
              },
            },
          }),
          node("model.cluster.cluster.2_1_1", {
            typeName: "Neuron",
            path: "model.cluster.cluster.2_1_1",
          }),
        ],
        [["model.cluster", "model.cluster.cluster.2_1_1"]],
      ),
      selectedNodeId: "model.cluster",
    });

    expect(scene?.activeCells[0]?.nodeMatch).toMatchObject({
      nodeId: "model.cluster.cluster.2_1_1",
      nodeType: "Neuron",
    });
  });
});
