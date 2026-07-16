import { performance } from "node:perf_hooks";

import { type GraphNode, type InspectResponse } from "@/lib/api/inspection";
import { buildCluster3DSceneModel } from "@/lib/graph/cluster-3d";

const ACTIVE_COORDINATE_COUNT = 240;
const WARMUP_COUNT = 5;
const SAMPLE_COUNT = 30;
const coordinates = Array.from(
  { length: ACTIVE_COORDINATE_COUNT },
  (_, index) =>
    [(index % 30) + 1, Math.floor(index / 30) + 1, 1] as [number, number, number],
);

function graphNode(
  id: string,
  typeName: string,
  coordinate: [number, number, number],
): GraphNode {
  return {
    id,
    label: id,
    typeName,
    path: `model.cluster.${id}`,
    graphRole: "architecture",
    parameterCount: 0,
    parameterSizeBytes: 0,
    details: {
      terminalReach: {
        position: coordinate,
        connections: [coordinate, [31, 8, 1]],
      },
    },
    config: null,
  };
}

const clusterNode: GraphNode = {
  id: "model.cluster",
  label: "Cluster",
  typeName: "NeuronCluster",
  path: "model.cluster",
  graphRole: "architecture",
  parameterCount: 0,
  parameterSizeBytes: 0,
  details: {
    cluster: {
      capacity: [30, 8, 1],
      coordinates,
    },
  },
  config: null,
};
const descendants = coordinates.flatMap((coordinate, index) => [
  graphNode(`terminal-${index}`, "Terminal", coordinate),
  graphNode(`neuron-${index}`, "Neuron", coordinate),
]);
const graph: InspectResponse = {
  modelType: "neurons",
  model: "neuron",
  preset: "baseline",
  parameterCount: 0,
  parameterSizeBytes: 0,
  nodes: [clusterNode, ...descendants],
  edges: descendants.map((node) => ({
    id: `cluster-${node.id}`,
    source: clusterNode.id,
    target: node.id,
  })),
};

function percentile(values: number[], ratio: number) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.min(ordered.length - 1, Math.floor(ordered.length * ratio))];
}

function buildSceneChecksum() {
  const scene = buildCluster3DSceneModel({
    graph,
    selectedNodeId: clusterNode.id,
  });
  if (!scene) {
    throw new Error("Cluster 3D benchmark did not build a scene.");
  }
  const matchedNeurons = scene.activeCells.filter(
    (cell) => cell.nodeMatch?.nodeType === "Neuron",
  ).length;
  const reachConnections = scene.activeCells.reduce(
    (total, cell) => total + (cell.reach?.connections.length ?? 0),
    0,
  );
  return scene.activeCells.length + matchedNeurons + reachConnections;
}

const expectedChecksum = ACTIVE_COORDINATE_COUNT * 4;
function measure() {
  for (let index = 0; index < WARMUP_COUNT; index += 1) {
    buildSceneChecksum();
  }
  const samples = Array.from({ length: SAMPLE_COUNT }, () => {
    const startedAt = performance.now();
    const checksum = buildSceneChecksum();
    const duration = performance.now() - startedAt;
    if (checksum !== expectedChecksum) {
      throw new Error(`Cluster 3D benchmark checksum changed: ${checksum}.`);
    }
    return duration;
  });
  return {
    medianMs: percentile(samples, 0.5),
    p95Ms: percentile(samples, 0.95),
  };
}

const result = measure();
console.log(
  `Cluster 3D scene benchmark (${ACTIVE_COORDINATE_COUNT} active coordinates, ${descendants.length} descendants, ${SAMPLE_COUNT} samples)`,
);
console.table([
  {
    implementation: "current scene builder",
    "median ms": result.medianMs.toFixed(3),
    "p95 ms": result.p95Ms.toFixed(3),
  },
]);
