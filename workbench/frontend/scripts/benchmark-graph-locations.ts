import { performance } from "node:perf_hooks";

import { type GraphNode, type InspectResponse } from "@/lib/api";
import { buildClusterDiagrams } from "@/lib/graph/cluster-diagrams";
import { buildGraphLocationSummaries } from "@/lib/graph/locations";
import { graphCoordinates, isRecord } from "@/lib/graph/helpers";

const CLUSTER_COUNT = 8;
const CAPACITY = [40, 40, 8] as const;
const ACTIVE_COORDINATE_COUNT = 120;
const WARMUP_COUNT = 5;
const SAMPLE_COUNT = 25;

const coordinates = Array.from(
  { length: ACTIVE_COORDINATE_COUNT },
  (_, index) =>
    [
      (index % CAPACITY[0]) + 1,
      (Math.floor(index / CAPACITY[0]) % CAPACITY[1]) + 1,
      1,
    ] as [number, number, number],
);
const nodes: GraphNode[] = Array.from({ length: CLUSTER_COUNT }, (_, index) => ({
  id: `cluster-${index}`,
  label: `Cluster ${index}`,
  typeName: "NeuronCluster",
  path: `model.cluster.${index}`,
  graphRole: "architecture",
  parameterCount: 0,
  parameterSizeBytes: 0,
  details: {
    cluster: {
      capacity: [...CAPACITY],
      instantiated: ACTIVE_COORDINATE_COUNT,
      coordinates,
    },
  },
  config: null,
}));
const graph: InspectResponse = {
  modelType: "neurons",
  model: "neuron",
  preset: "baseline",
  parameterCount: 0,
  parameterSizeBytes: 0,
  nodes,
  edges: [],
};

function fullGridBaseline() {
  const diagrams = buildClusterDiagrams(graph);
  return graph.nodes.reduce((checksum, node) => {
    const cluster = node.details.cluster;
    const diagram = diagrams.get(node.id);
    return !isRecord(cluster) || !diagram
      ? checksum
      : checksum +
          diagram.capacityTotal +
          diagram.instantiated +
          graphCoordinates(cluster.coordinates).length;
  }, 0);
}

function currentLocationSummaries() {
  return buildGraphLocationSummaries(graph).reduce(
    (checksum, summary) =>
      summary.kind === "cluster"
        ? checksum +
          summary.capacityTotal +
          summary.instantiated +
          summary.coordinates.length
        : checksum,
    0,
  );
}

function percentile(values: number[], ratio: number) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.min(ordered.length - 1, Math.floor(ordered.length * ratio))];
}

function measure(run: () => number) {
  for (let index = 0; index < WARMUP_COUNT; index += 1) {
    run();
  }
  const samples = Array.from({ length: SAMPLE_COUNT }, () => {
    const startedAt = performance.now();
    const checksum = run();
    const duration = performance.now() - startedAt;
    if (checksum !== fullGridBaselineChecksum) {
      throw new Error(`Location benchmark checksum changed: ${checksum}.`);
    }
    return duration;
  });
  return {
    medianMs: percentile(samples, 0.5),
    p95Ms: percentile(samples, 0.95),
  };
}

const fullGridBaselineChecksum = fullGridBaseline();
if (currentLocationSummaries() !== fullGridBaselineChecksum) {
  throw new Error("Location benchmark implementations disagree before measurement.");
}
const baseline = measure(fullGridBaseline);
const current = measure(currentLocationSummaries);

console.log(
  `Graph location benchmark (${CLUSTER_COUNT} clusters, ${CAPACITY.join("x")} capacity, ${ACTIVE_COORDINATE_COUNT} active each, ${SAMPLE_COUNT} samples)`,
);
console.table([
  {
    implementation: "full cell-grid baseline",
    "median ms": baseline.medianMs.toFixed(3),
    "p95 ms": baseline.p95Ms.toFixed(3),
  },
  {
    implementation: "current summaries",
    "median ms": current.medianMs.toFixed(3),
    "p95 ms": current.p95Ms.toFixed(3),
  },
]);
