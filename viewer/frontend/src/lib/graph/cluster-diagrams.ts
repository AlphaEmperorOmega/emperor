import { type GraphNode, type InspectResponse } from "@/lib/api";
import {
  CLUSTER_DIAGRAM_MAX_DIM,
  CLUSTER_DIAGRAM_MAX_PLANES,
} from "@/lib/graph/constants";
import { graphCoordinates, isRecord } from "@/lib/graph/helpers";
import {
  type ClusterDiagram,
  type ClusterDiagramCell,
  type ClusterDiagramPlane,
} from "@/lib/graph/types";

function clusterDetail(node: GraphNode): Record<string, unknown> | undefined {
  const cluster = node.details.cluster;
  return isRecord(cluster) ? cluster : undefined;
}

function axisValue(value: unknown, index: number): number {
  if (Array.isArray(value)) {
    const item = value[index];
    if (typeof item === "number" && item > 0) {
      return item;
    }
  }
  return 1;
}

function coordinateKeySet(coordinates: unknown): Set<string> {
  const keys = new Set<string>();
  for (const coordinate of graphCoordinates(coordinates)) {
    keys.add(coordinate.join(","));
  }
  return keys;
}

function clusterDiagramFromDetail(
  cluster: Record<string, unknown>,
  maxDim: number,
  maxPlanes: number,
): ClusterDiagram {
  const capacityColumns = axisValue(cluster.capacity, 0);
  const capacityRows = axisValue(cluster.capacity, 1);
  const capacityPlanes = axisValue(cluster.capacity, 2);
  const columns = Math.min(capacityColumns, maxDim);
  const rows = Math.min(capacityRows, maxDim);
  const planeCount = Math.min(capacityPlanes, maxPlanes);
  const filled = coordinateKeySet(cluster.coordinates);

  const planes: ClusterDiagramPlane[] = [];
  for (let z = 1; z <= planeCount; z += 1) {
    const cells: ClusterDiagramCell[] = [];
    for (let y = 1; y <= rows; y += 1) {
      for (let x = 1; x <= columns; x += 1) {
        const isFilled = filled.has(`${x},${y},${z}`);
        cells.push({
          x,
          y,
          filled: isFilled,
          title: `Neuron (${x}, ${y}, ${z}) — ${isFilled ? "active" : "empty"}`,
        });
      }
    }
    planes.push({ z, cells });
  }

  const instantiated =
    typeof cluster.instantiated === "number" ? cluster.instantiated : filled.size;

  return {
    columns,
    rows,
    planes,
    instantiated,
    capacityTotal: capacityColumns * capacityRows * capacityPlanes,
    maxSteps: typeof cluster.maxSteps === "number" ? cluster.maxSteps : null,
    growthThreshold:
      typeof cluster.growthThreshold === "number" ? cluster.growthThreshold : null,
    hasColumnOverflow: capacityColumns > columns,
    hasRowOverflow: capacityRows > rows,
    hasPlaneOverflow: capacityPlanes > planeCount,
  };
}

export function buildClusterDiagrams(
  graph: InspectResponse | undefined,
  options: { maxDim?: number; maxPlanes?: number } = {},
) {
  const maxDim = options.maxDim ?? CLUSTER_DIAGRAM_MAX_DIM;
  const maxPlanes = options.maxPlanes ?? CLUSTER_DIAGRAM_MAX_PLANES;
  const diagramsById = new Map<string, ClusterDiagram>();
  if (!graph) {
    return diagramsById;
  }

  for (const node of graph.nodes) {
    if (node.typeName !== "NeuronCluster") {
      continue;
    }
    const cluster = clusterDetail(node);
    if (!cluster) {
      continue;
    }
    diagramsById.set(node.id, clusterDiagramFromDetail(cluster, maxDim, maxPlanes));
  }

  return diagramsById;
}
