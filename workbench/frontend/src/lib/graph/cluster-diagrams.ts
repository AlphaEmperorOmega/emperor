import { type GraphNode, type InspectResponse } from "@/lib/api";
import { graphCoordinates, isRecord } from "@/lib/graph/helpers";
import { parseTerminalReachDetails } from "@/lib/graph/terminal-reach";
import {
  type ClusterDiagram,
  type ClusterDiagramCell,
  type ClusterDiagramPlane,
  type ClusterDiagramReach,
  type GraphCoordinate,
} from "@/lib/graph/types";

type ClusterCapacity = {
  columns: number;
  rows: number;
  planes: number;
};

type ClusterDiagramMetadata = Omit<ClusterDiagram, "planes"> & {
  planeCount: number;
};

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

function coordinateKey(coordinate: GraphCoordinate): string {
  return coordinate.join(",");
}

function coordinateKeySet(coordinates: unknown): Set<string> {
  const keys = new Set<string>();
  for (const coordinate of graphCoordinates(coordinates)) {
    keys.add(coordinateKey(coordinate));
  }
  return keys;
}

function uniqueCoordinates(coordinates: GraphCoordinate[]) {
  const byKey = new Map<string, GraphCoordinate>();
  for (const coordinate of coordinates) {
    byKey.set(coordinateKey(coordinate), coordinate);
  }
  return Array.from(byKey.values());
}

function clusterCapacity(cluster: Record<string, unknown>): ClusterCapacity {
  return {
    columns: axisValue(cluster.capacity, 0),
    rows: axisValue(cluster.capacity, 1),
    planes: axisValue(cluster.capacity, 2),
  };
}

function isInClusterBounds(
  coordinate: GraphCoordinate,
  capacity: ClusterCapacity,
) {
  const [x, y, z] = coordinate;
  return (
    x >= 1 &&
    x <= capacity.columns &&
    y >= 1 &&
    y <= capacity.rows &&
    z >= 1 &&
    z <= capacity.planes
  );
}

function graphChildIds(graph: InspectResponse) {
  const childrenById = new Map<string, string[]>();
  for (const edge of graph.edges) {
    const children = childrenById.get(edge.source) ?? [];
    children.push(edge.target);
    childrenById.set(edge.source, children);
  }
  return childrenById;
}

function descendantNodeIds(rootId: string, childrenById: Map<string, string[]>) {
  const descendantIds = new Set<string>();
  const queue = [...(childrenById.get(rootId) ?? [])];

  while (queue.length > 0) {
    const nodeId = queue.shift();
    if (!nodeId || descendantIds.has(nodeId)) {
      continue;
    }
    descendantIds.add(nodeId);
    queue.push(...(childrenById.get(nodeId) ?? []));
  }

  return descendantIds;
}

function terminalReachPriority(node: GraphNode) {
  if (node.typeName === "Neuron") {
    return 2;
  }
  if (node.typeName === "Terminal") {
    return 1;
  }
  return 0;
}

function clusterReachFromNode(
  node: GraphNode,
  activeCoordinateKeys: Set<string>,
  capacity: ClusterCapacity,
): ClusterDiagramReach | undefined {
  const reach = parseTerminalReachDetails(node.details);
  if (!reach || reach.connections.length === 0) {
    return undefined;
  }

  const positionKey = coordinateKey(reach.position);
  if (!activeCoordinateKeys.has(positionKey)) {
    return undefined;
  }

  const connections = uniqueCoordinates(reach.connections);
  const inBoundsConnections = connections.filter((coordinate) =>
    isInClusterBounds(coordinate, capacity),
  );
  const activeConnectionTotal = inBoundsConnections.filter((coordinate) => {
    const key = coordinateKey(coordinate);
    return key !== positionKey && activeCoordinateKeys.has(key);
  }).length;
  const emptyConnectionTotal = inBoundsConnections.filter(
    (coordinate) => !activeCoordinateKeys.has(coordinateKey(coordinate)),
  ).length;

  return {
    position: reach.position,
    connections,
    inBoundsConnections,
    activeConnectionTotal,
    emptyConnectionTotal,
    outOfBoundsTotal: connections.length - inBoundsConnections.length,
  };
}

function clusterReachByPosition(
  graph: InspectResponse,
  clusterNode: GraphNode,
  activeCoordinateKeys: Set<string>,
  capacity: ClusterCapacity,
  childrenById: Map<string, string[]>,
) {
  const descendantIds = descendantNodeIds(clusterNode.id, childrenById);
  const reachByPosition = new Map<string, ClusterDiagramReach>();
  const priorityByPosition = new Map<string, number>();

  for (const node of graph.nodes) {
    if (!descendantIds.has(node.id)) {
      continue;
    }

    const priority = terminalReachPriority(node);
    if (priority === 0) {
      continue;
    }

    const reach = clusterReachFromNode(node, activeCoordinateKeys, capacity);
    if (!reach) {
      continue;
    }

    const positionKey = coordinateKey(reach.position);
    const existingPriority = priorityByPosition.get(positionKey) ?? 0;
    if (existingPriority >= priority) {
      continue;
    }

    reachByPosition.set(positionKey, reach);
    priorityByPosition.set(positionKey, priority);
  }

  return reachByPosition;
}

function clusterDiagramMetadataFromDetail(
  cluster: Record<string, unknown>,
  filled = coordinateKeySet(cluster.coordinates),
): ClusterDiagramMetadata {
  const capacity = clusterCapacity(cluster);
  return {
    columns: capacity.columns,
    rows: capacity.rows,
    planeCount: capacity.planes,
    instantiated:
      typeof cluster.instantiated === "number" ? cluster.instantiated : filled.size,
    capacityTotal: capacity.columns * capacity.rows * capacity.planes,
    maxSteps: typeof cluster.maxSteps === "number" ? cluster.maxSteps : null,
    growthThreshold:
      typeof cluster.growthThreshold === "number" ? cluster.growthThreshold : null,
    hasColumnOverflow: false,
    hasRowOverflow: false,
    hasPlaneOverflow: false,
  };
}

export function buildClusterDiagramMetadata(node: GraphNode) {
  const cluster = clusterDetail(node);
  return cluster ? clusterDiagramMetadataFromDetail(cluster) : undefined;
}

function clusterDiagramFromDetail(
  cluster: Record<string, unknown>,
  reachByPosition: Map<string, ClusterDiagramReach>,
): ClusterDiagram {
  const filled = coordinateKeySet(cluster.coordinates);
  const metadata = clusterDiagramMetadataFromDetail(cluster, filled);
  const { columns, rows, planeCount, ...diagramMetadata } = metadata;

  const planes: ClusterDiagramPlane[] = [];
  for (let z = 1; z <= planeCount; z += 1) {
    const cells: ClusterDiagramCell[] = [];
    for (let y = 1; y <= rows; y += 1) {
      for (let x = 1; x <= columns; x += 1) {
        const isFilled = filled.has(`${x},${y},${z}`);
        const cell: ClusterDiagramCell = {
          x,
          y,
          filled: isFilled,
          title: `Neuron (${x}, ${y}, ${z}) — ${isFilled ? "active" : "empty"}`,
        };
        const reach = isFilled ? reachByPosition.get(`${x},${y},${z}`) : undefined;
        if (reach) {
          cell.reach = reach;
        }
        cells.push(cell);
      }
    }
    planes.push({ z, cells });
  }

  return {
    columns,
    rows,
    planes,
    ...diagramMetadata,
  };
}

export function buildClusterDiagrams(
  graph: InspectResponse | undefined,
) {
  const diagramsById = new Map<string, ClusterDiagram>();
  if (!graph) {
    return diagramsById;
  }
  const childrenById = graphChildIds(graph);

  for (const node of graph.nodes) {
    if (node.typeName !== "NeuronCluster") {
      continue;
    }
    const cluster = clusterDetail(node);
    if (!cluster) {
      continue;
    }
    const capacity = clusterCapacity(cluster);
    const activeCoordinateKeys = coordinateKeySet(cluster.coordinates);
    const reachByPosition = clusterReachByPosition(
      graph,
      node,
      activeCoordinateKeys,
      capacity,
      childrenById,
    );
    diagramsById.set(node.id, clusterDiagramFromDetail(cluster, reachByPosition));
  }

  return diagramsById;
}
