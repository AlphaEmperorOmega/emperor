import { type GraphNode, type InspectResponse, type TrainingJob } from "@/lib/api";
import { buildClusterGrowth } from "@/lib/cluster-growth";
import { asGraphCoordinate, graphCoordinates, isRecord } from "@/lib/graph/helpers";
import { buildGraphNavigation } from "@/lib/graph/navigation";
import { parseTerminalReachDetails } from "@/lib/graph/terminal-reach";
import { type GraphCoordinate } from "@/lib/graph/types";

const GHOST_CELL_CAPACITY_LIMIT = 512;

export type Cluster3DCellCategory = "initial" | "grown" | "recentAdded";
export type Cluster3DCellSource = "inspection" | "growth-overlay";

export type Cluster3DNodeMatch = {
  nodeId: string;
  nodePath: string;
  nodeType: string;
};

export type Cluster3DReach = {
  position: GraphCoordinate;
  connections: GraphCoordinate[];
  inBoundsConnections: GraphCoordinate[];
  activeConnectionTotal: number;
  emptyConnectionTotal: number;
  outOfBoundsTotal: number;
};

export type Cluster3DCell = {
  key: string;
  coordinate: GraphCoordinate;
  category: Cluster3DCellCategory;
  source: Cluster3DCellSource;
  isOverlayOnly: boolean;
  nodeMatch: Cluster3DNodeMatch | null;
  reach: Cluster3DReach | null;
};

export type Cluster3DSceneModel = {
  clusterNodeId: string;
  clusterNodePath: string;
  clusterNodeLabel: string;
  capacity: GraphCoordinate;
  initial: GraphCoordinate;
  initialStart: GraphCoordinate;
  capacityTotal: number;
  instantiated: number;
  activeCells: Cluster3DCell[];
  initialCount: number;
  grownCount: number;
  recentAddedCount: number;
  overlayOnlyCount: number;
  maxSteps: number | null;
  growthThreshold: number | null;
  renderGhostCells: boolean;
};

type ClusterCapacity = {
  x: number;
  y: number;
  z: number;
};

type ParsedTerminalReach = NonNullable<
  ReturnType<typeof parseTerminalReachDetails>
>;

type CoordinateNodeCandidate = {
  node: GraphNode;
  priority: number;
};

function clusterDetail(node: GraphNode): Record<string, unknown> | undefined {
  const cluster = node.details.cluster;
  return isRecord(cluster) ? cluster : undefined;
}

function axisCoordinate(value: unknown): GraphCoordinate | null {
  const coordinate = asGraphCoordinate(value);
  if (!coordinate) {
    return null;
  }
  if (coordinate.every((axis) => Number.isInteger(axis) && axis > 0)) {
    return coordinate;
  }
  return null;
}

function optionalAxisCoordinate(
  value: unknown,
  fallback: GraphCoordinate,
): GraphCoordinate {
  return axisCoordinate(value) ?? fallback;
}

function coordinateKey(coordinate: GraphCoordinate) {
  return coordinate.join(",");
}

function isInBounds(coordinate: GraphCoordinate, capacity: ClusterCapacity) {
  const [x, y, z] = coordinate;
  return (
    x >= 1 &&
    x <= capacity.x &&
    y >= 1 &&
    y <= capacity.y &&
    z >= 1 &&
    z <= capacity.z
  );
}

function isInsideInitialArea(
  coordinate: GraphCoordinate,
  initial: GraphCoordinate,
  initialStart: GraphCoordinate,
) {
  return coordinate.every((value, index) => {
    const start = initialStart[index] ?? 1;
    const length = initial[index] ?? 1;
    return value >= start && value < start + length;
  });
}

function uniqueCoordinateMap(coordinates: GraphCoordinate[]) {
  const byKey = new Map<string, GraphCoordinate>();
  for (const coordinate of coordinates) {
    byKey.set(coordinateKey(coordinate), coordinate);
  }
  return byKey;
}

function descendantNodeIds(graph: InspectResponse, clusterNodeId: string) {
  const navigation = buildGraphNavigation(graph);
  const descendants = new Set<string>();
  const queue = [...(navigation.childrenById.get(clusterNodeId) ?? [])];

  while (queue.length > 0) {
    const nodeId = queue.shift();
    if (!nodeId || descendants.has(nodeId)) {
      continue;
    }
    descendants.add(nodeId);
    queue.push(...(navigation.childrenById.get(nodeId) ?? []));
  }

  return descendants;
}

function terminalReachPriority(node: GraphNode) {
  if (node.typeName === "Neuron") {
    return 3;
  }
  if (node.typeName === "Terminal") {
    return 2;
  }
  return 1;
}

function coordinateSuffixes(coordinate: GraphCoordinate) {
  const suffix = coordinate.join("_");
  return [suffix, `neuron_${suffix}`];
}

function textMatchesCoordinateSuffix(text: string, coordinate: GraphCoordinate) {
  const normalized = text.toLowerCase();
  return coordinateSuffixes(coordinate).some((suffix) => {
    const lowerSuffix = suffix.toLowerCase();
    return (
      normalized === lowerSuffix ||
      normalized.endsWith(`.${lowerSuffix}`) ||
      normalized.endsWith(`_${lowerSuffix}`) ||
      normalized.endsWith(`/${lowerSuffix}`)
    );
  });
}

function nodeMatch(node: GraphNode): Cluster3DNodeMatch {
  return {
    nodeId: node.id,
    nodePath: node.path,
    nodeType: node.typeName,
  };
}

function shouldReplaceCandidate(
  current: CoordinateNodeCandidate | undefined,
  candidate: CoordinateNodeCandidate,
) {
  return !current || candidate.priority > current.priority;
}

function buildCoordinateNodeIndex(
  graph: InspectResponse,
  descendantIds: Set<string>,
  activeByKey: Map<string, GraphCoordinate>,
) {
  const descendants = graph.nodes
    .filter((node) => descendantIds.has(node.id))
    .map((node): CoordinateNodeCandidate => ({
      node,
      priority: terminalReachPriority(node),
    }));
  const exactCandidateByKey = new Map<string, CoordinateNodeCandidate>();
  const reachByNodeId = new Map<string, ParsedTerminalReach>();

  for (const candidate of descendants) {
    const reach = parseTerminalReachDetails(candidate.node.details);
    if (!reach) {
      continue;
    }
    reachByNodeId.set(candidate.node.id, reach);
    const key = coordinateKey(reach.position);
    if (
      activeByKey.has(key) &&
      shouldReplaceCandidate(exactCandidateByKey.get(key), candidate)
    ) {
      exactCandidateByKey.set(key, candidate);
    }
  }

  const unmatchedCoordinates = [...activeByKey.entries()].filter(
    ([key]) => !exactCandidateByKey.has(key),
  );
  const suffixCandidateByKey = new Map<string, CoordinateNodeCandidate>();
  for (const candidate of descendants) {
    const matchValues = [candidate.node.id, candidate.node.path, candidate.node.label];
    for (const [key, coordinate] of unmatchedCoordinates) {
      if (
        matchValues.some((value) =>
          textMatchesCoordinateSuffix(value, coordinate),
        ) &&
        shouldReplaceCandidate(suffixCandidateByKey.get(key), candidate)
      ) {
        suffixCandidateByKey.set(key, candidate);
      }
    }
  }

  const nodeByCoordinateKey = new Map<string, GraphNode>();
  for (const key of activeByKey.keys()) {
    const candidate = exactCandidateByKey.get(key) ?? suffixCandidateByKey.get(key);
    if (candidate) {
      nodeByCoordinateKey.set(key, candidate.node);
    }
  }
  return { nodeByCoordinateKey, reachByNodeId };
}

function reachFromDetails(
  reach: ParsedTerminalReach | undefined,
  activeCoordinateKeys: Set<string>,
  capacity: ClusterCapacity,
): Cluster3DReach | null {
  if (!reach || reach.connections.length === 0) {
    return null;
  }

  const positionKey = coordinateKey(reach.position);
  const connections = [...uniqueCoordinateMap(reach.connections).values()];
  const inBoundsConnections = connections.filter((coordinate) =>
    isInBounds(coordinate, capacity),
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

function selectedClusterGrowthCoordinates(
  activeTrainingJob: TrainingJob | undefined,
  selectedNodeId: string,
) {
  return (
    buildClusterGrowth(activeTrainingJob)
      .find((summary) => summary.node === selectedNodeId)
      ?.additions.map((addition) => addition.coord) ?? []
  );
}

function numberOrNull(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function buildCluster3DSceneModel({
  graph,
  selectedNodeId,
  activeTrainingJob,
}: {
  graph: InspectResponse | undefined;
  selectedNodeId: string | null | undefined;
  activeTrainingJob?: TrainingJob | undefined;
}): Cluster3DSceneModel | null {
  if (!graph || !selectedNodeId) {
    return null;
  }

  const clusterNode = graph.nodes.find((node) => node.id === selectedNodeId);
  if (!clusterNode || clusterNode.typeName !== "NeuronCluster") {
    return null;
  }

  const cluster = clusterDetail(clusterNode);
  if (!cluster) {
    return null;
  }

  const capacity = axisCoordinate(cluster.capacity);
  if (!capacity) {
    return null;
  }
  const initial = optionalAxisCoordinate(cluster.initial, capacity);
  const initialStart = optionalAxisCoordinate(cluster.initialStart, [1, 1, 1]);
  const capacityBounds = { x: capacity[0], y: capacity[1], z: capacity[2] };
  const capacityTotal = capacity[0] * capacity[1] * capacity[2];

  const inspectionCoordinates = graphCoordinates(cluster.coordinates).filter((coordinate) =>
    isInBounds(coordinate, capacityBounds),
  );
  const inspectionByKey = uniqueCoordinateMap(inspectionCoordinates);
  const recentAddedByKey = uniqueCoordinateMap(
    selectedClusterGrowthCoordinates(activeTrainingJob, selectedNodeId).filter(
      (coordinate) => isInBounds(coordinate, capacityBounds),
    ),
  );
  const activeByKey = new Map([...inspectionByKey, ...recentAddedByKey]);
  const activeCoordinateKeys = new Set(activeByKey.keys());
  const descendantIds = descendantNodeIds(graph, clusterNode.id);
  const { nodeByCoordinateKey, reachByNodeId } = buildCoordinateNodeIndex(
    graph,
    descendantIds,
    activeByKey,
  );

  const activeCells = [...activeByKey.entries()]
    .map(([key, coordinate]): Cluster3DCell => {
      const isRecentAdded = recentAddedByKey.has(key);
      const category: Cluster3DCellCategory = isRecentAdded
        ? "recentAdded"
        : isInsideInitialArea(coordinate, initial, initialStart)
          ? "initial"
          : "grown";
      const matchedNode = nodeByCoordinateKey.get(key);
      const match = matchedNode ? nodeMatch(matchedNode) : null;
      const reach = matchedNode
        ? reachFromDetails(
            reachByNodeId.get(matchedNode.id),
            activeCoordinateKeys,
            capacityBounds,
          )
        : null;

      return {
        key,
        coordinate,
        category,
        source: inspectionByKey.has(key) ? "inspection" : "growth-overlay",
        isOverlayOnly: !inspectionByKey.has(key),
        nodeMatch: match,
        reach,
      };
    })
    .sort((left, right) => {
      const [leftX, leftY, leftZ] = left.coordinate;
      const [rightX, rightY, rightZ] = right.coordinate;
      return leftZ - rightZ || leftY - rightY || leftX - rightX;
    });

  return {
    clusterNodeId: clusterNode.id,
    clusterNodePath: clusterNode.path,
    clusterNodeLabel: clusterNode.label,
    capacity,
    initial,
    initialStart,
    capacityTotal,
    instantiated:
      typeof cluster.instantiated === "number"
        ? cluster.instantiated
        : inspectionByKey.size,
    activeCells,
    initialCount: activeCells.filter((cell) => cell.category === "initial").length,
    grownCount: activeCells.filter((cell) => cell.category === "grown").length,
    recentAddedCount: activeCells.filter((cell) => cell.category === "recentAdded")
      .length,
    overlayOnlyCount: activeCells.filter((cell) => cell.isOverlayOnly).length,
    maxSteps: numberOrNull(cluster.maxSteps),
    growthThreshold: numberOrNull(cluster.growthThreshold),
    renderGhostCells: capacityTotal <= GHOST_CELL_CAPACITY_LIMIT,
  };
}
