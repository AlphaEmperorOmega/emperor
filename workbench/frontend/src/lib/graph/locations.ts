import { type GraphNode, type InspectResponse } from "@/lib/api";
import { buildClusterDiagramMetadata } from "@/lib/graph/cluster-diagrams";
import {
  buildTerminalReachGrid,
  parseTerminalReachDetails,
} from "@/lib/graph/terminal-reach";
import {
  graphCoordinates,
  isRecord,
} from "@/lib/graph/helpers";
import {
  type ClusterLocationSummary,
  type GraphLocationSummary,
} from "@/lib/graph/types";

function clusterDetail(node: GraphNode): Record<string, unknown> | undefined {
  const cluster = node.details.cluster;
  return isRecord(cluster) ? cluster : undefined;
}

function clusterLocationSummary(
  node: GraphNode,
): ClusterLocationSummary | undefined {
  if (node.typeName !== "NeuronCluster") {
    return undefined;
  }
  const cluster = clusterDetail(node);
  const metadata = buildClusterDiagramMetadata(node);
  const coordinates = cluster ? graphCoordinates(cluster.coordinates) : [];
  if (!metadata || coordinates.length === 0) {
    return undefined;
  }
  return {
    kind: "cluster",
    nodeId: node.id,
    nodePath: node.path,
    nodeLabel: node.label,
    nodeType: node.typeName,
    coordinates,
    instantiated: metadata.instantiated,
    capacityTotal: metadata.capacityTotal,
    hasOverflow:
      metadata.hasColumnOverflow ||
      metadata.hasRowOverflow ||
      metadata.hasPlaneOverflow,
  };
}

export function buildClusterLocationSummary(
  graph: InspectResponse | undefined,
  nodeId: string,
) {
  const node = graph?.nodes.find((candidate) => candidate.id === nodeId);
  return node ? clusterLocationSummary(node) : undefined;
}

export function buildGraphLocationSummaries(
  graph: InspectResponse | undefined,
): GraphLocationSummary[] {
  if (!graph) {
    return [];
  }

  const summaries: GraphLocationSummary[] = [];

  for (const node of graph.nodes) {
    const clusterSummary = clusterLocationSummary(node);
    if (clusterSummary) {
      summaries.push(clusterSummary);
    }

    const reach = parseTerminalReachDetails(node.details);
    if (!reach) {
      continue;
    }

    const reachGrid = buildTerminalReachGrid(node.details);

    summaries.push({
      kind: "terminalReach",
      nodeId: node.id,
      nodePath: node.path,
      nodeLabel: node.label,
      nodeType: node.typeName,
      position: reach.position,
      connections: reach.connections,
      total: reach.total,
      hasOverflow:
        Boolean(reachGrid?.hasOverflow) || reach.total > reach.connections.length,
    });
  }

  return summaries;
}
