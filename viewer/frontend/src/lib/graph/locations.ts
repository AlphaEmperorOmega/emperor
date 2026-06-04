import { type GraphNode, type InspectResponse } from "@/lib/api";
import { buildClusterDiagrams } from "@/lib/graph/cluster-diagrams";
import { buildTerminalReachGrid } from "@/lib/graph/terminal-reach";
import {
  asGraphCoordinate,
  graphCoordinates,
  isRecord,
} from "@/lib/graph/helpers";
import { type GraphLocationSummary } from "@/lib/graph/types";

function clusterDetail(node: GraphNode): Record<string, unknown> | undefined {
  const cluster = node.details.cluster;
  return isRecord(cluster) ? cluster : undefined;
}

function terminalReachDetail(node: GraphNode): Record<string, unknown> | undefined {
  const reach = node.details.terminalReach;
  return isRecord(reach) ? reach : undefined;
}

function connectionTotal(reach: Record<string, unknown>, connectionCount: number) {
  const total = reach.total;
  if (typeof total === "number" && Number.isFinite(total) && total >= 0) {
    return Math.trunc(total);
  }
  return connectionCount;
}

export function buildGraphLocationSummaries(
  graph: InspectResponse | undefined,
): GraphLocationSummary[] {
  if (!graph) {
    return [];
  }

  const clusterDiagramsById = buildClusterDiagrams(graph);
  const summaries: GraphLocationSummary[] = [];

  for (const node of graph.nodes) {
    if (node.typeName === "NeuronCluster") {
      const cluster = clusterDetail(node);
      const diagram = clusterDiagramsById.get(node.id);
      const coordinates = cluster ? graphCoordinates(cluster.coordinates) : [];

      if (diagram && coordinates.length > 0) {
        summaries.push({
          kind: "cluster",
          nodeId: node.id,
          nodePath: node.path,
          nodeLabel: node.label,
          nodeType: node.typeName,
          coordinates,
          instantiated: diagram.instantiated,
          capacityTotal: diagram.capacityTotal,
          hasOverflow:
            diagram.hasColumnOverflow ||
            diagram.hasRowOverflow ||
            diagram.hasPlaneOverflow,
        });
      }
    }

    const reach = terminalReachDetail(node);
    if (!reach) {
      continue;
    }

    const position = asGraphCoordinate(reach.position);
    if (!position) {
      continue;
    }

    const connections = graphCoordinates(reach.connections);
    const total = connectionTotal(reach, connections.length);
    const reachGrid = buildTerminalReachGrid(node.details);

    summaries.push({
      kind: "terminalReach",
      nodeId: node.id,
      nodePath: node.path,
      nodeLabel: node.label,
      nodeType: node.typeName,
      position,
      connections,
      total,
      hasOverflow: Boolean(reachGrid?.hasOverflow) || total > connections.length,
    });
  }

  return summaries;
}
