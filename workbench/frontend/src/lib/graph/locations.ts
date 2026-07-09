import { type GraphNode, type InspectResponse } from "@/lib/api";
import { buildClusterDiagrams } from "@/lib/graph/cluster-diagrams";
import {
  buildTerminalReachGrid,
  parseTerminalReachDetails,
} from "@/lib/graph/terminal-reach";
import {
  graphCoordinates,
  isRecord,
} from "@/lib/graph/helpers";
import { type GraphLocationSummary } from "@/lib/graph/types";

function clusterDetail(node: GraphNode): Record<string, unknown> | undefined {
  const cluster = node.details.cluster;
  return isRecord(cluster) ? cluster : undefined;
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
