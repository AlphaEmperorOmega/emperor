import type { GraphNode } from "@/lib/api/inspection";
import { graphDiagramLimits } from "@/lib/graph/constants";
import { asGraphCoordinate, graphCoordinates, isRecord } from "@/lib/graph/helpers";
import {
  type GraphCoordinate,
  type TerminalReachCell,
  type TerminalReachGrid,
  type TerminalReachPlane,
} from "@/lib/graph/types";

export type ParsedTerminalReach = {
  position: GraphCoordinate;
  connections: GraphCoordinate[];
  total: number;
};

function connectionTotal(value: unknown, connectionCount: number) {
  if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
    return Math.trunc(value);
  }
  return connectionCount;
}

export function parseTerminalReachDetails(
  details: GraphNode["details"],
): ParsedTerminalReach | undefined {
  const reach = details.terminalReach;
  if (!isRecord(reach)) {
    return undefined;
  }

  const position = asGraphCoordinate(reach.position);
  const connections = graphCoordinates(reach.connections);
  if (!position) {
    return undefined;
  }

  return {
    position,
    connections,
    total: connectionTotal(reach.total, connections.length),
  };
}

export function buildTerminalReachGrid(
  details: GraphNode["details"],
): TerminalReachGrid | undefined {
  const reach = parseTerminalReachDetails(details);
  if (!reach || reach.connections.length === 0) {
    return undefined;
  }

  const { connections, position } = reach;
  const xs = [position[0], ...connections.map((coordinate) => coordinate[0])];
  const ys = [position[1], ...connections.map((coordinate) => coordinate[1])];
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const fullColumns = Math.max(...xs) - minX + 1;
  const fullRows = Math.max(...ys) - minY + 1;
  const columns = Math.min(
    fullColumns,
    graphDiagramLimits.cluster.maxDimension,
  );
  const rows = Math.min(fullRows, graphDiagramLimits.cluster.maxDimension);

  const reachKeys = new Set(connections.map((coordinate) => coordinate.join(",")));
  const planeZs = Array.from(
    new Set([position[2], ...connections.map((coordinate) => coordinate[2])]),
  ).sort((a, b) => a - b);
  const visiblePlaneZs = planeZs.slice(
    0,
    graphDiagramLimits.cluster.maxPlanes,
  );

  const planes: TerminalReachPlane[] = visiblePlaneZs.map((z) => {
    const cells: TerminalReachCell[] = [];
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const x = minX + column;
        const y = minY + row;
        const isSelf = x === position[0] && y === position[1] && z === position[2];
        const isReach = reachKeys.has(`${x},${y},${z}`);
        const kind: TerminalReachCell["kind"] = isSelf
          ? "self"
          : isReach
            ? "reach"
            : "empty";
        cells.push({
          x,
          y,
          kind,
          title: `(${x}, ${y}, ${z}) — ${
            kind === "self" ? "this neuron" : kind === "reach" ? "reachable" : "out of reach"
          }`,
        });
      }
    }
    return { z, cells };
  });

  return {
    columns,
    rows,
    minX,
    minY,
    position,
    planes,
    total: reach.total,
    hasOverflow:
      fullColumns > columns ||
      fullRows > rows ||
      planeZs.length > visiblePlaneZs.length,
  };
}
