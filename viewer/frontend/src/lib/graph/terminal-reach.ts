import { type GraphNode } from "@/lib/api";
import {
  CLUSTER_DIAGRAM_MAX_DIM,
  CLUSTER_DIAGRAM_MAX_PLANES,
} from "@/lib/graph/constants";
import { asGraphCoordinate, graphCoordinates, isRecord } from "@/lib/graph/helpers";
import {
  type TerminalReachCell,
  type TerminalReachGrid,
  type TerminalReachPlane,
} from "@/lib/graph/types";

export function buildTerminalReachGrid(
  details: GraphNode["details"],
): TerminalReachGrid | undefined {
  const reach = details.terminalReach;
  if (!isRecord(reach)) {
    return undefined;
  }

  const position = asGraphCoordinate(reach.position);
  const connections = graphCoordinates(reach.connections);
  if (!position || connections.length === 0) {
    return undefined;
  }

  const xs = [position[0], ...connections.map((coordinate) => coordinate[0])];
  const ys = [position[1], ...connections.map((coordinate) => coordinate[1])];
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const fullColumns = Math.max(...xs) - minX + 1;
  const fullRows = Math.max(...ys) - minY + 1;
  const columns = Math.min(fullColumns, CLUSTER_DIAGRAM_MAX_DIM);
  const rows = Math.min(fullRows, CLUSTER_DIAGRAM_MAX_DIM);

  const reachKeys = new Set(connections.map((coordinate) => coordinate.join(",")));
  const planeZs = Array.from(
    new Set([position[2], ...connections.map((coordinate) => coordinate[2])]),
  ).sort((a, b) => a - b);
  const visiblePlaneZs = planeZs.slice(0, CLUSTER_DIAGRAM_MAX_PLANES);

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
    total: typeof reach.total === "number" ? reach.total : connections.length,
    hasOverflow:
      fullColumns > columns ||
      fullRows > rows ||
      planeZs.length > visiblePlaneZs.length,
  };
}
