import type { GraphCoordinate } from "@/lib/graph/types";

export const CLUSTER_CELL_SPACING = 1.12;

export function clusterCoordinatePosition(
  coordinate: GraphCoordinate,
  capacity: GraphCoordinate,
  spacing = CLUSTER_CELL_SPACING,
) {
  const [x, y, z] = coordinate;
  return [
    (x - (capacity[0] + 1) / 2) * spacing,
    (z - (capacity[2] + 1) / 2) * spacing,
    (y - (capacity[1] + 1) / 2) * spacing,
  ] as const;
}
