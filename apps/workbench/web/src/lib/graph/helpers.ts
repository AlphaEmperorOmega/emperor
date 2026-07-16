import type { GraphNode } from "@/lib/api/inspection";
import { type GraphCoordinate, type GraphNavigation } from "@/lib/graph/types";

export function lastPathSegment(path: string) {
  const pathParts = path.split(".");
  return pathParts[pathParts.length - 1] ?? path;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function asGraphCoordinate(value: unknown): GraphCoordinate | null {
  if (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every((item) => typeof item === "number" && Number.isFinite(item))
  ) {
    return [value[0], value[1], value[2]];
  }
  return null;
}

export function graphCoordinates(value: unknown): GraphCoordinate[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map(asGraphCoordinate)
    .filter((coordinate): coordinate is GraphCoordinate => coordinate !== null);
}

export function directChildNodes(
  nodeId: string,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
) {
  return (navigation.childrenById.get(nodeId) ?? [])
    .map((childId) => nodesById.get(childId))
    .filter((node): node is GraphNode => Boolean(node));
}

export function numericLastPathSegment(path: string) {
  const segment = lastPathSegment(path);
  if (!/^\d+$/.test(segment)) {
    return undefined;
  }
  return Number(segment);
}

export function directNumericChildNodes(
  nodeId: string,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
) {
  return directNumericChildEntries(nodeId, navigation, nodesById).map((entry) => entry.node);
}

export function directNumericChildEntries(
  nodeId: string,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
) {
  return directChildNodes(nodeId, navigation, nodesById)
    .map((node) => ({ node, index: numericLastPathSegment(node.path) }))
    .filter((entry): entry is { node: GraphNode; index: number } => entry.index !== undefined)
    .sort((left, right) => left.index - right.index);
}

export function detailCount(details: GraphNode["details"], key: "numExperts") {
  const value = details[key];
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return undefined;
  }
  return Math.trunc(value);
}
