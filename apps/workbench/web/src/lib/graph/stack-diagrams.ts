import type { GraphNode, InspectResponse } from "@/lib/api/inspection";
import {
  graphDiagramLimits,
  graphDisplayTypeNames,
} from "@/lib/graph/constants";
import {
  directChildNodes,
  directNumericChildEntries,
  lastPathSegment,
} from "@/lib/graph/helpers";
import { nodeDimRange, nodeDimsText, nodeTitle } from "@/lib/graph/formatting";
import { type GraphNavigation, type StackDiagram, type StackDiagramCell } from "@/lib/graph/types";

type StackLayerEntry = { node: GraphNode; index: number };

function isLayerLikeStackNode(node: GraphNode) {
  return node.typeName === "Layer" || node.typeName.endsWith("Layer");
}

function primaryLayerContentNode(
  layerNode: GraphNode,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
) {
  const children = directChildNodes(layerNode.id, navigation, nodesById);
  return (
    children.find((child) => lastPathSegment(child.path) === "model") ??
    children.find((child) => child.graphRole === "architecture") ??
    children[0]
  );
}

function stackLayerDimRange(
  entry: StackLayerEntry,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
  contentNode = primaryLayerContentNode(entry.node, navigation, nodesById),
) {
  return (
    nodeDimRange(entry.node.details, entry.node.config) ??
    (contentNode ? nodeDimRange(contentNode.details, contentNode.config) : undefined)
  );
}

function stackLayerDims(
  entry: StackLayerEntry,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
  contentNode = primaryLayerContentNode(entry.node, navigation, nodesById),
) {
  return stackLayerDimRange(entry, navigation, nodesById, contentNode)?.text;
}

function stackLayerCell(
  entry: StackLayerEntry,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
): StackDiagramCell {
  const contentNode = primaryLayerContentNode(entry.node, navigation, nodesById);
  const layerType = contentNode ? nodeTitle(contentNode) : nodeTitle(entry.node);
  const label = `Layer ${entry.index} · ${layerType}`;
  const dims = stackLayerDims(entry, navigation, nodesById, contentNode);

  return {
    label,
    title: [label, dims].filter(Boolean).join(" · "),
    ...(dims ? { dims } : {}),
    kind: "layer",
    layerIndex: entry.index,
  };
}

function stackRangeFromLayers(
  entries: StackLayerEntry[],
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
) {
  const firstRange = entries[0]
    ? stackLayerDimRange(entries[0], navigation, nodesById)
    : undefined;
  const lastEntry = entries[entries.length - 1];
  const lastRange = lastEntry
    ? stackLayerDimRange(lastEntry, navigation, nodesById)
    : undefined;

  if (!firstRange || !lastRange) {
    return undefined;
  }

  return `${firstRange.inputDim} -> ${lastRange.outputDim}`;
}

function createStackDiagramCells(
  entries: StackLayerEntry[],
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
): StackDiagramCell[] {
  const visibleEntries =
    entries.length > graphDiagramLimits.stack.total
      ? entries.slice(0, graphDiagramLimits.stack.visibleBeforeOverflow)
      : entries;
  const layerCells = visibleEntries.map((entry) => stackLayerCell(entry, navigation, nodesById));

  if (entries.length <= graphDiagramLimits.stack.total) {
    return layerCells;
  }

  const lastEntry = entries[entries.length - 1];
  const hiddenLayerCount =
    entries.length - graphDiagramLimits.stack.visibleBeforeOverflow - 1;

  return [
    ...layerCells,
    {
      label: "…",
      title: `${hiddenLayerCount} more layer${hiddenLayerCount === 1 ? "" : "s"}`,
      kind: "overflow" as const,
    },
    stackLayerCell(lastEntry, navigation, nodesById),
  ];
}

function stackDiagramFromContainer(
  containerNode: GraphNode,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
): StackDiagram | undefined {
  const layerEntries = directNumericChildEntries(containerNode.id, navigation, nodesById);
  if (layerEntries.length === 0) {
    return undefined;
  }
  if (!layerEntries.every((entry) => isLayerLikeStackNode(entry.node))) {
    return undefined;
  }

  const dims =
    nodeDimsText(containerNode.details, containerNode.config) ??
    stackRangeFromLayers(layerEntries, navigation, nodesById);

  return {
    cells: createStackDiagramCells(layerEntries, navigation, nodesById),
    ...(dims ? { dims } : {}),
    totalLayers: layerEntries.length,
    hasOverflow: layerEntries.length > graphDiagramLimits.stack.total,
  };
}

export function buildStackDiagrams(
  graph: InspectResponse | undefined,
  navigation: GraphNavigation,
) {
  const diagramsById = new Map<string, StackDiagram>();

  if (!graph) {
    return diagramsById;
  }

  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));

  for (const node of graph.nodes) {
    const pathSegment = lastPathSegment(node.path);
    if (node.typeName === "MixtureOfExpertsModel") {
      const expertStackNode = directChildNodes(node.id, navigation, nodesById).find(
        (child) => lastPathSegment(child.path) === "expert_stack",
      );
      if (!expertStackNode) {
        continue;
      }
      const diagram = stackDiagramFromContainer(expertStackNode, navigation, nodesById);
      if (diagram) {
        diagramsById.set(node.id, diagram);
      }
      continue;
    }

    if (
      !graphDisplayTypeNames.stackContainers.has(node.typeName) ||
      pathSegment === "expert_modules"
    ) {
      continue;
    }

    const diagram = stackDiagramFromContainer(node, navigation, nodesById);
    if (diagram) {
      diagramsById.set(node.id, diagram);
    }
  }

  return diagramsById;
}
