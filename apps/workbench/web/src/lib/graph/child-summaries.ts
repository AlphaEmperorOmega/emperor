import type { GraphNode, InspectResponse } from "@/lib/api/inspection";
import {
  graphDiagramLimits,
  graphDisplayLabels,
} from "@/lib/graph/constants";
import { directChildNodes, isRecord, lastPathSegment } from "@/lib/graph/helpers";
import { nodeDimsText, nodeTitle } from "@/lib/graph/formatting";
import { type ChildSummary, type GraphNavigation } from "@/lib/graph/types";

function detailFlag(details: GraphNode["details"], key: "gate" | "halting") {
  if (details[key] === true) {
    return true;
  }
  const recurrent = details.recurrent;
  return isRecord(recurrent) && recurrent[key] === true;
}

function childSummaryLabel(node: GraphNode) {
  const pathSegment = lastPathSegment(node.path);
  if (pathSegment === "gate" || pathSegment === "gate_model") {
    return graphDisplayLabels.gateSummary;
  }
  if (pathSegment === "halting" || pathSegment === "halting_model") {
    return graphDisplayLabels.haltingSummary;
  }
  return nodeTitle(node);
}

function layerStackLabel(node: GraphNode) {
  const pathSegment = lastPathSegment(node.path);
  if (/^\d+$/.test(pathSegment)) {
    return `Layer ${pathSegment}`;
  }
  return childSummaryLabel(node);
}

function childSummary(
  parent: GraphNode,
  child: GraphNode,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
): ChildSummary {
  const parentLayerInnerModelDims =
    parent.typeName === "Layer" && lastPathSegment(child.path) === "model"
      ? nodeDimsText(parent.details, parent.config)
      : undefined;
  const childDims = nodeDimsText(child.details, child.config);

  if (child.typeName !== "Layer") {
    return {
      label: childSummaryLabel(child),
      ...(parentLayerInnerModelDims || childDims
        ? { dims: parentLayerInnerModelDims ?? childDims }
        : {}),
      kind: "child",
      sourceNodeId: child.id,
    };
  }

  const label = layerStackLabel(child);
  const nestedChild = directChildNodes(child.id, navigation, nodesById)
    .find((node): node is GraphNode => Boolean(node));

  if (!nestedChild) {
    return {
      label,
      ...(childDims ? { dims: childDims } : {}),
      kind: "child",
      stackKind: "layer",
      sourceNodeId: child.id,
    };
  }

  return {
    label,
    nestedLabel: childSummaryLabel(nestedChild),
    ...(childDims ? { dims: childDims } : {}),
    kind: "child",
    stackKind: "layer",
    sourceNodeId: child.id,
  };
}

function collapseLayerStackSummaries(childSummaries: ChildSummary[]) {
  const layerSummaries = childSummaries.filter((summary) => summary.stackKind === "layer");
  if (layerSummaries.length <= graphDiagramLimits.layerSummary.total) {
    return childSummaries;
  }

  let emittedLayerCount = 0;
  let insertedOverflow = false;
  const hiddenLayerCount =
    layerSummaries.length -
    graphDiagramLimits.layerSummary.visibleBeforeOverflow -
    1;

  return childSummaries.flatMap((summary) => {
    if (summary.stackKind !== "layer") {
      return [summary];
    }

    emittedLayerCount += 1;
    if (
      emittedLayerCount <=
      graphDiagramLimits.layerSummary.visibleBeforeOverflow
    ) {
      return [summary];
    }

    if (emittedLayerCount === layerSummaries.length) {
      return [summary];
    }

    if (insertedOverflow) {
      return [];
    }

    insertedOverflow = true;
    return [
      {
        label: "…",
        kind: "overflow" as const,
        title: `${hiddenLayerCount} more layer${hiddenLayerCount === 1 ? "" : "s"}`,
      },
    ];
  });
}

export function buildChildSummaries(
  graph: InspectResponse | undefined,
  navigation: GraphNavigation,
) {
  const summariesById = new Map<string, ChildSummary[]>();

  if (!graph) {
    return summariesById;
  }

  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));

  for (const node of graph.nodes) {
    const childSummaries: ChildSummary[] = [];
    for (const childId of navigation.childrenById.get(node.id) ?? []) {
      const child = nodesById.get(childId);
      if (!child) {
        continue;
      }
      childSummaries.push(childSummary(node, child, navigation, nodesById));
    }

    const labels = new Set(childSummaries.map((summary) => summary.label));

    if (
      detailFlag(node.details, "gate") &&
      !labels.has(graphDisplayLabels.gateSummary)
    ) {
      childSummaries.push({
        label: graphDisplayLabels.gateSummary,
        kind: "mechanism",
      });
      labels.add(graphDisplayLabels.gateSummary);
    }
    if (
      detailFlag(node.details, "halting") &&
      !labels.has(graphDisplayLabels.haltingSummary)
    ) {
      childSummaries.push({
        label: graphDisplayLabels.haltingSummary,
        kind: "mechanism",
      });
    }

    summariesById.set(node.id, collapseLayerStackSummaries(childSummaries));
  }

  return summariesById;
}
