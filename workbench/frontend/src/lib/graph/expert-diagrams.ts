import { type GraphNode, type InspectResponse } from "@/lib/api";
import { graphDiagramLimits } from "@/lib/graph/constants";
import {
  detailCount,
  directChildNodes,
  directNumericChildNodes,
  lastPathSegment,
} from "@/lib/graph/helpers";
import { type ExpertDiagram, type ExpertDiagramCell, type GraphNavigation } from "@/lib/graph/types";

function expertCountFromExpertModules(
  expertModulesNode: GraphNode | undefined,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
) {
  if (!expertModulesNode) {
    return undefined;
  }
  const expertNodes = directNumericChildNodes(expertModulesNode.id, navigation, nodesById);
  return expertNodes.length > 0 ? expertNodes.length : undefined;
}

function descendantExpertCount(
  nodeId: string,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
) {
  const queue = [...(navigation.childrenById.get(nodeId) ?? [])];
  while (queue.length > 0) {
    const childId = queue.shift();
    if (!childId) {
      continue;
    }
    const child = nodesById.get(childId);
    if (!child) {
      continue;
    }

    if (child.typeName === "MixtureOfExperts") {
      const expertModulesNode = directChildNodes(child.id, navigation, nodesById).find(
        (candidate) => lastPathSegment(candidate.path) === "expert_modules",
      );
      return (
        detailCount(child.details, "numExperts") ??
        expertCountFromExpertModules(expertModulesNode, navigation, nodesById)
      );
    }

    queue.push(...(navigation.childrenById.get(child.id) ?? []));
  }
  return undefined;
}

function createExpertDiagramCells(totalExperts: number): ExpertDiagramCell[] {
  if (totalExperts <= graphDiagramLimits.expert.total) {
    return Array.from({ length: totalExperts }, (_, index) => ({
      label: `E${index}`,
      title: `Expert ${index}`,
      kind: "expert" as const,
      expertIndex: index,
    }));
  }

  return [
    ...Array.from(
      { length: graphDiagramLimits.expert.visibleBeforeOverflow },
      (_, index) => ({
        label: `E${index}`,
        title: `Expert ${index}`,
        kind: "expert" as const,
        expertIndex: index,
      }),
    ),
    {
      label: "…",
      title: `${
        totalExperts - graphDiagramLimits.expert.visibleBeforeOverflow
      } more experts`,
      kind: "overflow" as const,
    },
    {
      label: `${totalExperts} experts`,
      title: `${totalExperts} experts total`,
      kind: "total" as const,
    },
  ];
}

export function buildExpertDiagrams(
  graph: InspectResponse | undefined,
  navigation: GraphNavigation,
) {
  const diagramsById = new Map<string, ExpertDiagram>();

  if (!graph) {
    return diagramsById;
  }

  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));

  for (const node of graph.nodes) {
    const children = directChildNodes(node.id, navigation, nodesById);

    if (node.typeName === "MixtureOfExperts") {
      const samplerNode = children.find(
        (child) => lastPathSegment(child.path) === "sampler",
      );
      if (!samplerNode) {
        continue;
      }

      const expertModulesNode = children.find(
        (child) => lastPathSegment(child.path) === "expert_modules",
      );
      const totalExperts =
        detailCount(node.details, "numExperts") ??
        expertCountFromExpertModules(expertModulesNode, navigation, nodesById);
      if (!totalExperts) {
        continue;
      }

      diagramsById.set(node.id, {
        samplerLabel: "Sampler",
        samplerTitle: samplerNode.path,
        cells: createExpertDiagramCells(totalExperts),
        totalExperts,
        hasOverflow: totalExperts > graphDiagramLimits.expert.total,
      });
      continue;
    }

    if (node.typeName !== "MixtureOfExpertsModel") {
      continue;
    }

    const sharedSamplerNode = children.find(
      (child) => lastPathSegment(child.path) === "shared_sampler",
    );
    if (!sharedSamplerNode) {
      continue;
    }

    const expertStackNode = children.find(
      (child) => lastPathSegment(child.path) === "expert_stack",
    );
    const layerCount = expertStackNode
      ? directNumericChildNodes(expertStackNode.id, navigation, nodesById).length
      : undefined;
    const totalExperts =
      detailCount(node.details, "numExperts") ??
      detailCount(sharedSamplerNode.details, "numExperts") ??
      descendantExpertCount(node.id, navigation, nodesById);
    if (!totalExperts) {
      continue;
    }

    diagramsById.set(node.id, {
      samplerLabel: "Shared sampler",
      samplerTitle: sharedSamplerNode.path,
      cells: createExpertDiagramCells(totalExperts),
      totalExperts,
      layerCount: layerCount && layerCount > 0 ? layerCount : undefined,
      hasOverflow: totalExperts > graphDiagramLimits.expert.total,
    });
  }

  return diagramsById;
}
