import dagre from "dagre";
import { type Edge, type Node } from "@xyflow/react";
import { graphCardGeometry } from "@/lib/graph/constants";
import { type GraphDisplayProjection } from "@/lib/graph/display";
import { workbenchGraphEdgeVisual } from "@/lib/graph/visuals";
import { type WorkbenchNodeData } from "@/lib/graph/types";

export function layoutGraph(projection: GraphDisplayProjection) {
  if (projection.cards.length === 0) {
    return { nodes: [], edges: [] };
  }

  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({
    rankdir: "LR",
    nodesep: graphCardGeometry.layout.nodeGap,
    ranksep: graphCardGeometry.layout.rankGap,
  });

  projection.cards.forEach((card) => {
    dagreGraph.setNode(card.id, {
      width: card.width,
      height: card.height,
    });
  });
  projection.edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });
  dagre.layout(dagreGraph);

  const nodes: Array<Node<WorkbenchNodeData>> = projection.cards.map((card) => {
    const position = dagreGraph.node(card.id);
    return {
      id: card.id,
      type: "workbenchNode",
      position: {
        x: position.x - card.width / 2,
        y: position.y - card.height / 2,
      },
      selected: false,
      style: { width: card.width, height: card.height },
      data: card.data,
    };
  });
  const edges: Edge[] = projection.edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    ...workbenchGraphEdgeVisual(),
  }));

  return { nodes, edges };
}
