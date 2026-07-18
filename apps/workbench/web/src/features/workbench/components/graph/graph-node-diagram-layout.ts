import {
  type ClusterDiagram,
  type ExpertDiagram,
} from "@/lib/graph";
import {
  graphCardGeometry,
} from "@/lib/graph/constants";

export const GRAPH_NODE_HORIZONTAL_PADDING = 64;
export const EXPERT_DIAGRAM_WIDTH =
  graphCardGeometry.width - GRAPH_NODE_HORIZONTAL_PADDING;
export const EXPERT_DIAGRAM_GAP = 4;
export const EXPERT_DIAGRAM_OVERFLOW_WIDTH = 30;
export const EXPERT_DIAGRAM_TOTAL_WIDTH = 86;
export const EXPERT_DIAGRAM_SAMPLER_WIDTH = 148;
export const CLUSTER_DIAGRAM_WIDTH = EXPERT_DIAGRAM_WIDTH;

function diagramCellWidths({
  hasOverflow,
  cellsLength,
  overflowWidth,
  totalWidth,
  visibleBeforeOverflow,
  width,
  gap,
}: {
  hasOverflow: boolean;
  cellsLength: number;
  overflowWidth: number;
  totalWidth: number;
  visibleBeforeOverflow: number;
  width: number;
  gap: number;
}) {
  if (!hasOverflow) {
    const cellWidth = (width - (cellsLength - 1) * gap) / cellsLength;
    return Array.from({ length: cellsLength }, () => cellWidth);
  }

  const regularCellWidth =
    (width - (cellsLength - 1) * gap - overflowWidth - totalWidth) /
    visibleBeforeOverflow;
  return [
    ...Array.from({ length: visibleBeforeOverflow }, () => regularCellWidth),
    overflowWidth,
    totalWidth,
  ];
}

export function expertDiagramCellCenters(
  diagram: ExpertDiagram,
  visibleBeforeOverflow: number,
) {
  const widths = diagramCellWidths({
    hasOverflow: diagram.hasOverflow,
    cellsLength: diagram.cells.length,
    overflowWidth: EXPERT_DIAGRAM_OVERFLOW_WIDTH,
    totalWidth: EXPERT_DIAGRAM_TOTAL_WIDTH,
    visibleBeforeOverflow,
    width: EXPERT_DIAGRAM_WIDTH,
    gap: EXPERT_DIAGRAM_GAP,
  });
  let leftOffset = 0;
  return widths.map((cellWidth) => {
    const center = leftOffset + cellWidth / 2;
    leftOffset += cellWidth + EXPERT_DIAGRAM_GAP;
    return center;
  });
}

export function clusterDiagramGridHeight(diagram: ClusterDiagram) {
  return (
    diagram.rows * graphCardGeometry.clusterDiagram.cellSize +
    Math.max(diagram.rows - 1, 0) *
      graphCardGeometry.clusterDiagram.cellGap
  );
}

export function clusterDiagramPlaneWidth(diagram: ClusterDiagram) {
  return (
    diagram.columns * graphCardGeometry.clusterDiagram.cellSize +
    Math.max(diagram.columns - 1, 0) *
      graphCardGeometry.clusterDiagram.cellGap
  );
}
