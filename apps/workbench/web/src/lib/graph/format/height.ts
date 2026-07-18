import type { GraphNode } from "@/lib/api/inspection";
import { graphCardGeometry } from "@/lib/graph/constants";
import {
  nodeDetailEntries,
  parameterShapeEntries,
} from "@/lib/graph/format/badges";
import {
  type ChildSummary,
  type ClusterDiagram,
  type ExpertDiagram,
  type StackDiagram,
} from "@/lib/graph/types";

export type GraphNodeHeightInput = {
  details: GraphNode["details"];
  config: GraphNode["config"];
  childSummaries: ChildSummary[];
  expertDiagram?: ExpertDiagram;
  stackDiagram?: StackDiagram;
  clusterDiagram?: ClusterDiagram;
  isDetailsExpanded: boolean;
};

function childSummaryListHeight(childSummaries: ChildSummary[]) {
  return (
    childSummaries.length * graphCardGeometry.childSummary.rowHeight +
    Math.max(childSummaries.length - 1, 0) *
      graphCardGeometry.childSummary.rowGap
  );
}

function stackDiagramHeight(stackDiagram: StackDiagram) {
  return (
    stackDiagram.cells.length * graphCardGeometry.childSummary.rowHeight +
    Math.max(stackDiagram.cells.length - 1, 0) *
      graphCardGeometry.childSummary.rowGap
  );
}

function clusterDiagramHeight(clusterDiagram: ClusterDiagram) {
  return (
    graphCardGeometry.clusterDiagram.headerHeight +
    clusterDiagram.rows * graphCardGeometry.clusterDiagram.cellSize +
    Math.max(clusterDiagram.rows - 1, 0) *
      graphCardGeometry.clusterDiagram.cellGap
  );
}

function parameterShapeListHeight(input: GraphNodeHeightInput) {
  const parameterShapes = parameterShapeEntries(input.details);
  if (parameterShapes.length === 0) {
    return 0;
  }

  const rowCount = Math.ceil(parameterShapes.length / 2);
  return (
    graphCardGeometry.parameterShapes.marginBlockStart +
    rowCount * graphCardGeometry.parameterShapes.rowHeight +
    Math.max(rowCount - 1, 0) * graphCardGeometry.parameterShapes.rowGap
  );
}

function nonSimpleHeaderHeight() {
  return (
    graphCardGeometry.titleLineHeight +
    graphCardGeometry.subtitle.marginBlockStart +
    graphCardGeometry.subtitle.height
  );
}

function summaryBlockHeight(input: GraphNodeHeightInput) {
  const contentHeight = input.stackDiagram
    ? stackDiagramHeight(input.stackDiagram)
    : input.expertDiagram
      ? graphCardGeometry.expertDiagram.height
      : input.clusterDiagram
        ? clusterDiagramHeight(input.clusterDiagram)
        : input.childSummaries.length > 0
          ? childSummaryListHeight(input.childSummaries)
          : 0;

  if (contentHeight === 0) {
    return 0;
  }

  return graphCardGeometry.contentMarginBlockStart + contentHeight;
}

function detailRowsHeight(rowCount: number, isExpanded: boolean) {
  if (rowCount === 0 || !isExpanded) {
    return 0;
  }

  return (
    graphCardGeometry.details.marginBlockStart +
    rowCount * graphCardGeometry.details.rowHeight +
    Math.max(rowCount - 1, 0) * graphCardGeometry.details.rowGap
  );
}

export function graphNodeHeight(input: GraphNodeHeightInput) {
  const detailEntries = nodeDetailEntries(input.details, input.config);

  return (
    graphCardGeometry.paddingBlock * 2 +
    nonSimpleHeaderHeight() +
    parameterShapeListHeight(input) +
    summaryBlockHeight(input) +
    detailRowsHeight(detailEntries.length, input.isDetailsExpanded) +
    graphCardGeometry.actionBar.marginBlockStart +
    graphCardGeometry.actionBar.height
  );
}
