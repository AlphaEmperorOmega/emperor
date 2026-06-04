import { type GraphNode } from "@/lib/api";
import {
  CHILD_SUMMARY_EMPTY_HEIGHT,
  CHILD_SUMMARY_ROW_GAP,
  CHILD_SUMMARY_ROW_HEIGHT,
  DETAIL_LIST_MARGIN_TOP,
  DETAIL_ROW_GAP,
  DETAIL_ROW_HEIGHT,
  EXPERT_DIAGRAM_HEIGHT,
  CLUSTER_DIAGRAM_CELL_GAP,
  CLUSTER_DIAGRAM_CELL_HEIGHT,
  CLUSTER_DIAGRAM_HEADER_HEIGHT,
  GRAPH_NODE_BADGE_ROW_HEIGHT,
  GRAPH_NODE_BADGE_ROW_MARGIN_TOP,
  GRAPH_NODE_CONTENT_MARGIN_TOP,
  GRAPH_NODE_EXPANSION_TOGGLE_HEIGHT,
  GRAPH_NODE_METADATA_BORDER_TOP_WIDTH,
  GRAPH_NODE_METADATA_MARGIN_TOP,
  GRAPH_NODE_METADATA_PADDING_TOP,
  GRAPH_NODE_METADATA_TOGGLE_HEIGHT,
  GRAPH_NODE_SUBTITLE_HEIGHT,
  GRAPH_NODE_SUBTITLE_MARGIN_TOP,
  GRAPH_NODE_TITLE_LINE_HEIGHT,
  GRAPH_NODE_VERTICAL_PADDING,
  PARAMETER_SHAPE_LIST_MARGIN_TOP,
  PARAMETER_SHAPE_ROW_HEIGHT,
  PARAMETER_SHAPE_ROW_GAP,
  STACK_DIAGRAM_COMPACT_HEIGHT,
  STACK_DIAGRAM_DENSE_CELL_THRESHOLD,
  STACK_DIAGRAM_DENSE_HEIGHT,
} from "@/lib/graph/constants";
import {
  nodeDetailEntries,
  parameterShapeEntries,
} from "@/lib/graph/format/badges";
import {
  type ChildSummary,
  type ClusterDiagram,
  type ExpertDiagram,
  type GraphDetailMode,
  type StackDiagram,
} from "@/lib/graph/types";

export type GraphNodeHeightInput = {
  title: string;
  parameterCount: number;
  childCount: number;
  graphDetailMode: GraphDetailMode;
  canToggleExpansion: boolean;
  details: GraphNode["details"];
  config: GraphNode["config"];
  childSummaries: ChildSummary[];
  expertDiagram?: ExpertDiagram;
  stackDiagram?: StackDiagram;
  clusterDiagram?: ClusterDiagram;
  isDetailsExpanded: boolean;
};

function childSummaryListHeight(childSummaries: ChildSummary[]) {
  if (childSummaries.length === 0) {
    return CHILD_SUMMARY_EMPTY_HEIGHT;
  }

  return (
    childSummaries.length * CHILD_SUMMARY_ROW_HEIGHT +
    (childSummaries.length - 1) * CHILD_SUMMARY_ROW_GAP
  );
}

function stackDiagramHeight(stackDiagram: StackDiagram) {
  return stackDiagram.cells.length > STACK_DIAGRAM_DENSE_CELL_THRESHOLD
    ? STACK_DIAGRAM_DENSE_HEIGHT
    : STACK_DIAGRAM_COMPACT_HEIGHT;
}

function clusterDiagramHeight(clusterDiagram: ClusterDiagram) {
  return (
    CLUSTER_DIAGRAM_HEADER_HEIGHT +
    clusterDiagram.rows * CLUSTER_DIAGRAM_CELL_HEIGHT +
    Math.max(clusterDiagram.rows - 1, 0) * CLUSTER_DIAGRAM_CELL_GAP
  );
}

function parameterShapeListHeight(details: GraphNode["details"]) {
  const entries = parameterShapeEntries(details);
  if (entries.length === 0) {
    return 0;
  }

  const rowCount = Math.ceil(entries.length / 2);
  return (
    PARAMETER_SHAPE_LIST_MARGIN_TOP +
    rowCount * PARAMETER_SHAPE_ROW_HEIGHT +
    Math.max(rowCount - 1, 0) * PARAMETER_SHAPE_ROW_GAP
  );
}

function nonSimpleHeaderHeight(input: GraphNodeHeightInput) {
  const hasBadgeRow =
    input.graphDetailMode === "full" && (input.parameterCount > 0 || input.childCount > 0);
  return (
    Math.max(
      GRAPH_NODE_TITLE_LINE_HEIGHT,
      input.canToggleExpansion ? GRAPH_NODE_EXPANSION_TOGGLE_HEIGHT : 0,
    ) +
    (hasBadgeRow ? GRAPH_NODE_BADGE_ROW_MARGIN_TOP + GRAPH_NODE_BADGE_ROW_HEIGHT : 0) +
    GRAPH_NODE_SUBTITLE_MARGIN_TOP +
    GRAPH_NODE_SUBTITLE_HEIGHT
  );
}

function summaryBlockHeight(input: GraphNodeHeightInput) {
  const contentHeight = input.stackDiagram
    ? stackDiagramHeight(input.stackDiagram)
    : input.expertDiagram
      ? EXPERT_DIAGRAM_HEIGHT
      : input.clusterDiagram
        ? clusterDiagramHeight(input.clusterDiagram)
        : childSummaryListHeight(input.childSummaries);

  return GRAPH_NODE_CONTENT_MARGIN_TOP + contentHeight;
}

function metadataBlockHeight(rowCount: number, isExpanded: boolean) {
  if (rowCount === 0) {
    return 0;
  }

  return (
    GRAPH_NODE_METADATA_MARGIN_TOP +
    GRAPH_NODE_METADATA_BORDER_TOP_WIDTH +
    GRAPH_NODE_METADATA_PADDING_TOP +
    GRAPH_NODE_METADATA_TOGGLE_HEIGHT +
    (isExpanded
      ? DETAIL_LIST_MARGIN_TOP +
        rowCount * DETAIL_ROW_HEIGHT +
        Math.max(rowCount - 1, 0) * DETAIL_ROW_GAP
      : 0)
  );
}

export function graphNodeHeight(input: GraphNodeHeightInput) {
  const detailEntries = nodeDetailEntries(input.details, input.config);

  return (
    GRAPH_NODE_VERTICAL_PADDING +
    nonSimpleHeaderHeight(input) +
    parameterShapeListHeight(input.details) +
    summaryBlockHeight(input) +
    metadataBlockHeight(detailEntries.length, input.isDetailsExpanded)
  );
}
