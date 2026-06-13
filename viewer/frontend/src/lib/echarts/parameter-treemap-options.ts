import type {
  EChartsOption,
  TooltipComponentFormatterCallbackParams,
  TreemapSeriesOption,
} from "echarts";
import type { GraphNode, InspectResponse } from "@/lib/api";
import { scalarSeriesColors } from "@/lib/charts";
import {
  detailText,
  formatCompactCount,
  formatExactCount,
  formatModelSize,
  nodeBadges,
  nodeDimsText,
  structureNodeLabel,
} from "@/lib/graph/formatting";
import { lastPathSegment } from "@/lib/graph/helpers";

const SELECTED_BORDER_COLOR = "#f8fafc";
const SELECTED_SHADOW_COLOR = "rgba(34, 211, 238, 0.34)";
const DEFAULT_BORDER_COLOR = "#07070d";
const TILE_LABEL_COLOR = "#f4f2ff";
const TILE_LABEL_SHADOW = "rgba(0, 0, 0, 0.45)";
const GROUP_LABEL_COLOR = "#d7d3ee";
const GROUP_LABEL_BACKGROUND = "rgba(8, 8, 15, 0.78)";
const TOOLTIP_BACKGROUND = "#10101b";
const TOOLTIP_BORDER = "rgba(255, 255, 255, 0.08)";
const VIRTUAL_ROOT_ID = "__parameter_treemap_root__";

const ROLE_PALETTES: Record<GraphNode["graphRole"], string[]> = {
  architecture: ["#7768f5", "#5f8cf8", "#6d72ee", "#8b7cf6", "#5479d7"],
  internal: ["#16a7c8", "#20c7bd", "#2eb67d", "#38bdf8", "#5dd39e"],
  runtime: ["#f59e0b", "#ef6f91", "#facc15", "#fb923c", "#d946ef"],
};

const ROLE_COLORS: Record<GraphNode["graphRole"], string> = {
  architecture: scalarSeriesColors[0] ?? "#7c6dff",
  internal: scalarSeriesColors[1] ?? "#22d3ee",
  runtime: scalarSeriesColors[2] ?? "#f59e0b",
};

type TreemapDataItem = NonNullable<TreemapSeriesOption["data"]>[number];
type TreemapItemStyle = TreemapDataItem["itemStyle"];
type TreemapItemLabel = TreemapDataItem["label"];
type TreemapLevel = NonNullable<TreemapSeriesOption["levels"]>[number];
type TreemapUpperLabel = NonNullable<TreemapLevel["upperLabel"]>;

export type ParameterTreemapNodeSummary = {
  id: string;
  label: string;
  path: string;
  typeName: string;
  graphRole: GraphNode["graphRole"];
  parameterCount: number;
  parameterSizeBytes: number;
  childCount: number;
  hasParameters: boolean;
  dimText?: string;
  badges: Array<{ key: string; value: string }>;
  isVirtualRoot?: boolean;
};

export type ParameterTreemapItem = {
  id: string;
  name: string;
  value: number;
  nodeId: string;
  path: string;
  typeName: string;
  graphRole: GraphNode["graphRole"];
  parameterCount: number;
  parameterSizeBytes: number;
  focusedParameterCount: number;
  totalParameterCount: number;
  childCount: number;
  dimText?: string;
  canDrill: boolean;
  isDirectParameterBucket?: boolean;
  itemStyle?: TreemapItemStyle;
  label?: TreemapItemLabel;
  children?: ParameterTreemapItem[];
};

type DraftParameterTreemapItem = Omit<
  ParameterTreemapItem,
  "focusedParameterCount" | "totalParameterCount" | "children"
> & {
  children?: DraftParameterTreemapItem[];
};

export type ParameterTreemapData = {
  chartRoots: ParameterTreemapItem[];
  focusedNode: ParameterTreemapNodeSummary | null;
  focusNodeId: string | null;
  ancestors: ParameterTreemapNodeSummary[];
  immediateChildren: ParameterTreemapNodeSummary[];
  zeroParameterChildren: ParameterTreemapNodeSummary[];
  totalParameterCount: number;
  focusedParameterCount: number;
  hasParameters: boolean;
  hasChartParameters: boolean;
};

export type ParameterTreemapOptionOptions = {
  selectedNodeId?: string | null;
};

type GraphIndexes = {
  nodesById: Map<string, GraphNode>;
  childrenById: Map<string, string[]>;
  parentById: Map<string, string>;
  rootIds: string[];
};

function finiteNonNegative(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : 0;
}

function buildGraphIndexes(graph: InspectResponse): GraphIndexes {
  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));
  const childrenById = new Map(graph.nodes.map((node) => [node.id, [] as string[]]));
  const parentById = new Map<string, string>();
  const childIds = new Set<string>();
  const edgeKeys = new Set<string>();

  for (const edge of graph.edges) {
    if (
      edge.source === edge.target ||
      !nodesById.has(edge.source) ||
      !nodesById.has(edge.target)
    ) {
      continue;
    }

    const edgeKey = `${edge.source}\0${edge.target}`;
    if (edgeKeys.has(edgeKey)) {
      continue;
    }
    edgeKeys.add(edgeKey);
    childrenById.get(edge.source)?.push(edge.target);
    if (!parentById.has(edge.target)) {
      parentById.set(edge.target, edge.source);
    }
    childIds.add(edge.target);
  }

  const rootIds = graph.nodes
    .filter((node) => !childIds.has(node.id))
    .map((node) => node.id);

  return {
    nodesById,
    childrenById,
    parentById,
    rootIds: rootIds.length > 0 ? rootIds : graph.nodes[0] ? [graph.nodes[0].id] : [],
  };
}

function directParameterSizeBytes(
  parentSizeBytes: number,
  childSizeBytes: number,
  directParameterCount: number,
  parentParameterCount: number,
) {
  const exactRemainder = parentSizeBytes - childSizeBytes;
  if (exactRemainder > 0) {
    return exactRemainder;
  }
  if (parentSizeBytes > 0 && parentParameterCount > 0) {
    return Math.round(parentSizeBytes * (directParameterCount / parentParameterCount));
  }
  return 0;
}

function selectedStyle(nodeId: string, selectedNodeId: string | null | undefined) {
  if (!selectedNodeId || nodeId !== selectedNodeId) {
    return undefined;
  }

  return {
    borderColor: SELECTED_BORDER_COLOR,
    borderWidth: 3,
    shadowBlur: 14,
    shadowColor: SELECTED_SHADOW_COLOR,
  };
}

function hashString(value: string) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash;
}

function roleColor(graphRole: GraphNode["graphRole"], key = "") {
  const palette = ROLE_PALETTES[graphRole] ?? [ROLE_COLORS[graphRole]];
  return palette[hashString(key || graphRole) % palette.length] ?? ROLE_COLORS[graphRole];
}

function tileLabel(hasChildren: boolean): TreemapItemLabel {
  if (hasChildren) {
    return { show: false };
  }
  return {
    show: true,
    align: "left",
    verticalAlign: "top",
    color: TILE_LABEL_COLOR,
    fontSize: 10,
    fontWeight: 700,
    lineHeight: 13,
    overflow: "truncate",
    minMargin: 8,
    textBorderWidth: 0,
    textShadowColor: TILE_LABEL_SHADOW,
    textShadowBlur: 3,
    formatter: formatParameterTreemapLabel,
  };
}

function withFocusMetadata(
  item: DraftParameterTreemapItem,
  focusedParameterCount: number,
  totalParameterCount: number,
): ParameterTreemapItem {
  return {
    ...item,
    focusedParameterCount,
    totalParameterCount,
    children: item.children?.map((child) =>
      withFocusMetadata(child, focusedParameterCount, totalParameterCount),
    ),
  };
}

function withLayoutValue(
  item: DraftParameterTreemapItem,
  value: number,
): DraftParameterTreemapItem {
  return { ...item, value };
}

function pathSegments(path: string) {
  return path.split(".").filter(Boolean);
}

function uniqueLabelForDuplicate(node: GraphNode, pathSegmentCount: number) {
  const segments = pathSegments(node.path);
  const suffix = segments.slice(Math.max(0, segments.length - pathSegmentCount)).join(".");
  return `${suffix || lastPathSegment(node.path) || node.id}: ${node.typeName}`;
}

function uniqueTreemapLabels(nodes: GraphNode[]) {
  const labelsById = new Map<string, string>();
  const baseGroups = new Map<string, GraphNode[]>();

  for (const node of nodes) {
    const label = structureNodeLabel(node);
    const group = baseGroups.get(label) ?? [];
    group.push(node);
    baseGroups.set(label, group);
  }

  for (const [label, group] of baseGroups) {
    if (group.length === 1) {
      labelsById.set(group[0].id, label);
      continue;
    }

    let segmentCount = 2;
    let labels = group.map((node) => uniqueLabelForDuplicate(node, segmentCount));
    while (new Set(labels).size < labels.length && segmentCount < 8) {
      segmentCount += 1;
      labels = group.map((node) => uniqueLabelForDuplicate(node, segmentCount));
    }

    labels.forEach((candidate, index) => {
      const node = group[index];
      labelsById.set(
        node.id,
        labels.filter((other) => other === candidate).length === 1
          ? candidate
          : `${candidate} · ${node.id}`,
      );
    });
  }

  return labelsById;
}

function summarizeNode(
  node: GraphNode,
  childrenById: Map<string, string[]>,
  label = structureNodeLabel(node),
): ParameterTreemapNodeSummary {
  const parameterCount = finiteNonNegative(node.parameterCount);
  return {
    id: node.id,
    label,
    path: node.path,
    typeName: node.typeName,
    graphRole: node.graphRole,
    parameterCount,
    parameterSizeBytes: finiteNonNegative(node.parameterSizeBytes),
    childCount: childrenById.get(node.id)?.length ?? 0,
    hasParameters: parameterCount > 0,
    dimText: nodeDimsText(node.details, node.config),
    badges: nodeBadges(node.details).map(([key, value]) => ({
      key,
      value: detailText(value),
    })),
  };
}

function virtualRootSummary(
  graph: InspectResponse,
  childCount: number,
  totalParameterCount: number,
): ParameterTreemapNodeSummary {
  return {
    id: VIRTUAL_ROOT_ID,
    label: `${graph.model} / ${graph.preset}`,
    path: graph.model,
    typeName: "Root",
    graphRole: "architecture",
    parameterCount: totalParameterCount,
    parameterSizeBytes: finiteNonNegative(graph.parameterSizeBytes),
    childCount,
    hasParameters: totalParameterCount > 0,
    badges: [],
    isVirtualRoot: true,
  };
}

function ancestorSummaries(
  nodeId: string,
  indexes: GraphIndexes,
  labelsByParentId: Map<string, Map<string, string>>,
) {
  const ancestors: ParameterTreemapNodeSummary[] = [];
  const visited = new Set<string>([nodeId]);
  let parentId = indexes.parentById.get(nodeId);

  while (parentId) {
    if (visited.has(parentId)) {
      return [];
    }
    visited.add(parentId);
    const parent = indexes.nodesById.get(parentId);
    if (!parent) {
      break;
    }
    ancestors.push(
      summarizeNode(
        parent,
        indexes.childrenById,
        labelsByParentId.get(indexes.parentById.get(parent.id) ?? "")?.get(parent.id) ??
          structureNodeLabel(parent),
      ),
    );
    parentId = indexes.parentById.get(parentId);
  }

  return ancestors.reverse();
}

function childNodes(childIds: string[], nodesById: Map<string, GraphNode>) {
  return childIds
    .map((childId) => nodesById.get(childId))
    .filter((node): node is GraphNode => Boolean(node));
}

function labelsByParent(indexes: GraphIndexes) {
  const result = new Map<string, Map<string, string>>();
  result.set(
    VIRTUAL_ROOT_ID,
    uniqueTreemapLabels(childNodes(indexes.rootIds, indexes.nodesById)),
  );
  for (const [parentId, childIds] of indexes.childrenById) {
    result.set(parentId, uniqueTreemapLabels(childNodes(childIds, indexes.nodesById)));
  }
  return result;
}

export function fallbackParameterTreemapFocusNodeId(
  requestedFocusNodeId: string | null | undefined,
  visibleGraph: InspectResponse | undefined,
  sourceGraph: InspectResponse | undefined = visibleGraph,
) {
  if (!requestedFocusNodeId || !visibleGraph || visibleGraph.nodes.length === 0) {
    return null;
  }

  const visibleNodeIds = new Set(visibleGraph.nodes.map((node) => node.id));
  if (visibleNodeIds.has(requestedFocusNodeId)) {
    return requestedFocusNodeId;
  }
  if (!sourceGraph) {
    return null;
  }

  const parentById = buildGraphIndexes(sourceGraph).parentById;
  const visited = new Set<string>([requestedFocusNodeId]);
  let parentId = parentById.get(requestedFocusNodeId);

  while (parentId) {
    if (visited.has(parentId)) {
      return null;
    }
    visited.add(parentId);
    if (visibleNodeIds.has(parentId)) {
      return parentId;
    }
    parentId = parentById.get(parentId);
  }

  return null;
}

export function buildParameterTreemapData(
  graph: InspectResponse | undefined,
  focusNodeId: string | null = null,
): ParameterTreemapData {
  if (!graph || graph.nodes.length === 0) {
    return {
      chartRoots: [],
      focusedNode: null,
      focusNodeId: null,
      ancestors: [],
      immediateChildren: [],
      zeroParameterChildren: [],
      totalParameterCount: 0,
      focusedParameterCount: 0,
      hasParameters: false,
      hasChartParameters: false,
    };
  }

  const indexes = buildGraphIndexes(graph);
  const labelsByParentId = labelsByParent(indexes);
  const resolvedFocusNodeId = indexes.nodesById.has(focusNodeId ?? "")
    ? focusNodeId
    : null;
  const focusedGraphNode = resolvedFocusNodeId
    ? indexes.nodesById.get(resolvedFocusNodeId)
    : undefined;
  const totalParameterCount =
    finiteNonNegative(graph.parameterCount) ||
    graph.nodes.reduce((total, node) => total + finiteNonNegative(node.parameterCount), 0);
  const rootChildIds =
    indexes.rootIds.length === 1 &&
    (indexes.childrenById.get(indexes.rootIds[0])?.length ?? 0) > 0
      ? indexes.childrenById.get(indexes.rootIds[0]) ?? []
      : indexes.rootIds;
  const rootFocusNode =
    indexes.rootIds.length === 1 ? indexes.nodesById.get(indexes.rootIds[0]) : undefined;
  const focusedNode = focusedGraphNode
    ? summarizeNode(
        focusedGraphNode,
        indexes.childrenById,
        labelsByParentId
          .get(indexes.parentById.get(focusedGraphNode.id) ?? VIRTUAL_ROOT_ID)
          ?.get(focusedGraphNode.id) ?? structureNodeLabel(focusedGraphNode),
      )
    : rootFocusNode
      ? summarizeNode(rootFocusNode, indexes.childrenById)
      : virtualRootSummary(graph, rootChildIds.length, totalParameterCount);
  const focusedParameterCount =
    focusedGraphNode || rootFocusNode
      ? finiteNonNegative((focusedGraphNode ?? rootFocusNode)?.parameterCount) ||
        totalParameterCount
      : totalParameterCount;
  const focusedChildIds = focusedGraphNode
    ? indexes.childrenById.get(focusedGraphNode.id) ?? []
    : rootChildIds;
  const focusedChildNodes = childNodes(focusedChildIds, indexes.nodesById);
  const focusedChildLabels = focusedGraphNode
    ? labelsByParentId.get(focusedGraphNode.id) ?? new Map<string, string>()
    : indexes.rootIds.length === 1
      ? labelsByParentId.get(indexes.rootIds[0]) ?? new Map<string, string>()
      : labelsByParentId.get(VIRTUAL_ROOT_ID) ?? new Map<string, string>();
  const immediateChildren = focusedChildNodes.map((node) =>
    summarizeNode(
      node,
      indexes.childrenById,
      focusedChildLabels.get(node.id) ?? structureNodeLabel(node),
    ),
  );
  const zeroParameterChildren = immediateChildren.filter((child) => !child.hasParameters);

  const buildNode = (
    node: GraphNode,
    activeNodeIds: Set<string>,
    label: string,
  ): DraftParameterTreemapItem | undefined => {
    if (activeNodeIds.has(node.id)) {
      return undefined;
    }

    const nextActiveNodeIds = new Set(activeNodeIds);
    nextActiveNodeIds.add(node.id);
    const children = childNodes(indexes.childrenById.get(node.id) ?? [], indexes.nodesById);
    const childLabels = labelsByParentId.get(node.id) ?? new Map<string, string>();
    const childItems = children
      .map((child) =>
        buildNode(
          child,
          nextActiveNodeIds,
          childLabels.get(child.id) ?? structureNodeLabel(child),
        ),
      )
      .filter((child): child is DraftParameterTreemapItem => Boolean(child));

    const childParameterTotal = childItems.reduce(
      (total, child) => total + child.parameterCount,
      0,
    );
    const childSizeTotal = childItems.reduce(
      (total, child) => total + child.parameterSizeBytes,
      0,
    );
    const nodeParameterCount = finiteNonNegative(node.parameterCount);
    const parameterCount = nodeParameterCount > 0 ? nodeParameterCount : childParameterTotal;

    if (parameterCount <= 0 && childParameterTotal <= 0) {
      return undefined;
    }

    const nodeSizeBytes = finiteNonNegative(node.parameterSizeBytes);
    const parameterSizeBytes = nodeSizeBytes > 0 ? nodeSizeBytes : childSizeTotal;
    const layoutScale =
      childParameterTotal > parameterCount && parameterCount > 0
        ? parameterCount / childParameterTotal
        : 1;
    const layoutChildren = childItems.map((child) =>
      withLayoutValue(child, child.parameterCount * layoutScale),
    );
    // The inspection payload exposes recursive counts, not direct ownership.
    // When shared tensors make child totals exceed the parent, true direct
    // parent ownership cannot be recovered without backend metadata.
    const directParameterCount = parameterCount - childParameterTotal;

    if (childItems.length > 0 && directParameterCount > 0) {
      layoutChildren.push({
        id: `${node.id}::__direct_params`,
        name: "Direct parameters",
        value: directParameterCount,
        nodeId: node.id,
        path: node.path,
        typeName: "DirectParameters",
        graphRole: node.graphRole,
        parameterCount: directParameterCount,
        parameterSizeBytes: directParameterSizeBytes(
          parameterSizeBytes,
          childSizeTotal,
          directParameterCount,
          parameterCount,
        ),
        childCount: 0,
        canDrill: false,
        isDirectParameterBucket: true,
        label: tileLabel(false),
        itemStyle: {
          color: roleColor(node.graphRole, `${node.id}:direct`),
          opacity: 0.46,
        },
      });
    }

    return {
      id: node.id,
      name: label,
      value: parameterCount,
      nodeId: node.id,
      path: node.path,
      typeName: node.typeName,
      graphRole: node.graphRole,
      parameterCount,
      parameterSizeBytes,
      childCount: indexes.childrenById.get(node.id)?.length ?? 0,
      dimText: nodeDimsText(node.details, node.config),
      canDrill: (indexes.childrenById.get(node.id)?.length ?? 0) > 0,
      label: tileLabel(layoutChildren.length > 0),
      children: layoutChildren.length > 0 ? layoutChildren : undefined,
      itemStyle: {
        color: roleColor(node.graphRole, node.path || node.id),
      },
    };
  };

  const chartRoots = focusedChildNodes
    .map((node) =>
      buildNode(
        node,
        new Set(focusedGraphNode ? [focusedGraphNode.id] : []),
        focusedChildLabels.get(node.id) ?? structureNodeLabel(node),
      ),
    )
    .filter((item): item is DraftParameterTreemapItem => Boolean(item));

  const focusDirectSource = focusedGraphNode ?? rootFocusNode;
  if (focusDirectSource && focusedChildNodes.length > 0) {
    const childParameterTotal = chartRoots.reduce(
      (total, child) => total + child.parameterCount,
      0,
    );
    const childSizeTotal = chartRoots.reduce(
      (total, child) => total + child.parameterSizeBytes,
      0,
    );
    const directParameterCount = focusedParameterCount - childParameterTotal;
    if (directParameterCount > 0) {
      chartRoots.push({
        id: `${focusDirectSource.id}::__direct_params`,
        name: "Direct parameters",
        value: directParameterCount,
        nodeId: focusDirectSource.id,
        path: focusDirectSource.path,
        typeName: "DirectParameters",
        graphRole: focusDirectSource.graphRole,
        parameterCount: directParameterCount,
        parameterSizeBytes: directParameterSizeBytes(
          finiteNonNegative(focusDirectSource.parameterSizeBytes),
          childSizeTotal,
          directParameterCount,
          focusedParameterCount,
        ),
        childCount: 0,
        canDrill: false,
        isDirectParameterBucket: true,
        label: tileLabel(false),
        itemStyle: {
          color: roleColor(focusDirectSource.graphRole, `${focusDirectSource.id}:direct`),
          opacity: 0.46,
        },
      });
    } else if (childParameterTotal > focusedParameterCount && focusedParameterCount > 0) {
      const layoutScale = focusedParameterCount / childParameterTotal;
      for (const child of chartRoots) {
        child.value = child.parameterCount * layoutScale;
      }
    }
  }

  const rootsToHydrate =
    chartRoots.length > 0 || focusedChildNodes.length > 0
      ? chartRoots
      : focusDirectSource
        ? [
            buildNode(
              focusDirectSource,
              new Set(),
              focusedNode.label,
            ),
          ].filter((item): item is DraftParameterTreemapItem => Boolean(item))
        : chartRoots;
  const hydratedRoots = rootsToHydrate.map((root) =>
    withFocusMetadata(root, focusedParameterCount, totalParameterCount),
  );

  return {
    chartRoots: hydratedRoots,
    focusedNode,
    focusNodeId: resolvedFocusNodeId,
    ancestors: focusedGraphNode
      ? ancestorSummaries(focusedGraphNode.id, indexes, labelsByParentId)
      : [],
    immediateChildren,
    zeroParameterChildren,
    totalParameterCount,
    focusedParameterCount,
    hasParameters: totalParameterCount > 0,
    hasChartParameters: hydratedRoots.length > 0,
  };
}

function formatPercent(part: number, whole: number) {
  if (whole <= 0) {
    return "0%";
  }
  const percent = (part / whole) * 100;
  if (percent > 0 && percent < 0.1) {
    return "<0.1%";
  }
  if (percent < 10) {
    return `${percent.toFixed(1).replace(/\.0$/, "")}%`;
  }
  return `${percent.toFixed(0)}%`;
}

function escapeHtml(value: string) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function isParameterTreemapItem(data: unknown): data is ParameterTreemapItem {
  if (!data || typeof data !== "object") {
    return false;
  }
  const candidate = data as Partial<ParameterTreemapItem>;
  return (
    typeof candidate.name === "string" &&
    typeof candidate.path === "string" &&
    typeof candidate.typeName === "string" &&
    typeof candidate.parameterCount === "number" &&
    typeof candidate.parameterSizeBytes === "number" &&
    typeof candidate.focusedParameterCount === "number" &&
    typeof candidate.totalParameterCount === "number"
  );
}

function tooltipData(
  params: TooltipComponentFormatterCallbackParams,
): ParameterTreemapItem | undefined {
  const firstParam = Array.isArray(params) ? params[0] : params;
  const data = (firstParam as { data?: unknown }).data;
  if (!isParameterTreemapItem(data)) {
    return undefined;
  }
  return data;
}

function formatParameterTreemapTooltip(params: TooltipComponentFormatterCallbackParams) {
  const data = tooltipData(params);
  if (!data) {
    return "";
  }

  const modelSize = formatModelSize(data.parameterSizeBytes) ?? "0 MB";
  return [
    `<strong>${escapeHtml(data.name)}</strong>`,
    `Path: ${escapeHtml(data.path)}`,
    `Type: ${escapeHtml(data.typeName)}`,
    `Exact params: ${formatExactCount(data.parameterCount)}`,
    `Compact params: ${formatCompactCount(data.parameterCount)}`,
    `Memory: ${modelSize}`,
    `Share of focus: ${formatPercent(data.parameterCount, data.focusedParameterCount)}`,
    `Share of model: ${formatPercent(data.parameterCount, data.totalParameterCount)}`,
  ].join("<br/>");
}

function formatParameterTreemapLabel(params: { data?: unknown; name?: string }) {
  const data = params.data as Partial<ParameterTreemapItem> | undefined;
  if (!data) {
    return params.name ?? "";
  }
  const parameterCount =
    typeof data.parameterCount === "number" ? data.parameterCount : undefined;
  if (parameterCount === undefined) {
    return params.name ?? "";
  }

  const details = data.dimText
    ? data.dimText
    : typeof data.childCount === "number" && data.childCount > 0
      ? `${data.childCount} children`
      : undefined;
  return [
    params.name ?? "",
    data.typeName && data.typeName !== "DirectParameters" ? data.typeName : undefined,
    formatCompactCount(parameterCount),
    details,
  ].filter(Boolean).join("\n");
}

function formatParameterTreemapGroupLabel(params: { data?: unknown; name?: string }) {
  const data = params.data as Partial<ParameterTreemapItem> | undefined;
  if (!data?.children || data.children.length === 0) {
    return "";
  }
  const count = typeof data.parameterCount === "number"
    ? ` · ${formatCompactCount(data.parameterCount)}`
    : "";
  return `${params.name ?? data.name ?? ""}${count}`;
}

function groupUpperLabel(height: number): TreemapUpperLabel {
  return {
    show: true,
    height,
    color: GROUP_LABEL_COLOR,
    backgroundColor: GROUP_LABEL_BACKGROUND,
    fontSize: 10,
    fontWeight: 700,
    lineHeight: height,
    overflow: "truncate",
    minMargin: 6,
    padding: [0, 6],
    textBorderWidth: 0,
    formatter: formatParameterTreemapGroupLabel,
  };
}

export function buildParameterTreemapOption(
  data: ParameterTreemapData,
  options: ParameterTreemapOptionOptions = {},
): EChartsOption {
  const selectedNodeId = options.selectedNodeId ?? null;
  const roots = selectedNodeId
    ? data.chartRoots.map((root) => markSelected(root, selectedNodeId))
    : data.chartRoots;

  return {
    animation: false,
    backgroundColor: "transparent",
    tooltip: {
      trigger: "item",
      formatter: formatParameterTreemapTooltip,
      backgroundColor: TOOLTIP_BACKGROUND,
      borderColor: TOOLTIP_BORDER,
      borderWidth: 1,
      padding: [10, 12],
      extraCssText:
        "border-radius:10px;box-shadow:0 18px 44px rgba(0,0,0,0.38);backdrop-filter:blur(10px);",
    },
    series: [
      {
        type: "treemap",
        animation: false,
        roam: false,
        nodeClick: false,
        childrenVisibleMin: 0,
        breadcrumb: { show: false },
        left: 10,
        right: 10,
        top: 10,
        bottom: 10,
        sort: false,
        label: {
          show: false,
          color: TILE_LABEL_COLOR,
          fontSize: 10,
          fontWeight: 700,
          lineHeight: 13,
          overflow: "truncate",
          minMargin: 8,
          textBorderWidth: 0,
          textShadowColor: TILE_LABEL_SHADOW,
          textShadowBlur: 3,
          formatter: formatParameterTreemapLabel,
        },
        upperLabel: {
          show: false,
        },
        itemStyle: {
          borderColor: DEFAULT_BORDER_COLOR,
          borderWidth: 2,
          gapWidth: 4,
          borderRadius: 5,
        },
        emphasis: {
          focus: "self",
          label: {
            show: false,
            color: "#ffffff",
            textBorderWidth: 0,
            textShadowColor: "rgba(0, 0, 0, 0.55)",
            textShadowBlur: 4,
          },
          itemStyle: {
            borderColor: SELECTED_BORDER_COLOR,
            borderWidth: 2,
            shadowBlur: 12,
            shadowColor: SELECTED_SHADOW_COLOR,
          },
        },
        levels: [
          {
            upperLabel: { show: false },
            itemStyle: {
              borderColor: "#05050a",
              borderWidth: 3,
              gapWidth: 6,
            },
          },
          {
            upperLabel: groupUpperLabel(18),
            itemStyle: {
              borderColor: "#080811",
              borderWidth: 3,
              gapWidth: 5,
            },
          },
          {
            upperLabel: groupUpperLabel(17),
            itemStyle: {
              borderColor: "#0b0b14",
              borderWidth: 2,
              gapWidth: 4,
            },
          },
          {
            upperLabel: groupUpperLabel(16),
            itemStyle: {
              borderColor: "#11111d",
              borderWidth: 2,
              gapWidth: 3,
            },
          },
        ],
        data: roots,
      } satisfies TreemapSeriesOption,
    ],
  };
}

function markSelected(
  item: ParameterTreemapItem,
  selectedNodeId: string,
): ParameterTreemapItem {
  return {
    ...item,
    itemStyle: {
      ...item.itemStyle,
      ...selectedStyle(item.nodeId, selectedNodeId),
    },
    children: item.children?.map((child) => markSelected(child, selectedNodeId)),
  };
}
