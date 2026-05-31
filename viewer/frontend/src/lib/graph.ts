import dagre from "dagre";
import { MarkerType, type Edge, type Node } from "@xyflow/react";
import { type GraphNode, type InspectResponse } from "@/lib/api";

export type GraphDetailMode = "basic" | "full";
export type GraphScope = "opened" | "entire";

export type ViewerNodeData = {
  nodeId: string;
  label: string;
  subtitle: string;
  path: string;
  parameterCount: number;
  details: GraphNode["details"];
  childCount: number;
  childSummaries: ChildSummary[];
  expertDiagram?: ExpertDiagram;
  stackDiagram?: StackDiagram;
  height: number;
  isExpanded: boolean;
  canToggleExpansion: boolean;
  isDetailsExpanded: boolean;
  onActivateNode: () => void;
  onToggleDetails: () => void;
};

export type HierarchyNode = {
  node: GraphNode;
  children: HierarchyNode[];
};

export type GraphNavigation = {
  childrenById: Map<string, string[]>;
  rootIds: Set<string>;
};

export type ChildSummary = {
  label: string;
  nestedLabel?: string;
  count?: number;
  kind: "child" | "mechanism" | "overflow";
  stackKind?: "layer";
  title?: string;
};

export type ExpertDiagramCell = {
  label: string;
  title: string;
  kind: "expert" | "overflow" | "total";
  expertIndex?: number;
};

export type ExpertDiagram = {
  samplerLabel: "Sampler" | "Shared sampler";
  samplerTitle: string;
  cells: ExpertDiagramCell[];
  totalExperts: number;
  layerCount?: number;
  hasOverflow: boolean;
};

export type StackDiagramCell = {
  label: string;
  title: string;
  kind: "layer" | "overflow" | "total";
  layerIndex?: number;
};

export type StackDiagram = {
  cells: StackDiagramCell[];
  totalLayers: number;
  hasOverflow: boolean;
};

const NODE_WIDTH = 220;
const COLLAPSED_NODE_HEIGHT_WITH_METADATA = 148;
const COLLAPSED_NODE_HEIGHT_WITHOUT_METADATA = 116;
const COLLAPSED_NODE_STATIC_HEIGHT_WITH_METADATA = 112;
const COLLAPSED_NODE_STATIC_HEIGHT_WITHOUT_METADATA = 80;
const CHILD_SUMMARY_EMPTY_HEIGHT = 20;
const CHILD_SUMMARY_ROW_HEIGHT = 24;
const CHILD_SUMMARY_ROW_GAP = 4;
const EXPERT_DIAGRAM_HEIGHT = 82;
const EXPERT_DIAGRAM_LIMIT = 7;
const EXPERT_DIAGRAM_VISIBLE_BEFORE_OVERFLOW = 5;
const STACK_DIAGRAM_HEIGHT = 82;
const STACK_DIAGRAM_LIMIT = 7;
const STACK_DIAGRAM_VISIBLE_BEFORE_OVERFLOW = 5;
const LAYER_STACK_SUMMARY_LIMIT = 7;
const LAYER_STACK_VISIBLE_BEFORE_OVERFLOW = 5;
const DETAIL_LIST_EXTRA_HEIGHT = 4;
const DETAIL_ROW_HEIGHT = 26;
const GRAPH_VERTICAL_CARD_GAP = 16;
const GRAPH_HORIZONTAL_CARD_GAP = 64;
const SEMANTIC_LABEL_TYPE_NAMES = new Set(["ModuleList", "Sequential"]);
const STACK_CONTAINER_TYPE_NAMES = new Set(["ModuleList", "Sequential"]);
const GATE_SUMMARY_LABEL = "Gate";
const HALTING_SUMMARY_LABEL = "Halting mechanism";

const exactCountFormatter = new Intl.NumberFormat("en-US");

export function detailText(value: unknown) {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

export function formatExactCount(count: number) {
  return exactCountFormatter.format(count);
}

export function formatCompactCount(count: number) {
  const absoluteCount = Math.abs(count);
  if (absoluteCount < 1000) {
    return formatExactCount(count);
  }

  const units = [
    { suffix: "B", value: 1_000_000_000 },
    { suffix: "M", value: 1_000_000 },
    { suffix: "K", value: 1_000 },
  ];
  const unit = units.find((candidate) => absoluteCount >= candidate.value);
  if (!unit) {
    return formatExactCount(count);
  }

  const value = count / unit.value;
  const formatted = value >= 100 ? value.toFixed(0) : value.toFixed(1);
  return `${formatted.replace(/\.0$/, "")}${unit.suffix}`;
}

export function nodeBadges(details: GraphNode["details"]) {
  const badges: Array<[string, unknown]> = [];
  if (typeof details.dims === "string") {
    badges.push(["dims", details.dims]);
  }
  if (typeof details.activation === "string" && details.activation !== "DISABLED") {
    badges.push(["act", details.activation]);
  }
  if (typeof details.layerNorm === "string" && details.layerNorm !== "DISABLED") {
    badges.push(["norm", details.layerNorm]);
  }
  if (typeof details.dropout === "number" && details.dropout > 0) {
    badges.push(["drop", details.dropout]);
  }
  if (details.gate === true) {
    badges.push(["gate", "on"]);
  }
  if (details.halting === true) {
    badges.push(["halt", "on"]);
  }
  if (isRecord(details.recurrent)) {
    badges.push(["steps", details.recurrent.maxSteps]);
  }
  return badges;
}

export function nodeDetailEntries(details: GraphNode["details"]) {
  return Object.entries(details);
}

function childSummaryListHeight(childSummaries: ChildSummary[]) {
  if (childSummaries.length === 0) {
    return CHILD_SUMMARY_EMPTY_HEIGHT;
  }

  return (
    childSummaries.length * CHILD_SUMMARY_ROW_HEIGHT +
    (childSummaries.length - 1) * CHILD_SUMMARY_ROW_GAP
  );
}

function graphNodeHeight(
  details: GraphNode["details"],
  isDetailsExpanded: boolean,
  childSummaries: ChildSummary[],
  expertDiagram?: ExpertDiagram,
  stackDiagram?: StackDiagram,
) {
  const detailEntries = nodeDetailEntries(details);
  const hasMetadata = detailEntries.length > 0;
  const summaryHeight = stackDiagram
    ? STACK_DIAGRAM_HEIGHT
    : expertDiagram
      ? EXPERT_DIAGRAM_HEIGHT
      : childSummaryListHeight(childSummaries);
  const collapsedHeight = Math.max(
    hasMetadata ? COLLAPSED_NODE_HEIGHT_WITH_METADATA : COLLAPSED_NODE_HEIGHT_WITHOUT_METADATA,
    (hasMetadata
      ? COLLAPSED_NODE_STATIC_HEIGHT_WITH_METADATA
      : COLLAPSED_NODE_STATIC_HEIGHT_WITHOUT_METADATA) + summaryHeight,
  );

  if (!isDetailsExpanded || !hasMetadata) {
    return collapsedHeight;
  }

  return (
    collapsedHeight +
    DETAIL_LIST_EXTRA_HEIGHT +
    detailEntries.length * DETAIL_ROW_HEIGHT
  );
}

function humanizePathSegment(segment: string) {
  return segment
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function nodeTitle(node: GraphNode) {
  const pathParts = node.path.split(".");
  const pathSegment = pathParts[pathParts.length - 1] ?? node.path;
  const hasSemanticPathSegment = pathSegment.length > 0 && !/^\d+$/.test(pathSegment);
  if (SEMANTIC_LABEL_TYPE_NAMES.has(node.typeName) && hasSemanticPathSegment) {
    return humanizePathSegment(pathSegment);
  }
  return node.typeName;
}

export function nodeSubtitle(node: GraphNode) {
  const title = nodeTitle(node);
  return title === node.typeName ? node.path : `${node.typeName} · ${node.path}`;
}

function lastPathSegment(path: string) {
  const pathParts = path.split(".");
  return pathParts[pathParts.length - 1] ?? path;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

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
    return GATE_SUMMARY_LABEL;
  }
  if (pathSegment === "halting" || pathSegment === "halting_model") {
    return HALTING_SUMMARY_LABEL;
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
  child: GraphNode,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
): ChildSummary {
  if (child.typeName !== "Layer") {
    return { label: childSummaryLabel(child), kind: "child" };
  }

  const label = layerStackLabel(child);
  const nestedChild = (navigation.childrenById.get(child.id) ?? [])
    .map((childId) => nodesById.get(childId))
    .find((node): node is GraphNode => Boolean(node));

  if (!nestedChild) {
    return { label, kind: "child", stackKind: "layer" };
  }

  return {
    label,
    nestedLabel: childSummaryLabel(nestedChild),
    kind: "child",
    stackKind: "layer",
  };
}

function collapseLayerStackSummaries(childSummaries: ChildSummary[]) {
  const layerSummaries = childSummaries.filter((summary) => summary.stackKind === "layer");
  if (layerSummaries.length <= LAYER_STACK_SUMMARY_LIMIT) {
    return childSummaries;
  }

  let emittedLayerCount = 0;
  let insertedOverflow = false;

  return childSummaries.flatMap((summary) => {
    if (summary.stackKind !== "layer") {
      return [summary];
    }

    emittedLayerCount += 1;
    if (emittedLayerCount <= LAYER_STACK_VISIBLE_BEFORE_OVERFLOW) {
      return [summary];
    }

    if (insertedOverflow) {
      return [];
    }

    insertedOverflow = true;
    return [
      {
        label: "...",
        kind: "overflow" as const,
        title: `${layerSummaries.length - LAYER_STACK_VISIBLE_BEFORE_OVERFLOW} more layers`,
      },
      {
        label: `${layerSummaries.length} layers`,
        kind: "child" as const,
        title: `${layerSummaries.length} layers total`,
      },
    ];
  });
}

function directChildNodes(
  nodeId: string,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
) {
  return (navigation.childrenById.get(nodeId) ?? [])
    .map((childId) => nodesById.get(childId))
    .filter((node): node is GraphNode => Boolean(node));
}

function numericLastPathSegment(path: string) {
  const segment = lastPathSegment(path);
  if (!/^\d+$/.test(segment)) {
    return undefined;
  }
  return Number(segment);
}

function directNumericChildNodes(
  nodeId: string,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
) {
  return directNumericChildEntries(nodeId, navigation, nodesById).map((entry) => entry.node);
}

function directNumericChildEntries(
  nodeId: string,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
) {
  return directChildNodes(nodeId, navigation, nodesById)
    .map((node) => ({ node, index: numericLastPathSegment(node.path) }))
    .filter((entry): entry is { node: GraphNode; index: number } => entry.index !== undefined)
    .sort((left, right) => left.index - right.index);
}

function detailCount(details: GraphNode["details"], key: "numExperts") {
  const value = details[key];
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return undefined;
  }
  return Math.trunc(value);
}

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
    const childId = queue.shift()!;
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
  if (totalExperts <= EXPERT_DIAGRAM_LIMIT) {
    return Array.from({ length: totalExperts }, (_, index) => ({
      label: `E${index}`,
      title: `Expert ${index}`,
      kind: "expert" as const,
      expertIndex: index,
    }));
  }

  return [
    ...Array.from({ length: EXPERT_DIAGRAM_VISIBLE_BEFORE_OVERFLOW }, (_, index) => ({
      label: `E${index}`,
      title: `Expert ${index}`,
      kind: "expert" as const,
      expertIndex: index,
    })),
    {
      label: "...",
      title: `${totalExperts - EXPERT_DIAGRAM_VISIBLE_BEFORE_OVERFLOW} more experts`,
      kind: "overflow" as const,
    },
    {
      label: `${totalExperts} experts`,
      title: `${totalExperts} experts total`,
      kind: "total" as const,
    },
  ];
}

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

function stackLayerTitle(
  entry: { node: GraphNode; index: number },
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
) {
  const contentNode = primaryLayerContentNode(entry.node, navigation, nodesById);
  const layerType = contentNode ? nodeTitle(contentNode) : nodeTitle(entry.node);
  const dims =
    typeof entry.node.details.dims === "string"
      ? entry.node.details.dims
      : contentNode && typeof contentNode.details.dims === "string"
        ? contentNode.details.dims
        : undefined;

  return [`Layer ${entry.index}`, layerType, dims].filter(Boolean).join(" · ");
}

function stackLayerLabel(
  entry: { node: GraphNode; index: number },
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
) {
  const contentNode = primaryLayerContentNode(entry.node, navigation, nodesById);
  const layerType = contentNode ? nodeTitle(contentNode) : nodeTitle(entry.node);
  return `Layer ${entry.index} · ${layerType}`;
}

function createStackDiagramCells(
  entries: Array<{ node: GraphNode; index: number }>,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
): StackDiagramCell[] {
  const visibleEntries =
    entries.length > STACK_DIAGRAM_LIMIT
      ? entries.slice(0, STACK_DIAGRAM_VISIBLE_BEFORE_OVERFLOW)
      : entries;
  const layerCells = visibleEntries.map((entry) => ({
    label: stackLayerLabel(entry, navigation, nodesById),
    title: stackLayerTitle(entry, navigation, nodesById),
    kind: "layer" as const,
    layerIndex: entry.index,
  }));

  if (entries.length <= STACK_DIAGRAM_LIMIT) {
    return layerCells;
  }

  return [
    ...layerCells,
    {
      label: "...",
      title: `${entries.length - STACK_DIAGRAM_VISIBLE_BEFORE_OVERFLOW} more layers`,
      kind: "overflow" as const,
    },
    {
      label: `${entries.length} layers`,
      title: `${entries.length} layers total`,
      kind: "total" as const,
    },
  ];
}

function stackDiagramFromContainer(
  containerNode: GraphNode,
  navigation: GraphNavigation,
  nodesById: Map<string, GraphNode>,
) {
  const layerEntries = directNumericChildEntries(containerNode.id, navigation, nodesById);
  if (layerEntries.length === 0) {
    return undefined;
  }
  if (!layerEntries.every((entry) => isLayerLikeStackNode(entry.node))) {
    return undefined;
  }

  return {
    cells: createStackDiagramCells(layerEntries, navigation, nodesById),
    totalLayers: layerEntries.length,
    hasOverflow: layerEntries.length > STACK_DIAGRAM_LIMIT,
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

    if (!STACK_CONTAINER_TYPE_NAMES.has(node.typeName) || pathSegment === "expert_modules") {
      continue;
    }

    const diagram = stackDiagramFromContainer(node, navigation, nodesById);
    if (diagram) {
      diagramsById.set(node.id, diagram);
    }
  }

  return diagramsById;
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
        hasOverflow: totalExperts > EXPERT_DIAGRAM_LIMIT,
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
      hasOverflow: totalExperts > EXPERT_DIAGRAM_LIMIT,
    });
  }

  return diagramsById;
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
      childSummaries.push(childSummary(child, navigation, nodesById));
    }

    const labels = new Set(childSummaries.map((summary) => summary.label));

    if (detailFlag(node.details, "gate") && !labels.has(GATE_SUMMARY_LABEL)) {
      childSummaries.push({ label: GATE_SUMMARY_LABEL, kind: "mechanism" });
      labels.add(GATE_SUMMARY_LABEL);
    }
    if (detailFlag(node.details, "halting") && !labels.has(HALTING_SUMMARY_LABEL)) {
      childSummaries.push({ label: HALTING_SUMMARY_LABEL, kind: "mechanism" });
    }

    summariesById.set(node.id, collapseLayerStackSummaries(childSummaries));
  }

  return summariesById;
}

export function buildHierarchy(graph: InspectResponse | undefined) {
  if (!graph) {
    return [];
  }

  const nodesById = new Map(
    graph.nodes.map((node) => [node.id, { node, children: [] as HierarchyNode[] }]),
  );
  const childIds = new Set<string>();

  for (const edge of graph.edges) {
    const parent = nodesById.get(edge.source);
    const child = nodesById.get(edge.target);
    if (!parent || !child) {
      continue;
    }
    parent.children.push(child);
    childIds.add(edge.target);
  }

  return graph.nodes
    .filter((node) => !childIds.has(node.id))
    .map((node) => nodesById.get(node.id))
    .filter((node): node is HierarchyNode => Boolean(node));
}

export function buildGraphNavigation(graph: InspectResponse | undefined): GraphNavigation {
  const childrenById = new Map<string, string[]>();
  const childIds = new Set<string>();

  if (!graph) {
    return { childrenById, rootIds: new Set() };
  }

  for (const node of graph.nodes) {
    childrenById.set(node.id, []);
  }

  for (const edge of graph.edges) {
    if (!childrenById.has(edge.source) || !childrenById.has(edge.target)) {
      continue;
    }
    childrenById.get(edge.source)?.push(edge.target);
    childIds.add(edge.target);
  }

  const rootIds = new Set(
    graph.nodes
      .filter((node) => !childIds.has(node.id))
      .map((node) => node.id),
  );

  if (rootIds.size === 0 && graph.nodes[0]) {
    rootIds.add(graph.nodes[0].id);
  }

  return { childrenById, rootIds };
}

export function filterGraphByDetail(
  graph: InspectResponse | undefined,
  detailMode: GraphDetailMode,
): InspectResponse | undefined {
  if (!graph || detailMode === "full") {
    return graph;
  }

  const visibleNodeIds = new Set(
    graph.nodes
      .filter((node) => node.graphRole === "architecture")
      .map((node) => node.id),
  );
  const parentByChildId = new Map<string, string>();

  for (const edge of graph.edges) {
    if (!parentByChildId.has(edge.target)) {
      parentByChildId.set(edge.target, edge.source);
    }
  }

  const edges: InspectResponse["edges"] = [];
  const edgeIds = new Set<string>();

  for (const node of graph.nodes) {
    if (!visibleNodeIds.has(node.id)) {
      continue;
    }

    let ancestorId = parentByChildId.get(node.id);
    while (ancestorId && !visibleNodeIds.has(ancestorId)) {
      ancestorId = parentByChildId.get(ancestorId);
    }

    if (!ancestorId) {
      continue;
    }

    const edgeId = `${ancestorId}-${node.id}`;
    if (edgeIds.has(edgeId)) {
      continue;
    }
    edgeIds.add(edgeId);
    edges.push({
      id: edgeId,
      source: ancestorId,
      target: node.id,
    });
  }

  return {
    ...graph,
    nodes: graph.nodes.filter((node) => visibleNodeIds.has(node.id)),
    edges,
  };
}

export function filterGraphByExpansion(
  graph: InspectResponse | undefined,
  navigation: GraphNavigation,
  expandedNodeIds: Set<string>,
  scope: GraphScope,
): InspectResponse | undefined {
  if (!graph || scope === "entire") {
    return graph;
  }

  const visibleNodeIds = new Set<string>();

  const addVisibleNode = (nodeId: string) => {
    if (visibleNodeIds.has(nodeId)) {
      return;
    }
    visibleNodeIds.add(nodeId);

    const shouldShowChildren =
      navigation.rootIds.has(nodeId) || expandedNodeIds.has(nodeId);
    if (!shouldShowChildren) {
      return;
    }

    for (const childId of navigation.childrenById.get(nodeId) ?? []) {
      addVisibleNode(childId);
    }
  };

  for (const rootId of navigation.rootIds) {
    addVisibleNode(rootId);
  }

  return {
    ...graph,
    nodes: graph.nodes.filter((node) => visibleNodeIds.has(node.id)),
    edges: graph.edges.filter(
      (edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target),
    ),
  };
}

export function layoutGraph(
  graph: InspectResponse | undefined,
  options: {
    navigation: GraphNavigation;
    childSummariesById: Map<string, ChildSummary[]>;
    expertDiagramsById?: Map<string, ExpertDiagram>;
    stackDiagramsById?: Map<string, StackDiagram>;
    expandedGraphNodeIds: Set<string>;
    expandedDetailNodeIds: Set<string>;
    enableExpansion: boolean;
    selectedNodeId: string | null;
    onActivateNode: (nodeId: string) => void;
    onToggleDetails: (nodeId: string) => void;
  },
) {
  if (!graph) {
    return { nodes: [], edges: [] };
  }

  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({
    rankdir: "LR",
    nodesep: GRAPH_VERTICAL_CARD_GAP,
    ranksep: GRAPH_HORIZONTAL_CARD_GAP,
  });

  graph.nodes.forEach((node) => {
    const childSummaries = options.childSummariesById.get(node.id) ?? [];
    const expertDiagram = options.expertDiagramsById?.get(node.id);
    const stackDiagram = options.stackDiagramsById?.get(node.id);
    dagreGraph.setNode(node.id, {
      width: NODE_WIDTH,
      height: graphNodeHeight(
        node.details,
        options.expandedDetailNodeIds.has(node.id),
        childSummaries,
        expertDiagram,
        stackDiagram,
      ),
    });
  });
  graph.edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });
  dagre.layout(dagreGraph);

  const nodes: Array<Node<ViewerNodeData>> = graph.nodes.map((node) => {
    const position = dagreGraph.node(node.id);
    const childCount = options.navigation.childrenById.get(node.id)?.length ?? 0;
    const childSummaries = options.childSummariesById.get(node.id) ?? [];
    const expertDiagram = options.expertDiagramsById?.get(node.id);
    const stackDiagram = options.stackDiagramsById?.get(node.id);
    const isRoot = options.navigation.rootIds.has(node.id);
    const isExpanded = isRoot || options.expandedGraphNodeIds.has(node.id);
    const isDetailsExpanded = options.expandedDetailNodeIds.has(node.id);
    const height = graphNodeHeight(
      node.details,
      isDetailsExpanded,
      childSummaries,
      expertDiagram,
      stackDiagram,
    );
    const canToggleExpansion = options.enableExpansion && !isRoot && childCount > 0;

    return {
      id: node.id,
      type: "viewerNode",
      position: {
        x: position.x - NODE_WIDTH / 2,
        y: position.y - height / 2,
      },
      selected: options.selectedNodeId === node.id,
      style: { width: NODE_WIDTH, height },
      data: {
        nodeId: node.id,
        label: nodeTitle(node),
        subtitle: nodeSubtitle(node),
        path: node.path,
        parameterCount: node.parameterCount,
        details: node.details,
        childCount,
        childSummaries,
        expertDiagram,
        stackDiagram,
        height,
        isExpanded,
        canToggleExpansion,
        isDetailsExpanded,
        onActivateNode: () => options.onActivateNode(node.id),
        onToggleDetails: () => options.onToggleDetails(node.id),
      },
    };
  });

  const edges: Edge[] = graph.edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    markerEnd: { type: MarkerType.ArrowClosed, color: "#7c877c" },
    style: { stroke: "#7c877c", strokeWidth: 1.4 },
  }));

  return { nodes, edges };
}
