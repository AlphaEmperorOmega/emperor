export const graphCardGeometry = {
  width: 360,
  simpleHeight: 118,
  paddingBlock: 16,
  titleLineHeight: 24,
  subtitle: {
    marginBlockStart: 6,
    height: 20,
  },
  actionBar: {
    marginBlockStart: 6,
    height: 32,
  },
  contentMarginBlockStart: 8,
  details: {
    marginBlockStart: 12,
    rowGap: 4,
    rowHeight: 32,
  },
  childSummary: {
    rowGap: 8,
    rowHeight: 36,
  },
  parameterShapes: {
    marginBlockStart: 8,
    rowGap: 4,
    rowHeight: 28,
  },
  expertDiagram: {
    height: 104,
  },
  clusterDiagram: {
    cellSize: 20,
    cellGap: 4,
    headerHeight: 52,
  },
  layout: {
    nodeGap: 24,
    rankGap: 72,
  },
} as const;

export const parameterActivityMinimapGeometry = {
  activityNodeWidth: 104,
  branchNodeWidth: 42,
  nodeHeight: 42,
  nodeGap: graphCardGeometry.layout.nodeGap,
  rankGap: Math.round(
    (graphCardGeometry.layout.rankGap / graphCardGeometry.width) * 104,
  ),
} as const;

export const graphDiagramLimits = {
  expert: {
    total: 7,
    visibleBeforeOverflow: 5,
  },
  stack: {
    total: 3,
    visibleBeforeOverflow: 2,
  },
  layerSummary: {
    total: 3,
    visibleBeforeOverflow: 2,
  },
  cluster: {
    maxDimension: 8,
    maxPlanes: 4,
  },
} as const;

export const graphDisplayTypeNames = {
  semanticLabels: new Set(["LayerStack", "ModuleList", "Sequential"]),
  stackContainers: new Set(["LayerStack", "ModuleList", "Sequential"]),
} as const;

export const graphDisplayLabels = {
  gateSummary: "Gate",
  haltingSummary: "Halting mechanism",
} as const;
