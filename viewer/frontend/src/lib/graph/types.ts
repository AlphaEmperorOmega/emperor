import { type GraphNode } from "@/lib/api";

export type GraphDetailMode = "simple" | "basic" | "full";
export type GraphScope = "opened" | "entire";
export type GraphCoordinate = [number, number, number];
export type GraphParameterActivityStatus =
  | "updated"
  | "unchanged"
  | "missing"
  | "unknown"
  | "mixed";
export type GraphParameterActivitySource = "active-job" | "historical";

export type GraphParameterActivityChannel = {
  status: GraphParameterActivityStatus;
  source: GraphParameterActivitySource;
  sourceLabel: string;
  metric?: string | null;
  lastStep?: number | null;
  observedPoints: number;
  updatedRuns?: number;
  unchangedRuns?: number;
  missingRuns?: number;
  unknownRuns?: number;
  totalRuns?: number;
};

export type GraphParameterActivity = {
  targetPath: string;
  weights: GraphParameterActivityChannel;
  bias?: GraphParameterActivityChannel;
};

export type ViewerNodeData = {
  nodeId: string;
  label: string;
  subtitle: string;
  path: string;
  parameterCount: number;
  details: GraphNode["details"];
  config: GraphNode["config"];
  childCount: number;
  childSummaries: ChildSummary[];
  expertDiagram?: ExpertDiagram;
  stackDiagram?: StackDiagram;
  clusterDiagram?: ClusterDiagram;
  graphDetailMode: GraphDetailMode;
  height: number;
  isExpanded: boolean;
  canToggleExpansion: boolean;
  canOpenMonitor?: boolean;
  parameterActivity?: GraphParameterActivity;
  isDetailsExpanded: boolean;
  onActivateNode: () => void;
  onToggleExpansion: () => void;
  onOpenMonitor?: () => void;
  onToggleDetails: () => void;
};

export type HierarchyNode = {
  node: GraphNode;
  children: HierarchyNode[];
};

export type GraphNavigation = {
  childrenById: Map<string, string[]>;
  parentById: Map<string, string>;
  rootIds: Set<string>;
};

export type ChildSummary = {
  label: string;
  nestedLabel?: string;
  dims?: string;
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
  dims?: string;
  kind: "layer" | "overflow" | "total";
  layerIndex?: number;
};

export type StackDiagram = {
  cells: StackDiagramCell[];
  dims?: string;
  totalLayers: number;
  hasOverflow: boolean;
};

export type ClusterDiagramCell = {
  x: number;
  y: number;
  filled: boolean;
  title: string;
};

export type ClusterDiagramPlane = {
  z: number;
  cells: ClusterDiagramCell[];
};

export type ClusterDiagram = {
  columns: number;
  rows: number;
  planes: ClusterDiagramPlane[];
  instantiated: number;
  capacityTotal: number;
  maxSteps: number | null;
  growthThreshold: number | null;
  hasColumnOverflow: boolean;
  hasRowOverflow: boolean;
  hasPlaneOverflow: boolean;
};

export type TerminalReachCell = {
  x: number;
  y: number;
  kind: "self" | "reach" | "empty";
  title: string;
};

export type TerminalReachPlane = {
  z: number;
  cells: TerminalReachCell[];
};

export type TerminalReachGrid = {
  columns: number;
  rows: number;
  minX: number;
  minY: number;
  position: GraphCoordinate;
  planes: TerminalReachPlane[];
  total: number;
  hasOverflow: boolean;
};

export type ClusterLocationSummary = {
  kind: "cluster";
  nodeId: string;
  nodePath: string;
  nodeLabel: string;
  nodeType: string;
  coordinates: GraphCoordinate[];
  instantiated: number;
  capacityTotal: number;
  hasOverflow: boolean;
};

export type TerminalReachLocationSummary = {
  kind: "terminalReach";
  nodeId: string;
  nodePath: string;
  nodeLabel: string;
  nodeType: string;
  position: GraphCoordinate;
  connections: GraphCoordinate[];
  total: number;
  hasOverflow: boolean;
};

export type GraphLocationSummary =
  | ClusterLocationSummary
  | TerminalReachLocationSummary;
