export type {
  ChildSummary,
  ClusterDiagram,
  ClusterDiagramCell,
  ClusterDiagramPlane,
  ClusterDiagramReach,
  ExpertDiagram,
  ExpertDiagramCell,
  GraphDetailMode,
  GraphNavigation,
  GraphParameterActivity,
  GraphParameterActivityChannel,
  GraphParameterActivitySource,
  GraphParameterActivityStatus,
  GraphScope,
  HierarchyNode,
  StackDiagram,
  StackDiagramCell,
  TerminalReachCell,
  TerminalReachGrid,
  TerminalReachLocationSummary,
  TerminalReachPlane,
  WorkbenchNodeData,
  ClusterLocationSummary,
  GraphCoordinate,
  GraphLocationSummary,
} from "@/lib/graph/types";
export type {
  Cluster3DCell,
  Cluster3DCellCategory,
  Cluster3DCellSource,
  Cluster3DNodeMatch,
  Cluster3DReach,
  Cluster3DSceneModel,
} from "@/lib/graph/cluster-3d";
export type {
  LinearMonitorComparisonCandidateGroups,
  LinearMonitorComparisonScope,
  LinearMonitorTargetResolver,
  MonitorComparisonCandidateGroups,
  MonitorComparisonScope,
  MonitorName,
  MonitorTargetResolver,
  ResolvedMonitorTarget,
} from "@/lib/graph/monitor-targets";
export {
  configDetailText,
  detailText,
  formatCompactCount,
  formatExactCount,
  formatGraphCoordinate,
  formatModelSize,
  graphNodeHeight,
  nodeDimsText,
  nodeDetailEntryText,
  nodeBadges,
  nodeDetailEntries,
  nodeSubtitle,
  nodeTitle,
  parameterShapeDimsText,
  parameterShapeEntries,
  simpleGraphParamText,
  structureNodeLabel,
} from "@/lib/graph/formatting";
export {
  ancestorNodeIds,
  buildHierarchy,
  buildGraphNavigation,
  expandableSubtreeNodeIds,
} from "@/lib/graph/navigation";
export { filterGraphByDetail, filterGraphByExpansion } from "@/lib/graph/filtering";
export {
  decorateGraphSelection,
  deriveGraphDisplayModel,
  projectGraphDisplay,
} from "@/lib/graph/display";
export type {
  GraphDisplayCard,
  GraphDisplayModel,
  GraphDisplayProjection,
  ProjectGraphDisplayOptions,
} from "@/lib/graph/display";
export {
  collapseParameterActivityMinimapNodes,
  deriveParameterActivityMinimapModel,
  expandAllParameterActivityMinimapNodes,
  filterParameterActivityMinimapGraphByExpansion,
} from "@/lib/graph/parameter-activity-minimap";
export type {
  ParameterActivityMinimapInput,
  ParameterActivityMinimapModel,
} from "@/lib/graph/parameter-activity-minimap";
export { buildChildSummaries } from "@/lib/graph/child-summaries";
export { buildExpertDiagrams } from "@/lib/graph/expert-diagrams";
export { buildStackDiagrams } from "@/lib/graph/stack-diagrams";
export { buildClusterDiagrams } from "@/lib/graph/cluster-diagrams";
export {
  buildTerminalReachGrid,
  parseTerminalReachDetails,
} from "@/lib/graph/terminal-reach";
export {
  buildClusterLocationSummary,
  buildGraphLocationSummaries,
} from "@/lib/graph/locations";
export { buildCluster3DSceneModel } from "@/lib/graph/cluster-3d";
// `layoutGraph` is intentionally NOT re-exported here: it pulls in `dagre`, and
// re-exporting it keeps `dagre` in the first-load bundle even when unused.
// Consumers dynamically import the leaf layout module instead.
export {
  buildMonitorComparisonCandidateGroups,
  buildMonitorComparisonCandidates,
  buildLinearMonitorComparisonCandidateGroups,
  buildLinearMonitorComparisonCandidates,
  createMonitorTargetNodeResolver,
  createMonitorTargetResolver,
  createLinearMonitorTargetResolver,
  resolveMonitorTarget,
  resolveLinearMonitorTarget,
} from "@/lib/graph/monitor-targets";
