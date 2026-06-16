export type {
  ChildSummary,
  ClusterDiagram,
  ClusterDiagramCell,
  ClusterDiagramPlane,
  ClusterDiagramReach,
  ExpertDiagram,
  ExpertDiagramCell,
  GraphDetailMode,
  GraphKind,
  GraphNavigation,
  GraphParameterActivity,
  GraphParameterActivityChannel,
  GraphParameterActivitySource,
  GraphParameterActivityStatus,
  GraphScope,
  PreviewVisualizationMode,
  HierarchyNode,
  StackDiagram,
  StackDiagramCell,
  TerminalReachCell,
  TerminalReachGrid,
  TerminalReachLocationSummary,
  TerminalReachPlane,
  ViewerNodeData,
  OperationFlowNodeData,
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
  formatModelSize,
  graphNodeHeight,
  nodeDimsText,
  nodeDetailEntryText,
  nodeBadges,
  nodeDetailEntries,
  nodeSubtitle,
  nodeTitle,
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
export { buildChildSummaries } from "@/lib/graph/child-summaries";
export { buildExpertDiagrams } from "@/lib/graph/expert-diagrams";
export { buildStackDiagrams } from "@/lib/graph/stack-diagrams";
export { buildClusterDiagrams } from "@/lib/graph/cluster-diagrams";
export {
  buildTerminalReachGrid,
  parseTerminalReachDetails,
} from "@/lib/graph/terminal-reach";
export { buildGraphLocationSummaries } from "@/lib/graph/locations";
export { buildCluster3DSceneModel } from "@/lib/graph/cluster-3d";
// `layoutGraph` / `layoutOperationGraph` are intentionally NOT re-exported here:
// they pull in `dagre`, and re-exporting them keeps `dagre` in the first-load
// bundle even when unused. Consumers dynamically import the leaf layout modules
// instead. Only the dagre-free node-id prefix is re-exported.
export { OPERATION_GROUP_NODE_PREFIX } from "@/lib/graph/operation-graph-constants";
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
