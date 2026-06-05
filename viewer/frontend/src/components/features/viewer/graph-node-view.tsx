import { memo } from "react";
import { ChevronRight, LineChart } from "lucide-react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { GraphIconButton } from "@/components/features/viewer/graph/graph-icon-button";
import { GraphNodeDetails } from "@/components/features/viewer/graph/graph-node-details";
import { GraphNodeHeader } from "@/components/features/viewer/graph/graph-node-header";
import { GraphNodeChildSummaries } from "@/components/features/viewer/graph-node-child-summaries";
import {
  ClusterDiagramView,
  ExpertDiagramView,
  StackDiagramView,
} from "@/components/features/viewer/graph-node-diagrams";
import { GraphNodeParameterShapes } from "@/components/features/viewer/graph-node-parameter-shapes";
import {
  type ViewerNodeData,
  nodeDimsText,
  nodeDetailEntries,
  parameterShapeEntries,
  simpleGraphParamText,
} from "@/lib/graph";
import { cn } from "@/lib/utils";

const GraphNodeView = memo(function GraphNodeView({
  data,
  selected,
}: NodeProps<Node<ViewerNodeData>>) {
  const isSimpleMode = data.graphDetailMode === "simple";
  const entries = nodeDetailEntries(data.details, data.config);
  const parameterShapes = parameterShapeEntries(data.details);
  const hasMetadata = entries.length > 0;
  const simpleParamText = isSimpleMode ? simpleGraphParamText(data.parameterCount) : undefined;
  const simpleDimsText = isSimpleMode
    ? nodeDimsText(data.details, data.config) ?? data.stackDiagram?.dims
    : undefined;
  const detailToggleLabel = data.config ? "Config options" : "Details";
  const detailsId = `graph-node-details-${data.nodeId.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  const activateLabel = data.canToggleExpansion
    ? `Select and ${data.isExpanded ? "collapse" : "expand"} ${data.path}`
    : `Select ${data.path}`;
  const handleActivate = () => {
    data.onActivateNode();
  };
  const expansionButton = data.canToggleExpansion ? (
    <GraphIconButton
      className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[8px] border border-line bg-white/[0.03] text-ink-dim transition hover:border-white/20 hover:bg-white/[0.07] hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
      label={`${data.isExpanded ? "Collapse" : "Expand"} tree ${data.path}`}
      onClick={data.onToggleExpansion}
      icon={
        <ChevronRight
          className={cn("h-3.5 w-3.5 transition-transform", data.isExpanded && "rotate-90")}
          aria-hidden
        />
      }
    />
  ) : null;
  const monitorButton = data.canOpenMonitor && data.onOpenMonitor ? (
    <GraphIconButton
      className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[8px] border border-violet/25 bg-violet/10 text-[#cdbcff] transition hover:border-violet/45 hover:bg-violet/20 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
      label={`Open monitor charts for ${data.path}`}
      onClick={() => data.onOpenMonitor?.()}
      icon={<LineChart className="h-3.5 w-3.5" aria-hidden />}
    />
  ) : null;

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={activateLabel}
      aria-expanded={data.canToggleExpansion ? data.isExpanded : undefined}
      onClick={handleActivate}
      onKeyDown={(event) => {
        if (event.key !== "Enter" && event.key !== " ") {
          return;
        }
        event.preventDefault();
        handleActivate();
      }}
      className={cn(
        "nodrag nopan edge flex w-full flex-col overflow-hidden rounded-card px-8 pb-4 pt-4 shadow-[0_18px_40px_-28px_rgba(0,0,0,0.95)] transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
        selected ? "edge-sel" : "hover:brightness-110",
      )}
      style={{ height: data.height }}
    >
      <Handle type="target" position={Position.Left} />
      <GraphNodeHeader
        nodeId={data.nodeId}
        label={data.label}
        subtitle={data.subtitle}
        graphDetailMode={data.graphDetailMode}
        parameterCount={data.parameterCount}
        childCount={data.childCount}
        simpleParameterText={simpleParamText}
        simpleDimsText={simpleDimsText}
        expansionButton={expansionButton}
        monitorButton={monitorButton}
      />
      {!isSimpleMode && parameterShapes.length > 0 && (
        <GraphNodeParameterShapes nodeId={data.nodeId} entries={parameterShapes} />
      )}
      {!isSimpleMode && data.clusterDiagram ? (
        <ClusterDiagramView diagram={data.clusterDiagram} nodeId={data.nodeId} />
      ) : !isSimpleMode && data.stackDiagram ? (
        <StackDiagramView diagram={data.stackDiagram} nodeId={data.nodeId} />
      ) : !isSimpleMode && data.expertDiagram ? (
        <ExpertDiagramView diagram={data.expertDiagram} nodeId={data.nodeId} />
      ) : !isSimpleMode ? (
        <GraphNodeChildSummaries
          nodeId={data.nodeId}
          summaries={data.childSummaries}
        />
      ) : null}
      {!isSimpleMode && hasMetadata && (
        <GraphNodeDetails
          detailsId={detailsId}
          toggleLabel={detailToggleLabel}
          path={data.path}
          entries={entries}
          isExpanded={data.isDetailsExpanded}
          onToggleDetails={data.onToggleDetails}
        />
      )}
      <Handle type="source" position={Position.Right} />
    </div>
  );
});

export const nodeTypes = {
  viewerNode: GraphNodeView,
};
