import { createContext, memo, useContext, type ReactNode } from "react";
import { ChevronRight, LineChart } from "lucide-react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { GraphIconButton } from "@/features/viewer/components/graph/graph-icon-button";
import { GraphNodeDetails } from "@/features/viewer/components/graph/graph-node-details";
import { GraphNodeHeader } from "@/features/viewer/components/graph/graph-node-header";
import { GraphParameterIndicators } from "@/features/viewer/components/graph/graph-parameter-indicators";
import { GraphNodeChildSummaries } from "@/features/viewer/components/graph/graph-node-child-summaries";
import {
  ClusterDiagramView,
  ExpertDiagramView,
  StackDiagramView,
} from "@/features/viewer/components/graph/graph-node-diagrams";
import { GraphNodeParameterShapes } from "@/features/viewer/components/graph/graph-node-parameter-shapes";
import { OperationGraphNodeView } from "@/features/viewer/components/graph/operation-graph-node-view";
import {
  type ViewerNodeData,
  formatModelSize,
  nodeDimsText,
  nodeDetailEntries,
  parameterShapeEntries,
  simpleGraphParamText,
} from "@/lib/graph";
import { cn } from "@/lib/utils";

const GraphNodeViewportMovingContext = createContext(false);

export function GraphNodeRenderModeProvider({
  children,
  isViewportMoving,
}: {
  children: ReactNode;
  isViewportMoving: boolean;
}) {
  return (
    <GraphNodeViewportMovingContext.Provider value={isViewportMoving}>
      {children}
    </GraphNodeViewportMovingContext.Provider>
  );
}

const GraphNodeView = memo(function GraphNodeView({
  data,
  selected,
}: NodeProps<Node<ViewerNodeData>>) {
  const isViewportMoving = useContext(GraphNodeViewportMovingContext);
  const isSimpleMode = data.graphDetailMode === "simple";
  const simpleParamText = isSimpleMode ? simpleGraphParamText(data.parameterCount) : undefined;
  const modelSizeText = data.isRootNode ? formatModelSize(data.parameterSizeBytes) : undefined;
  const simpleDimsText = isSimpleMode
    ? nodeDimsText(data.details, data.config) ?? data.stackDiagram?.dims
    : undefined;
  const activateLabel = data.canToggleExpansion
    ? `Select and ${data.isExpanded ? "collapse" : "expand"} ${data.path}`
    : `Select ${data.path}`;
  const handleActivate = () => {
    data.onActivateNode();
  };

  if (isViewportMoving) {
    return (
      <MovingGraphNodeShell
        data={data}
        selected={selected}
        activateLabel={activateLabel}
        onActivate={handleActivate}
        modelSizeText={modelSizeText}
        simpleParamText={simpleParamText}
        simpleDimsText={simpleDimsText}
      />
    );
  }

  const entries = nodeDetailEntries(data.details, data.config);
  const parameterShapes = parameterShapeEntries(data.details);
  const hasMetadata = entries.length > 0;
  const detailToggleLabel = data.config ? "Config options" : "Details";
  const detailsId = `graph-node-details-${data.nodeId.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
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
      className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[8px] border border-violet/25 bg-violet/10 text-violet-muted transition hover:border-violet/45 hover:bg-violet/20 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
      label={`Open monitor charts for ${data.path}`}
      onClick={() => data.onOpenMonitor?.()}
      icon={<LineChart className="h-3.5 w-3.5" aria-hidden />}
    />
  ) : null;
  const parameterIndicators = (
    <GraphParameterIndicators activity={data.parameterActivity} />
  );

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
        parameterSizeBytes={data.parameterSizeBytes}
        modelSizeText={modelSizeText}
        childCount={data.childCount}
        simpleParameterText={simpleParamText}
        simpleDimsText={simpleDimsText}
        expansionButton={expansionButton}
        parameterIndicators={parameterIndicators}
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

function MovingGraphNodeShell({
  data,
  selected,
  activateLabel,
  onActivate,
  modelSizeText,
  simpleParamText,
  simpleDimsText,
}: {
  data: ViewerNodeData;
  selected: boolean;
  activateLabel: string;
  onActivate: () => void;
  modelSizeText?: string;
  simpleParamText?: string;
  simpleDimsText?: string;
}) {
  const parameterText = simpleParamText ?? simpleGraphParamText(data.parameterCount);

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={activateLabel}
      aria-expanded={data.canToggleExpansion ? data.isExpanded : undefined}
      data-testid={`graph-node-moving-${data.nodeId}`}
      onClick={onActivate}
      onKeyDown={(event) => {
        if (event.key !== "Enter" && event.key !== " ") {
          return;
        }
        event.preventDefault();
        onActivate();
      }}
      className={cn(
        "nodrag nopan edge flex w-full flex-col justify-center overflow-hidden rounded-card px-5 py-4 transition-none focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
        selected && "edge-sel",
      )}
      style={{ height: data.height }}
    >
      <Handle type="target" position={Position.Left} />
      <div className="min-w-0 truncate text-[16px] font-bold leading-5 text-ink">
        {data.label}
      </div>
      <div className="mt-1 min-w-0 truncate font-mono text-[11px] leading-4 text-ink-dim">
        {data.path}
      </div>
      <div className="mt-2 flex min-w-0 flex-wrap items-center gap-1.5 overflow-hidden font-mono text-[10px] font-semibold text-ink-faint">
        {parameterText && <span className="shrink-0">{parameterText}</span>}
        {modelSizeText && <span className="shrink-0">{modelSizeText}</span>}
        {simpleDimsText && <span className="shrink-0">{simpleDimsText}</span>}
        {data.childCount > 0 && (
          <span className="shrink-0">
            {data.childCount} {data.childCount === 1 ? "child" : "children"}
          </span>
        )}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

export const nodeTypes = {
  viewerNode: GraphNodeView,
  operationNode: OperationGraphNodeView,
};
