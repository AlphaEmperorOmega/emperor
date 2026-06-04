import { memo } from "react";
import { ChevronRight, LineChart } from "lucide-react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { Badge } from "@/components/ui/badge";
import { GraphNodeChildSummaries } from "@/components/features/viewer/graph-node-child-summaries";
import {
  ClusterDiagramView,
  ExpertDiagramView,
  StackDiagramView,
} from "@/components/features/viewer/graph-node-diagrams";
import { GraphNodeParameterShapes } from "@/components/features/viewer/graph-node-parameter-shapes";
import {
  type ViewerNodeData,
  formatCompactCount,
  formatExactCount,
  nodeDimsText,
  nodeDetailEntryText,
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
  const isBasicMode = data.graphDetailMode === "basic";
  const entries = nodeDetailEntries(data.details, data.config);
  const parameterShapes = parameterShapeEntries(data.details);
  const hasMetadata = entries.length > 0;
  const hasParameters = data.parameterCount > 0;
  const simpleParamText = isSimpleMode ? simpleGraphParamText(data.parameterCount) : undefined;
  const simpleDimsText = isSimpleMode
    ? nodeDimsText(data.details, data.config) ?? data.stackDiagram?.dims
    : undefined;
  const hasGraphBadges = hasParameters || data.childCount > 0;
  const detailToggleLabel = data.config ? "Config options" : "Details";
  const detailsId = `graph-node-details-${data.nodeId.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  const activateLabel = data.canToggleExpansion
    ? `Select and ${data.isExpanded ? "collapse" : "expand"} ${data.path}`
    : `Select ${data.path}`;
  const handleActivate = () => {
    data.onActivateNode();
  };
  const expansionButton = data.canToggleExpansion ? (
    <button
      type="button"
      className="nodrag nopan flex h-7 w-7 shrink-0 items-center justify-center rounded-[8px] border border-line bg-white/[0.03] text-ink-dim transition hover:border-white/20 hover:bg-white/[0.07] hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
      title={`${data.isExpanded ? "Collapse" : "Expand"} tree ${data.path}`}
      aria-label={`${data.isExpanded ? "Collapse" : "Expand"} tree ${data.path}`}
      onClick={(event) => {
        event.stopPropagation();
        data.onToggleExpansion();
      }}
      onKeyDown={(event) => {
        event.stopPropagation();
      }}
    >
      <ChevronRight
        className={cn("h-3.5 w-3.5 transition-transform", data.isExpanded && "rotate-90")}
        aria-hidden
      />
    </button>
  ) : null;
  const monitorButton = data.canOpenMonitor && data.onOpenMonitor ? (
    <button
      type="button"
      className="nodrag nopan flex h-7 w-7 shrink-0 items-center justify-center rounded-[8px] border border-violet/25 bg-violet/10 text-[#cdbcff] transition hover:border-violet/45 hover:bg-violet/20 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
      title={`Open monitor charts for ${data.path}`}
      aria-label={`Open monitor charts for ${data.path}`}
      onClick={(event) => {
        event.stopPropagation();
        data.onOpenMonitor?.();
      }}
      onKeyDown={(event) => {
        event.stopPropagation();
      }}
    >
      <LineChart className="h-3.5 w-3.5" aria-hidden />
    </button>
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
      {isSimpleMode ? (
        <div className="flex shrink-0 items-center gap-2">
          {expansionButton}
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 flex-nowrap items-center gap-2">
              <div className="min-w-0 flex-1 truncate text-[18px] font-bold leading-6 text-ink">
                {data.label}
              </div>
              {simpleParamText && (
                <Badge
                  className="shrink-0 whitespace-nowrap border-violet/25 bg-violet/15 px-1.5 py-0.5 font-mono text-[11px] leading-none text-[#cdbcff] [overflow-wrap:normal]"
                  title={`${formatExactCount(data.parameterCount)} parameters`}
                >
                  {simpleParamText}
                </Badge>
              )}
              {simpleDimsText && (
                <Badge
                  className="shrink-0 whitespace-nowrap border-line bg-white/[0.04] px-1.5 py-0.5 font-mono text-[11px] leading-none text-ink-dim [overflow-wrap:normal]"
                  title={`input/output: ${simpleDimsText}`}
                >
                  {simpleDimsText}
                </Badge>
              )}
            </div>
          </div>
          {monitorButton}
        </div>
      ) : (
        <div className="shrink-0">
          <div
            className="flex min-w-0 items-start gap-2"
            data-testid={`graph-node-title-row-${data.nodeId}`}
          >
            {expansionButton}
            <div className="flex min-w-0 flex-1 flex-nowrap items-center gap-1.5">
              <div className="min-w-0 flex-1 truncate text-[18px] font-bold leading-6 text-ink">
                {data.label}
              </div>
              {isBasicMode && hasParameters && (
                <Badge
                  className="h-6 shrink-0 items-center whitespace-nowrap border-violet/25 bg-violet/15 px-2 py-0 font-mono text-xs leading-none text-[#cdbcff] [overflow-wrap:normal]"
                  title={`${formatExactCount(data.parameterCount)} parameters`}
                >
                  {formatCompactCount(data.parameterCount)}
                </Badge>
              )}
              {isBasicMode && data.childCount > 0 && (
                <Badge className="h-6 shrink-0 items-center whitespace-nowrap border-line bg-white/[0.04] px-2 py-0 font-sans text-xs font-medium leading-none text-ink-dim [overflow-wrap:normal]">
                  {data.childCount} {data.childCount === 1 ? "child" : "children"}
                </Badge>
              )}
            </div>
            {monitorButton}
          </div>
          {!isBasicMode && hasGraphBadges && (
            <div
              className="mt-1 flex h-6 min-w-0 items-center gap-1.5 overflow-hidden"
              data-testid={`graph-node-badges-${data.nodeId}`}
            >
              {hasParameters && (
                <Badge
                  className="h-6 shrink-0 items-center whitespace-nowrap border-violet/25 bg-violet/15 px-2 py-0 font-mono text-xs leading-none text-[#cdbcff] [overflow-wrap:normal]"
                  title={`${formatExactCount(data.parameterCount)} parameters`}
                >
                  {formatCompactCount(data.parameterCount)}
                </Badge>
              )}
              {data.childCount > 0 && (
                <Badge className="h-6 shrink-0 items-center whitespace-nowrap border-line bg-white/[0.04] px-2 py-0 font-sans text-xs font-medium leading-none text-ink-dim [overflow-wrap:normal]">
                  {data.childCount} {data.childCount === 1 ? "child" : "children"}
                </Badge>
              )}
            </div>
          )}
          <div className="mt-1.5 truncate font-mono text-[13px] leading-5 text-ink-dim">
            {data.subtitle}
          </div>
        </div>
      )}
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
        <div className="mt-3 shrink-0 border-t border-line-soft pt-3">
          <button
            type="button"
            aria-expanded={data.isDetailsExpanded}
            aria-controls={detailsId}
            aria-label={`${detailToggleLabel} for ${data.path}`}
            onClick={(event) => {
              event.stopPropagation();
              data.onToggleDetails();
            }}
            onKeyDown={(event) => {
              event.stopPropagation();
            }}
            className="nodrag nopan flex h-9 w-full items-center justify-between gap-2 rounded-[10px] border border-line-soft bg-white/[0.015] px-3 text-left text-sm font-semibold text-ink-dim transition hover:border-line hover:bg-white/[0.04] hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          >
            <span>{detailToggleLabel}</span>
            <ChevronRight
              className={cn("h-3 w-3 transition-transform", data.isDetailsExpanded && "rotate-90")}
              aria-hidden
            />
          </button>
          {data.isDetailsExpanded && (
            <div id={detailsId} className="mt-2 grid gap-1">
              {entries.map((entry) => (
                <div
                  key={entry.key}
                  className="grid h-8 grid-cols-[96px_minmax(0,1fr)] items-center gap-2 rounded-[8px] border border-line-soft bg-black/20 px-2.5 text-[12.5px]"
                >
                  <span className="truncate font-medium text-ink-dim">{entry.key}</span>
                  <span className="truncate font-mono text-ink">
                    {nodeDetailEntryText(entry)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      <Handle type="source" position={Position.Right} />
    </div>
  );
});

export const nodeTypes = {
  viewerNode: GraphNodeView,
};
