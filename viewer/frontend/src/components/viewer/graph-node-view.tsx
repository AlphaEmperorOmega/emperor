import { ChevronRight } from "lucide-react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { Badge } from "@/components/ui/badge";
import {
  type ExpertDiagram,
  type StackDiagram,
  type ViewerNodeData,
  detailText,
  formatCompactCount,
  formatExactCount,
  nodeDetailEntries,
} from "@/lib/graph";
import { cn } from "@/lib/utils";

const EXPERT_DIAGRAM_WIDTH = 196;
const EXPERT_DIAGRAM_GAP = 3;
const EXPERT_DIAGRAM_OVERFLOW_WIDTH = 20;
const EXPERT_DIAGRAM_TOTAL_WIDTH = 58;
const EXPERT_DIAGRAM_VISIBLE_BEFORE_OVERFLOW = 5;
const STACK_DIAGRAM_WIDTH = EXPERT_DIAGRAM_WIDTH;
const STACK_DIAGRAM_HEIGHT = 82;
const STACK_DIAGRAM_CELL_WIDTH = 116;

function diagramCellWidths({
  hasOverflow,
  cellsLength,
  overflowWidth,
  totalWidth,
  visibleBeforeOverflow,
  width,
  gap,
}: {
  hasOverflow: boolean;
  cellsLength: number;
  overflowWidth: number;
  totalWidth: number;
  visibleBeforeOverflow: number;
  width: number;
  gap: number;
}) {
  if (!hasOverflow) {
    const cellWidth = (width - (cellsLength - 1) * gap) / cellsLength;
    return Array.from({ length: cellsLength }, () => cellWidth);
  }

  const regularCellWidth =
    (width - (cellsLength - 1) * gap - overflowWidth - totalWidth) /
    visibleBeforeOverflow;
  return [
    ...Array.from({ length: visibleBeforeOverflow }, () => regularCellWidth),
    overflowWidth,
    totalWidth,
  ];
}

function expertDiagramCellCenters(diagram: ExpertDiagram) {
  const widths = diagramCellWidths({
    hasOverflow: diagram.hasOverflow,
    cellsLength: diagram.cells.length,
    overflowWidth: EXPERT_DIAGRAM_OVERFLOW_WIDTH,
    totalWidth: EXPERT_DIAGRAM_TOTAL_WIDTH,
    visibleBeforeOverflow: EXPERT_DIAGRAM_VISIBLE_BEFORE_OVERFLOW,
    width: EXPERT_DIAGRAM_WIDTH,
    gap: EXPERT_DIAGRAM_GAP,
  });
  let offset = 0;
  return widths.map((width) => {
    const center = offset + width / 2;
    offset += width + EXPERT_DIAGRAM_GAP;
    return center;
  });
}

function stackDiagramCellMetrics(diagram: StackDiagram) {
  const cellHeight = diagram.cells.length > 5 ? 10 : diagram.cells.length > 4 ? 13 : 16;
  const gap = diagram.cells.length > 5 ? 2 : diagram.cells.length > 4 ? 3 : 4;
  const totalHeight = diagram.cells.length * cellHeight + (diagram.cells.length - 1) * gap;
  let offset = (STACK_DIAGRAM_HEIGHT - totalHeight) / 2;
  return diagram.cells.map(() => {
    const metrics = {
      top: offset,
      center: offset + cellHeight / 2,
      height: cellHeight,
    };
    offset += cellHeight + gap;
    return metrics;
  });
}

function ExpertDiagramView({
  diagram,
  nodeId,
}: {
  diagram: ExpertDiagram;
  nodeId: string;
}) {
  const cellCenters = expertDiagramCellCenters(diagram);
  const gridTemplateColumns = diagram.hasOverflow
    ? `repeat(${EXPERT_DIAGRAM_VISIBLE_BEFORE_OVERFLOW}, minmax(0, 1fr)) ${EXPERT_DIAGRAM_OVERFLOW_WIDTH}px ${EXPERT_DIAGRAM_TOTAL_WIDTH}px`
    : `repeat(${diagram.cells.length}, minmax(0, 1fr))`;

  return (
    <div
      className="mt-2 h-[82px] shrink-0"
      data-testid={`expert-diagram-${nodeId}`}
      aria-label={`Expert routing diagram for ${nodeId}`}
    >
      <div className="relative h-full w-full">
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full text-[#9aa79f]"
          viewBox={`0 0 ${EXPERT_DIAGRAM_WIDTH} 82`}
          fill="none"
          aria-hidden
        >
          {cellCenters.map((x, index) => (
            <path
              key={`${diagram.cells[index]?.label ?? index}-${x}`}
              d={`M 98 58 C 98 48 ${x.toFixed(2)} 42 ${x.toFixed(2)} 25`}
              stroke="currentColor"
              strokeWidth="1.1"
              strokeLinecap="round"
            />
          ))}
        </svg>
        <div
          className="absolute left-0 top-0 grid w-full gap-[3px]"
          style={{ gridTemplateColumns }}
        >
          {diagram.cells.map((cell) => (
            <div
              key={`${cell.kind}-${cell.label}`}
              title={cell.title}
              className={cn(
                "flex h-6 min-w-0 items-center justify-center rounded border px-1 text-center text-[10px] font-semibold leading-none",
                cell.kind === "expert"
                  ? "border-accent-edge bg-accent-soft font-mono text-accent shadow-[inset_0_-1px_0_#b9cfc7]"
                  : cell.kind === "overflow"
                    ? "border-subtle bg-panel font-mono text-muted shadow-[inset_0_-1px_0_#d8ded9]"
                    : "border-subtle bg-surface text-[9px] text-muted shadow-[inset_0_-1px_0_#d8ded9]",
              )}
            >
              <span className="truncate">{cell.label}</span>
            </div>
          ))}
        </div>
        <div
          title={diagram.samplerTitle}
          className="absolute bottom-0 left-1/2 flex h-6 w-[92px] -translate-x-1/2 items-center justify-center rounded border border-border bg-surface px-2 text-[10px] font-semibold leading-none text-ink shadow-[inset_0_-1px_0_#d8ded9]"
        >
          <span className="truncate">{diagram.samplerLabel}</span>
        </div>
      </div>
    </div>
  );
}

function StackDiagramView({
  diagram,
  nodeId,
}: {
  diagram: StackDiagram;
  nodeId: string;
}) {
  const cellMetrics = stackDiagramCellMetrics(diagram);
  const firstCell = cellMetrics[0];
  const lastCell = cellMetrics[cellMetrics.length - 1];
  const cellLeft = (STACK_DIAGRAM_WIDTH - STACK_DIAGRAM_CELL_WIDTH) / 2;
  const railX = cellLeft - 16;
  const compactText = diagram.cells.length > 5;

  return (
    <div
      className="mt-2 h-[82px] shrink-0"
      data-testid={`stack-diagram-${nodeId}`}
      aria-label={`Layer stack diagram for ${nodeId}`}
    >
      <div className="relative h-full w-full">
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full text-[#9aa79f]"
          viewBox={`0 0 ${STACK_DIAGRAM_WIDTH} 82`}
          fill="none"
          aria-hidden
        >
          {firstCell && lastCell && (
            <path
              d={`M ${railX} ${firstCell.center.toFixed(2)} L ${railX} ${lastCell.center.toFixed(2)}`}
              stroke="currentColor"
              strokeWidth="1.2"
              strokeLinecap="round"
            />
          )}
          {cellMetrics.map((cell, index) => (
            <path
              key={`${diagram.cells[index]?.label ?? index}-tick`}
              d={`M ${railX} ${cell.center.toFixed(2)} L ${(cellLeft - 2).toFixed(2)} ${cell.center.toFixed(2)}`}
              stroke="currentColor"
              strokeWidth="1.1"
              strokeLinecap="round"
            />
          ))}
        </svg>
        {diagram.cells.map((cell, index) => {
          const metrics = cellMetrics[index];

          return (
            <div
              key={`${cell.kind}-${cell.label}`}
              title={cell.title}
              style={{
                height: metrics.height,
                left: cellLeft,
                top: metrics.top,
                width: STACK_DIAGRAM_CELL_WIDTH,
              }}
              className={cn(
                "absolute flex min-w-0 items-center justify-center rounded border px-1 text-center font-semibold leading-none",
                compactText ? "text-[8px]" : "text-[10px]",
                cell.kind === "layer"
                  ? "border-accent-edge bg-accent-soft font-mono text-accent shadow-[inset_0_-1px_0_#b9cfc7]"
                  : cell.kind === "overflow"
                    ? "border-subtle bg-panel font-mono text-muted shadow-[inset_0_-1px_0_#d8ded9]"
                    : "border-subtle bg-surface text-[9px] text-muted shadow-[inset_0_-1px_0_#d8ded9]",
              )}
            >
              <span className="truncate">{cell.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function GraphNodeView({ data, selected }: NodeProps<Node<ViewerNodeData>>) {
  const entries = nodeDetailEntries(data.details);
  const hasMetadata = entries.length > 0;
  const hasParameters = data.parameterCount > 0;
  const detailsId = `graph-node-details-${data.nodeId.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  const activateLabel = data.canToggleExpansion
    ? `${data.isExpanded ? "Select and collapse" : "Select and expand"} ${data.path}`
    : `Select ${data.path}`;
  const handleActivate = () => {
    data.onActivateNode();
  };

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
        "flex w-[220px] flex-col overflow-hidden rounded-md border bg-panel px-3 pb-4 pt-3 shadow-panel transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
        selected
          ? "border-accent ring-2 ring-[#15705f24]"
          : "border-border hover:border-[#aeb8b2]",
      )}
      style={{ height: data.height }}
    >
      <Handle type="target" position={Position.Left} />
      <div className="flex shrink-0 items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-1.5">
            <div className="min-w-0 truncate text-sm font-semibold text-ink">
              {data.label}
            </div>
            {hasParameters && (
              <Badge
                className="shrink-0 border-accent-edge bg-accent-soft px-1 py-0.5 font-mono text-[10px] text-accent"
                title={`${formatExactCount(data.parameterCount)} parameters`}
              >
                {formatCompactCount(data.parameterCount)}
              </Badge>
            )}
            {data.childCount > 0 && (
              <Badge className="shrink-0 px-1 py-0.5 text-[10px]">
                {data.childCount} {data.childCount === 1 ? "child" : "children"}
              </Badge>
            )}
          </div>
          <div className="mt-1 truncate font-mono text-[11px] text-muted">{data.subtitle}</div>
        </div>
        {data.canToggleExpansion && (
          <span
            className="flex h-6 w-6 shrink-0 items-center justify-center rounded border border-border bg-surface text-muted"
            title={`${data.isExpanded ? "Collapse" : "Expand"} ${data.path}`}
            aria-hidden
          >
            <ChevronRight
              className={cn("h-3.5 w-3.5 transition-transform", data.isExpanded && "rotate-90")}
              aria-hidden
            />
          </span>
        )}
      </div>
      {data.stackDiagram ? (
        <StackDiagramView diagram={data.stackDiagram} nodeId={data.nodeId} />
      ) : data.expertDiagram ? (
        <ExpertDiagramView diagram={data.expertDiagram} nodeId={data.nodeId} />
      ) : (
        <div
          className="mt-2 grid min-h-5 shrink-0 content-start gap-1"
          data-testid={`child-summaries-${data.nodeId}`}
        >
          {data.childSummaries.map((summary, index) => {
            const summaryLabel = summary.nestedLabel
              ? `${summary.label} ${summary.nestedLabel}`
              : summary.label;
            const summaryTitle = summary.title ?? summaryLabel;

            return (
              <div
                key={`${summary.kind}-${summary.label}-${index}`}
                aria-label={summaryLabel}
                title={summaryTitle}
                className={cn(
                  "relative flex h-6 items-center overflow-hidden rounded-md border px-2 text-[11px] font-medium leading-none",
                  summary.kind === "mechanism"
                    ? "border-accent-edge bg-accent-soft text-accent shadow-[inset_0_-1px_0_#b9cfc7]"
                    : summary.kind === "overflow"
                      ? "border-subtle bg-panel text-muted shadow-[inset_0_-1px_0_#d8ded9]"
                      : "border-subtle bg-surface text-muted shadow-[inset_0_-1px_0_#d8ded9]",
                )}
              >
                {summary.kind === "overflow" ? (
                  <span className="w-full text-center tracking-[0.18em]">{summary.label}</span>
                ) : summary.nestedLabel ? (
                  <span className="flex min-w-0 items-center gap-1">
                    <span className="shrink-0">{summary.label}</span>
                    <ChevronRight className="h-3 w-3 shrink-0 text-muted" aria-hidden />
                    <span className="min-w-0 truncate">{summary.nestedLabel}</span>
                  </span>
                ) : (
                  <span className="truncate">
                    {summary.count ? `${summary.label} x${summary.count}` : summary.label}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
      {hasMetadata && (
        <div className="mt-2 shrink-0 border-t border-border pt-2">
          <button
            type="button"
            aria-expanded={data.isDetailsExpanded}
            aria-controls={detailsId}
            aria-label={`Details for ${data.path}`}
            onClick={(event) => {
              event.stopPropagation();
              data.onToggleDetails();
            }}
            onKeyDown={(event) => {
              event.stopPropagation();
            }}
            className="nodrag nopan flex h-6 w-full items-center justify-between gap-2 rounded border border-border bg-surface px-2 text-left text-[11px] font-semibold text-muted transition hover:bg-panel hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          >
            <span>Details</span>
            <ChevronRight
              className={cn("h-3 w-3 transition-transform", data.isDetailsExpanded && "rotate-90")}
              aria-hidden
            />
          </button>
          {data.isDetailsExpanded && (
            <div id={detailsId} className="mt-2 grid gap-1">
              {entries.map(([key, value]) => (
                <div
                  key={key}
                  className="grid h-[22px] grid-cols-[72px_minmax(0,1fr)] items-center gap-1 rounded border border-subtle bg-surface px-1.5 text-[10px]"
                >
                  <span className="truncate font-medium text-muted">{key}</span>
                  <span className="truncate font-mono text-ink">{detailText(value)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

export const nodeTypes = {
  viewerNode: GraphNodeView,
};
