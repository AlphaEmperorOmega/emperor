import { type StackDiagram } from "@/lib/graph";
import {
  STACK_DIAGRAM_CELL_WIDTH,
  isDenseStackDiagram,
  stackDiagramCellMetrics,
} from "@/components/features/viewer/graph-node-diagram-layout";
import { GraphChip } from "@/components/features/viewer/graph/graph-chip";
import { cn } from "@/lib/utils";

export function StackDiagramView({
  diagram,
  nodeId,
}: {
  diagram: StackDiagram;
  nodeId: string;
}) {
  const cellMetrics = stackDiagramCellMetrics(diagram);

  return (
    <div
      className={cn(
        "mt-2 shrink-0",
        isDenseStackDiagram(diagram) ? "h-[160px]" : "h-[112px]",
      )}
      data-testid={`stack-diagram-${nodeId}`}
      aria-label={`Layer stack diagram for ${nodeId}`}
    >
      <div className="relative h-full w-full">
        {diagram.cells.map((cell, index) => {
          const metrics = cellMetrics[index];

          return (
            <GraphChip
              key={`${cell.kind}-${cell.label}`}
              title={cell.title}
              aria-label={cell.title}
              tone={cell.kind === "layer" ? "violet" : "default"}
              style={{
                height: metrics.height,
                left: 0,
                top: metrics.top,
                width: STACK_DIAGRAM_CELL_WIDTH,
              }}
              className={cn(
                "absolute flex min-w-0 items-center rounded-[8px] px-2.5 text-[12px] font-semibold",
                cell.kind === "layer" && cell.dims
                  ? "justify-between gap-2 text-left"
                  : "justify-center text-center",
                cell.kind === "layer"
                  ? "bg-[linear-gradient(135deg,rgba(146,113,255,0.14),rgba(111,168,255,0.08))] font-mono"
                  : cell.kind === "overflow"
                    ? "bg-white/[0.035] font-mono"
                    : "bg-black/25",
              )}
            >
              {cell.kind === "layer" && cell.dims ? (
                <>
                  <span className="min-w-0 flex-1 truncate text-left">{cell.label}</span>
                  <span className="shrink-0 text-right font-mono">{cell.dims}</span>
                </>
              ) : (
                <span className="truncate">{cell.label}</span>
              )}
            </GraphChip>
          );
        })}
      </div>
    </div>
  );
}
