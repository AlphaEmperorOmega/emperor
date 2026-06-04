import { type StackDiagram } from "@/lib/graph";
import {
  STACK_DIAGRAM_CELL_WIDTH,
  isDenseStackDiagram,
  stackDiagramCellMetrics,
} from "@/components/features/viewer/graph-node-diagram-layout";
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
            <div
              key={`${cell.kind}-${cell.label}`}
              title={cell.title}
              aria-label={cell.title}
              style={{
                height: metrics.height,
                left: 0,
                top: metrics.top,
                width: STACK_DIAGRAM_CELL_WIDTH,
              }}
              className={cn(
                "absolute flex min-w-0 items-center rounded-[8px] border px-2.5 text-[12px] font-semibold leading-none",
                cell.kind === "layer" && cell.dims
                  ? "justify-between gap-2 text-left"
                  : "justify-center text-center",
                cell.kind === "layer"
                  ? "border-violet/30 bg-[linear-gradient(135deg,rgba(146,113,255,0.14),rgba(111,168,255,0.08))] font-mono text-[#d7c9ff] shadow-[inset_0_-1px_0_rgba(146,113,255,0.24)]"
                  : cell.kind === "overflow"
                    ? "border-line-soft bg-white/[0.035] font-mono text-ink-dim"
                    : "border-line-soft bg-black/25 text-ink-dim",
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
            </div>
          );
        })}
      </div>
    </div>
  );
}
