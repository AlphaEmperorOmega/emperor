import { type StackDiagram } from "@/lib/graph";
import { GraphChip } from "@/features/workbench/components/graph/graph-chip";
import { graphCardGeometry } from "@/lib/graph/constants";
import { cn } from "@/lib/utils";

export function StackDiagramView({
  diagram,
  nodeId,
}: {
  diagram: StackDiagram;
  nodeId: string;
}) {
  return (
    <div
      className="grid shrink-0 content-start"
      style={{
        marginTop: graphCardGeometry.contentMarginBlockStart,
        rowGap: graphCardGeometry.childSummary.rowGap,
      }}
      data-testid={`stack-diagram-${nodeId}`}
      aria-label={`Layer stack diagram for ${nodeId}`}
    >
      {diagram.cells.map((cell) => (
        <GraphChip
          key={`${cell.kind}-${cell.label}`}
          title={cell.title}
          aria-label={cell.title}
          tone="default"
          className={cn(
            "relative flex min-w-0 items-center gap-2 overflow-hidden rounded-[10px] px-3 text-[13px] font-medium",
            cell.kind === "overflow"
              ? "justify-center text-center font-mono tracking-[0.18em]"
              : "justify-between text-left",
          )}
          style={{ height: graphCardGeometry.childSummary.rowHeight }}
        >
          {cell.kind === "overflow" ? (
            <span className="w-full truncate text-center">{cell.label}</span>
          ) : (
            <>
              <span className="min-w-0 flex-1 truncate">{cell.label}</span>
              {cell.dims && (
                <span className="shrink-0 text-right font-mono text-[12px] font-semibold text-ink">
                  {cell.dims}
                </span>
              )}
            </>
          )}
        </GraphChip>
      ))}
    </div>
  );
}
