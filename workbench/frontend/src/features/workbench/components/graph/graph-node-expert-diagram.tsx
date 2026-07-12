import { type ExpertDiagram } from "@/lib/graph";
import {
  graphCardGeometry,
  graphDiagramLimits,
} from "@/lib/graph/constants";
import {
  EXPERT_DIAGRAM_OVERFLOW_WIDTH,
  EXPERT_DIAGRAM_SAMPLER_WIDTH,
  EXPERT_DIAGRAM_TOTAL_WIDTH,
  EXPERT_DIAGRAM_WIDTH,
  expertDiagramCellCenters,
} from "@/features/workbench/components/graph/graph-node-diagram-layout";
import { GraphChip } from "@/features/workbench/components/graph/graph-chip";
import { cn } from "@/lib/utils";

export function ExpertDiagramView({
  diagram,
  nodeId,
}: {
  diagram: ExpertDiagram;
  nodeId: string;
}) {
  const cellCenters = expertDiagramCellCenters(
    diagram,
    graphDiagramLimits.expert.visibleBeforeOverflow,
  );
  const gridTemplateColumns = diagram.hasOverflow
    ? `repeat(${graphDiagramLimits.expert.visibleBeforeOverflow}, minmax(0, 1fr)) ${EXPERT_DIAGRAM_OVERFLOW_WIDTH}px ${EXPERT_DIAGRAM_TOTAL_WIDTH}px`
    : `repeat(${diagram.cells.length}, minmax(0, 1fr))`;

  return (
    <div
      className="shrink-0"
      style={{
        height: graphCardGeometry.expertDiagram.height,
        marginTop: graphCardGeometry.contentMarginBlockStart,
      }}
      data-testid={`expert-diagram-${nodeId}`}
      aria-label={`Expert routing diagram for ${nodeId}`}
    >
      <div className="relative h-full w-full">
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full text-violet/50"
          viewBox={`0 0 ${EXPERT_DIAGRAM_WIDTH} ${graphCardGeometry.expertDiagram.height}`}
          fill="none"
          aria-hidden
        >
          {diagram.cells.map((cell, index) => {
            const x = cellCenters[index];
            if (x === undefined) {
              return null;
            }
            return (
              <path
                key={`${cell.kind}-${cell.label}-${x}`}
                d={`M ${EXPERT_DIAGRAM_WIDTH / 2} 76 C ${EXPERT_DIAGRAM_WIDTH / 2} 64 ${x.toFixed(
                  2,
                )} 54 ${x.toFixed(2)} 34`}
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
              />
            );
          })}
        </svg>
        <div
          className="absolute left-0 top-0 grid w-full gap-1"
          style={{ gridTemplateColumns }}
        >
          {diagram.cells.map((cell) => (
            <GraphChip
              key={`${cell.kind}-${cell.label}`}
              title={cell.title}
              tone={cell.kind === "expert" ? "violet" : "default"}
              className={cn(
                "flex h-8 min-w-0 items-center justify-center rounded-[8px] px-1.5 text-center text-[12px] font-semibold",
                cell.kind === "expert"
                  ? "font-mono"
                  : cell.kind === "overflow"
                    ? "bg-white/[0.035] font-mono"
                    : "bg-black/25",
              )}
            >
              <span className="truncate">{cell.label}</span>
            </GraphChip>
          ))}
        </div>
        <GraphChip
          title={diagram.samplerTitle}
          className="absolute bottom-0 left-1/2 flex h-8 -translate-x-1/2 items-center justify-center rounded-[8px] border-line bg-black/25 px-2 text-[12px] font-semibold text-ink shadow-[inset_0_-1px_0_rgba(255,255,255,0.05)]"
          style={{ width: EXPERT_DIAGRAM_SAMPLER_WIDTH }}
        >
          <span className="truncate">{diagram.samplerLabel}</span>
        </GraphChip>
      </div>
    </div>
  );
}
