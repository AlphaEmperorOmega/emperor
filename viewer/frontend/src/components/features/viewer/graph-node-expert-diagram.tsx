import { type ExpertDiagram } from "@/lib/graph";
import {
  EXPERT_DIAGRAM_HEIGHT,
  EXPERT_DIAGRAM_VISIBLE_BEFORE_OVERFLOW,
} from "@/lib/graph/constants";
import {
  EXPERT_DIAGRAM_OVERFLOW_WIDTH,
  EXPERT_DIAGRAM_SAMPLER_WIDTH,
  EXPERT_DIAGRAM_TOTAL_WIDTH,
  EXPERT_DIAGRAM_WIDTH,
  expertDiagramCellCenters,
} from "@/components/features/viewer/graph-node-diagram-layout";
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
    EXPERT_DIAGRAM_VISIBLE_BEFORE_OVERFLOW,
  );
  const gridTemplateColumns = diagram.hasOverflow
    ? `repeat(${EXPERT_DIAGRAM_VISIBLE_BEFORE_OVERFLOW}, minmax(0, 1fr)) ${EXPERT_DIAGRAM_OVERFLOW_WIDTH}px ${EXPERT_DIAGRAM_TOTAL_WIDTH}px`
    : `repeat(${diagram.cells.length}, minmax(0, 1fr))`;

  return (
    <div
      className="mt-2 h-[104px] shrink-0"
      data-testid={`expert-diagram-${nodeId}`}
      aria-label={`Expert routing diagram for ${nodeId}`}
    >
      <div className="relative h-full w-full">
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full text-violet/50"
          viewBox={`0 0 ${EXPERT_DIAGRAM_WIDTH} ${EXPERT_DIAGRAM_HEIGHT}`}
          fill="none"
          aria-hidden
        >
          {cellCenters.map((x, index) => (
            <path
              key={`${diagram.cells[index]?.label ?? index}-${x}`}
              d={`M ${EXPERT_DIAGRAM_WIDTH / 2} 76 C ${EXPERT_DIAGRAM_WIDTH / 2} 64 ${x.toFixed(
                2,
              )} 54 ${x.toFixed(2)} 34`}
              stroke="currentColor"
              strokeWidth="1.2"
              strokeLinecap="round"
            />
          ))}
        </svg>
        <div
          className="absolute left-0 top-0 grid w-full gap-1"
          style={{ gridTemplateColumns }}
        >
          {diagram.cells.map((cell) => (
            <div
              key={`${cell.kind}-${cell.label}`}
              title={cell.title}
              className={cn(
                "flex h-8 min-w-0 items-center justify-center rounded-[8px] border px-1.5 text-center text-[12px] font-semibold leading-none",
                cell.kind === "expert"
                  ? "border-violet/30 bg-violet/15 font-mono text-[#d7c9ff] shadow-[inset_0_-1px_0_rgba(146,113,255,0.24)]"
                  : cell.kind === "overflow"
                    ? "border-line-soft bg-white/[0.035] font-mono text-ink-dim"
                    : "border-line-soft bg-black/25 text-ink-dim",
              )}
            >
              <span className="truncate">{cell.label}</span>
            </div>
          ))}
        </div>
        <div
          title={diagram.samplerTitle}
          className="absolute bottom-0 left-1/2 flex h-8 -translate-x-1/2 items-center justify-center rounded-[8px] border border-line bg-black/25 px-2 text-[12px] font-semibold leading-none text-ink shadow-[inset_0_-1px_0_rgba(255,255,255,0.05)]"
          style={{ width: EXPERT_DIAGRAM_SAMPLER_WIDTH }}
        >
          <span className="truncate">{diagram.samplerLabel}</span>
        </div>
      </div>
    </div>
  );
}
