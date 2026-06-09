import { type ClusterDiagram, formatExactCount } from "@/lib/graph";
import {
  CLUSTER_DIAGRAM_CELL_GAP,
  CLUSTER_DIAGRAM_CELL_HEIGHT,
  CLUSTER_DIAGRAM_HEADER_HEIGHT,
} from "@/lib/graph/constants";
import {
  CLUSTER_DIAGRAM_WIDTH,
  clusterDiagramGridHeight,
  clusterDiagramPlaneWidth,
} from "@/features/viewer/components/graph/graph-node-diagram-layout";
import { GraphChip } from "@/features/viewer/components/graph/graph-chip";
import { cn } from "@/lib/utils";

export function ClusterDiagramView({
  diagram,
  nodeId,
}: {
  diagram: ClusterDiagram;
  nodeId: string;
}) {
  const gridHeight = clusterDiagramGridHeight(diagram);
  const planeWidth = clusterDiagramPlaneWidth(diagram);
  const isTruncated =
    diagram.hasColumnOverflow || diagram.hasRowOverflow || diagram.hasPlaneOverflow;

  return (
    <div
      className="mt-2 shrink-0 overflow-hidden"
      style={{ height: CLUSTER_DIAGRAM_HEADER_HEIGHT + gridHeight }}
      data-testid={`cluster-diagram-${nodeId}`}
      aria-label={`Neuron cluster map for ${nodeId}`}
    >
      <div className="mb-3 flex h-10 items-center justify-between gap-2 rounded-[10px] border border-line-soft bg-black/20 px-2.5">
        <div className="min-w-0">
          <div className="truncate text-[12px] font-bold leading-4 text-ink">Cluster map</div>
          <div className="mt-0.5 truncate font-mono text-[11px] leading-3 text-ink-faint">
            {formatExactCount(diagram.instantiated)} / {formatExactCount(diagram.capacityTotal)}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <GraphChip
            compact
            tone="violet"
            className="border-violet/25 bg-violet/10 font-mono font-bold"
          >
            {diagram.planes.length}z
          </GraphChip>
          {isTruncated && (
            <GraphChip compact className="bg-white/[0.035] font-bold uppercase">
              clipped
            </GraphChip>
          )}
        </div>
      </div>
      <div
        className="flex gap-2 overflow-x-auto overflow-y-hidden pb-1"
        style={{ height: gridHeight + 4, width: CLUSTER_DIAGRAM_WIDTH }}
      >
        {diagram.planes.map((plane) => (
          <div
            key={plane.z}
            className="relative shrink-0"
            style={{ width: planeWidth }}
            title={`Z plane ${plane.z}`}
          >
            <span className="pointer-events-none absolute left-1 top-1 z-10 rounded-[6px] border border-line-soft bg-black/60 px-1.5 py-0.5 font-mono text-[10px] font-bold leading-none text-ink-dim">
              z{plane.z}
            </span>
            <div
              className="grid"
              style={{
                gap: CLUSTER_DIAGRAM_CELL_GAP,
                gridTemplateColumns: `repeat(${diagram.columns}, ${CLUSTER_DIAGRAM_CELL_HEIGHT}px)`,
              }}
            >
              {plane.cells.map((cell) => (
                <div
                  key={`${plane.z}-${cell.x}-${cell.y}`}
                  title={cell.title}
                  aria-label={cell.title}
                  className={cn(
                    "rounded-[5px] border shadow-[inset_0_-1px_0_rgba(255,255,255,0.06)]",
                    cell.filled
                      ? "border-violet/45 bg-[linear-gradient(135deg,rgba(146,113,255,0.88),rgba(111,168,255,0.58))]"
                      : "border-line-soft bg-white/[0.035]",
                  )}
                  style={{
                    height: CLUSTER_DIAGRAM_CELL_HEIGHT,
                    width: CLUSTER_DIAGRAM_CELL_HEIGHT,
                  }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
