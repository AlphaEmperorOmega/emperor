import { useMemo, useState } from "react";
import { type ClusterDiagram, formatExactCount } from "@/lib/graph";
import { type ClusterDiagramReach } from "@/lib/graph/types";
import { graphCardGeometry } from "@/lib/graph/constants";
import {
  CLUSTER_DIAGRAM_WIDTH,
  clusterDiagramGridHeight,
  clusterDiagramPlaneWidth,
} from "@/features/workbench/components/graph/graph-node-diagram-layout";
import { GraphChip } from "@/features/workbench/components/graph/graph-chip";
import { cn } from "@/lib/utils";

function coordinateKey([x, y, z]: [number, number, number]) {
  return `${x},${y},${z}`;
}

function coordinateText([x, y, z]: [number, number, number]) {
  return `(${x}, ${y}, ${z})`;
}

function reachSummary(reach: ClusterDiagramReach) {
  const outsideText =
    reach.outOfBoundsTotal > 0
      ? ` · ${formatExactCount(reach.outOfBoundsTotal)} outside`
      : "";
  return `${coordinateText(reach.position)} · ${formatExactCount(
    reach.connections.length,
  )} reach · ${formatExactCount(reach.activeConnectionTotal)} active${outsideText}`;
}

export function ClusterDiagramView({
  diagram,
  nodeId,
}: {
  diagram: ClusterDiagram;
  nodeId: string;
}) {
  const [hoveredReach, setHoveredReach] = useState<ClusterDiagramReach | null>(null);
  const gridHeight = clusterDiagramGridHeight(diagram);
  const planeWidth = clusterDiagramPlaneWidth(diagram);
  const isTruncated =
    diagram.hasColumnOverflow || diagram.hasRowOverflow || diagram.hasPlaneOverflow;
  const hoveredPositionKey = hoveredReach ? coordinateKey(hoveredReach.position) : null;
  const reachableKeys = useMemo(
    () =>
      new Set(
        hoveredReach?.inBoundsConnections.map((coordinate) => coordinateKey(coordinate)) ??
          [],
      ),
    [hoveredReach],
  );
  const summaryText = hoveredReach
    ? reachSummary(hoveredReach)
    : `${formatExactCount(diagram.instantiated)} / ${formatExactCount(diagram.capacityTotal)}`;

  return (
    <div
      className="shrink-0 overflow-hidden"
      style={{
        height: graphCardGeometry.clusterDiagram.headerHeight + gridHeight,
        marginTop: graphCardGeometry.contentMarginBlockStart,
      }}
      data-testid={`cluster-diagram-${nodeId}`}
      aria-label={`Neuron cluster map for ${nodeId}`}
    >
      <div className="mb-panel flex h-control-lg items-center justify-between gap-2 rounded-panel border border-line-soft bg-control-field px-2.5">
        <div className="min-w-0">
          <div className="truncate type-label font-bold leading-4 text-ink">Cluster map</div>
          <div
            className="mt-0.5 truncate font-mono type-meta leading-3 text-ink-faint"
            data-testid={`cluster-diagram-summary-${nodeId}`}
          >
            {summaryText}
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
        onMouseLeave={() => setHoveredReach(null)}
      >
        {diagram.planes.map((plane) => (
          <div
            key={plane.z}
            className="relative shrink-0"
            style={{ width: planeWidth }}
            title={`Z plane ${plane.z}`}
          >
            <span className="pointer-events-none absolute left-1 top-1 z-10 rounded-chip border border-line-soft bg-black/60 px-1.5 py-0.5 font-mono type-caption font-bold leading-none text-ink-dim">
              z{plane.z}
            </span>
            <div
              className="grid"
              style={{
                gap: graphCardGeometry.clusterDiagram.cellGap,
                gridTemplateColumns: `repeat(${diagram.columns}, ${graphCardGeometry.clusterDiagram.cellSize}px)`,
              }}
            >
              {plane.cells.map((cell) => {
                const cellKey = `${cell.x},${cell.y},${plane.z}`;
                const isSource = hoveredPositionKey === cellKey;
                const isReachable = Boolean(hoveredReach && reachableKeys.has(cellKey));
                const isDimmed = Boolean(hoveredReach && !isSource && !isReachable);
                const isReachableActive = isReachable && cell.filled && !isSource;
                const isReachableEmpty = isReachable && !cell.filled;

                return (
                  <div
                    key={`${plane.z}-${cell.x}-${cell.y}`}
                    title={cell.title}
                    aria-label={cell.title}
                    onMouseEnter={() => {
                      if (cell.filled && cell.reach) {
                        setHoveredReach(cell.reach);
                      }
                    }}
                    onMouseLeave={() => {
                      if (cell.reach === hoveredReach) {
                        setHoveredReach(null);
                      }
                    }}
                    className={cn(
                      "rounded-chip border shadow-divider transition duration-100",
                      cell.filled
                        ? "border-violet/45 bg-cluster-active"
                        : "border-line-soft bg-white/[0.035]",
                      cell.filled && cell.reach ? "cursor-crosshair" : "cursor-default",
                      isSource &&
                        "border-violet-text ring-2 ring-violet-text/80 ring-offset-1 ring-offset-bg",
                      isReachableActive &&
                        "border-cyan/90 ring-1 ring-cyan/60 shadow-cyan-selection",
                      isReachableEmpty &&
                        "border-cyan/55 bg-cyan/15 shadow-cyan-inset",
                      isDimmed && "opacity-35",
                    )}
                    style={{
                      height: graphCardGeometry.clusterDiagram.cellSize,
                      width: graphCardGeometry.clusterDiagram.cellSize,
                    }}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
