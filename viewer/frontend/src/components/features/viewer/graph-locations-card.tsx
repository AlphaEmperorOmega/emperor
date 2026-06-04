import { useMemo, type HTMLAttributes, type ReactNode } from "react";
import { MapPin } from "lucide-react";
import { EdgeCard } from "@/components/ui/edge-card";
import {
  type GraphCoordinate,
  type GraphLocationSummary,
  buildGraphLocationSummaries,
  formatExactCount,
} from "@/lib/graph";
import { type InspectResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

function coordinateLabel(coordinate: GraphCoordinate) {
  return `(${coordinate[0]}, ${coordinate[1]}, ${coordinate[2]})`;
}

function StatText({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-[7px] border border-line-soft bg-black/20 px-2 py-1 font-mono text-[11px] font-semibold leading-none text-ink-dim">
      {children}
    </span>
  );
}

function CoordinateChip({
  coordinate,
  label,
  onClick,
}: {
  coordinate: GraphCoordinate;
  label: string;
  onClick: () => void;
}) {
  const text = coordinateLabel(coordinate);

  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="min-h-7 rounded-[8px] border border-violet/25 bg-violet/10 px-2 font-mono text-[11.5px] font-semibold leading-none text-[#d7c9ff] transition hover:border-violet/45 hover:bg-violet/20 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
    >
      {text}
    </button>
  );
}

function LocationSummarySection({
  summary,
  selected,
  onRevealNode,
}: {
  summary: GraphLocationSummary;
  selected: boolean;
  onRevealNode: (nodeId: string) => void;
}) {
  const revealNode = () => onRevealNode(summary.nodeId);
  const title = summary.nodeLabel || summary.nodeType;

  return (
    <section
      role="group"
      aria-label={`${summary.nodePath} locations`}
      data-testid={`location-summary-${summary.nodeId}`}
      className={cn(
        "rounded-[10px] border border-line-soft bg-white/[0.018] p-2.5",
        selected && "border-violet/35 bg-violet/10 shadow-[inset_0_0_0_1px_rgba(146,113,255,0.22)]",
      )}
    >
      <button
        type="button"
        aria-label={`Reveal ${summary.nodePath} locations`}
        aria-current={selected ? "true" : undefined}
        onClick={revealNode}
        className="flex w-full min-w-0 items-start justify-between gap-2 rounded-[8px] p-1 text-left transition hover:bg-white/[0.04] focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
      >
        <span className="min-w-0">
          <span className="block truncate text-[13px] font-bold leading-4 text-ink">
            {title}
          </span>
          <span className="mt-1 block truncate font-mono text-[11px] leading-4 text-ink-faint">
            {summary.nodePath}
          </span>
        </span>
        <span className="shrink-0 rounded-[7px] border border-line-soft bg-black/20 px-1.5 py-1 text-[10px] font-bold uppercase leading-none tracking-[0.08em] text-ink-dim">
          {summary.kind === "cluster" ? "Cluster" : "Terminal"}
        </span>
      </button>

      {summary.kind === "cluster" ? (
        <div className="mt-2 grid gap-2">
          <div className="flex flex-wrap gap-1.5">
            <StatText>{formatExactCount(summary.instantiated)} instantiated</StatText>
            <StatText>{formatExactCount(summary.capacityTotal)} capacity</StatText>
            {summary.hasOverflow && <StatText>display truncated</StatText>}
          </div>
          <div className="flex flex-wrap gap-1.5" aria-label="Cluster coordinates">
            {summary.coordinates.map((coordinate, index) => (
              <CoordinateChip
                key={`${coordinate.join(",")}-${index}`}
                coordinate={coordinate}
                label={`Reveal ${summary.nodePath} coordinate ${coordinateLabel(coordinate)}`}
                onClick={revealNode}
              />
            ))}
          </div>
        </div>
      ) : (
        <div className="mt-2 grid gap-2">
          <div className="flex flex-wrap gap-1.5">
            <StatText>{formatExactCount(summary.total)} connections</StatText>
            {summary.hasOverflow && <StatText>display truncated</StatText>}
          </div>
          <div className="grid gap-1.5">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="w-[58px] shrink-0 text-[11px] font-bold uppercase tracking-[0.08em] text-ink-faint">
                Position
              </span>
              <CoordinateChip
                coordinate={summary.position}
                label={`Reveal ${summary.nodePath} terminal position ${coordinateLabel(
                  summary.position,
                )}`}
                onClick={revealNode}
              />
            </div>
            {summary.connections.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5" aria-label="Reachable coordinates">
                <span className="w-[58px] shrink-0 text-[11px] font-bold uppercase tracking-[0.08em] text-ink-faint">
                  Reach
                </span>
                {summary.connections.map((coordinate, index) => (
                  <CoordinateChip
                    key={`${coordinate.join(",")}-${index}`}
                    coordinate={coordinate}
                    label={`Reveal ${summary.nodePath} reachable coordinate ${coordinateLabel(
                      coordinate,
                    )}`}
                    onClick={revealNode}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

type GraphLocationsCardProps = HTMLAttributes<HTMLDivElement> & {
  graph: InspectResponse | undefined;
  selectedNodeId: string | null;
  onRevealNode: (nodeId: string) => void;
};

export function GraphLocationsCard({
  graph,
  selectedNodeId,
  onRevealNode,
  className,
  ...props
}: GraphLocationsCardProps) {
  const summaries = useMemo(() => buildGraphLocationSummaries(graph), [graph]);

  return (
    <EdgeCard
      aria-label="Locations"
      className={cn(
        "flex min-h-0 flex-col overflow-hidden rounded-card p-3.5 shadow-[0_20px_50px_-30px_rgba(0,0,0,0.95)] backdrop-blur-md",
        className,
      )}
      data-testid="graph-locations-card"
      {...props}
    >
      <div className="mb-3 flex shrink-0 items-center gap-2">
        <MapPin className="h-[15px] w-[15px] text-violet" aria-hidden />
        <h2 className="text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
          Locations
        </h2>
      </div>
      {summaries.length === 0 ? (
        <div className="p-3 text-sm text-ink-faint">No analysed locations found.</div>
      ) : (
        <div className="min-h-0 flex-1 overflow-auto pr-1">
          <div className="grid gap-2">
            {summaries.map((summary) => (
              <LocationSummarySection
                key={`${summary.kind}-${summary.nodeId}`}
                summary={summary}
                selected={summary.nodeId === selectedNodeId}
                onRevealNode={onRevealNode}
              />
            ))}
          </div>
        </div>
      )}
    </EdgeCard>
  );
}
