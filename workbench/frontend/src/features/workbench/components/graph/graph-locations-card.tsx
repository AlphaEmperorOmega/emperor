import {
  useMemo,
  useState,
  type HTMLAttributes,
  type ReactNode,
} from "react";
import { ChevronDown, MapPin } from "lucide-react";
import { EdgeCard } from "@/components/ui/edge-card";
import { GraphChip } from "@/features/workbench/components/graph/graph-chip";
import { SectionHeading } from "@/components/ui/section-heading";
import { StatChip } from "@/features/workbench/components/shared/stat-chip";
import {
  type ClusterLocationSummary,
  type GraphCoordinate,
  buildClusterLocationSummary,
  formatExactCount,
  formatGraphCoordinate,
} from "@/lib/graph";
import { type InspectResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

function StatText({ children }: { children: ReactNode }) {
  return <StatChip tone="soft">{children}</StatChip>;
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
  const text = formatGraphCoordinate(coordinate);

  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="min-h-touch rounded-control-md border border-violet/25 bg-violet/10 px-2 font-mono type-meta font-semibold leading-none text-violet-text transition-colors hover:border-violet/45 hover:bg-violet/20 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:min-h-control-sm"
    >
      {text}
    </button>
  );
}

function LocationSummarySection({
  summary,
  onRevealNode,
}: {
  summary: ClusterLocationSummary;
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
        "rounded-panel border border-line-soft bg-panel-2/70 p-2.5",
        "border-violet/35 bg-violet/10 shadow-control-selected",
      )}
    >
      <button
        type="button"
        aria-label={`Reveal ${summary.nodePath} locations`}
        aria-current="true"
        onClick={revealNode}
        className="flex w-full min-w-0 items-start justify-between gap-2 rounded-control-md p-1 text-left transition hover:bg-white/[0.04] focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
      >
        <span className="min-w-0">
          <span className="block truncate type-compact font-bold leading-4 text-ink">
            {title}
          </span>
          <span className="mt-1 block truncate font-mono type-meta leading-4 text-ink-faint">
            {summary.nodePath}
          </span>
        </span>
        <GraphChip compact className="shrink-0 bg-black/20 font-bold uppercase tracking-label">
          Cluster
        </GraphChip>
      </button>

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
              label={`Reveal ${summary.nodePath} coordinate ${formatGraphCoordinate(coordinate)}`}
              onClick={revealNode}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

type GraphLocationsCardProps = HTMLAttributes<HTMLDivElement> & {
  graph: InspectResponse | undefined;
  selectedNodeId: string | null;
  onRevealNode: (nodeId: string) => void;
};

function SelectedGraphLocationsCard({
  summary,
  onRevealNode,
  className,
  ...props
}: HTMLAttributes<HTMLDivElement> & {
  summary: ClusterLocationSummary;
  onRevealNode: (nodeId: string) => void;
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!isExpanded) {
    return (
      <div
        className={cn(
          "nodrag nopan h-touch w-touch md:h-control-lg md:w-control-lg",
          className,
        )}
        data-testid="graph-locations-card"
        onPointerDown={(event) => event.stopPropagation()}
        {...props}
      >
        <button
          type="button"
          aria-label="Show cluster locations"
          title="Show cluster locations"
          onClick={(event) => {
            event.stopPropagation();
            setIsExpanded(true);
          }}
          className="grid h-touch w-touch place-items-center rounded-control border border-violet/35 bg-panel/90 text-violet shadow-panel backdrop-blur-md transition-colors hover:border-violet/55 hover:bg-violet/15 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:h-control-lg md:w-control-lg"
        >
          <MapPin className="h-[17px] w-[17px]" aria-hidden />
        </button>
      </div>
    );
  }

  return (
    <EdgeCard
      aria-label="Locations"
      className={cn(
        "nodrag nopan flex min-h-0 w-[312px] flex-col overflow-hidden rounded-card p-3.5 shadow-panel backdrop-blur-md",
        className,
      )}
      data-testid="graph-locations-card"
      onClick={(event) => event.stopPropagation()}
      onPointerDown={(event) => event.stopPropagation()}
      {...props}
    >
      <SectionHeading
        as="h2"
        className="mb-3 shrink-0"
        icon={<MapPin className="h-[15px] w-[15px] text-violet" aria-hidden />}
        title="Locations"
        actions={
          <button
            type="button"
            aria-label="Hide cluster locations"
            title="Hide cluster locations"
            onClick={() => setIsExpanded(false)}
            className="ml-auto grid h-touch w-touch place-items-center rounded-control-md border border-line bg-control text-ink-dim transition-colors hover:border-violet/45 hover:bg-violet/15 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:h-control-sm md:w-control-sm"
          >
            <ChevronDown className="h-4 w-4" aria-hidden />
          </button>
        }
      />
      <div className="min-h-0 flex-1 overflow-auto pr-1">
        <LocationSummarySection summary={summary} onRevealNode={onRevealNode} />
      </div>
    </EdgeCard>
  );
}

export function GraphLocationsCard({
  graph,
  selectedNodeId,
  onRevealNode,
  className,
  ...props
}: GraphLocationsCardProps) {
  const summary = useMemo(() => {
    if (!selectedNodeId) {
      return null;
    }

    return buildClusterLocationSummary(graph, selectedNodeId) ?? null;
  }, [graph, selectedNodeId]);

  if (!summary) {
    return null;
  }

  return (
    <SelectedGraphLocationsCard
      key={summary.nodeId}
      summary={summary}
      onRevealNode={onRevealNode}
      className={className}
      {...props}
    />
  );
}
