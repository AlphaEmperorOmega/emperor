import { LineChart } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { KeyValueRow } from "@/features/workbench/components/shared/key-value-row";
import { SurfacePanel } from "@/components/ui/surface-panel";
import type { GraphNode } from "@/lib/api/inspection";
import {
  type TerminalReachGrid,
  buildTerminalReachGrid,
  detailText,
  formatExactCount,
  nodeBadges,
  nodeDetailEntries,
  nodeDetailEntryText,
} from "@/lib/graph";
import { cn } from "@/lib/utils";

function TerminalReachView({ grid }: { grid: TerminalReachGrid }) {
  return (
    <SurfacePanel>
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="font-semibold text-ink">Sampler reach</span>
        <span className="font-mono text-ink-dim">{grid.total} connections</span>
      </div>
      <div className="mt-3 flex items-start gap-3 overflow-x-auto">
        {grid.planes.map((plane) => (
          <div key={plane.z} className="flex shrink-0 flex-col items-center gap-1">
            <span className="font-mono type-meta leading-none text-ink-dim">z={plane.z}</span>
            <div
              className="grid"
              style={{ gridTemplateColumns: `repeat(${grid.columns}, 18px)`, gap: 3 }}
            >
              {plane.cells.map((cell) => (
                <div
                  key={`${cell.x}-${cell.y}`}
                  title={cell.title}
                  style={{ height: 18, width: 18 }}
                  className={cn(
                    "rounded-indicator border",
                    cell.kind === "self"
                      ? "border-violet-text bg-violet-text"
                      : cell.kind === "reach"
                        ? "border-violet/40 bg-violet/30"
                        : "border-line-soft bg-white/[0.02]",
                  )}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap gap-3 type-meta text-ink-dim">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-indicator bg-violet-text" /> this neuron
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-indicator border border-violet/40 bg-violet/30" />{" "}
          reachable
        </span>
      </div>
      {grid.hasOverflow && (
        <div className="mt-2 type-meta text-ink-dim">reach truncated for display</div>
      )}
    </SurfacePanel>
  );
}

// Pure presentation of a selected node's metadata. Modal state lives in the
// SelectedNodeDetails container; this view only signals intent via onOpenMonitors.
export function SelectedNodeDetailsView({
  node,
  canOpenMonitors,
  monitorButtonTitle,
  onOpenMonitors,
}: {
  node: GraphNode;
  canOpenMonitors: boolean;
  monitorButtonTitle?: string;
  onOpenMonitors: () => void;
}) {
  const entries = nodeDetailEntries(node.details, node.config);
  const badges = nodeBadges(node.details);
  const hasParameters = node.parameterCount > 0;
  const reachGrid = buildTerminalReachGrid(node.details);

  return (
    <div className="grid gap-4">
      <SurfacePanel>
        <div className="grid min-w-0 gap-2">
          <div className="min-w-0">
            <div className="truncate type-heading font-bold text-ink">{node.typeName}</div>
            <div className="mt-1 break-words font-mono text-xs text-ink-dim">{node.path}</div>
          </div>
          <div
            className="grid min-w-0 grid-cols-[24px_minmax(0,1fr)] items-center gap-2 rounded-control border border-line bg-black/25 px-3 py-2 text-xs"
            title={node.id}
          >
            <span className="font-bold uppercase tracking-label text-ink-dim">ID</span>
            <span className="min-w-0 truncate font-mono text-ink">{node.id}</span>
          </div>
        </div>
        {badges.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1">
            {badges.map(([key, value]) => (
              <Badge key={`${key}-${value}`} className="border-violet/25 bg-violet/10 text-ink">
                {`${key}: ${detailText(value)}`}
              </Badge>
            ))}
          </div>
        )}
        <Button
          className="mt-3 w-full"
          variant="secondary"
          disabled={!canOpenMonitors}
          title={
            monitorButtonTitle ??
            (canOpenMonitors
              ? "Open monitor charts for this node"
              : "Start a workbench training job before opening monitor charts")
          }
          onClick={onOpenMonitors}
        >
          <LineChart className="h-4 w-4" aria-hidden />
        Monitor Charts
        </Button>
      </SurfacePanel>
      {reachGrid && <TerminalReachView grid={reachGrid} />}
      {(hasParameters || entries.length > 0) && (
        <div className="grid gap-0">
          {hasParameters && (
            <KeyValueRow
              label={<span className="font-mono">Params</span>}
              value={formatExactCount(node.parameterCount)}
              valueClassName="text-violet-text"
            />
          )}
          {entries.map((entry) => {
            const value = nodeDetailEntryText(entry);
            const isNumeric = typeof entry.value === "number";
            return (
              <KeyValueRow
                key={entry.key}
                label={<span className="font-mono">{entry.key}</span>}
                value={value}
                className="last:border-b-0"
                valueClassName={isNumeric ? "text-violet-text" : undefined}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
