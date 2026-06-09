import { ChevronRight } from "lucide-react";
import {
  nodeDetailEntryText,
  type NodeDetailEntry,
} from "@/lib/graph/format/text";
import { cn } from "@/lib/utils";

export function GraphNodeDetails({
  detailsId,
  toggleLabel,
  path,
  entries,
  isExpanded,
  onToggleDetails,
}: {
  detailsId: string;
  toggleLabel: string;
  path: string;
  entries: NodeDetailEntry[];
  isExpanded: boolean;
  onToggleDetails: () => void;
}) {
  return (
    <div className="mt-3 shrink-0 border-t border-line-soft pt-3">
      <button
        type="button"
        aria-expanded={isExpanded}
        aria-controls={detailsId}
        aria-label={`${toggleLabel} for ${path}`}
        onClick={(event) => {
          event.stopPropagation();
          onToggleDetails();
        }}
        onKeyDown={(event) => {
          event.stopPropagation();
        }}
        className="nodrag nopan flex h-9 w-full items-center justify-between gap-2 rounded-[10px] border border-line-soft bg-white/[0.015] px-3 text-left text-sm font-semibold text-ink-dim transition hover:border-line hover:bg-white/[0.04] hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
      >
        <span>{toggleLabel}</span>
        <ChevronRight
          className={cn("h-3 w-3 transition-transform", isExpanded && "rotate-90")}
          aria-hidden
        />
      </button>
      {isExpanded && (
        <div id={detailsId} className="mt-2 grid gap-1">
          {entries.map((entry) => (
            <div
              key={entry.key}
              className="grid h-8 grid-cols-[96px_minmax(0,1fr)] items-center gap-2 rounded-[8px] border border-line-soft bg-black/20 px-2.5 text-[12.5px]"
            >
              <span className="truncate font-medium text-ink-dim">{entry.key}</span>
              <span className="truncate font-mono text-ink">
                {nodeDetailEntryText(entry)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
