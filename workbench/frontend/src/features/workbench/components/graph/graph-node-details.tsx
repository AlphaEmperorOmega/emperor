import { List, Settings2 } from "lucide-react";
import { GraphIconButton } from "@/features/workbench/components/graph/graph-icon-button";
import {
  nodeDetailEntryText,
  type NodeDetailEntry,
} from "@/lib/graph/format/text";
import { graphCardGeometry } from "@/lib/graph/constants";

export function GraphNodeDetails({
  detailsId,
  entries,
  isExpanded,
}: {
  detailsId: string;
  entries: NodeDetailEntry[];
  isExpanded: boolean;
}) {
  if (!isExpanded) {
    return null;
  }

  return (
    <div
      id={detailsId}
      className="grid shrink-0"
      style={{
        marginTop: graphCardGeometry.details.marginBlockStart,
        rowGap: graphCardGeometry.details.rowGap,
      }}
    >
      {entries.map((entry) => (
        <div
          key={entry.key}
          className="grid grid-cols-[96px_minmax(0,1fr)] items-center gap-2 rounded-control-md border border-line-soft bg-black/20 px-2.5 type-compact"
          style={{ height: graphCardGeometry.details.rowHeight }}
        >
          <span className="truncate font-medium text-ink-dim">{entry.key}</span>
          <span className="truncate font-mono text-ink">
            {nodeDetailEntryText(entry)}
          </span>
        </div>
      ))}
    </div>
  );
}

export function GraphNodeDetailsToggle({
  detailsId,
  toggleLabel,
  path,
  hasConfig,
  isExpanded,
  onToggleDetails,
}: {
  detailsId: string;
  toggleLabel: string;
  path: string;
  hasConfig: boolean;
  isExpanded: boolean;
  onToggleDetails: () => void;
}) {
  const Icon = hasConfig ? Settings2 : List;

  return (
    <GraphIconButton
      aria-controls={detailsId}
      aria-expanded={isExpanded}
      active={isExpanded}
      className="flex h-touch w-touch shrink-0 items-center justify-center rounded-control-md border border-line bg-control text-ink-dim transition-colors hover:border-line-hover hover:bg-control-hover hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:h-control-sm md:w-control-sm"
      label={`${toggleLabel} for ${path}`}
      onClick={onToggleDetails}
      icon={<Icon className="h-3.5 w-3.5" aria-hidden />}
    />
  );
}
