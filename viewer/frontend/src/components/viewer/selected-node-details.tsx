import { Badge } from "@/components/ui/badge";
import { type GraphNode } from "@/lib/api";
import { detailText, formatExactCount, nodeBadges } from "@/lib/graph";

export function SelectedNodeDetails({ node }: { node: GraphNode | undefined }) {
  if (!node) {
    return (
      <div className="rounded-md border border-dashed border-faint bg-surface p-4 text-sm text-muted">
        No node selected
      </div>
    );
  }
  const entries = Object.entries(node.details);
  const badges = nodeBadges(node.details);
  const hasParameters = node.parameterCount > 0;
  return (
    <div className="grid gap-4">
      <div className="rounded-md border border-border bg-surface p-3">
        <div className="grid min-w-0 gap-2">
          <div className="min-w-0">
            <div className="truncate text-base font-semibold text-ink">{node.typeName}</div>
            <div className="mt-1 break-words font-mono text-xs text-muted">{node.path}</div>
          </div>
          <div
            className="grid min-w-0 grid-cols-[24px_minmax(0,1fr)] items-center gap-2 rounded border border-subtle bg-panel px-2 py-1.5 text-[11px]"
            title={node.id}
          >
            <span className="font-semibold uppercase tracking-[0.08em] text-muted">ID</span>
            <span className="min-w-0 truncate font-mono text-ink">{node.id}</span>
          </div>
        </div>
        {badges.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1">
            {badges.map(([key, value]) => (
              <Badge key={`${key}-${value}`} className="border-accent-edge bg-panel text-ink">
                {`${key}: ${detailText(value)}`}
              </Badge>
            ))}
          </div>
        )}
      </div>
      {(hasParameters || entries.length > 0) && (
        <div className="grid gap-2">
          {hasParameters && (
            <div className="grid grid-cols-[104px_minmax(0,1fr)] gap-2 rounded-md border border-border bg-panel p-2.5 text-xs shadow-panel">
              <span className="truncate font-medium text-muted">Params</span>
              <span className="break-words font-mono text-ink">
                {formatExactCount(node.parameterCount)}
              </span>
            </div>
          )}
          {entries.map(([key, value]) => (
            <div
              key={key}
              className="grid grid-cols-[104px_minmax(0,1fr)] gap-2 rounded-md border border-border bg-panel p-2.5 text-xs shadow-panel"
            >
              <span className="truncate font-medium text-muted">{key}</span>
              <span className="break-words font-mono text-ink">{detailText(value)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
