import { useMemo, useState, type ReactNode, type SyntheticEvent } from "react";
import { ChevronRight, X } from "lucide-react";
import { IconButton } from "@/components/ui/icon-button";
import type { InspectResponse } from "@/lib/api/inspection";
import {
  ancestorNodeIds,
  buildGraphNavigation,
  buildHierarchy,
  formatCompactCount,
  structureNodeLabel,
  type HierarchyNode,
} from "@/lib/graph";
import { cn } from "@/lib/utils";

type GraphStructurePanelProps = {
  panelId: string;
  graph: InspectResponse | undefined;
  selectedNodeId: string | null;
  onRevealNode: (nodeId: string) => void;
  onClose: () => void;
};

function stopPanelEvent(event: SyntheticEvent<HTMLElement>) {
  event.stopPropagation();
}

function nextExpandedNodeIds(current: Set<string>, nodeId: string) {
  const next = new Set(current);
  if (next.has(nodeId)) {
    next.delete(nodeId);
  } else {
    next.add(nodeId);
  }
  return next;
}

function defaultExpandedNodeIds(
  graph: InspectResponse | undefined,
  selectedNodeId: string | null,
) {
  const navigation = buildGraphNavigation(graph);
  const expandedNodeIds = new Set(navigation.rootIds);
  if (selectedNodeId) {
    for (const ancestorId of ancestorNodeIds(selectedNodeId, navigation)) {
      expandedNodeIds.add(ancestorId);
    }
  }
  return expandedNodeIds;
}

function StructureBadge({ children }: { children: ReactNode }) {
  return (
    <span className="shrink-0 rounded-chip border border-white/[0.08] bg-white/[0.045] px-1.5 py-0.5 font-mono type-caption font-semibold uppercase tracking-caption text-ink-faint">
      {children}
    </span>
  );
}

function StructureTreeItem({
  item,
  depth,
  expandedNodeIds,
  selectedNodeId,
  onToggleNode,
  onRevealNode,
}: {
  item: HierarchyNode;
  depth: number;
  expandedNodeIds: Set<string>;
  selectedNodeId: string | null;
  onToggleNode: (nodeId: string) => void;
  onRevealNode: (nodeId: string) => void;
}) {
  const { node, children } = item;
  const hasChildren = children.length > 0;
  const isExpanded = expandedNodeIds.has(node.id);
  const isSelected = selectedNodeId === node.id;

  return (
    <li className="min-w-0">
      <div
        className="flex min-w-0 items-center gap-1.5 rounded-control-md pr-1"
        style={{ paddingLeft: depth * 14 }}
      >
        {hasChildren ? (
          <button
            type="button"
            aria-label={`${isExpanded ? "Collapse" : "Expand"} structure ${node.path}`}
            aria-expanded={isExpanded}
            onClick={() => onToggleNode(node.id)}
            className="flex h-touch w-touch shrink-0 items-center justify-center rounded-control-sm text-ink-faint transition-colors hover:bg-control-hover hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:h-control-sm md:w-control-sm"
          >
            <ChevronRight
              className={cn("h-3.5 w-3.5 transition-transform", isExpanded && "rotate-90")}
              aria-hidden
            />
          </button>
        ) : (
          <span className="h-touch w-touch shrink-0 md:h-control-sm md:w-control-sm" aria-hidden />
        )}
        <button
          type="button"
          aria-label={`Reveal ${node.path} in graph`}
          aria-current={isSelected ? "true" : undefined}
          onClick={() => onRevealNode(node.id)}
          className={cn(
            "grid min-h-touch min-w-0 flex-1 grid-cols-[minmax(0,1fr)_auto] items-center gap-x-2 rounded-control-md border px-2 py-1.5 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:min-h-control-sm",
            isSelected
              ? "border-cyan/55 bg-cyan/[0.12] text-ink shadow-cyan-selection"
              : "border-transparent bg-transparent text-ink-dim hover:border-white/[0.08] hover:bg-white/[0.055] hover:text-ink",
          )}
        >
          <span className="min-w-0 truncate type-label font-semibold">
            {structureNodeLabel(node)}
          </span>
          <span className="shrink-0 font-mono type-caption text-ink-faint">
            {children.length}
          </span>
          <span className="min-w-0 truncate font-mono type-caption text-ink-faint">
            {node.path}
          </span>
          {node.parameterCount > 0 ? (
            <StructureBadge>{formatCompactCount(node.parameterCount)}</StructureBadge>
          ) : (
            <span aria-hidden />
          )}
        </button>
      </div>
      {hasChildren && isExpanded && (
        <ol className="mt-1 grid gap-1">
          {children.map((child) => (
            <StructureTreeItem
              key={child.node.id}
              item={child}
              depth={depth + 1}
              expandedNodeIds={expandedNodeIds}
              selectedNodeId={selectedNodeId}
              onToggleNode={onToggleNode}
              onRevealNode={onRevealNode}
            />
          ))}
        </ol>
      )}
    </li>
  );
}

export function GraphStructurePanel({
  panelId,
  graph,
  selectedNodeId,
  onRevealNode,
  onClose,
}: GraphStructurePanelProps) {
  const roots = useMemo(() => buildHierarchy(graph), [graph]);
  const defaultExpanded = useMemo(
    () => defaultExpandedNodeIds(graph, selectedNodeId),
    [graph, selectedNodeId],
  );
  const [expansion, setExpansion] = useState(() => ({
    graph,
    selectedNodeId,
    nodeIds: defaultExpanded,
  }));
  const expandedNodeIds = useMemo(() => {
    if (
      expansion.graph === graph &&
      expansion.selectedNodeId === selectedNodeId
    ) {
      return expansion.nodeIds;
    }
    return new Set([...expansion.nodeIds, ...defaultExpanded]);
  }, [defaultExpanded, expansion, graph, selectedNodeId]);

  return (
    <section
      id={panelId}
      data-testid="graph-structure-panel"
      aria-labelledby={`${panelId}-title`}
      onClick={stopPanelEvent}
      onMouseDown={stopPanelEvent}
      onPointerDown={stopPanelEvent}
      className="nowheel nodrag nopan edge w-[min(360px,calc(100vw-2rem))] overflow-hidden rounded-card bg-structure-overlay shadow-popover backdrop-blur-xl"
    >
      <div className="flex min-w-0 items-center justify-between gap-3 border-b border-white/[0.08] px-3 py-2.5">
        <div className="min-w-0">
          <h2
            id={`${panelId}-title`}
            className="truncate type-compact font-bold text-ink"
          >
            Structure
          </h2>
          <div className="mt-0.5 font-mono type-caption text-ink-faint">
            {formatCompactCount(graph?.nodes.length ?? 0)} nodes
          </div>
        </div>
        <IconButton
          label="Close graph structure"
          title="Close graph structure"
          icon={<X className="h-3.5 w-3.5" aria-hidden />}
          size="sm"
          variant="ghost"
          onClick={onClose}
        />
      </div>
      <div className="nowheel max-h-[min(62vh,560px)] overflow-y-auto px-2 py-2">
        {roots.length > 0 ? (
          <ol aria-label="Graph structure" className="grid gap-1">
            {roots.map((root) => (
              <StructureTreeItem
                key={root.node.id}
                item={root}
                depth={0}
                expandedNodeIds={expandedNodeIds}
                selectedNodeId={selectedNodeId}
                onToggleNode={(nodeId) =>
                  setExpansion({
                    graph,
                    selectedNodeId,
                    nodeIds: nextExpandedNodeIds(expandedNodeIds, nodeId),
                  })
                }
                onRevealNode={onRevealNode}
              />
            ))}
          </ol>
        ) : (
          <div className="px-2 py-8 text-center type-label text-ink-faint">
            No graph nodes
          </div>
        )}
      </div>
    </section>
  );
}
