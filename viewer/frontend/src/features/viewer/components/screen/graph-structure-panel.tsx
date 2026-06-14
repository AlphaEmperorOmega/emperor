import { useEffect, useMemo, useState, type ReactNode, type SyntheticEvent } from "react";
import { ChevronRight, X } from "lucide-react";
import { IconButton } from "@/components/ui/icon-button";
import { type InspectResponse } from "@/lib/api";
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
    <span className="shrink-0 rounded-[5px] border border-white/[0.08] bg-white/[0.045] px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.04em] text-ink-faint">
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
        className="flex min-w-0 items-center gap-1.5 rounded-[7px] pr-1"
        style={{ paddingLeft: depth * 14 }}
      >
        {hasChildren ? (
          <button
            type="button"
            aria-label={`${isExpanded ? "Collapse" : "Expand"} structure ${node.path}`}
            aria-expanded={isExpanded}
            onClick={() => onToggleNode(node.id)}
            className="flex h-6 w-6 shrink-0 items-center justify-center rounded-[6px] text-ink-faint transition hover:bg-white/[0.07] hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          >
            <ChevronRight
              className={cn("h-3.5 w-3.5 transition-transform", isExpanded && "rotate-90")}
              aria-hidden
            />
          </button>
        ) : (
          <span className="h-6 w-6 shrink-0" aria-hidden />
        )}
        <button
          type="button"
          aria-label={`Reveal ${node.path} in graph`}
          aria-current={isSelected ? "true" : undefined}
          onClick={() => onRevealNode(node.id)}
          className={cn(
            "grid min-w-0 flex-1 grid-cols-[minmax(0,1fr)_auto] items-center gap-x-2 rounded-[7px] border px-2 py-1.5 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
            isSelected
              ? "border-cyan-200/55 bg-cyan-300/[0.12] text-ink shadow-[0_0_22px_rgba(34,211,238,0.11)]"
              : "border-transparent bg-transparent text-ink-dim hover:border-white/[0.08] hover:bg-white/[0.055] hover:text-ink",
          )}
        >
          <span className="min-w-0 truncate text-[12px] font-semibold">
            {structureNodeLabel(node)}
          </span>
          <span className="shrink-0 font-mono text-[10px] text-ink-faint">
            {children.length}
          </span>
          <span className="min-w-0 truncate font-mono text-[10px] text-ink-faint">
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
  const [expandedNodeIds, setExpandedNodeIds] = useState(
    () => defaultExpandedNodeIds(graph, selectedNodeId),
  );

  useEffect(() => {
    setExpandedNodeIds((current) => {
      const next = new Set(current);
      let changed = false;
      for (const nodeId of defaultExpandedNodeIds(graph, selectedNodeId)) {
        if (!next.has(nodeId)) {
          next.add(nodeId);
          changed = true;
        }
      }
      return changed ? next : current;
    });
  }, [graph, selectedNodeId]);

  return (
    <aside
      id={panelId}
      data-testid="graph-structure-panel"
      aria-label="Graph structure"
      onClick={stopPanelEvent}
      onMouseDown={stopPanelEvent}
      onPointerDown={stopPanelEvent}
      className="nowheel nodrag nopan edge w-[min(360px,calc(100vw-2rem))] overflow-hidden rounded-card bg-[rgba(8,9,15,0.92)] shadow-[0_22px_70px_rgba(0,0,0,0.48)] backdrop-blur-xl"
    >
      <div className="flex min-w-0 items-center justify-between gap-3 border-b border-white/[0.08] px-3 py-2.5">
        <div className="min-w-0">
          <h2 className="truncate text-[13px] font-bold text-ink">Structure</h2>
          <div className="mt-0.5 font-mono text-[10px] text-ink-faint">
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
                  setExpandedNodeIds((current) => nextExpandedNodeIds(current, nodeId))
                }
                onRevealNode={onRevealNode}
              />
            ))}
          </ol>
        ) : (
          <div className="px-2 py-8 text-center text-[12px] text-ink-faint">
            No graph nodes
          </div>
        )}
      </div>
    </aside>
  );
}
