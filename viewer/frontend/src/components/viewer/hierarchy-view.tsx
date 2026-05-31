"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronRight, ListTree } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/viewer/empty-state";
import { type HierarchyNode, buildHierarchy, detailText, nodeBadges } from "@/lib/graph";
import { type InspectResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

function HierarchyNodeRow({
  item,
  depth,
  isSelected,
  isOpen,
  onToggle,
  onSelect,
}: {
  item: HierarchyNode;
  depth: number;
  isSelected: boolean;
  isOpen: boolean;
  onToggle: () => void;
  onSelect: () => void;
}) {
  const badges = nodeBadges(item.node.details);
  const hasChildren = item.children.length > 0;

  return (
    <button
      type="button"
      aria-expanded={hasChildren ? isOpen : undefined}
      onClick={() => {
        onSelect();
        if (hasChildren) {
          onToggle();
        }
      }}
      className={cn(
        "grid min-h-[48px] w-full grid-cols-[24px_minmax(0,1fr)_auto] items-center gap-2 border-b border-border px-3 py-2 text-left transition",
        isSelected
          ? "bg-accent-soft text-ink shadow-[inset_3px_0_0_#15705f]"
          : "bg-panel text-ink hover:bg-surface",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus",
      )}
      style={{ paddingLeft: 12 + depth * 18 }}
    >
      <span className="flex h-6 w-6 items-center justify-center text-muted">
        {hasChildren && (
          <ChevronRight
            className={cn("h-4 w-4 transition-transform", isOpen && "rotate-90")}
            aria-hidden
          />
        )}
      </span>
      <span className="min-w-0">
        <span className="block truncate text-sm font-semibold">{item.node.typeName}</span>
        <span className="block truncate font-mono text-[11px] text-muted">
          {item.node.path}
        </span>
      </span>
      <span className="flex min-w-0 max-w-[360px] flex-wrap justify-end gap-1 max-md:hidden">
        {hasChildren && <Badge>{item.children.length} children</Badge>}
        {badges.slice(0, 3).map(([key, value]) => (
          <Badge key={`${item.node.id}-${key}-${value}`}>{`${key}: ${detailText(value)}`}</Badge>
        ))}
      </span>
    </button>
  );
}

function HierarchyBranch({
  item,
  depth,
  openNodeIds,
  selectedNodeId,
  onToggleNode,
  onSelectNode,
}: {
  item: HierarchyNode;
  depth: number;
  openNodeIds: Set<string>;
  selectedNodeId: string | null;
  onToggleNode: (nodeId: string) => void;
  onSelectNode: (nodeId: string) => void;
}) {
  const isOpen = openNodeIds.has(item.node.id);

  return (
    <div>
      <HierarchyNodeRow
        item={item}
        depth={depth}
        isSelected={selectedNodeId === item.node.id}
        isOpen={isOpen}
        onToggle={() => onToggleNode(item.node.id)}
        onSelect={() => onSelectNode(item.node.id)}
      />
      {isOpen &&
        item.children.map((child) => (
          <HierarchyBranch
            key={child.node.id}
            item={child}
            depth={depth + 1}
            openNodeIds={openNodeIds}
            selectedNodeId={selectedNodeId}
            onToggleNode={onToggleNode}
            onSelectNode={onSelectNode}
          />
        ))}
    </div>
  );
}

export function HierarchyView({
  graph,
  selectedNodeId,
  onSelectNode,
}: {
  graph: InspectResponse | undefined;
  selectedNodeId: string | null;
  onSelectNode: (nodeId: string) => void;
}) {
  const roots = useMemo(() => buildHierarchy(graph), [graph]);
  const [openNodeIds, setOpenNodeIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    setOpenNodeIds(new Set(roots.map((root) => root.node.id)));
  }, [roots]);

  const toggleNode = (nodeId: string) => {
    setOpenNodeIds((current) => {
      const next = new Set(current);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  };

  if (!graph) {
    return (
      <div className="relative h-full">
        <EmptyState
          title="No hierarchy loaded"
          detail="Preview data has not returned yet."
          icon={<ListTree className="h-4 w-4" aria-hidden />}
        />
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto bg-surface p-4">
      <div className="overflow-hidden rounded-md border border-border bg-panel shadow-panel">
        {roots.length === 0 ? (
          <div className="p-6 text-center text-sm text-muted">No graph roots found.</div>
        ) : (
          roots.map((root) => (
            <HierarchyBranch
              key={root.node.id}
              item={root}
              depth={0}
              openNodeIds={openNodeIds}
              selectedNodeId={selectedNodeId}
              onToggleNode={toggleNode}
              onSelectNode={onSelectNode}
            />
          ))
        )}
      </div>
    </div>
  );
}
