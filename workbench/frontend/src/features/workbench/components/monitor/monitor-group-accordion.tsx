import { useEffect, useRef, useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { SurfacePanel } from "@/components/ui/surface-panel";
import {
  formatGroupCount,
  monitorGroupPanelId,
} from "@/lib/monitor/grouping";
import { cn } from "@/lib/utils";
import { type MonitorGroup, monitorGroupOrder } from "@/types/monitor";

export function MonitorGroupAccordion({
  idPrefix,
  group,
  count,
  countUnit,
  isOpen,
  onToggle,
  children,
}: {
  idPrefix: string;
  group: MonitorGroup;
  count: number;
  countUnit: "chart" | "pair";
  isOpen: boolean;
  onToggle: () => void;
  children: ReactNode;
}) {
  const panelId = monitorGroupPanelId(idPrefix, group);

  return (
    <SurfacePanel as="section" padding="none" className="overflow-hidden">
      <h3>
        <button
          type="button"
          aria-expanded={isOpen}
          aria-controls={panelId}
          onClick={onToggle}
          className="flex w-full items-center justify-between gap-3 p-3 text-left transition hover:bg-white/[0.035] focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          <span className="flex min-w-0 items-center gap-2">
            <ChevronDown
              className={cn(
                "h-4 w-4 shrink-0 text-ink-faint transition-transform",
                !isOpen && "-rotate-90",
              )}
              aria-hidden
            />
            <span className="truncate text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
              {group}
            </span>
          </span>
          <Badge>{formatGroupCount(count, countUnit)}</Badge>
        </button>
      </h3>
      {isOpen && (
        <div id={panelId} className="border-t border-line-soft p-3">
          {children}
        </div>
      )}
    </SurfacePanel>
  );
}

export function useMonitorGroupAccordion(
  groupCounts: Record<MonitorGroup, number>,
  { startCollapsed = false }: { startCollapsed?: boolean } = {},
) {
  const firstNonEmptyGroup = monitorGroupOrder.find((group) => groupCounts[group] > 0);
  const [openGroups, setOpenGroups] = useState<MonitorGroup[]>(
    startCollapsed || !firstNonEmptyGroup ? [] : [firstNonEmptyGroup],
  );
  const countsKey = monitorGroupOrder
    .map((group) => `${group}:${groupCounts[group]}`)
    .join("|");
  const previousCountsKey = useRef<string | undefined>(undefined);

  useEffect(() => {
    if (previousCountsKey.current === countsKey) {
      return;
    }
    previousCountsKey.current = countsKey;
    setOpenGroups((currentGroups) => {
      const retainedGroups = currentGroups.filter((group) => groupCounts[group] > 0);
      if (retainedGroups.length > 0 || startCollapsed || !firstNonEmptyGroup) {
        return retainedGroups;
      }
      return [firstNonEmptyGroup];
    });
  }, [countsKey, firstNonEmptyGroup, groupCounts, startCollapsed]);

  return {
    isGroupOpen: (group: MonitorGroup) => openGroups.includes(group),
    toggleGroup: (group: MonitorGroup) =>
      setOpenGroups((currentGroups) =>
        currentGroups.includes(group)
          ? currentGroups.filter((currentGroup) => currentGroup !== group)
          : [...currentGroups, group],
      ),
  };
}
