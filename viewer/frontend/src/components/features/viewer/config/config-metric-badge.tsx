import { useState } from "react";
import { ListChecks, PencilLine, type LucideIcon } from "lucide-react";
import { badgeVariantClassNames } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type ConfigMetricKind = "fields" | "overrides";
type ConfigMetricVariant = "default" | "override";
type TooltipPosition = "top" | "bottom";

const metricIcons: Record<ConfigMetricKind, LucideIcon> = {
  fields: ListChecks,
  overrides: PencilLine,
};

const metricLabels: Record<ConfigMetricKind, { tooltip: string; noun: string }> = {
  fields: { tooltip: "Fields", noun: "field" },
  overrides: { tooltip: "Overrides", noun: "override" },
};

export function ConfigMetricBadge({
  count,
  kind,
  variant = "default",
  focusable = true,
  tooltipPosition = "top",
}: {
  count: number;
  kind: ConfigMetricKind;
  variant?: ConfigMetricVariant;
  focusable?: boolean;
  tooltipPosition?: TooltipPosition;
}) {
  const [isTooltipVisible, setIsTooltipVisible] = useState(false);
  const Icon = metricIcons[kind];
  const label = metricLabels[kind];
  const accessibleLabel = `${count} ${label.noun}${count === 1 ? "" : "s"}`;

  return (
    <span
      aria-label={accessibleLabel}
      tabIndex={focusable ? 0 : undefined}
      onBlur={() => setIsTooltipVisible(false)}
      onFocus={() => setIsTooltipVisible(true)}
      onMouseEnter={() => setIsTooltipVisible(true)}
      onMouseLeave={() => setIsTooltipVisible(false)}
      className={cn(
        "relative inline-flex h-[23px] shrink-0 items-center gap-1.5 whitespace-nowrap rounded-[7px] border border-line bg-white/[0.04] px-2 font-mono text-xs font-semibold leading-none text-ink-dim transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
        badgeVariantClassNames[variant],
      )}
    >
      <Icon className="h-3.5 w-3.5" aria-hidden />
      <span>{count}</span>
      {isTooltipVisible && (
        <span
          role="tooltip"
          className={cn(
            "pointer-events-none absolute left-1/2 z-30 -translate-x-1/2 rounded-[7px] border border-line-soft bg-panel px-2 py-1 font-sans text-[11px] font-bold leading-none text-ink shadow-panel",
            tooltipPosition === "top" ? "bottom-[calc(100%+6px)]" : "top-[calc(100%+6px)]",
          )}
        >
          {label.tooltip}
        </span>
      )}
    </span>
  );
}
