import { type ReactNode, useId, useState } from "react";
import { cn } from "@/lib/utils";

export type HoverTooltipTriggerProps = {
  "aria-describedby"?: string;
  onBlur: () => void;
  onFocus: () => void;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
};

export function HoverTooltip({
  children,
  className,
  tooltip,
  tooltipClassName,
}: {
  children: (props: HoverTooltipTriggerProps) => ReactNode;
  className?: string;
  tooltip: string;
  tooltipClassName?: string;
}) {
  const [isTooltipVisible, setIsTooltipVisible] = useState(false);
  const tooltipId = useId();
  const triggerProps = {
    "aria-describedby": tooltipId,
    onBlur: () => setIsTooltipVisible(false),
    onFocus: () => setIsTooltipVisible(true),
    onMouseEnter: () => setIsTooltipVisible(true),
    onMouseLeave: () => setIsTooltipVisible(false),
  };

  return (
    <span className={cn("relative inline-flex", className)}>
      {children(triggerProps)}
      <span
        id={tooltipId}
        role="tooltip"
        className={cn(
          isTooltipVisible
            ? "pointer-events-none absolute top-[calc(100%+6px)] z-30 whitespace-nowrap rounded-[7px] border border-line-soft bg-panel px-2 py-1 font-sans text-[11px] font-bold leading-none text-ink shadow-panel"
            : "sr-only",
          isTooltipVisible && tooltipClassName,
        )}
      >
        {tooltip}
      </span>
    </span>
  );
}
