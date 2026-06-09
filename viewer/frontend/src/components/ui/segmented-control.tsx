import { type HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

// Styled `role="tablist"` container for a row of segmented tab buttons. Pass the
// group label via `aria-label` and the segment buttons (e.g. ViewModeButton) as
// children.
export function SegmentedControl({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      role="tablist"
      className={cn(
        "inline-flex rounded-control border border-line bg-control-subtle p-[3px]",
        className,
      )}
      {...props}
    />
  );
}
