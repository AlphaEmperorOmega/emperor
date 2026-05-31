import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

export function ViewModeButton({
  active,
  children,
  onClick,
}: {
  active: boolean;
  children: ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={cn(
        "inline-flex h-8 items-center gap-2 rounded-md px-3 text-sm font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
        active
          ? "bg-panel text-ink shadow-panel ring-1 ring-border"
          : "text-muted hover:bg-panel hover:text-ink",
      )}
    >
      {children}
    </button>
  );
}
