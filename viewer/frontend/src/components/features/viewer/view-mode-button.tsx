import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

export function ViewModeButton({
  active,
  children,
  disabled = false,
  onClick,
}: {
  active: boolean;
  children: ReactNode;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "inline-flex h-[30px] items-center gap-2 rounded-[7px] px-3.5 text-[12.5px] font-semibold transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
        active
          ? "bg-grad text-white shadow-[0_4px_12px_-4px_rgba(124,109,255,0.7)]"
          : "text-ink-faint hover:bg-white/[0.045] hover:text-ink",
        disabled && "cursor-not-allowed opacity-50 hover:bg-transparent hover:text-ink-faint",
      )}
    >
      {children}
    </button>
  );
}
