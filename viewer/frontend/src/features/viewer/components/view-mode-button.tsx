import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

export function ViewModeButton({
  active,
  children,
  controls,
  disabled = false,
  id,
  onClick,
}: {
  active: boolean;
  children: ReactNode;
  controls?: string;
  disabled?: boolean;
  id?: string;
  onClick: () => void;
}) {
  return (
    <button
      id={id}
      type="button"
      role="tab"
      aria-selected={active}
      aria-controls={controls}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "inline-flex h-[30px] items-center gap-2 rounded-control-sm px-3.5 text-[12.5px] font-semibold transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
        active
          ? "bg-grad text-white shadow-control-active"
          : "text-ink-faint hover:bg-control-active hover:text-ink",
        disabled && "cursor-not-allowed opacity-50 hover:bg-transparent hover:text-ink-faint",
      )}
    >
      {children}
    </button>
  );
}
