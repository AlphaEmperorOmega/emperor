import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

export function ViewModeButton({
  active,
  children,
  controls,
  disabled = false,
  id,
  onClick,
  variant = "radio",
}: {
  active: boolean;
  children: ReactNode;
  controls?: string;
  disabled?: boolean;
  id?: string;
  onClick: () => void;
  variant?: "radio" | "tab";
}) {
  const isTab = variant === "tab";

  return (
    <button
      id={id}
      type="button"
      role={isTab ? "tab" : "radio"}
      aria-selected={isTab ? active : undefined}
      aria-checked={isTab ? undefined : active}
      aria-controls={isTab ? controls : undefined}
      disabled={disabled}
      onClick={onClick}
      tabIndex={isTab ? undefined : active ? 0 : -1}
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
