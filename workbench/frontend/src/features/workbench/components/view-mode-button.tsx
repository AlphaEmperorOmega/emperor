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
      tabIndex={active ? 0 : -1}
      className={cn(
        "inline-flex h-touch min-w-touch items-center gap-2 rounded-control-sm px-3 text-xs font-semibold transition-[color,background-color,border-color,box-shadow,transform] duration-150 ease-out focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-1 focus-visible:ring-offset-bg md:h-control-sm md:min-w-0",
        active
          ? "bg-violet-deep text-white shadow-control-active"
          : "text-ink-faint hover:bg-control-active hover:text-ink",
        disabled && "cursor-not-allowed opacity-50 hover:bg-transparent hover:text-ink-faint",
      )}
    >
      {children}
    </button>
  );
}
