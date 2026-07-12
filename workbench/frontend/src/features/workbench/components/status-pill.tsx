import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

export function StatusPill({
  icon,
  label,
  value,
  tone = "neutral",
  live = false,
  className,
}: {
  icon: ReactNode;
  label: string;
  value: ReactNode;
  tone?: "neutral" | "good" | "warn" | "danger";
  live?: boolean;
  className?: string;
}) {
  const toneClass = {
    neutral: "text-ink",
    good: "text-ok",
    warn: "text-amber",
    danger: "text-danger",
  }[tone];

  return (
    <div
      role={live ? "status" : undefined}
      aria-live={live ? "polite" : undefined}
      className={cn(
        "inline-flex h-control max-w-full min-w-0 items-center gap-2 rounded-control-md border border-line bg-panel-2/80 px-2.5 text-xs font-medium text-ink-dim shadow-control",
        toneClass,
        className,
      )}
    >
      <span className="shrink-0 text-ink-faint" aria-hidden>
        {icon}
      </span>
      <span className="min-w-0 truncate text-ink-dim">{label}</span>
      <span className="min-w-0 break-words font-mono font-bold tabular-nums text-current">
        {value}
      </span>
    </div>
  );
}
