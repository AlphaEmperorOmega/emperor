import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

export function StatusPill({
  icon,
  label,
  value,
  tone = "neutral",
  className,
}: {
  icon: ReactNode;
  label: string;
  value: ReactNode;
  tone?: "neutral" | "good" | "warn" | "danger";
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
      className={cn(
        "inline-flex h-[34px] items-center gap-2 rounded-[9px] border border-line bg-white/[0.025] px-2.5 text-xs font-medium text-ink-dim",
        toneClass,
        className,
      )}
    >
      <span className="shrink-0 text-ink-faint" aria-hidden>
        {icon}
      </span>
      <span className="text-ink-dim">{label}</span>
      <span className="font-mono font-bold text-current">{value}</span>
    </div>
  );
}
