import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

export function StatusPill({
  icon,
  label,
  value,
  tone = "neutral",
}: {
  icon: ReactNode;
  label: string;
  value: ReactNode;
  tone?: "neutral" | "good" | "warn" | "danger";
}) {
  const toneClass = {
    neutral: "border-border bg-panel text-ink",
    good: "border-[#a5d6c5] bg-[#edf9f3] text-accent",
    warn: "border-[#efcf91] bg-[#fff9ec] text-[#8a5714]",
    danger: "border-danger-line bg-danger-soft text-danger",
  }[tone];

  return (
    <div
      className={cn(
        "inline-flex h-9 items-center gap-2 rounded-md border px-2.5 text-xs font-medium",
        toneClass,
      )}
    >
      <span className="shrink-0" aria-hidden>
        {icon}
      </span>
      <span className="text-muted">{label}</span>
      <span className="font-semibold text-current">{value}</span>
    </div>
  );
}
