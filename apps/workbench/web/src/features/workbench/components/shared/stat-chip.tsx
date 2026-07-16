import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

export type StatChipTone =
  | "default"
  | "soft"
  | "violet"
  | "success"
  | "warning"
  | "danger";

export type StatChipProps = {
  tone?: StatChipTone;
  size?: "xs" | "sm";
  children: ReactNode;
  className?: string;
};

const toneClasses: Record<StatChipTone, string> = {
  default: "border-line bg-control text-ink-dim",
  soft: "border-line-soft bg-control-field type-meta font-semibold leading-none text-ink-dim",
  violet: "border-violet/30 bg-violet/10 text-violet",
  success: "border-ok/30 bg-ok/10 text-ok",
  warning: "border-amber/40 bg-amber/[0.12] text-amber",
  danger: "border-danger-line bg-danger-soft text-danger-text",
};

const sizeClasses: Record<NonNullable<StatChipProps["size"]>, string> = {
  xs: "px-1.5 py-0.5 type-meta",
  sm: "px-2 py-1 text-xs",
};

export function StatChip({
  tone = "default",
  size = "sm",
  children,
  className,
}: StatChipProps) {
  return (
    <span
      className={cn(
        "rounded-chip border font-mono tabular-nums",
        sizeClasses[size],
        toneClasses[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
