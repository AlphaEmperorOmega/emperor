import { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export type BadgeVariant =
  | "default"
  | "override"
  | "preset"
  | "success"
  | "warning"
  | "danger"
  | "info"
  | "violet";

type BadgeProps = HTMLAttributes<HTMLSpanElement> & {
  variant?: BadgeVariant;
};

export const badgeVariantClassNames: Readonly<Record<BadgeVariant, string>> = {
  default: "",
  override: "border-violet/30 bg-violet/15 text-violet",
  preset: "border-amber/40 bg-amber/[0.12] text-amber",
  success: "border-ok/30 bg-ok/10 text-ok",
  warning: "border-amber/40 bg-amber/[0.12] text-amber",
  danger: "border-danger-line bg-danger-soft text-danger-text",
  info: "border-blue/30 bg-blue/10 text-blue",
  violet: "border-violet/30 bg-violet/15 text-violet",
};

export function Badge({ className, variant = "default", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex max-w-full min-w-0 whitespace-normal rounded-chip border border-line bg-control px-2 py-1 font-mono type-meta font-semibold leading-4 text-ink-dim [overflow-wrap:anywhere]",
        badgeVariantClassNames[variant],
        className,
      )}
      {...props}
    />
  );
}
