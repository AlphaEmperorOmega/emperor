import { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

type BadgeVariant = "default" | "override" | "preset";

type BadgeProps = HTMLAttributes<HTMLSpanElement> & {
  variant?: BadgeVariant;
};

const badgeVariants: Record<BadgeVariant, string> = {
  default: "",
  override: "border-violet/30 bg-violet/15 text-violet",
  preset: "border-amber/40 bg-amber/[0.12] text-amber",
};

export function Badge({ className, variant = "default", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex max-w-full min-w-0 whitespace-normal rounded-[7px] border border-line bg-white/[0.04] px-2 py-0.5 font-mono text-xs font-semibold leading-tight text-ink-dim [overflow-wrap:anywhere]",
        badgeVariants[variant],
        className,
      )}
      {...props}
    />
  );
}
