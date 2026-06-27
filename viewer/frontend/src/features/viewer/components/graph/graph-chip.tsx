import { type HTMLAttributes, type ReactNode, type Ref } from "react";
import { cn } from "@/lib/utils";

type GraphChipTone = "default" | "violet" | "success" | "warning" | "none";

export type GraphChipProps = Omit<
  HTMLAttributes<HTMLSpanElement>,
  "children" | "className" | "title"
> & {
  tone?: GraphChipTone;
  compact?: boolean;
  title?: string;
  children: ReactNode;
  className?: string;
  elementRef?: Ref<HTMLSpanElement>;
};

const graphChipToneClassNames: Readonly<Record<GraphChipTone, string>> = {
  default: "border-line-soft bg-white/[0.02] text-ink-dim",
  violet:
    "border-violet/30 bg-violet/15 text-violet-text shadow-[inset_0_-1px_0_rgba(146,113,255,0.24)]",
  success: "border-ok/30 bg-ok/10 text-ok",
  warning: "border-amber/40 bg-amber/[0.12] text-amber",
  none: "",
};

export function GraphChip({
  tone = "default",
  compact = false,
  title,
  children,
  className,
  elementRef,
  ...props
}: GraphChipProps) {
  return (
    <span
      ref={elementRef}
      title={title}
      className={cn(
        "border",
        compact ? "rounded-[7px] px-1.5 py-1 text-[10px]" : "rounded-[8px] px-2 text-[12px]",
        graphChipToneClassNames[tone],
        className,
        "leading-none",
      )}
      {...props}
    >
      {children}
    </span>
  );
}
