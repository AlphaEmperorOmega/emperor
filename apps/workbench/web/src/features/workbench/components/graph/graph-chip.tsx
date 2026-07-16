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
  default: "border-line-soft bg-control text-ink-dim",
  violet:
    "border-violet/30 bg-accent-soft text-violet-text",
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
        compact ? "rounded-chip px-1.5 py-1 type-caption" : "rounded-control-md px-2 text-xs",
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
