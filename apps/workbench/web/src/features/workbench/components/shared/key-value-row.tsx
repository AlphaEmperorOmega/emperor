import { type HTMLAttributes, type ReactNode } from "react";
import { cn } from "@/lib/utils";

export type KeyValueRowProps = Omit<HTMLAttributes<HTMLDivElement>, "children"> & {
  label: ReactNode;
  value: ReactNode;
  variant?: "line" | "card";
  labelClassName?: string;
  valueClassName?: string;
};

const rowClasses: Record<NonNullable<KeyValueRowProps["variant"]>, string> = {
  line: "grid grid-cols-[104px_minmax(0,1fr)] gap-panel border-b border-line-soft py-panel text-xs",
  card: "grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2 rounded-control-md border border-line-soft bg-control-field px-panel py-2 text-xs",
};

const valueClasses: Record<NonNullable<KeyValueRowProps["variant"]>, string> = {
  line: "break-words text-right font-mono font-semibold tabular-nums text-ink",
  card: "font-mono font-semibold tabular-nums text-ink",
};

export function KeyValueRow({
  label,
  value,
  variant = "line",
  className,
  labelClassName,
  valueClassName,
  ...props
}: KeyValueRowProps) {
  return (
    <div className={cn(rowClasses[variant], className)} {...props}>
      <span className={cn("truncate text-ink-dim", labelClassName)}>{label}</span>
      <span className={cn(valueClasses[variant], valueClassName)}>{value}</span>
    </div>
  );
}
