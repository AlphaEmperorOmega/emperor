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
  line: "grid grid-cols-[104px_minmax(0,1fr)] gap-3 border-b border-line-soft py-3 text-xs",
  card: "grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2 rounded-[9px] border border-line-soft bg-black/20 px-3 py-2 text-xs",
};

const valueClasses: Record<NonNullable<KeyValueRowProps["variant"]>, string> = {
  line: "break-words text-right font-mono font-semibold text-ink",
  card: "font-mono font-semibold text-ink",
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
