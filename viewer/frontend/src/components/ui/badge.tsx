import { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Badge({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn(
        "inline-flex max-w-full min-w-0 whitespace-normal rounded border border-border bg-surface px-1.5 py-1 text-[11px] font-medium leading-tight text-muted [overflow-wrap:anywhere]",
        className,
      )}
      {...props}
    />
  );
}
