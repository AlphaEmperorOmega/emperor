import { SelectHTMLAttributes, forwardRef } from "react";
import { cn } from "@/lib/utils";

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, ...props }, ref) => (
    <select
      ref={ref}
      className={cn(
        "h-9 w-full min-w-0 rounded-md border border-border bg-panel px-2.5 font-mono text-sm text-ink outline-none transition focus-visible:border-accent focus-visible:ring-2 focus-visible:ring-focus",
        className,
      )}
      {...props}
    />
  ),
);

Select.displayName = "Select";
