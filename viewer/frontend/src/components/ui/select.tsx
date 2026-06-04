import { SelectHTMLAttributes, forwardRef } from "react";
import { cn } from "@/lib/utils";

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, ...props }, ref) => (
    <select
      ref={ref}
      className={cn(
        "h-10 w-full min-w-0 rounded-[10px] border border-line bg-[linear-gradient(155deg,#161622,#0e0e18)] px-3 font-sans text-[13.5px] font-semibold text-ink outline-none transition hover:border-white/15 focus-visible:border-violet/60 focus-visible:ring-2 focus-visible:ring-focus",
        className,
      )}
      {...props}
    />
  ),
);

Select.displayName = "Select";
