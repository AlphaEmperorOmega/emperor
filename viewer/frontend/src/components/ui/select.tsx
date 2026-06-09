import { SelectHTMLAttributes, forwardRef } from "react";
import { selectControlClassName } from "@/components/ui/control-styles";
import { cn } from "@/lib/utils";

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, ...props }, ref) => (
    <select
      ref={ref}
      className={cn(
        selectControlClassName,
        className,
      )}
      {...props}
    />
  ),
);

Select.displayName = "Select";
