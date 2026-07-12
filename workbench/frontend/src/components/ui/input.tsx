import { InputHTMLAttributes, forwardRef } from "react";
import { fieldControlClassName } from "@/components/ui/control-styles";
import { cn } from "@/lib/utils";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        fieldControlClassName,
        "h-touch px-3 font-mono type-body tabular-nums placeholder:text-ink-faint md:h-control",
        className,
      )}
      {...props}
    />
  ),
);

Input.displayName = "Input";
