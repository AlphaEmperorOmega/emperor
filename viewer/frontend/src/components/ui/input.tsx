import { InputHTMLAttributes, forwardRef } from "react";
import { fieldControlClassName } from "@/components/ui/control-styles";
import { cn } from "@/lib/utils";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        fieldControlClassName,
        "h-9 px-2.5 font-mono text-sm placeholder:text-ink-faint",
        className,
      )}
      {...props}
    />
  ),
);

Input.displayName = "Input";
