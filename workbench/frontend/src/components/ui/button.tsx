import { ButtonHTMLAttributes, forwardRef } from "react";
import { controlButtonClassName } from "@/components/ui/control-styles";
import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
};

const variants: Record<ButtonVariant, string> = {
  primary:
    "border-0 bg-grad text-white shadow-primary hover:brightness-110 active:translate-y-px",
  secondary:
    "border border-line bg-control text-ink-dim shadow-none hover:border-line-hover hover:bg-control-hover hover:text-ink active:translate-y-px",
  ghost:
    "border border-transparent text-ink-faint hover:bg-control-active hover:text-ink active:translate-y-px",
  danger:
    "border border-danger-line bg-danger-soft text-danger-text hover:border-danger/60 hover:bg-danger-hover/40 hover:text-white active:translate-y-px",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "secondary", type = "button", ...props }, ref) => (
    <button
      ref={ref}
      type={type}
      className={cn(
        controlButtonClassName,
        variants[variant],
        className,
      )}
      {...props}
    />
  ),
);

Button.displayName = "Button";
