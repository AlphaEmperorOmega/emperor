import { ButtonHTMLAttributes, forwardRef } from "react";
import { controlButtonClassName } from "@/components/ui/control-styles";
import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
};

const variants: Record<ButtonVariant, string> = {
  primary:
    "border border-violet/70 bg-selected-control text-white shadow-primary hover:border-violet hover:bg-selected-control/90 active:translate-y-px",
  secondary:
    "border border-line bg-panel-2/90 text-ink-dim shadow-none hover:border-line-hover hover:bg-control-hover hover:text-ink active:translate-y-px",
  ghost:
    "border border-transparent bg-transparent text-ink-faint hover:border-line-soft hover:bg-control-active hover:text-ink active:translate-y-px",
  danger:
    "border border-danger-line bg-danger-soft text-danger-text hover:border-danger/70 hover:bg-danger-hover/45 hover:text-white active:translate-y-px",
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
