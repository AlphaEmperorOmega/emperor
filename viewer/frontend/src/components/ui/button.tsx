import { ButtonHTMLAttributes, forwardRef } from "react";
import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
};

const variants: Record<ButtonVariant, string> = {
  primary:
    "border-0 bg-grad text-white shadow-primary hover:brightness-110 active:translate-y-px",
  secondary:
    "border border-line bg-white/[0.035] text-ink-dim shadow-none hover:border-white/15 hover:bg-white/[0.07] hover:text-ink active:translate-y-px",
  ghost:
    "border border-transparent text-ink-faint hover:bg-white/[0.055] hover:text-ink active:translate-y-px",
  danger:
    "border border-danger-line bg-danger-soft text-[#fda4af] hover:border-[#fb7185]/60 hover:bg-[#7f1d2d]/40 hover:text-white active:translate-y-px",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "secondary", type = "button", ...props }, ref) => (
    <button
      ref={ref}
      type={type}
      className={cn(
        "inline-flex h-9 items-center justify-center gap-2 rounded-[10px] px-3 text-sm font-semibold transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50",
        variants[variant],
        className,
      )}
      {...props}
    />
  ),
);

Button.displayName = "Button";
