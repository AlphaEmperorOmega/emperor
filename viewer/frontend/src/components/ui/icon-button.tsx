import {
  type ButtonHTMLAttributes,
  type ReactNode,
  forwardRef,
} from "react";
import { cn } from "@/lib/utils";

type IconButtonSize = "sm" | "md";
type IconButtonVariant = "ghost" | "edge" | "danger";

export type IconButtonProps = Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  "children" | "aria-label"
> & {
  label: string;
  icon: ReactNode;
  size?: IconButtonSize;
  variant?: IconButtonVariant;
};

const sizes: Record<IconButtonSize, string> = {
  sm: "h-8 w-8 rounded-[8px]",
  md: "h-9 w-9 rounded-[10px]",
};

const variants: Record<IconButtonVariant, string> = {
  ghost: "border-transparent text-ink-faint hover:bg-white/[0.055] hover:text-ink",
  edge:
    "border-line bg-white/[0.035] text-ink-faint hover:bg-white/[0.07] hover:text-ink",
  danger:
    "border-transparent text-ink-faint hover:border-danger-line hover:bg-danger-soft hover:text-[#fda4af]",
};

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  (
    {
      className,
      icon,
      label,
      size = "md",
      type = "button",
      variant = "ghost",
      ...props
    },
    ref,
  ) => (
    <button
      {...props}
      ref={ref}
      type={type}
      aria-label={label}
      className={cn(
        "inline-flex shrink-0 items-center justify-center border transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50",
        sizes[size],
        variants[variant],
        className,
      )}
    >
      {icon}
    </button>
  ),
);

IconButton.displayName = "IconButton";
