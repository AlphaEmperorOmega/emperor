import {
  type ButtonHTMLAttributes,
  type ReactNode,
  forwardRef,
} from "react";
import { iconButtonClassName } from "@/components/ui/control-styles";
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
  sm: "h-touch w-touch rounded-control-md md:h-control-sm md:w-control-sm",
  md: "h-touch w-touch rounded-control md:h-control md:w-control",
};

const variants: Record<IconButtonVariant, string> = {
  ghost:
    "border-transparent text-ink-faint hover:bg-control-active hover:text-ink disabled:hover:bg-transparent disabled:hover:text-ink-faint",
  edge:
    "border-line bg-panel-2/90 text-ink-faint shadow-control hover:border-line-hover hover:bg-control-hover hover:text-ink disabled:hover:border-line disabled:hover:bg-panel-2 disabled:hover:text-ink-faint",
  danger:
    "border-transparent text-ink-faint hover:border-danger-line hover:bg-danger-soft hover:text-danger-text disabled:hover:border-transparent disabled:hover:bg-transparent disabled:hover:text-ink-faint",
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
        iconButtonClassName,
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
