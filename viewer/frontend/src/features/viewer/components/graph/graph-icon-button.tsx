import {
  type ButtonHTMLAttributes,
  type KeyboardEventHandler,
  type MouseEventHandler,
  type ReactNode,
} from "react";
import { IconButton } from "@/components/ui/icon-button";
import { cn } from "@/lib/utils";

export type GraphIconButtonProps = Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  "aria-label" | "children" | "onClick"
> & {
  label: string;
  icon: ReactNode;
  onClick?: () => void;
  active?: boolean;
  disabled?: boolean;
  className?: string;
};

export function GraphIconButton({
  label,
  icon,
  onClick,
  active = false,
  disabled = false,
  className,
  onKeyDown,
  title,
  ...buttonProps
}: GraphIconButtonProps) {
  const handleClick: MouseEventHandler<HTMLButtonElement> = (event) => {
    event.stopPropagation();
    onClick?.();
  };
  const handleKeyDown: KeyboardEventHandler<HTMLButtonElement> = (event) => {
    event.stopPropagation();
    onKeyDown?.(event);
  };

  return (
    <IconButton
      {...buttonProps}
      label={label}
      icon={icon}
      type="button"
      title={title ?? label}
      disabled={disabled}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      className={cn(
        "nodrag nopan",
        active && "border-violet/35 bg-violet/10 text-white",
        className,
      )}
    />
  );
}
