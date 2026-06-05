import { type MouseEventHandler, type ReactNode } from "react";
import { IconButton } from "@/components/ui/icon-button";
import { cn } from "@/lib/utils";

export type GraphIconButtonProps = {
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
}: GraphIconButtonProps) {
  const handleClick: MouseEventHandler<HTMLButtonElement> = (event) => {
    event.stopPropagation();
    onClick?.();
  };

  return (
    <IconButton
      label={label}
      icon={icon}
      type="button"
      title={label}
      disabled={disabled}
      onClick={handleClick}
      onKeyDown={(event) => {
        event.stopPropagation();
      }}
      className={cn(
        "nodrag nopan",
        active && "border-violet/35 bg-violet/10 text-white",
        className,
      )}
    />
  );
}
