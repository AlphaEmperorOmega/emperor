import {
  type AriaRole,
  type CSSProperties,
  type UIEventHandler,
  forwardRef,
  type ReactNode,
} from "react";
import { cn } from "@/lib/utils";

export type DropdownShellProps = {
  id?: string;
  role?: AriaRole;
  labelledBy?: string;
  ariaLabel?: string;
  searchSlot?: ReactNode;
  children: ReactNode;
  className?: string;
  onScroll?: UIEventHandler<HTMLDivElement>;
  style?: CSSProperties;
};

const dropdownShellClassName =
  "absolute left-0 right-0 top-full mt-2 overflow-hidden rounded-panel border border-line-hover bg-panel/95 shadow-popover backdrop-blur-xl";

export const DropdownShell = forwardRef<HTMLDivElement, DropdownShellProps>(
  (
    {
      id,
      role,
      labelledBy,
      ariaLabel,
      searchSlot,
      children,
      className,
      onScroll,
      style,
    },
    ref,
  ) => (
    <div
      ref={ref}
      id={id}
      role={role}
      aria-labelledby={labelledBy}
      aria-label={ariaLabel}
      onScroll={onScroll}
      style={style}
      className={cn(dropdownShellClassName, className)}
    >
      {searchSlot}
      {children}
    </div>
  ),
);

DropdownShell.displayName = "DropdownShell";
