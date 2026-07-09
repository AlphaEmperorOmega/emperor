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
  "absolute left-0 right-0 top-full mt-2 rounded-[12px] border border-line bg-panel/95 shadow-[0_22px_50px_-30px_rgba(0,0,0,0.98)] backdrop-blur";

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
