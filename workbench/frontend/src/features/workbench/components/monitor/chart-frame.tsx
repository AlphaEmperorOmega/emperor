import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

type ChartFrameProps = {
  title: ReactNode;
  subtitle?: ReactNode;
  badge?: ReactNode;
  actions?: ReactNode;
  footer?: ReactNode;
  children: ReactNode;
  className?: string;
};

export function ChartFrame({
  title,
  subtitle,
  badge,
  actions,
  footer,
  children,
  className,
}: ChartFrameProps) {
  let headerAside = null;
  if (badge !== undefined && actions !== undefined) {
    headerAside = (
      <div className="flex shrink-0 items-center gap-2">
        {badge}
        {actions}
      </div>
    );
  } else if (badge !== undefined) {
    headerAside = badge;
  } else if (actions !== undefined) {
    headerAside = actions;
  }

  return (
    <div className={cn("grid min-w-0 gap-2 p-3", className)}>
      <div className="flex min-w-0 items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold text-ink">{title}</div>
          {subtitle !== undefined && (
            <div className="truncate font-mono text-xs text-ink-dim">{subtitle}</div>
          )}
        </div>
        {headerAside}
      </div>
      {children}
      {footer !== undefined && (
        <div className="flex items-center justify-between gap-2 font-mono text-xs text-ink-dim">
          {footer}
        </div>
      )}
    </div>
  );
}
