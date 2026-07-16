import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

export type SidePanelProps = {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  headerClassName?: string;
};

const sidePanelClassName = "min-w-0";

export function SidePanel({
  title,
  subtitle,
  actions,
  children,
  className,
  headerClassName,
}: SidePanelProps) {
  const hasHeader = title !== undefined || subtitle !== undefined || actions !== undefined;

  return (
    <div className={cn(sidePanelClassName, className)}>
      {hasHeader && (
        <div className={cn("mb-region grid gap-panel", headerClassName)}>
          {(title !== undefined || actions !== undefined) && (
            <div className="flex items-center justify-between gap-3">
              {title !== undefined && (
                <h2 className="type-title text-balance font-bold text-ink">{title}</h2>
              )}
              {actions}
            </div>
          )}
          {subtitle}
        </div>
      )}
      {children}
    </div>
  );
}
