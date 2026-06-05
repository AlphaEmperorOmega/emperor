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

const sidePanelClassName =
  "min-h-0 overflow-y-auto border-t border-line bg-[linear-gradient(180deg,rgba(13,12,22,0.6),rgba(8,8,13,0.4))] px-[18px] pb-8 pt-5 backdrop-blur lg:border-l lg:border-t-0";

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
    <aside className={cn(sidePanelClassName, className)}>
      {hasHeader && (
        <div className={cn("mb-4 grid gap-3", headerClassName)}>
          {(title !== undefined || actions !== undefined) && (
            <div className="flex items-center justify-between gap-3">
              {title !== undefined && (
                <h2 className="text-base font-bold text-ink">{title}</h2>
              )}
              {actions}
            </div>
          )}
          {subtitle}
        </div>
      )}
      {children}
    </aside>
  );
}
