import { type ReactNode } from "react";
import { SurfacePanel } from "@/features/workbench/components/shared/surface-panel";
import { cn } from "@/lib/utils";

export type MetricCardProps = {
  icon?: ReactNode;
  label: ReactNode;
  value: ReactNode;
  detail?: ReactNode;
  className?: string;
  valueTitle?: string;
  valueClassName?: string;
  detailClassName?: string;
};

export function MetricCard({
  icon,
  label,
  value,
  detail,
  className,
  valueTitle,
  valueClassName,
  detailClassName,
}: MetricCardProps) {
  return (
    <SurfacePanel className={className}>
      <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
        {icon}
        {label}
      </div>
      <div
        className={cn("mt-1 font-mono text-xl font-extrabold text-ink", valueClassName)}
        title={valueTitle}
      >
        {value}
      </div>
      {detail !== undefined && (
        <div className={cn("mt-1 text-xs text-ink-faint", detailClassName)}>
          {detail}
        </div>
      )}
    </SurfacePanel>
  );
}
