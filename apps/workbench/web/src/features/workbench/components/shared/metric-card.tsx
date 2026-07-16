import { type ReactNode } from "react";
import { SurfacePanel } from "@/components/ui/surface-panel";
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
      <div className="flex min-w-0 items-center gap-2 type-label font-bold uppercase text-ink-dim">
        {icon}
        {label}
      </div>
      <div
        className={cn("font-mono type-display font-bold tabular-nums text-ink", valueClassName)}
        title={valueTitle}
      >
        {value}
      </div>
      {detail !== undefined && (
        <div className={cn("text-xs text-ink-faint", detailClassName)}>
          {detail}
        </div>
      )}
    </SurfacePanel>
  );
}
