import { type ReactNode } from "react";
import { EdgeCard } from "@/components/ui/edge-card";
import { cn } from "@/lib/utils";

export type MetricCardProps = {
  label: ReactNode;
  value: ReactNode;
  detail?: ReactNode;
  className?: string;
  valueTitle?: string;
  valueClassName?: string;
  detailClassName?: string;
};

export function MetricCard({
  label,
  value,
  detail,
  className,
  valueTitle,
  valueClassName,
  detailClassName,
}: MetricCardProps) {
  return (
    <EdgeCard className={cn("rounded-[12px] px-3 py-3", className)}>
      <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
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
    </EdgeCard>
  );
}
