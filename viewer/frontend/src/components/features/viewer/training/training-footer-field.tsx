import { type ReactNode } from "react";
import { SectionHeading } from "@/components/features/viewer/shared/section-heading";
import { cn } from "@/lib/utils";

type TrainingFooterFieldProps = {
  icon?: ReactNode;
  label: ReactNode;
  detail?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
};

export function TrainingFooterField({
  icon,
  label,
  detail,
  actions,
  children,
  className,
}: TrainingFooterFieldProps) {
  const hasIcon = icon !== null && icon !== undefined;
  const hasLabel = label !== null && label !== undefined;
  const hasDetail = detail !== null && detail !== undefined;
  const hasActions = actions !== null && actions !== undefined;
  const hasHeader = hasLabel || hasDetail || hasActions;

  return (
    <div
      className={cn(
        "grid content-start gap-1.5 rounded-[10px] border border-line bg-white/[0.018] px-2.5 py-2",
        className,
      )}
    >
      {hasHeader && (
        <div className="flex min-h-[38px] flex-wrap items-center justify-between gap-2">
          {hasIcon && hasLabel ? (
            <SectionHeading icon={icon} title={label} />
          ) : (
            label
          )}
          {(hasDetail || hasActions) && (
            <div className="flex shrink-0 flex-wrap items-center gap-1.5">
              {detail}
              {actions}
            </div>
          )}
        </div>
      )}
      {children}
    </div>
  );
}
