import { type ReactNode } from "react";
import { SectionHeading } from "@/features/viewer/components/shared/section-heading";
import { cn } from "@/lib/utils";

export const surfacePanelClassName =
  "grid content-start gap-1.5 rounded-[10px] border border-line bg-white/[0.018] px-2.5 py-2";

export const surfacePanelHeaderClassName =
  "flex min-h-[38px] flex-wrap items-center justify-between gap-2";

export type SurfacePanelProps = {
  icon?: ReactNode;
  title?: ReactNode;
  detail?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
  className?: string;
  headerClassName?: string;
};

export function SurfacePanel({
  icon,
  title,
  detail,
  actions,
  children,
  className,
  headerClassName,
}: SurfacePanelProps) {
  const hasIcon = icon !== null && icon !== undefined;
  const hasTitle = title !== null && title !== undefined;
  const hasDetail = detail !== null && detail !== undefined;
  const hasActions = actions !== null && actions !== undefined;
  const hasHeader = hasTitle || hasDetail || hasActions;

  return (
    <div className={cn(surfacePanelClassName, className)}>
      {hasHeader && (
        <div className={cn(surfacePanelHeaderClassName, headerClassName)}>
          {hasIcon && hasTitle ? (
            <SectionHeading icon={icon} title={title} />
          ) : (
            title
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
