import {
  forwardRef,
  type ElementType,
  type HTMLAttributes,
  type ReactNode,
} from "react";
import { SectionHeading } from "@/features/viewer/components/shared/section-heading";
import { cn } from "@/lib/utils";

export type SurfacePanelPadding = "compact" | "roomy" | "spacious" | "none";

export const surfacePanelBaseClassName =
  "grid content-start rounded-[10px] border border-line bg-white/[0.018]";

export const surfacePanelPaddingClassNames: Record<SurfacePanelPadding, string> = {
  compact: "gap-1.5 px-2.5 py-2",
  roomy: "gap-3 p-3",
  spacious: "gap-4 p-4",
  none: "gap-0 p-0",
};

export const surfacePanelClassName = cn(
  surfacePanelBaseClassName,
  surfacePanelPaddingClassNames.compact,
);

export const surfacePanelHeaderClassName =
  "flex min-h-[38px] flex-wrap items-center justify-between gap-2";

export type SurfacePanelProps = Omit<HTMLAttributes<HTMLElement>, "title"> & {
  as?: "div" | "section" | "article" | "aside";
  padding?: SurfacePanelPadding;
  icon?: ReactNode;
  title?: ReactNode;
  detail?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
  headerClassName?: string;
};

export const SurfacePanel = forwardRef<HTMLElement, SurfacePanelProps>(
  function SurfacePanel(
    {
      as = "div",
      padding = "compact",
      icon,
      title,
      detail,
      actions,
      children,
      className,
      headerClassName,
      ...props
    },
    ref,
  ) {
    const Component: ElementType = as;
    const hasIcon = icon !== null && icon !== undefined;
    const hasTitle = title !== null && title !== undefined;
    const hasDetail = detail !== null && detail !== undefined;
    const hasActions = actions !== null && actions !== undefined;
    const hasHeader = hasTitle || hasDetail || hasActions;

    return (
      <Component
        ref={ref as never}
        className={cn(
          surfacePanelBaseClassName,
          surfacePanelPaddingClassNames[padding],
          className,
        )}
        {...props}
      >
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
      </Component>
    );
  },
);
