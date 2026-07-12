import {
  forwardRef,
  type ElementType,
  type HTMLAttributes,
  type ReactNode,
} from "react";
import { SectionHeading } from "@/components/ui/section-heading";
import { cn } from "@/lib/utils";

export type SurfacePanelPadding = "compact" | "roomy" | "spacious" | "none";

export const surfacePanelBaseClassName =
  "grid min-w-0 content-start rounded-panel border border-line bg-panel-2/70 shadow-control";

export const surfacePanelPaddingClassNames: Record<SurfacePanelPadding, string> = {
  compact: "gap-2 px-panel py-2",
  roomy: "gap-panel p-panel",
  spacious: "gap-region p-region",
  none: "gap-0 p-0",
};

export const surfacePanelClassName = cn(
  surfacePanelBaseClassName,
  surfacePanelPaddingClassNames.compact,
);

export const surfacePanelHeaderClassName =
  "flex min-h-control flex-wrap items-center justify-between gap-2";

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
