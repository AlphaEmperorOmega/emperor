import { type ReactNode } from "react";
import { SectionHeading, type SectionHeadingProps } from "@/components/ui/section-heading";
import { cn } from "@/lib/utils";

export function WorkbenchSidebarStack({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      data-workbench-sidebar=""
      className={cn("grid min-w-0 content-start gap-region", className)}
    >
      {children}
    </div>
  );
}

function SidebarHeadingIcon({ children }: { children: ReactNode }) {
  return (
    <span
      className="inline-flex shrink-0 text-violet [&>svg]:h-[15px] [&>svg]:w-[15px]"
      aria-hidden
    >
      {children}
    </span>
  );
}

export function WorkbenchSidebarHeader({
  title,
  icon,
  actions,
  as = "h2",
  className,
}: {
  title: ReactNode;
  icon?: ReactNode;
  actions?: ReactNode;
  as?: SectionHeadingProps["as"];
  className?: string;
}) {
  return (
    <header
      data-workbench-sidebar-header=""
      className={cn(
        "flex min-h-control items-center justify-between gap-2 border-b border-line-soft pb-panel",
        className,
      )}
    >
      <SectionHeading
        as={as}
        icon={icon ? <SidebarHeadingIcon>{icon}</SidebarHeadingIcon> : undefined}
        title={title}
      />
      {actions}
    </header>
  );
}

export function WorkbenchSidebarSection({
  title,
  icon,
  aside,
  children,
  as,
  divider = "none",
  className,
}: {
  title: ReactNode;
  icon?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  as?: SectionHeadingProps["as"];
  divider?: "none" | "before" | "after";
  className?: string;
}) {
  return (
    <section
      data-workbench-sidebar-section=""
      className={cn(
        "grid min-w-0 gap-2",
        divider === "before" && "border-t border-line-soft pt-panel",
        divider === "after" && "border-b border-line-soft pb-panel",
        className,
      )}
    >
      <div className="flex min-h-[28px] flex-wrap items-center justify-between gap-2">
        <SectionHeading
          as={as}
          icon={icon ? <SidebarHeadingIcon>{icon}</SidebarHeadingIcon> : undefined}
          title={title}
        />
        {aside}
      </div>
      {children}
    </section>
  );
}
