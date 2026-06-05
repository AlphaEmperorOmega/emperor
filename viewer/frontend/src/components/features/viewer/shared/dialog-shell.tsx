import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

export type DialogShellProps = {
  titleId?: string;
  labelledBy?: string;
  describedBy?: string;
  size?: "sm" | "md" | "lg" | "fullscreen";
  header?: ReactNode;
  footer?: ReactNode;
  overlayChildren?: ReactNode;
  children: ReactNode;
  className?: string;
  panelClassName?: string;
};

const dialogOverlayClassName =
  "fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-3 backdrop-blur-sm sm:p-6";

const dialogPanelClassNames: Record<NonNullable<DialogShellProps["size"]>, string> = {
  sm: "edge flex max-h-[calc(100vh-1.5rem)] w-full max-w-lg flex-col overflow-hidden rounded-card shadow-[0_24px_80px_rgba(0,0,0,0.58)] sm:max-h-[calc(100vh-3rem)]",
  md: "edge flex max-h-[calc(100vh-1.5rem)] w-full max-w-3xl flex-col overflow-hidden rounded-card shadow-[0_24px_80px_rgba(0,0,0,0.58)] sm:max-h-[calc(100vh-3rem)]",
  lg: "edge flex max-h-[calc(100vh-1.5rem)] w-full max-w-4xl flex-col overflow-hidden rounded-card shadow-[0_24px_80px_rgba(0,0,0,0.58)] sm:max-h-[calc(100vh-3rem)]",
  fullscreen:
    "edge flex max-h-[calc(100vh-1.5rem)] w-full max-w-[92rem] flex-col overflow-hidden rounded-card shadow-[0_24px_80px_rgba(0,0,0,0.58)] sm:max-h-[calc(100vh-3rem)]",
};

export function DialogShell({
  titleId,
  labelledBy,
  describedBy,
  size = "lg",
  header,
  footer,
  overlayChildren,
  children,
  className,
  panelClassName,
}: DialogShellProps) {
  return (
    <div className={cn(dialogOverlayClassName, className)}>
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy ?? titleId}
        aria-describedby={describedBy}
        className={cn(dialogPanelClassNames[size], panelClassName)}
      >
        {header}
        {children}
        {footer}
      </section>
      {overlayChildren}
    </div>
  );
}
