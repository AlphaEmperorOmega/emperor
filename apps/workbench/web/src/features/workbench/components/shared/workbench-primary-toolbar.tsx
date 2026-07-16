import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

export function WorkbenchPrimaryToolbar({
  children,
  className,
  detail,
  title,
  titleAs: Title = "h1",
  workspaceTitle,
}: {
  children: ReactNode;
  className?: string;
  detail: ReactNode;
  title: string;
  titleAs?: "h1" | "h2";
  workspaceTitle?: string;
}) {
  return (
    <header
      className={cn(
        "flex h-14 min-w-0 items-center justify-between gap-panel overflow-hidden border-b border-line bg-panel/55 px-region shadow-divider backdrop-blur-sm sm:px-shell",
        className,
      )}
    >
      <div className="min-w-0 shrink-0">
        {workspaceTitle && <h1 className="sr-only">{workspaceTitle}</h1>}
        <Title className="sr-only type-label font-bold uppercase text-ink-dim sm:not-sr-only">
          {title}
        </Title>
        <div className="mt-0.5 hidden truncate font-mono text-xs text-ink-dim sm:block">
          {detail}
        </div>
      </div>
      <div className="flex min-w-0 flex-nowrap items-center justify-start gap-2 overflow-x-auto overscroll-x-contain [scrollbar-width:none] 2xl:justify-end">
        {children}
      </div>
    </header>
  );
}
