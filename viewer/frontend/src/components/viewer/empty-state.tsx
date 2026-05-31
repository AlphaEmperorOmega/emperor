import { type ReactNode } from "react";

export function EmptyState({
  title,
  detail,
  icon,
}: {
  title: string;
  detail?: string;
  icon: ReactNode;
}) {
  return (
    <div className="absolute inset-0 flex items-center justify-center p-6">
      <div className="grid max-w-[360px] justify-items-center gap-3 rounded-md border border-dashed border-faint bg-panel/90 p-6 text-center shadow-panel">
        <div className="flex h-10 w-10 items-center justify-center rounded-md border border-border bg-surface text-accent">
          {icon}
        </div>
        <div>
          <div className="text-sm font-semibold text-ink">{title}</div>
          {detail && <div className="mt-1 text-xs leading-5 text-muted">{detail}</div>}
        </div>
      </div>
    </div>
  );
}
