import { Loader2 } from "lucide-react";
import { type ReactNode } from "react";

export type StatusCardProps = {
  title: ReactNode;
  detail?: ReactNode;
  icon?: ReactNode;
  busy?: boolean;
  tone?: "default" | "danger";
  layout?: "page" | "overlay" | "chart" | "inline";
  actions?: ReactNode;
};

function IconFrame({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-10 w-10 items-center justify-center rounded-[10px] border border-line bg-white/[0.04] text-violet">
      {children}
    </div>
  );
}

function defaultBusyIcon(className = "h-4 w-4 animate-spin") {
  return <Loader2 className={className} aria-hidden />;
}

export function StatusCard({
  title,
  detail,
  icon,
  busy = false,
  tone = "default",
  layout = "inline",
  actions,
}: StatusCardProps) {
  const framedIcon = icon ?? (busy ? defaultBusyIcon() : null);

  if (layout === "page") {
    return (
      <main className="grid min-h-screen place-items-center bg-bg p-6 text-ink">
        <div className="edge grid max-w-md justify-items-center gap-3 rounded-card p-6 text-center">
          {framedIcon && <IconFrame>{framedIcon}</IconFrame>}
          <div>
            <h1 className="text-lg font-bold">{title}</h1>
            {detail && <p className="mt-1 text-sm leading-6 text-ink-dim">{detail}</p>}
          </div>
          {actions}
        </div>
      </main>
    );
  }

  if (layout === "overlay") {
    return (
      <div className="absolute inset-0 flex items-center justify-center p-6">
        <div className="edge grid max-w-[360px] justify-items-center gap-3 rounded-card p-6 text-center shadow-panel">
          {framedIcon && <IconFrame>{framedIcon}</IconFrame>}
          <div>
            <div className="text-sm font-semibold text-ink">{title}</div>
            {detail && <div className="mt-1 text-xs leading-5 text-ink-faint">{detail}</div>}
          </div>
          {actions}
        </div>
      </div>
    );
  }

  if (layout === "chart") {
    return (
      <div className="grid h-full min-h-[360px] place-items-center p-6">
        <div className="edge grid max-w-md justify-items-center gap-3 rounded-card p-6 text-center shadow-panel">
          {busy && defaultBusyIcon("h-5 w-5 animate-spin text-violet")}
          {icon && <IconFrame>{icon}</IconFrame>}
          <div>
            <div className="text-sm font-semibold text-ink">{title}</div>
            {detail && <div className="mt-1 text-xs leading-5 text-ink-faint">{detail}</div>}
          </div>
          {actions}
        </div>
      </div>
    );
  }

  const inlineClasses =
    tone === "danger"
      ? "rounded-card border border-danger-line bg-danger-soft p-3 text-sm text-[#fda4af] shadow-panel"
      : "rounded-card border border-line-soft bg-white/[0.018] p-3 text-sm text-ink-faint shadow-panel";
  const inlineDetailClasses = tone === "danger" ? "mt-1 text-[#fecdd3]" : "mt-1 text-ink-faint";

  return (
    <div role={tone === "danger" ? "alert" : undefined} className={inlineClasses}>
      <div className="flex items-center gap-2 font-semibold">
        {busy && defaultBusyIcon("h-4 w-4 animate-spin")}
        {icon}
        {title}
      </div>
      {detail && <div className={inlineDetailClasses}>{detail}</div>}
      {actions}
    </div>
  );
}
