import { Loader2 } from "lucide-react";
import { type ReactNode } from "react";
import { SurfacePanel } from "@/components/ui/surface-panel";

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
    <div className="flex h-touch w-touch items-center justify-center rounded-control border border-accent-line bg-accent-soft text-violet shadow-control-accent">
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
      <main className="grid min-h-screen place-items-center bg-bg p-shell-wide text-ink">
        <SurfacePanel
          padding="spacious"
          role={busy ? "status" : undefined}
          aria-busy={busy || undefined}
          aria-live={busy ? "polite" : undefined}
          className="max-w-md justify-items-center border-line-hover bg-panel text-center shadow-panel"
        >
          {framedIcon && <IconFrame>{framedIcon}</IconFrame>}
          <div>
            <h1 className="type-display text-balance font-bold">{title}</h1>
            {detail && <p className="mt-2 text-pretty type-body text-ink-dim">{detail}</p>}
          </div>
          {actions}
        </SurfacePanel>
      </main>
    );
  }

  if (layout === "overlay") {
    return (
      <div className="absolute inset-0 flex items-center justify-center p-shell-wide">
        <SurfacePanel
          padding="spacious"
          role={busy ? "status" : undefined}
          aria-busy={busy || undefined}
          aria-live={busy ? "polite" : undefined}
          className="max-w-[360px] justify-items-center border-line-hover bg-panel/95 text-center shadow-panel backdrop-blur-sm"
        >
          {framedIcon && <IconFrame>{framedIcon}</IconFrame>}
          <div>
            <div className="type-title text-balance font-semibold text-ink">{title}</div>
            {detail && <div className="mt-2 text-pretty text-xs leading-5 text-ink-faint">{detail}</div>}
          </div>
          {actions}
        </SurfacePanel>
      </div>
    );
  }

  if (layout === "chart") {
    return (
      <div className="grid h-full min-h-[360px] place-items-center p-shell-wide">
        <SurfacePanel
          padding="spacious"
          role={busy ? "status" : undefined}
          aria-busy={busy || undefined}
          aria-live={busy ? "polite" : undefined}
          className="max-w-md justify-items-center text-center shadow-panel"
        >
          {busy && defaultBusyIcon("h-5 w-5 animate-spin text-violet")}
          {icon && <IconFrame>{icon}</IconFrame>}
          <div>
            <div className="type-title text-balance font-semibold text-ink">{title}</div>
            {detail && <div className="mt-2 text-pretty text-xs leading-5 text-ink-faint">{detail}</div>}
          </div>
          {actions}
        </SurfacePanel>
      </div>
    );
  }

  const inlineClasses =
    "rounded-panel border border-danger-line bg-danger-soft p-panel type-body text-danger-text";
  const inlineDetailClasses =
    tone === "danger" ? "mt-1 text-danger-detail" : "mt-1 text-ink-faint";

  if (tone !== "danger") {
    return (
      <SurfacePanel padding="roomy" className="text-sm text-ink-faint">
        <div className="flex items-center gap-2 font-semibold">
          {busy && defaultBusyIcon("h-4 w-4 animate-spin")}
          {icon}
          {title}
        </div>
        {detail && <div className={inlineDetailClasses}>{detail}</div>}
        {actions}
      </SurfacePanel>
    );
  }

  return (
    <div role="alert" className={inlineClasses}>
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
