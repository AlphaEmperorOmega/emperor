import { type ReactNode } from "react";
import { Loader2 } from "lucide-react";
import { StatusCard } from "@/components/ui/status-card";

export function WorkbenchWorkspaceFrame({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <div
      id="workbench-workspace-content"
      tabIndex={-1}
      data-workbench-layout="workspace-frame"
      className="block min-h-0 overflow-x-hidden overflow-y-auto xl:grid xl:grid-cols-[320px_minmax(0,1fr)_320px] xl:overflow-hidden 2xl:grid-cols-[344px_minmax(0,1fr)_332px]"
    >
      {children}
    </div>
  );
}

export function WorkbenchThreeRegionLayout({
  sidebar,
  primary,
  details,
  sidebarLabel = "Workspace setup",
  primaryLabel = "Workspace primary content",
  detailsLabel = "Workspace details",
}: {
  sidebar: ReactNode;
  primary: ReactNode;
  details: ReactNode;
  sidebarLabel?: string;
  primaryLabel?: string;
  detailsLabel?: string;
}) {
  return (
    <>
      <div
        role="region"
        aria-label={sidebarLabel}
        data-workbench-region="sidebar"
        className="min-h-0 overflow-visible border-b border-line bg-panel/55 px-region pb-shell-wide pt-region backdrop-blur-sm xl:overflow-y-auto xl:border-b-0 xl:border-r"
      >
        <div className="grid gap-shell">{sidebar}</div>
      </div>
      <div
        role="region"
        aria-label={primaryLabel}
        data-workbench-region="primary"
        className="grid min-h-[520px] min-w-0 sm:min-h-[640px] xl:min-h-0"
      >
        {primary}
      </div>
      <div
        role="region"
        aria-label={detailsLabel}
        data-workbench-region="details"
        tabIndex={0}
        className="min-h-0 min-w-0 overflow-x-hidden overflow-y-visible border-t border-line bg-panel/55 px-region pb-[max(1.5rem,env(safe-area-inset-bottom))] pt-region backdrop-blur-sm xl:overflow-y-auto xl:border-l xl:border-t-0 xl:px-shell"
      >
        {details}
      </div>
    </>
  );
}

export function WorkbenchWideWorkspaceRegion({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <div
      data-workbench-region="wide"
      className="grid h-full min-h-[520px] min-w-0 grid-rows-[minmax(0,1fr)] overflow-hidden sm:min-h-[640px] xl:col-span-3 xl:min-h-0"
    >
      {children}
    </div>
  );
}

export function WorkbenchWorkspaceLoadingStatus({
  label,
}: {
  label: string;
}) {
  return (
    <div
      className="h-full min-h-0"
      role="status"
      aria-label={label}
      aria-busy="true"
      aria-live="polite"
    >
      <StatusCard
        layout="chart"
        title={label}
        detail="Preparing this workspace…"
        icon={<Loader2 className="h-5 w-5 animate-spin" aria-hidden />}
      />
    </div>
  );
}
