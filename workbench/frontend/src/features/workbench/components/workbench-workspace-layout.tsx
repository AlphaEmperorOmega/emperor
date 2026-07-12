import { type ReactNode } from "react";

export type WorkbenchWorkspaceBoundary = (
  content: ReactNode,
) => ReactNode;

export function WorkbenchWorkspaceFrame({
  children,
  workspaceBoundary,
}: {
  children: ReactNode;
  workspaceBoundary?: WorkbenchWorkspaceBoundary;
}) {
  const content = workspaceBoundary ? workspaceBoundary(children) : children;

  return (
    <section
      id="workbench-workspace-content"
      tabIndex={-1}
      data-workbench-layout="workspace-frame"
      className="grid min-h-0 grid-cols-1 overflow-auto lg:grid-cols-[344px_minmax(0,1fr)_332px] lg:overflow-hidden"
    >
      {content}
    </section>
  );
}

export function WorkbenchThreeRegionLayout({
  sidebar,
  primary,
  details,
}: {
  sidebar: ReactNode;
  primary: ReactNode;
  details: ReactNode;
}) {
  return (
    <>
      <aside
        data-workbench-region="sidebar"
        className="min-h-0 overflow-y-auto border-b border-line bg-[linear-gradient(180deg,rgba(13,12,22,0.6),rgba(8,8,13,0.4))] px-4 pb-7 pt-[18px] backdrop-blur lg:border-b-0 lg:border-r"
      >
        <div className="grid gap-[22px]">{sidebar}</div>
      </aside>
      <div
        data-workbench-region="primary"
        className="grid min-h-[560px] min-w-0 lg:min-h-0"
      >
        {primary}
      </div>
      <aside
        data-workbench-region="details"
        className="min-h-0 min-w-0 overflow-x-hidden overflow-y-auto border-t border-line bg-[linear-gradient(180deg,rgba(13,12,22,0.6),rgba(8,8,13,0.4))] px-[18px] pb-8 pt-5 backdrop-blur lg:border-l lg:border-t-0"
      >
        {details}
      </aside>
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
      className="grid h-full min-h-[560px] min-w-0 grid-rows-[minmax(0,1fr)] overflow-hidden lg:col-span-3 lg:min-h-0"
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
      className="grid h-full place-items-center"
      role="status"
      aria-label={label}
    >
      <span className="text-xs text-ink-faint">{label}</span>
    </div>
  );
}
