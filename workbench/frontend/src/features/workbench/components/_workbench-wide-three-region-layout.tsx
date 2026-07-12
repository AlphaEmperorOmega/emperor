import { type ReactNode } from "react";

// Private deferred leaf of the Workbench Layout Module. Keeping Training's
// wide-region Implementation here prevents it entering the initial workspace
// chunk while the shared frame and region protocol remain eagerly available.
export function WorkbenchWideThreeRegionLayout({
  notices,
  leading,
  primary,
  trailing,
  leadingLabel,
  primaryLabel,
  trailingLabel,
}: {
  notices?: ReactNode;
  leading: ReactNode;
  primary: ReactNode;
  trailing: ReactNode;
  leadingLabel: string;
  primaryLabel: string;
  trailingLabel: string;
}) {
  return (
    <div
      data-workbench-layout="wide-three-region"
      className="grid h-full min-h-0 grid-rows-[auto_minmax(0,1fr)] overflow-x-auto overflow-y-hidden bg-bg-2/90 px-4 py-3 sm:px-5"
    >
      <div className="mb-3 grid gap-2 empty:hidden">{notices}</div>
      <div className="row-start-2 grid h-full min-h-0 min-w-[920px] grid-cols-[minmax(300px,340px)_minmax(22rem,1fr)_minmax(280px,360px)] items-stretch gap-3 overflow-y-hidden">
        <aside
          aria-label={leadingLabel}
          data-workbench-region="wide-leading"
          className="grid h-full min-h-0 content-start gap-4 overflow-y-auto pr-1"
        >
          {leading}
        </aside>
        <main
          aria-label={primaryLabel}
          data-workbench-region="wide-primary"
          className="grid h-full min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-3 overflow-hidden"
        >
          {primary}
        </main>
        <aside
          aria-label={trailingLabel}
          aria-live="polite"
          data-workbench-region="wide-trailing"
          className="grid h-full min-h-0 content-start gap-3 overflow-y-auto pr-1"
        >
          {trailing}
        </aside>
      </div>
    </div>
  );
}
