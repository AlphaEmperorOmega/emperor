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
      className="block h-full min-h-0 overflow-x-hidden overflow-y-auto bg-bg-2/80 px-region py-panel sm:px-shell xl:grid xl:grid-rows-[auto_minmax(0,1fr)] xl:overflow-y-hidden"
    >
      <div className="mb-panel grid gap-2 empty:hidden">{notices}</div>
      <div className="row-start-2 block h-auto min-h-0 min-w-0 space-y-panel pb-[env(safe-area-inset-bottom)] xl:grid xl:h-full xl:grid-cols-[minmax(280px,320px)_minmax(0,1fr)_minmax(280px,340px)] xl:items-stretch xl:gap-panel xl:space-y-0 xl:overflow-y-hidden xl:pb-0 2xl:grid-cols-[minmax(300px,340px)_minmax(0,1fr)_minmax(300px,360px)]">
        <aside
          aria-label={leadingLabel}
          data-workbench-region="wide-leading"
          className="grid h-auto min-h-0 content-start gap-region overflow-visible xl:h-full xl:overflow-y-auto xl:pr-1"
        >
          {leading}
        </aside>
        <section
          aria-label={primaryLabel}
          data-workbench-region="wide-primary"
          className="grid min-h-[600px] grid-rows-[auto_minmax(0,1fr)] gap-panel overflow-hidden xl:h-full xl:min-h-0"
        >
          {primary}
        </section>
        <aside
          aria-label={trailingLabel}
          aria-live="polite"
          data-workbench-region="wide-trailing"
          className="grid h-auto min-h-0 content-start gap-panel overflow-visible xl:h-full xl:overflow-y-auto xl:pr-1"
        >
          {trailing}
        </aside>
      </div>
    </div>
  );
}
