import { ChevronDown, ListTree } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { type RuntimeDefaultsSectionPresentation } from "@/features/workbench/state/runtime-defaults/runtime-defaults-presentation";
import { cn } from "@/lib/utils";
import { ConfigMetricBadge } from "@/features/workbench/components/config/config-metric-badge";
import { SectionHeading } from "@/components/ui/section-heading";
import { surfacePanelClassName } from "@/components/ui/surface-panel";

export function SectionNavigation({
  sections,
  openSectionTitles,
  areAllSectionsOpen,
  emptyMessage,
  variant = "sidebar",
  ariaLabel = "Full config sections",
  title = "Sections",
  onJumpToSection,
  onToggleSection,
  onToggleAllSections,
}: {
  sections: RuntimeDefaultsSectionPresentation[];
  openSectionTitles: Set<string>;
  areAllSectionsOpen: boolean;
  emptyMessage?: string;
  variant?: "sidebar" | "inline";
  ariaLabel?: string;
  title?: string;
  onJumpToSection: (title: string) => void;
  onToggleSection: (title: string) => void;
  onToggleAllSections: () => void;
}) {
  const isInline = variant === "inline";

  return (
    <nav
      aria-label={ariaLabel}
      className={cn(
        surfacePanelClassName,
        "min-w-0 p-2",
        isInline
          ? "border-line-soft bg-black/15"
          : "lg:sticky lg:top-0 lg:self-start",
      )}
    >
      <div className="mb-2 flex min-w-0 items-center justify-between gap-2 px-1">
        <SectionHeading
          className="min-w-0 truncate"
          icon={<ListTree className="h-[15px] w-[15px] text-violet" aria-hidden />}
          title={title}
        />
        <button
          type="button"
          onClick={onToggleAllSections}
          disabled={sections.length === 0}
          className="h-touch min-w-[5.75rem] shrink-0 whitespace-nowrap rounded-control-md border border-line bg-control px-2 text-xs font-semibold uppercase tracking-label text-ink-dim transition-colors hover:border-violet/30 hover:bg-violet/10 hover:text-violet focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:h-control-sm"
        >
          {areAllSectionsOpen ? "Close All" : "Open All"}
        </button>
      </div>
      <div
        className={cn(
          "flex gap-2 overflow-auto pb-1",
          !isInline && "lg:grid lg:max-h-[calc(100dvh-16rem)] lg:gap-0 lg:pb-0",
        )}
      >
        {sections.length === 0 && (
          <div
            className={cn(
              surfacePanelClassName,
              "min-w-[13rem] border-dashed border-line-soft bg-black/20 px-3 py-2 text-sm text-ink-dim",
            )}
          >
            {emptyMessage ?? "No sections"}
          </div>
        )}
        {sections.map((section) => {
          const {
            fieldCount,
            overrideCount: sectionModifiedCount,
            presetCount: sectionPresetOwnedCount,
            state,
          } = section.treeMetrics;
          const displayTitle = section.displayTitle;
          const hasOverride = sectionModifiedCount > 0;
          const hasPreset = sectionPresetOwnedCount > 0;
          const hasBoth = state === "override-and-preset";
          const rowStateClass = hasBoth
            ? "border-amber/35 bg-config-navigation ring-1 ring-violet/20 hover:bg-config-navigation-hover"
            : hasOverride
              ? "border-violet/30 bg-violet/[0.055] hover:bg-violet/15"
              : hasPreset
                ? "border-amber/30 bg-amber/[0.055] hover:bg-amber/[0.09]"
                : "";
          const sectionId = section.id;
          const panelId = `${sectionId}-fields`;
          const isSectionDisabled = section.isDisabled;
          const isSectionOpen =
            !isSectionDisabled && openSectionTitles.has(section.title);
          return (
            <div
              key={section.title}
              className={cn(
                "group/section-row grid min-h-11 min-w-[13rem] grid-cols-[minmax(0,1fr)_2.25rem] border-b border-line-soft transition focus-within:ring-2 focus-within:ring-focus hover:bg-violet/10 last:border-b-0",
                isInline
                  ? "rounded-control border border-line-soft last:border-b"
                  : "lg:min-w-0",
                isSectionDisabled && "opacity-70",
                rowStateClass,
              )}
            >
              <div className="relative grid min-w-0 grid-cols-[minmax(0,1fr)_auto]">
                <button
                  type="button"
                  aria-label={`Jump to ${displayTitle}`}
                  onClick={() => onJumpToSection(section.title)}
                  className="min-h-11 min-w-0 px-2.5 py-2 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus"
                >
                  <span className="block min-w-0 truncate text-sm font-semibold text-ink transition-opacity group-hover/section-row:opacity-0 group-focus-within/section-row:opacity-0">
                    {displayTitle}
                  </span>
                </button>
                <span className="flex shrink-0 items-center gap-1 px-2 transition-opacity group-hover/section-row:pointer-events-none group-hover/section-row:opacity-0 group-focus-within/section-row:pointer-events-none group-focus-within/section-row:opacity-0">
                  <ConfigMetricBadge
                    count={fieldCount}
                    kind="fields"
                    focusable={false}
                    tooltipPosition="bottom"
                  />
                  <ConfigMetricBadge
                    count={sectionModifiedCount}
                    kind="overrides"
                    variant={sectionModifiedCount > 0 ? "override" : "default"}
                    focusable={false}
                    tooltipPosition="bottom"
                  />
                  {hasPreset && (
                    <Badge
                      variant="preset"
                      className="h-[23px] items-center px-1.5 py-0"
                    >
                      {sectionPresetOwnedCount} preset
                    </Badge>
                  )}
                </span>
                <span
                  aria-hidden
                  className="pointer-events-none absolute left-2.5 right-2.5 top-1/2 z-20 -translate-y-1/2 truncate text-sm font-semibold text-ink opacity-0 transition-opacity group-hover/section-row:opacity-100 group-focus-within/section-row:opacity-100"
                >
                  {displayTitle}
                </span>
              </div>
              <button
                type="button"
                aria-label={`${isSectionOpen ? "Close" : "Open"} ${displayTitle}`}
                aria-expanded={isSectionOpen}
                aria-controls={panelId}
                disabled={isSectionDisabled}
                onClick={() => onToggleSection(section.title)}
                className="flex min-h-full items-center justify-center border-l border-line-soft text-ink-faint transition-colors hover:bg-control-hover hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-ink-faint"
              >
                <ChevronDown
                  className={cn(
                    "h-4 w-4 transition-transform",
                    !isSectionOpen && "-rotate-90",
                  )}
                  aria-hidden
                />
              </button>
            </div>
          );
        })}
      </div>
    </nav>
  );
}
