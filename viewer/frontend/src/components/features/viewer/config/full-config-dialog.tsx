import { useMemo, useState } from "react";
import { FilePlus2, RotateCcw, Terminal, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  type ConfigSearchOption,
  filterConfigSectionsForSearch,
  flattenConfigSearchOptions,
  presetOwnedCount,
  sectionElementId,
} from "@/lib/config";
import { buildTrainingCommand } from "@/lib/training-command";
import { ConfigFieldSearch } from "@/components/features/viewer/config/config-field-search";
import { ConfigSectionAccordion } from "@/components/features/viewer/config/config-section-accordion";
import { ConfigMetricBadge } from "@/components/features/viewer/config/config-metric-badge";
import { SectionNavigation } from "@/components/features/viewer/config/section-navigation";
import { TrainingCommandDialog } from "@/components/features/viewer/config/training-command-dialog";
import {
  AddConfigSnapshotDialog,
  ConfigSnapshotsTray,
} from "@/components/features/viewer/config/config-snapshots-tray";
import { useConfigDialogSections } from "@/components/features/viewer/config/use-config-dialog-sections";
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";
import { useTargetConfig } from "@/components/features/viewer/providers/viewer-providers";

export function FullConfigDialog({ onClose }: { onClose: () => void }) {
  const {
    selectedModel: model,
    selectedPreset: preset,
    configSections: sections,
    fieldCount,
    overrideCount,
    overrides,
    selectedDatasets,
    configSnapshots,
    configSnapshotGroups,
    configSnapshotCount,
    schemaQuery,
    addConfigSnapshot,
    removeConfigSnapshot,
    renameConfigSnapshot,
    loadConfigSnapshot,
    updateOverride: onFieldChange,
    clearOverride: onFieldReset,
    resetOverrides: onResetOverrides,
    updatePreview: onUpdatePreview,
  } = useTargetConfig();
  const isLoading = schemaQuery.isLoading;
  const canUpdate = Boolean(model && preset && selectedDatasets.length > 0);
  const [isTrainingCommandOpen, setIsTrainingCommandOpen] = useState(false);
  const [isAddSnapshotOpen, setIsAddSnapshotOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFieldKey, setSelectedFieldKey] = useState<string | null>(null);
  const isSearchActive = searchQuery.trim().length > 0 || selectedFieldKey !== null;
  const presetOwnedFieldCount = useMemo(
    () =>
      sections.reduce(
        (count, section) => count + presetOwnedCount(section.fields),
        0,
      ),
    [sections],
  );
  const searchOptions = useMemo(
    () => flattenConfigSearchOptions(sections, overrides),
    [sections, overrides],
  );
  const configFields = useMemo(
    () => sections.flatMap((section) => section.fields),
    [sections],
  );
  const visibleSections = useMemo(
    () =>
      filterConfigSectionsForSearch(sections, {
        query: searchQuery,
        selectedFieldKey,
      }),
    [sections, searchQuery, selectedFieldKey],
  );
  const searchOpenKey = isSearchActive
    ? `${selectedFieldKey ?? ""}\u0000${searchQuery.trim()}`
    : "all";
  const {
    openSectionTitles,
    areAllSectionsOpen,
    sectionRefs,
    toggleSection,
    toggleAllSections,
    jumpToSection,
  } = useConfigDialogSections(visibleSections, searchOpenKey);
  const trainingCommand = useMemo(
    () => buildTrainingCommand({ model, preset, sections, overrides }),
    [model, preset, sections, overrides],
  );
  const { status: copyStatus, copy: copyTrainingCommand } =
    useCopyToClipboard(trainingCommand);

  function handleSearchQueryChange(query: string) {
    setSearchQuery(query);
    setSelectedFieldKey(null);
  }

  function handleSearchClear() {
    setSearchQuery("");
    setSelectedFieldKey(null);
  }

  function handleSearchSelect(option: ConfigSearchOption) {
    setSearchQuery(option.label);
    setSelectedFieldKey(option.key);
    jumpToSection(option.sectionTitle);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-3 backdrop-blur-sm sm:p-6">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="full-config-title"
        className="edge full-config-dialog-shell flex max-h-[calc(100vh-1.5rem)] w-full max-w-[92rem] flex-col overflow-hidden rounded-card shadow-[0_24px_80px_rgba(0,0,0,0.58)] sm:max-h-[calc(100vh-3rem)]"
      >
        <header className="full-config-dialog-chrome full-config-dialog-header sticky top-0 z-10 border-b border-line-soft px-4 py-3 backdrop-blur sm:px-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <h2 id="full-config-title" className="text-base font-semibold text-ink">
                Full Configuration
              </h2>
              <div className="mt-1 flex min-w-0 flex-wrap items-center gap-1.5 text-xs text-ink-faint">
                <span className="max-w-full truncate font-mono">{model || "No model"}</span>
                {preset && <span aria-hidden>/</span>}
                {preset && <span className="max-w-full truncate font-mono">{preset}</span>}
              </div>
            </div>
            <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
              <Badge>{sections.length} sections</Badge>
              <ConfigMetricBadge
                count={fieldCount}
                kind="fields"
                tooltipPosition="bottom"
              />
              <ConfigMetricBadge
                count={overrideCount}
                kind="overrides"
                variant={overrideCount > 0 ? "override" : "default"}
                tooltipPosition="bottom"
              />
              {presetOwnedFieldCount > 0 && (
                <Badge variant="preset">{presetOwnedFieldCount} preset</Badge>
              )}
              {configSnapshotCount > 0 && (
                <Badge className="border-ok/30 bg-ok/10 text-ok">
                  {configSnapshotCount} snapshots
                </Badge>
              )}
              <button
                type="button"
                aria-label="Close full config"
                onClick={onClose}
                className="flex h-9 w-9 items-center justify-center rounded-[10px] border border-line-soft bg-white/[0.025] text-ink-faint shadow-[inset_0_1px_0_rgba(255,255,255,0.035)] transition hover:border-line hover:bg-white/[0.055] hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                <X className="h-4 w-4" aria-hidden />
              </button>
            </div>
          </div>
        </header>

        <div className="full-config-dialog-body min-h-0 flex-1 overflow-y-auto px-4 py-4 sm:px-5">
          {isLoading ? (
            <div className="rounded-[10px] border border-dashed border-faint bg-white/[0.018] p-4 text-sm text-ink-faint">
              Loading config schema...
            </div>
          ) : fieldCount === 0 ? (
            <div className="rounded-[10px] border border-dashed border-faint bg-white/[0.018] p-4 text-sm text-ink-faint">
              No config fields
            </div>
          ) : (
            <div className="grid min-h-0 gap-5 lg:grid-cols-[300px_minmax(0,1fr)]">
              {configSnapshotGroups.length > 0 && (
                <div className="lg:col-span-2">
                  <ConfigSnapshotsTray
                    groups={configSnapshotGroups}
                    selectedPreset={preset}
                    overrides={overrides}
                    onLoad={loadConfigSnapshot}
                    onRename={renameConfigSnapshot}
                    onRemove={removeConfigSnapshot}
                  />
                </div>
              )}
              <div className="lg:col-span-2">
                <ConfigFieldSearch
                  options={searchOptions}
                  query={searchQuery}
                  selectedFieldKey={selectedFieldKey}
                  overrides={overrides}
                  onQueryChange={handleSearchQueryChange}
                  onClear={handleSearchClear}
                  onSelect={handleSearchSelect}
                  onFieldChange={onFieldChange}
                  onFieldReset={onFieldReset}
                />
              </div>
              <SectionNavigation
                sections={visibleSections}
                overrides={overrides}
                openSectionTitles={openSectionTitles}
                areAllSectionsOpen={areAllSectionsOpen}
                emptyMessage={isSearchActive ? "No matching sections" : undefined}
                onJumpToSection={jumpToSection}
                onToggleSection={toggleSection}
                onToggleAllSections={toggleAllSections}
              />
              <div className="grid auto-rows-max items-start gap-3">
                {visibleSections.length > 0 ? (
                  visibleSections.map((section, index) => {
                    const sectionId = sectionElementId(index, section.title);
                    return (
                      <ConfigSectionAccordion
                        key={section.title}
                        id={sectionId}
                        refCallback={(element) => {
                          sectionRefs.current[section.title] = element;
                        }}
                        title={section.title}
                        fields={section.fields}
                        overrides={overrides}
                        isOpen={openSectionTitles.has(section.title)}
                        onToggle={() => toggleSection(section.title)}
                        onFieldChange={onFieldChange}
                        onFieldReset={onFieldReset}
                      />
                    );
                  })
                ) : (
                  <div className="rounded-[12px] border border-dashed border-line-soft bg-black/20 px-4 py-6 text-sm text-ink-dim">
                    {`No config fields match "${searchQuery.trim()}".`}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        <footer className="full-config-dialog-chrome full-config-dialog-footer sticky bottom-0 z-10 flex flex-wrap items-center justify-between gap-2 border-t border-line-soft px-4 py-3 backdrop-blur sm:px-5">
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="secondary"
              onClick={onResetOverrides}
              disabled={!model || overrideCount === 0}
            >
              <RotateCcw className="h-4 w-4" aria-hidden />
              Reset Overrides
            </Button>
            <Button
              variant="secondary"
              onClick={() => setIsAddSnapshotOpen(true)}
              disabled={!model || !preset || fieldCount === 0}
            >
              <FilePlus2 className="h-4 w-4" aria-hidden />
              Add Config Snapshot
            </Button>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
            <Button variant="secondary" onClick={() => setIsTrainingCommandOpen(true)}>
              <Terminal className="h-4 w-4" aria-hidden />
              Training Command
            </Button>
            <Button variant="primary" onClick={onUpdatePreview} disabled={!canUpdate}>
              Update Preview
            </Button>
          </div>
        </footer>
      </section>
      {isTrainingCommandOpen && (
        <TrainingCommandDialog
          model={model}
          preset={preset}
          trainingCommand={trainingCommand}
          copyStatus={copyStatus}
          onCopy={copyTrainingCommand}
          onClose={() => setIsTrainingCommandOpen(false)}
        />
      )}
      {isAddSnapshotOpen && (
        <AddConfigSnapshotDialog
          model={model}
          preset={preset}
          fields={configFields}
          overrides={overrides}
          snapshots={configSnapshots}
          onAdd={addConfigSnapshot}
          onClose={() => setIsAddSnapshotOpen(false)}
        />
      )}
    </div>
  );
}
