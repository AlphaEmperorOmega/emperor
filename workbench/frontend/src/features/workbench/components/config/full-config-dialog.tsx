import { useCallback, useMemo, useState } from "react";
import { FilePlus2, RotateCcw, SlidersHorizontal, Terminal, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import {
  presentRuntimeDefaultsSchema,
  type RuntimeDefaultsSearchOptionPresentation,
} from "@/features/workbench/state/full-config/runtime-defaults-schema-presentation";
import { ConfigFieldSearch } from "@/features/workbench/components/config/config-field-search";
import { ConfigSectionAccordion } from "@/features/workbench/components/config/config-section-accordion";
import { ConfigMetricBadge } from "@/features/workbench/components/config/config-metric-badge";
import { SectionNavigation } from "@/features/workbench/components/config/section-navigation";
import { TrainingCommandDialog } from "@/features/workbench/components/config/training-command-dialog";
import {
  AddConfigSnapshotDialog,
  ConfigSnapshotsTray,
} from "@/features/workbench/components/config/config-snapshots-tray";
import { DialogShell } from "@/features/workbench/components/shared/dialog-shell";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { surfacePanelClassName } from "@/components/ui/surface-panel";
import {
  configFieldsNoMatchCopy,
  workbenchStatusCopy,
} from "@/features/workbench/components/shared/status-copy";
import { useConfigDialogSections } from "@/features/workbench/components/config/use-config-dialog-sections";
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";
import { useFullConfigSession } from "@/features/workbench/state/full-config/use-full-config-session";
import {
  type FullConfigDialogMode,
  type FullConfigDialogScope,
} from "@/features/workbench/state/use-workbench-workspace-shell";
import { cn } from "@/lib/utils";

export function FullConfigDialog({
  mode = "default",
  scope = "model",
  onClose,
}: {
  mode?: FullConfigDialogMode;
  scope?: FullConfigDialogScope;
  onClose: () => void;
}) {
  const session = useFullConfigSession({ mode, scope, onClose });
  const isSnapshotDraftMode = session.kind === "snapshot-draft";
  const isSnapshotEditMode = session.kind === "snapshot-edit";
  const isSnapshotSaveMode = isSnapshotDraftMode || isSnapshotEditMode;
  const isTrainingScope = session.kind === "training";
  const { modelType, model, preset } = session.identity;
  const {
    sections,
    fieldCount,
    overrides: dialogOverrides,
    overrideCount: dialogOverrideCount,
    lockedOverrideCount,
    isLoading,
  } = session.runtimeDefaults;
  const {
    records: displayedSnapshots,
    recordGroups: displayedSnapshotGroups,
    selected: selectedConfigSnapshot,
    selectedTrainingIds: selectedTrainingSnapshotIds,
    mutation: configSnapshotMutation,
    saveMutation,
  } = session.snapshots;
  const snapshotModeLabel = isSnapshotDraftMode
    ? "Snapshot draft"
    : isSnapshotEditMode
      ? "Snapshot edit"
      : "";
  const displayedSnapshotCount = displayedSnapshots.length;
  const [isTrainingCommandOpen, setIsTrainingCommandOpen] = useState(false);
  const [isAddSnapshotOpen, setIsAddSnapshotOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFieldKey, setSelectedFieldKey] = useState<string | null>(null);
  const schemaPresentation = useMemo(
    () =>
      presentRuntimeDefaultsSchema({
        sections,
        overrides: dialogOverrides,
        search: { query: searchQuery, selectedFieldKey },
      }),
    [dialogOverrides, searchQuery, sections, selectedFieldKey],
  );
  const {
    schemaFields: configFields,
    presetOwnedFieldCount,
    sections: visibleRenderSections,
    defaultOpenSectionTitles,
    searchOpenKey,
    isSearchActive,
    search,
  } = schemaPresentation;
  const {
    openSectionTitles,
    sectionRefs,
    toggleSection,
    setOpenSections,
    jumpToSection,
  } = useConfigDialogSections(
    visibleRenderSections,
    searchOpenKey,
    defaultOpenSectionTitles,
  );
  const effectiveOpenSectionTitles = useMemo(() => {
    const titles = new Set<string>();
    for (const title of openSectionTitles) {
      const section = visibleRenderSections.find((candidate) => candidate.title === title);
      if (section && !section.isDisabled) {
        titles.add(title);
      }
    }
    return titles;
  }, [openSectionTitles, visibleRenderSections]);
  const enabledVisibleSections = useMemo(
    () =>
      visibleRenderSections.filter(
        (section) => !section.isDisabled,
      ),
    [visibleRenderSections],
  );
  const areAllEnabledSectionsOpen =
    enabledVisibleSections.length > 0 &&
    enabledVisibleSections.every((section) =>
      effectiveOpenSectionTitles.has(section.title),
    );
  const trainingCommand = session.trainingCommand;
  const handleToggleAllSections = useCallback(() => {
    setOpenSections(
      areAllEnabledSectionsOpen
        ? []
        : enabledVisibleSections.map((section) => section.title),
    );
  }, [areAllEnabledSectionsOpen, enabledVisibleSections, setOpenSections]);
  const { status: copyStatus, copy: copyTrainingCommand } =
    useCopyToClipboard(trainingCommand);
  const handleFieldChange = session.actions.editOverride;
  const handleFieldReset = session.actions.clearOverride;
  const handleResetOverrides = session.actions.resetOverrides;
  const closeFullConfig = session.actions.close;

  function handleSearchQueryChange(query: string) {
    setSearchQuery(query);
    setSelectedFieldKey(null);
  }

  function handleSearchClear() {
    setSearchQuery("");
    setSelectedFieldKey(null);
  }

  function handleSearchSelect(option: RuntimeDefaultsSearchOptionPresentation) {
    setSearchQuery(option.label);
    setSelectedFieldKey(option.key);
    jumpToSection(option.rootSectionTitle);
  }

  return (
    <DialogShell
      size="fullscreen"
      panelVariant="surface"
      titleId="full-config-title"
      onClose={closeFullConfig}
      panelClassName="full-config-dialog-shell"
      header={
        <header className="full-config-dialog-chrome full-config-dialog-header sticky top-0 z-10 border-b border-line-soft px-region py-panel backdrop-blur-xl sm:px-shell">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <h2
                id="full-config-title"
                className="flex items-center gap-2 type-title text-balance font-semibold text-ink"
              >
                <SlidersHorizontal
                  className="h-[15px] w-[15px] text-violet"
                  aria-hidden
                />
                {isTrainingScope ? "Training Full Configuration" : "Full Configuration"}
              </h2>
              <div className="mt-1 flex min-w-0 flex-wrap items-center gap-1.5 text-xs text-ink-faint">
                <span className="max-w-full truncate font-mono">{model || "No model"}</span>
                {preset && <span aria-hidden>/</span>}
                {preset && <span className="max-w-full truncate font-mono">{preset}</span>}
              </div>
            </div>
            <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
              <ConfigMetricBadge
                count={fieldCount}
                kind="fields"
                tooltipPosition="bottom"
              />
              <ConfigMetricBadge
                count={dialogOverrideCount}
                kind="overrides"
                variant={dialogOverrideCount > 0 ? "override" : "default"}
                tooltipPosition="bottom"
              />
              {snapshotModeLabel && <Badge>{snapshotModeLabel}</Badge>}
              {lockedOverrideCount > 0 && !isSnapshotSaveMode && (
                <Badge variant="preset">
                  {lockedOverrideCount} inactive
                </Badge>
              )}
              {presetOwnedFieldCount > 0 && (
                <Badge variant="preset">{presetOwnedFieldCount} preset</Badge>
              )}
              {!isTrainingScope && displayedSnapshotCount > 0 && (
                <Badge variant="success">
                  {displayedSnapshotCount} snapshots
                </Badge>
              )}
              <IconButton
                label="Close full config"
                onClick={closeFullConfig}
                variant="edge"
                className="border-line-soft bg-white/[0.025] shadow-control hover:border-line hover:bg-white/[0.055]"
                icon={<X className="h-4 w-4" aria-hidden />}
              />
            </div>
          </div>
        </header>
      }
      footer={
        <footer className="full-config-dialog-chrome full-config-dialog-footer sticky bottom-0 z-10 flex flex-wrap items-center justify-between gap-2 border-t border-line-soft px-region py-panel backdrop-blur-xl sm:px-shell">
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="secondary"
              onClick={handleResetOverrides}
              disabled={!session.controls.canReset}
            >
              <RotateCcw className="h-4 w-4" aria-hidden />
              Reset Overrides
            </Button>
            {!isSnapshotSaveMode && !isTrainingScope && (
              <Button
                variant="secondary"
                onClick={() => {
                  if (session.actions.openSnapshotSave()) {
                    setIsAddSnapshotOpen(true);
                  }
                }}
                disabled={!session.controls.canAddSnapshot}
              >
                <FilePlus2 className="h-4 w-4" aria-hidden />
                Add Config Snapshot
              </Button>
            )}
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            <Button variant="ghost" onClick={closeFullConfig}>
              Close
            </Button>
            {!isTrainingScope && (
              <Button
                variant="secondary"
                onClick={() => setIsTrainingCommandOpen(true)}
              >
                <Terminal className="h-4 w-4" aria-hidden />
                Training Command
              </Button>
            )}
            {isTrainingScope ? (
              <Button variant="primary" onClick={closeFullConfig}>
                Done
              </Button>
            ) : isSnapshotSaveMode ? (
              <Button
                variant="primary"
                onClick={() => setIsAddSnapshotOpen(true)}
                disabled={!session.controls.canSaveSnapshot}
              >
                <FilePlus2 className="h-4 w-4" aria-hidden />
                {isSnapshotEditMode ? "Save Snapshot Changes" : "Save as Snapshot"}
              </Button>
            ) : (
              <Button
                variant="primary"
                onClick={session.actions.updatePreview}
                disabled={!session.controls.canUpdatePreview}
              >
                Update Preview
              </Button>
            )}
          </div>
        </footer>
      }
      overlayChildren={
        <>
          {!isTrainingScope && isTrainingCommandOpen && (
            <TrainingCommandDialog
              model={model}
              preset={preset}
              trainingCommand={trainingCommand}
              copyStatus={copyStatus}
              onCopy={copyTrainingCommand}
              onClose={() => setIsTrainingCommandOpen(false)}
            />
          )}
          {!isTrainingScope && isAddSnapshotOpen && (
            <AddConfigSnapshotDialog
              modelType={modelType}
              model={model}
              preset={preset}
              fields={configFields}
              overrides={dialogOverrides}
              snapshots={displayedSnapshots}
              title={
                isSnapshotEditMode
                  ? "Save Snapshot Changes"
                  : isSnapshotDraftMode
                    ? "Save as Snapshot"
                    : undefined
              }
              actionLabel={
                isSnapshotEditMode
                  ? "Save Snapshot Changes"
                  : isSnapshotDraftMode
                    ? "Save Snapshot"
                    : undefined
              }
              initialName={
                isSnapshotEditMode ? selectedConfigSnapshot?.name : undefined
              }
              excludeSnapshotId={
                isSnapshotEditMode ? selectedConfigSnapshot?.id : undefined
              }
              mutation={saveMutation}
              onAdd={session.actions.saveSnapshot}
              onRetry={session.actions.retrySnapshotSave}
              onDismissMutation={session.actions.dismissSaveMutation}
              onClose={() => {
                session.actions.closeSnapshotSave();
                setIsAddSnapshotOpen(false);
              }}
            />
          )}
        </>
      }
    >
      <div className="full-config-dialog-body min-h-0 flex-1 overflow-y-auto p-region sm:px-shell">
        {isLoading ? (
          <InlineStatus busy>
            {workbenchStatusCopy.loading.configSchema}
          </InlineStatus>
        ) : fieldCount === 0 ? (
          <InlineStatus>
            {workbenchStatusCopy.empty.configFields}
          </InlineStatus>
        ) : (
          <div className="grid min-h-0 gap-shell lg:grid-cols-[300px_minmax(0,1fr)]">
            {!isTrainingScope && displayedSnapshotGroups.length > 0 && (
              <div className="lg:col-span-2">
                <ConfigSnapshotsTray
                  groups={displayedSnapshotGroups}
                  selectedPreset={preset}
                  selectedTrainingSnapshotIds={selectedTrainingSnapshotIds}
                  overrides={dialogOverrides}
                  canManage={session.snapshots.canManage}
                  mutation={configSnapshotMutation}
                  onLoad={session.actions.loadSnapshot}
                  onRename={session.actions.renameSnapshot}
                  onRemove={session.actions.removeSnapshot}
                  onRetryMutation={session.actions.retryMutation}
                  onDismissMutation={session.actions.dismissMutation}
                  onToggleSelection={session.actions.toggleSnapshotRunSelection}
                />
              </div>
            )}
            <div className="lg:col-span-2">
              <ConfigFieldSearch
                options={search.options}
                query={searchQuery}
                selectedFieldKey={selectedFieldKey}
                matchesQuery={search.matchesQuery}
                onQueryChange={handleSearchQueryChange}
                onClear={handleSearchClear}
                onSelect={handleSearchSelect}
                onFieldChange={handleFieldChange}
                onFieldReset={handleFieldReset}
              />
            </div>
            <SectionNavigation
              sections={visibleRenderSections}
              openSectionTitles={effectiveOpenSectionTitles}
              areAllSectionsOpen={areAllEnabledSectionsOpen}
              emptyMessage={isSearchActive ? "No matching sections" : undefined}
              onJumpToSection={jumpToSection}
              onToggleSection={toggleSection}
              onToggleAllSections={handleToggleAllSections}
            />
            <div className="grid auto-rows-max items-start gap-3">
              {visibleRenderSections.length > 0 ? (
                visibleRenderSections.map((section) => {
                  const isSectionOpen = effectiveOpenSectionTitles.has(section.title);
                  return (
                    <ConfigSectionAccordion
                      key={section.title}
                      section={section}
                      refCallback={(element) => {
                        sectionRefs.current[section.title] = element;
                      }}
                      isOpen={isSectionOpen}
                      autoOpenKey={searchOpenKey}
                      onToggle={() => toggleSection(section.title)}
                      onFieldChange={handleFieldChange}
                      onFieldReset={handleFieldReset}
                    />
                  );
                })
              ) : (
                <div
                  className={cn(
                    surfacePanelClassName,
                    "border-dashed border-line-soft bg-black/20 px-4 py-6 text-sm text-ink-dim",
                  )}
                >
                  {configFieldsNoMatchCopy(searchQuery.trim())}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </DialogShell>
  );
}
