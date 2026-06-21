import { useCallback, useMemo, useState } from "react";
import { FilePlus2, RotateCcw, SlidersHorizontal, Terminal, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import {
  type ConfigSearchOption,
  controlledSectionState,
  deriveNestedConfigSections,
  disabledConfigFieldReasons,
  filterConfigSectionsForSearch,
  flattenConfigSearchOptions,
  isDefaultConfigFieldValue,
  modifiedCount,
  presetOwnedCount,
  sectionElementId,
} from "@/lib/config";
import { buildTrainingCommand } from "@/lib/training-command";
import { ConfigFieldSearch } from "@/features/viewer/components/config/config-field-search";
import { ConfigSectionAccordion } from "@/features/viewer/components/config/config-section-accordion";
import { ConfigMetricBadge } from "@/features/viewer/components/config/config-metric-badge";
import { SectionNavigation } from "@/features/viewer/components/config/section-navigation";
import { TrainingCommandDialog } from "@/features/viewer/components/config/training-command-dialog";
import {
  AddConfigSnapshotDialog,
  ConfigSnapshotsTray,
} from "@/features/viewer/components/config/config-snapshots-tray";
import { DialogShell } from "@/features/viewer/components/shared/dialog-shell";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { surfacePanelClassName } from "@/features/viewer/components/shared/surface-panel";
import {
  configFieldsNoMatchCopy,
  viewerStatusCopy,
} from "@/features/viewer/components/shared/status-copy";
import { useConfigDialogSections } from "@/features/viewer/components/config/use-config-dialog-sections";
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";
import { useTargetConfig } from "@/features/viewer/providers/viewer-providers";
import {
  type FullConfigDialogMode,
} from "@/features/viewer/state/use-viewer-workspace-shell";
import { cn } from "@/lib/utils";

export function FullConfigDialog({
  mode = "default",
  onClose,
}: {
  mode?: FullConfigDialogMode;
  onClose: () => void;
}) {
  const {
    selectedModelType: modelType,
    selectedModel: model,
    selectedPreset: preset,
    configSections: sections,
    fieldCount,
    overrides,
    snapshotEditorDraft,
    activeOverrideScopeLabel,
    inactiveLockedOverrideCount,
    snapshotOverrideWarning,
    selectedConfigSnapshot,
    selectedTrainingSnapshotIds,
    selectedDatasets,
    allConfigSnapshots,
    allConfigSnapshotGroups,
    allConfigSnapshotCount,
    capabilities,
    schemaLoading,
    addConfigSnapshot,
    removeConfigSnapshot,
    renameConfigSnapshot,
    updateSelectedConfigSnapshot,
    loadConfigSnapshot,
    toggleConfigSnapshotRunSelection,
    updateOverride,
    clearOverride,
    updateSnapshotEditorDraftOverride,
    clearSnapshotEditorDraftOverride,
    resetSnapshotEditorDraft,
    resetOverrides,
    updatePreview: onUpdatePreview,
  } = useTargetConfig();
  const isLoading = schemaLoading;
  const isSnapshotDraftMode = mode === "snapshotDraft";
  const isSnapshotEditMode = mode === "snapshotEdit";
  const isSnapshotSaveMode = isSnapshotDraftMode || isSnapshotEditMode;
  const dialogOverrides = isSnapshotSaveMode ? snapshotEditorDraft : overrides;
  const dialogOverrideCount = Object.keys(dialogOverrides).length;
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
    () => flattenConfigSearchOptions(sections, dialogOverrides),
    [sections, dialogOverrides],
  );
  const configFields = useMemo(
    () => sections.flatMap((section) => section.fields),
    [sections],
  );
  const configFieldsByKey = useMemo(
    () => new Map(configFields.map((field) => [field.key, field])),
    [configFields],
  );
  const visibleSections = useMemo(
    () =>
      filterConfigSectionsForSearch(sections, {
        query: searchQuery,
        selectedFieldKey,
      }),
    [sections, searchQuery, selectedFieldKey],
  );
  const visibleRenderSections = useMemo(
    () => deriveNestedConfigSections(visibleSections, sections),
    [sections, visibleSections],
  );
  const sectionsByTitle = useMemo(
    () => new Map(sections.map((section) => [section.title, section])),
    [sections],
  );
  const disabledFieldReasons = useMemo(
    () => disabledConfigFieldReasons(sections, dialogOverrides),
    [sections, dialogOverrides],
  );
  const disabledSectionTitles = useMemo(() => {
    const titles = new Set<string>();
    for (const section of visibleSections) {
      const sourceSection = sectionsByTitle.get(section.title) ?? section;
      const state = controlledSectionState(sourceSection, dialogOverrides);
      if (state.isControlled && !state.isEnabled) {
        titles.add(section.title);
      }
    }
    return titles;
  }, [dialogOverrides, sectionsByTitle, visibleSections]);
  const searchOpenKey = isSearchActive
    ? `${selectedFieldKey ?? ""}\u0000${searchQuery.trim()}`
    : "all";
  const defaultOpenSectionTitles = useMemo(() => {
    if (isSearchActive) {
      return visibleSections.map((section) => section.title);
    }
    const titles = new Set<string>();
    const firstSection = visibleSections[0];
    if (firstSection) {
      titles.add(firstSection.title);
    }
    for (const section of visibleSections) {
      const sourceSection = sectionsByTitle.get(section.title) ?? section;
      if (modifiedCount(sourceSection.fields, dialogOverrides) > 0) {
        titles.add(section.title);
      }
    }
    return Array.from(titles);
  }, [dialogOverrides, isSearchActive, sectionsByTitle, visibleSections]);
  const {
    openSectionTitles,
    sectionRefs,
    toggleSection,
    setOpenSections,
    jumpToSection,
  } = useConfigDialogSections(
    visibleSections,
    searchOpenKey,
    defaultOpenSectionTitles,
  );
  const effectiveOpenSectionTitles = useMemo(() => {
    const titles = new Set<string>();
    for (const title of openSectionTitles) {
      if (!disabledSectionTitles.has(title)) {
        titles.add(title);
      }
    }
    return titles;
  }, [disabledSectionTitles, openSectionTitles]);
  const enabledVisibleSections = useMemo(
    () =>
      visibleSections.filter(
        (section) => !disabledSectionTitles.has(section.title),
      ),
    [disabledSectionTitles, visibleSections],
  );
  const areAllEnabledSectionsOpen =
    enabledVisibleSections.length > 0 &&
    enabledVisibleSections.every((section) =>
      effectiveOpenSectionTitles.has(section.title),
    );
  const trainingCommand = useMemo(
    () =>
      isTrainingCommandOpen
        ? buildTrainingCommand({
            modelType,
            model,
            preset,
            sections,
            overrides: dialogOverrides,
          })
        : "",
    [dialogOverrides, isTrainingCommandOpen, modelType, model, preset, sections],
  );
  const handleToggleAllSections = useCallback(() => {
    setOpenSections(
      areAllEnabledSectionsOpen
        ? []
        : enabledVisibleSections.map((section) => section.title),
    );
  }, [areAllEnabledSectionsOpen, enabledVisibleSections, setOpenSections]);
  const { status: copyStatus, copy: copyTrainingCommand } =
    useCopyToClipboard(trainingCommand);
  const handleFieldChange = useCallback(
    (key: string, value: string) => {
      const field = configFieldsByKey.get(key);
      if (field && isDefaultConfigFieldValue(field, value)) {
        if (isSnapshotSaveMode) {
          clearSnapshotEditorDraftOverride(key);
          return;
        }
        clearOverride(key);
        return;
      }
      if (isSnapshotSaveMode) {
        updateSnapshotEditorDraftOverride(key, value);
        return;
      }
      updateOverride(key, value);
    },
    [
      clearOverride,
      clearSnapshotEditorDraftOverride,
      configFieldsByKey,
      isSnapshotSaveMode,
      updateOverride,
      updateSnapshotEditorDraftOverride,
    ],
  );
  const handleFieldReset = useCallback(
    (key: string) => {
      if (isSnapshotSaveMode) {
        clearSnapshotEditorDraftOverride(key);
        return;
      }
      clearOverride(key);
    },
    [clearOverride, clearSnapshotEditorDraftOverride, isSnapshotSaveMode],
  );
  const handleResetOverrides = useCallback(() => {
    if (isSnapshotSaveMode) {
      resetSnapshotEditorDraft();
      return;
    }
    resetOverrides();
  }, [
    isSnapshotSaveMode,
    resetOverrides,
    resetSnapshotEditorDraft,
  ]);

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
    <DialogShell
      size="fullscreen"
      panelVariant="surface"
      titleId="full-config-title"
      onClose={onClose}
      panelClassName="full-config-dialog-shell"
      header={
        <header className="full-config-dialog-chrome full-config-dialog-header sticky top-0 z-10 border-b border-line-soft px-4 py-3 backdrop-blur sm:px-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <h2
                id="full-config-title"
                className="flex items-center gap-2 text-base font-semibold text-ink"
              >
                <SlidersHorizontal
                  className="h-[15px] w-[15px] text-violet"
                  aria-hidden
                />
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
                count={dialogOverrideCount}
                kind="overrides"
                variant={dialogOverrideCount > 0 ? "override" : "default"}
                tooltipPosition="bottom"
              />
              <Badge>{isSnapshotSaveMode ? "Snapshot draft" : activeOverrideScopeLabel}</Badge>
              {inactiveLockedOverrideCount > 0 && !isSnapshotSaveMode && (
                <Badge variant="preset">
                  {inactiveLockedOverrideCount} inactive
                </Badge>
              )}
              {presetOwnedFieldCount > 0 && (
                <Badge variant="preset">{presetOwnedFieldCount} preset</Badge>
              )}
              {allConfigSnapshotCount > 0 && (
                <Badge variant="success">
                  {allConfigSnapshotCount} snapshots
                </Badge>
              )}
              <IconButton
                label="Close full config"
                onClick={onClose}
                variant="edge"
                className="border-line-soft bg-white/[0.025] shadow-[inset_0_1px_0_rgba(255,255,255,0.035)] hover:border-line hover:bg-white/[0.055]"
                icon={<X className="h-4 w-4" aria-hidden />}
              />
            </div>
          </div>
        </header>
      }
      footer={
        <footer className="full-config-dialog-chrome full-config-dialog-footer sticky bottom-0 z-10 flex flex-wrap items-center justify-between gap-2 border-t border-line-soft px-4 py-3 backdrop-blur sm:px-5">
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="secondary"
              onClick={handleResetOverrides}
              disabled={
                !model ||
                (dialogOverrideCount === 0 &&
                  (isSnapshotSaveMode || inactiveLockedOverrideCount === 0))
              }
            >
              <RotateCcw className="h-4 w-4" aria-hidden />
              Reset Overrides
            </Button>
            {!isSnapshotSaveMode && (
              <Button
                variant="secondary"
                onClick={() => setIsAddSnapshotOpen(true)}
                disabled={
                  !model ||
                  !preset ||
                  fieldCount === 0 ||
                  !capabilities.configSnapshotsEnabled
                }
              >
                <FilePlus2 className="h-4 w-4" aria-hidden />
                Add Config Snapshot
              </Button>
            )}
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
            <Button variant="secondary" onClick={() => setIsTrainingCommandOpen(true)}>
              <Terminal className="h-4 w-4" aria-hidden />
              Training Command
            </Button>
            {isSnapshotSaveMode ? (
              <Button
                variant="primary"
                onClick={() => setIsAddSnapshotOpen(true)}
                disabled={
                  !model ||
                  !preset ||
                  fieldCount === 0 ||
                  !capabilities.configSnapshotsEnabled ||
                  (isSnapshotEditMode && !selectedConfigSnapshot)
                }
              >
                <FilePlus2 className="h-4 w-4" aria-hidden />
                {isSnapshotEditMode ? "Save Snapshot Changes" : "Save as Snapshot"}
              </Button>
            ) : (
              <Button variant="primary" onClick={onUpdatePreview} disabled={!canUpdate}>
                Update Preview
              </Button>
            )}
          </div>
        </footer>
      }
      overlayChildren={
        <>
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
              modelType={modelType}
              model={model}
              preset={preset}
              fields={configFields}
              overrides={dialogOverrides}
              snapshots={allConfigSnapshots}
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
              onAdd={
                isSnapshotEditMode
                  ? (name) => updateSelectedConfigSnapshot(name, dialogOverrides)
                  : (name) => addConfigSnapshot(name, dialogOverrides)
              }
              onClose={() => setIsAddSnapshotOpen(false)}
            />
          )}
        </>
      }
    >
      <div className="full-config-dialog-body min-h-0 flex-1 overflow-y-auto px-4 py-4 sm:px-5">
        {isLoading ? (
          <InlineStatus>
            {viewerStatusCopy.loading.configSchema}
          </InlineStatus>
        ) : fieldCount === 0 ? (
          <InlineStatus>
            {viewerStatusCopy.empty.configFields}
          </InlineStatus>
        ) : (
          <div className="grid min-h-0 gap-5 lg:grid-cols-[300px_minmax(0,1fr)]">
            {snapshotOverrideWarning && !isSnapshotSaveMode && (
              <div className="lg:col-span-2">
                <InlineStatus tone="warning" compact>
                  {snapshotOverrideWarning}
                </InlineStatus>
              </div>
            )}
            {allConfigSnapshotGroups.length > 0 && (
              <div className="lg:col-span-2">
                <ConfigSnapshotsTray
                  groups={allConfigSnapshotGroups}
                  selectedPreset={preset}
                  selectedTrainingSnapshotIds={selectedTrainingSnapshotIds}
                  overrides={dialogOverrides}
                  canManage={capabilities.configSnapshotsEnabled}
                  onLoad={loadConfigSnapshot}
                  onRename={renameConfigSnapshot}
                  onRemove={removeConfigSnapshot}
                  onToggleSelection={toggleConfigSnapshotRunSelection}
                />
              </div>
            )}
            <div className="lg:col-span-2">
              <ConfigFieldSearch
                options={searchOptions}
                query={searchQuery}
                selectedFieldKey={selectedFieldKey}
                overrides={dialogOverrides}
                disabledFieldReasons={disabledFieldReasons}
                onQueryChange={handleSearchQueryChange}
                onClear={handleSearchClear}
                onSelect={handleSearchSelect}
                onFieldChange={handleFieldChange}
                onFieldReset={handleFieldReset}
              />
            </div>
            <SectionNavigation
              sections={visibleSections}
              overrides={dialogOverrides}
              openSectionTitles={effectiveOpenSectionTitles}
              disabledSectionTitles={disabledSectionTitles}
              areAllSectionsOpen={areAllEnabledSectionsOpen}
              emptyMessage={isSearchActive ? "No matching sections" : undefined}
              onJumpToSection={jumpToSection}
              onToggleSection={toggleSection}
              onToggleAllSections={handleToggleAllSections}
            />
            <div className="grid auto-rows-max items-start gap-3">
              {visibleSections.length > 0 ? (
                visibleRenderSections.map((section, index) => {
                  const sectionId = sectionElementId(index, section.title);
                  const sourceSection = sectionsByTitle.get(section.title) ?? section;
                  const sectionState = controlledSectionState(
                    sourceSection,
                    dialogOverrides,
                  );
                  const isSectionOpen = effectiveOpenSectionTitles.has(section.title);
                  return (
                    <ConfigSectionAccordion
                      key={section.title}
                      id={sectionId}
                      refCallback={(element) => {
                        sectionRefs.current[section.title] = element;
                      }}
                      title={section.title}
                      fields={section.fields}
                      childSections={section.children}
                      overrides={dialogOverrides}
                      isOpen={isSectionOpen}
                      controlField={sectionState.controlField}
                      disabledReason={sectionState.disabledReason}
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
