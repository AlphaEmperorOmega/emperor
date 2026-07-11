import { useCallback, useMemo, useState } from "react";
import { FilePlus2, RotateCcw, SlidersHorizontal, Terminal, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import {
  type ConfigSearchOption,
  type ConfigSection,
  configSectionsFields,
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
import {
  useConfigSnapshotEditor,
  useModelPackageInspection,
  useConfigSnapshotRecords,
} from "@/features/workbench/providers/workbench-providers";
import { useWorkbenchCapabilities } from "@/features/workbench/providers/workbench-connection-provider";
import { useTrainingConfiguration } from "@/features/workbench/providers/training-provider";
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
  const { capabilities } = useWorkbenchCapabilities();
  const modelPackageInspection = useModelPackageInspection();
  const { browser, options, runtimeDefaults, status, actions } =
    modelPackageInspection;
  const selectedModelType = browser.selectedModelType;
  const selectedModel = browser.selectedModel;
  const selectedPreset = browser.selectedPreset;
  const configSections = options.configSections;
  const modelFieldCount = runtimeDefaults.fieldCount;
  const modelOverrides = runtimeDefaults.active;
  const inactiveLockedOverrideCount = runtimeDefaults.inactiveLockedCount;
  const selectedDatasets = browser.selectedDatasets;
  const updateOverride = actions.editRuntimeDefault;
  const clearOverride = actions.clearRuntimeDefault;
  const resetOverrides = actions.resetRuntimeDefaults;
  const onUpdatePreview = actions.refreshInspection;
  const schemaLoading = status.schema.isLoading;
  const snapshotLibrary = useConfigSnapshotRecords();
  const allConfigSnapshots = snapshotLibrary.records.all;
  const allConfigSnapshotGroups = snapshotLibrary.records.allGroups;
  const removeConfigSnapshot = snapshotLibrary.actions.remove;
  const renameConfigSnapshot = snapshotLibrary.actions.rename;
  const loadConfigSnapshot = snapshotLibrary.actions.selectTarget;
  const snapshotEditor = useConfigSnapshotEditor();
  const editorSession = snapshotEditor.session;
  const selectedConfigSnapshot = editorSession.selectedSnapshot;
  const training = useTrainingConfiguration();
  const selectedTrainingModelType = training.selectedModelType;
  const selectedTrainingModel = training.selectedModel;
  const selectedTrainingPrimaryPreset = training.selectedPrimaryPreset;
  const selectedTrainingSnapshotIds = training.selectedSnapshotIds;
  const trainingConfigSections = training.configSections;
  const trainingFieldCount = training.fieldCount;
  const trainingOverrides = training.bulkOverrides;
  const trainingInactiveLockedOverrideCount =
    training.inactiveLockedOverrideCount;
  const selectedMonitors = training.selectedMonitors;
  const updateTrainingOverride = training.updateOverride;
  const clearTrainingOverride = training.clearOverride;
  const resetTrainingOverrides = training.resetOverrides;
  const trainingSchemaLoading = training.schemaLoading;
  const snapshotOverrideWarning = "";
  const toggleConfigSnapshotRunSelection = useCallback(
    (snapshotId: string) => {
      if (training.selectedSnapshotIds.includes(snapshotId)) {
        training.excludeSnapshot(snapshotId);
        return;
      }
      training.includeSnapshot(snapshotId);
    },
    [training],
  );
  const isSnapshotDraftMode = mode === "snapshotDraft";
  const isSnapshotEditMode = mode === "snapshotEdit";
  const isSnapshotSaveMode = isSnapshotDraftMode || isSnapshotEditMode;
  const isTrainingScope = scope === "training" && !isSnapshotSaveMode;
  const modelType = isSnapshotSaveMode
    ? editorSession.modelType
    : isTrainingScope
      ? selectedTrainingModelType
      : selectedModelType;
  const model = isSnapshotSaveMode
    ? editorSession.model
    : isTrainingScope
      ? selectedTrainingModel
      : selectedModel;
  const preset = isSnapshotSaveMode
    ? editorSession.preset
    : isTrainingScope
      ? selectedTrainingPrimaryPreset
      : selectedPreset;
  const sections = isSnapshotSaveMode
    ? editorSession.configSections
    : isTrainingScope
      ? trainingConfigSections
      : configSections;
  const fieldCount = isSnapshotSaveMode
    ? editorSession.fieldCount
    : isTrainingScope
      ? trainingFieldCount
      : modelFieldCount;
  const overrides = isTrainingScope ? trainingOverrides : modelOverrides;
  const lockedOverrideCount = isTrainingScope
    ? trainingInactiveLockedOverrideCount
    : inactiveLockedOverrideCount;
  const snapshotModeLabel = isSnapshotDraftMode
    ? "Snapshot draft"
    : isSnapshotEditMode
      ? "Snapshot edit"
      : "";
  const isLoading = isSnapshotSaveMode
    ? editorSession.status.isLoading
    : isTrainingScope
      ? trainingSchemaLoading
      : schemaLoading;
  const dialogOverrides = isSnapshotSaveMode ? editorSession.draft : overrides;
  const displayedSnapshots = isSnapshotSaveMode
    ? editorSession.records
    : allConfigSnapshots;
  const displayedSnapshotGroups = isSnapshotSaveMode
    ? editorSession.recordGroups
    : allConfigSnapshotGroups;
  const displayedSnapshotCount = displayedSnapshots.length;
  const dialogOverrideCount = Object.keys(dialogOverrides).length;
  const canUpdate = Boolean(model && preset && selectedDatasets.length > 0);
  const [isTrainingCommandOpen, setIsTrainingCommandOpen] = useState(false);
  const [isAddSnapshotOpen, setIsAddSnapshotOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFieldKey, setSelectedFieldKey] = useState<string | null>(null);
  const isSearchActive = searchQuery.trim().length > 0 || selectedFieldKey !== null;
  const presetOwnedFieldCount = useMemo(
    () => presetOwnedCount(configSectionsFields(sections)),
    [sections],
  );
  const searchOptions = useMemo(
    () => flattenConfigSearchOptions(sections),
    [sections],
  );
  const configFields = useMemo(
    () => configSectionsFields(sections),
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
    for (const section of visibleRenderSections) {
      const state = controlledSectionState(section, dialogOverrides);
      if (state.isControlled && !state.isEnabled) {
        titles.add(section.title);
      }
    }
    return titles;
  }, [dialogOverrides, visibleRenderSections]);
  const rootSectionTitleBySectionTitle = useMemo(() => {
    const titleMap = new Map<string, string>();

    function collect(section: ConfigSection, rootTitle: string) {
      titleMap.set(section.title, rootTitle);
      for (const child of section.children ?? []) {
        collect(child, rootTitle);
      }
    }

    for (const section of visibleRenderSections) {
      collect(section, section.title);
    }

    return titleMap;
  }, [visibleRenderSections]);
  const searchOpenKey = isSearchActive
    ? `${selectedFieldKey ?? ""}\u0000${searchQuery.trim()}`
    : "all";
  const defaultOpenSectionTitles = useMemo(() => {
    if (isSearchActive) {
      return visibleRenderSections.map((section) => section.title);
    }
    const titles = new Set<string>();
    const firstSection = visibleRenderSections[0];
    if (firstSection) {
      titles.add(firstSection.title);
    }
    for (const section of visibleRenderSections) {
      if (modifiedCount(configSectionsFields([section]), dialogOverrides) > 0) {
        titles.add(section.title);
      }
    }
    return Array.from(titles);
  }, [dialogOverrides, isSearchActive, visibleRenderSections]);
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
      if (!disabledSectionTitles.has(title)) {
        titles.add(title);
      }
    }
    return titles;
  }, [disabledSectionTitles, openSectionTitles]);
  const enabledVisibleSections = useMemo(
    () =>
      visibleRenderSections.filter(
        (section) => !disabledSectionTitles.has(section.title),
      ),
    [disabledSectionTitles, visibleRenderSections],
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
            monitors: selectedMonitors,
            sections,
            overrides: dialogOverrides,
          })
        : "",
    [
      dialogOverrides,
      isTrainingCommandOpen,
      modelType,
      model,
      preset,
      sections,
      selectedMonitors,
    ],
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
          snapshotEditor.actions.clearOverride(key);
          return;
        }
        if (isTrainingScope) {
          clearTrainingOverride(key);
          return;
        }
        clearOverride(key);
        return;
      }
      if (isSnapshotSaveMode) {
        snapshotEditor.actions.updateOverride(key, value);
        return;
      }
      if (isTrainingScope) {
        updateTrainingOverride(key, value);
        return;
      }
      updateOverride(key, value);
    },
    [
      clearOverride,
      clearTrainingOverride,
      configFieldsByKey,
      isSnapshotSaveMode,
      isTrainingScope,
      updateOverride,
      updateTrainingOverride,
      snapshotEditor.actions,
    ],
  );
  const handleFieldReset = useCallback(
    (key: string) => {
      if (isSnapshotSaveMode) {
        snapshotEditor.actions.clearOverride(key);
        return;
      }
      if (isTrainingScope) {
        clearTrainingOverride(key);
        return;
      }
      clearOverride(key);
    },
    [
      clearOverride,
      clearTrainingOverride,
      isSnapshotSaveMode,
      isTrainingScope,
      snapshotEditor.actions,
    ],
  );
  const handleResetOverrides = useCallback(() => {
    if (isSnapshotSaveMode) {
      snapshotEditor.actions.reset();
      return;
    }
    if (isTrainingScope) {
      resetTrainingOverrides();
      return;
    }
    resetOverrides();
  }, [
    isSnapshotSaveMode,
    isTrainingScope,
    resetOverrides,
    resetTrainingOverrides,
    snapshotEditor.actions,
  ]);
  const closeFullConfig = useCallback(() => {
    if (isSnapshotSaveMode) {
      snapshotEditor.actions.close();
    }
    onClose();
  }, [isSnapshotSaveMode, onClose, snapshotEditor.actions]);

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
    jumpToSection(
      rootSectionTitleBySectionTitle.get(option.sectionTitle) ?? option.sectionTitle,
    );
  }

  return (
    <DialogShell
      size="fullscreen"
      panelVariant="surface"
      titleId="full-config-title"
      onClose={closeFullConfig}
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
                  (isSnapshotSaveMode || lockedOverrideCount === 0))
              }
            >
              <RotateCcw className="h-4 w-4" aria-hidden />
              Reset Overrides
            </Button>
            {!isSnapshotSaveMode && !isTrainingScope && (
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
              onAdd={(name) => snapshotEditor.actions.save(name)}
              onClose={() => setIsAddSnapshotOpen(false)}
            />
          )}
        </>
      }
    >
      <div className="full-config-dialog-body min-h-0 flex-1 overflow-y-auto px-4 py-4 sm:px-5">
        {isLoading ? (
          <InlineStatus>
            {workbenchStatusCopy.loading.configSchema}
          </InlineStatus>
        ) : fieldCount === 0 ? (
          <InlineStatus>
            {workbenchStatusCopy.empty.configFields}
          </InlineStatus>
        ) : (
          <div className="grid min-h-0 gap-5 lg:grid-cols-[300px_minmax(0,1fr)]">
            {snapshotOverrideWarning && !isSnapshotSaveMode && !isTrainingScope && (
              <div className="lg:col-span-2">
                <InlineStatus tone="warning" compact>
                  {snapshotOverrideWarning}
                </InlineStatus>
              </div>
            )}
            {!isTrainingScope && displayedSnapshotGroups.length > 0 && (
              <div className="lg:col-span-2">
                <ConfigSnapshotsTray
                  groups={displayedSnapshotGroups}
                  selectedPreset={preset}
                  selectedTrainingSnapshotIds={selectedTrainingSnapshotIds}
                  overrides={dialogOverrides}
                  canManage={capabilities.configSnapshotsEnabled}
                  onLoad={
                    isSnapshotSaveMode
                      ? snapshotEditor.actions.load
                      : loadConfigSnapshot
                  }
                  onRename={
                    isSnapshotSaveMode
                      ? snapshotEditor.actions.rename
                      : renameConfigSnapshot
                  }
                  onRemove={
                    isSnapshotSaveMode
                      ? snapshotEditor.actions.remove
                      : removeConfigSnapshot
                  }
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
              sections={visibleRenderSections}
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
              {visibleRenderSections.length > 0 ? (
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
                      allFields={configFields}
                      showInheritedFields={!isSearchActive}
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
