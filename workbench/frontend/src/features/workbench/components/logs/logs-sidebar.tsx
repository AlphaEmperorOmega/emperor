import { type ReactNode, useMemo } from "react";
import {
  Cpu,
  Database,
  FolderTree,
  Layers,
  LineChart,
  Loader2,
  RefreshCw,
  Tag,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { ErrorPanel } from "@/features/workbench/components/error-panel";
import {
  DeleteExperimentDialog,
  DeleteSubsetRunsDialog,
  type SubsetDeleteKind,
} from "@/features/workbench/components/logs/delete-dialogs";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { StatChip } from "@/features/workbench/components/shared/stat-chip";
import {
  WorkbenchSidebarHeader,
  WorkbenchSidebarSection,
} from "@/features/workbench/components/shared/workbench-sidebar";
import {
  MultiSelectDropdown,
  type MultiSelectDropdownOption,
  type MultiSelectDropdownOptionAction,
} from "@/features/workbench/components/screen/multi-select-dropdown";
import {
  type LogsBrowser,
  type LogsDeletion,
} from "@/features/workbench/providers/logs-workspace-provider";
import { type ChecklistOption } from "@/features/workbench/state/logs/logs-selectors";
import { cn, errorMessage } from "@/lib/utils";

export type LogsSidebarProps = {
  browser: LogsBrowser;
  deletion: LogsDeletion;
};

function runCountLabel(count: number) {
  return `${count} ${count === 1 ? "run" : "runs"}`;
}

function LogFilterSection({
  title,
  icon,
  options,
  selectedValues,
  onToggle,
  onAll,
  onNone,
  optionActions,
  beforeDropdown,
  divided = false,
  optionCountDisplay = "visible",
}: {
  title: string;
  icon: ReactNode;
  options: ChecklistOption[];
  selectedValues: string[];
  onToggle: (value: string) => void;
  onAll: () => void;
  onNone: () => void;
  optionActions?: (
    option: ChecklistOption,
  ) => MultiSelectDropdownOptionAction[] | undefined;
  beforeDropdown?: ReactNode;
  divided?: boolean;
  optionCountDisplay?: "visible" | "hover";
}) {
  const selected = useMemo(() => new Set(selectedValues), [selectedValues]);
  const dropdownOptions = useMemo<MultiSelectDropdownOption[]>(
    () =>
      options.map((option) => ({
        value: option.value,
        label: option.label,
        description: option.detail,
        meta:
          option.count === undefined || optionCountDisplay === "hover" ? undefined : (
            <StatChip size="xs" className="shrink-0">
              {runCountLabel(option.count)}
            </StatChip>
          ),
        metaTooltip:
          option.count === undefined || optionCountDisplay !== "hover"
            ? undefined
            : runCountLabel(option.count),
        wrapLabel: optionCountDisplay === "hover",
        actions: optionActions?.(option),
      })),
    [optionActions, optionCountDisplay, options],
  );

  function changeSelection(nextValues: string[]) {
    const nextSelected = new Set(nextValues);
    const changedOption = options.find(
      (option) => selected.has(option.value) !== nextSelected.has(option.value),
    );
    if (changedOption) {
      onToggle(changedOption.value);
    }
  }

  return (
    <WorkbenchSidebarSection
      title={title}
      icon={icon}
      divider={divided ? "before" : "none"}
      aside={
        <StatChip>
          {selectedValues.length} / {options.length}
        </StatChip>
      }
    >
      {beforeDropdown}
      <MultiSelectDropdown
        label={title}
        values={selectedValues}
        options={dropdownOptions}
        onChange={changeSelection}
        placeholder={`Select ${title.toLowerCase()}`}
        emptyMessage="No options"
        noResultsMessage="No options"
      />
      <div className="grid grid-cols-2 gap-2">
        <Button
          variant="secondary"
          className="h-touch type-compact md:h-control"
          onClick={onAll}
        >
          All
        </Button>
        <Button
          variant="ghost"
          className="h-touch border border-line bg-panel-2/80 type-compact md:h-control"
          onClick={onNone}
        >
          None
        </Button>
      </div>
    </WorkbenchSidebarSection>
  );
}

function SidebarStatus({
  title,
  detail,
  busy,
}: {
  title: string;
  detail: string;
  busy?: boolean;
}) {
  return (
    <div className="rounded-panel border border-line bg-panel-2/70 px-panel py-region text-center shadow-control">
      <div className="grid justify-items-center gap-2">
        {busy && <Loader2 className="h-5 w-5 animate-spin text-violet" aria-hidden />}
        <div className="text-sm font-semibold text-ink">{title}</div>
        <div className="text-xs leading-5 text-ink-faint">{detail}</div>
      </div>
    </div>
  );
}

export function LogsSidebar({ browser, deletion }: LogsSidebarProps) {
  const { actions, filters, pagination, results, scope, status } = browser;
  const experimentOptions = filters.experiments.options;
  const datasetOptions = filters.datasets.options;
  const modelOptions = filters.models.options;
  const presetOptions = filters.presets.options;
  const tagOptions = filters.tags.options;
  const selectedExperimentValues = filters.experiments.selectedValues;
  const {
    enabled: logDeletionEnabled,
    presetTargetExperiment,
    operation: deleteOperation,
    actions: deleteActions,
  } = deletion;
  const toggleExperiment = (value: string) =>
    actions.toggleFilter("experiments", value);
  const toggleDataset = (value: string) => actions.toggleFilter("datasets", value);
  const toggleModel = (value: string) => actions.toggleFilter("models", value);
  const togglePreset = (value: string) => actions.toggleFilter("presets", value);
  const toggleTag = (value: string) => actions.toggleFilter("tags", value);
  const selectAllExperiments = () => actions.selectAll("experiments");
  const selectNoExperiments = () => actions.selectNone("experiments");
  const selectAllDatasets = () => actions.selectAll("datasets");
  const selectNoDatasets = () => actions.selectNone("datasets");
  const selectAllModels = () => actions.selectAll("models");
  const selectNoModels = () => actions.selectNone("models");
  const selectAllPresets = () => actions.selectAll("presets");
  const selectNoPresets = () => actions.selectNone("presets");
  const selectAllTags = () => actions.selectAll("tags");
  const selectNoTags = () => actions.selectNone("tags");
  const isScanning = status.isScanning;
  const isRefreshing = status.isRefreshing;

  async function confirmDeletion() {
    try {
      await deleteActions.confirm();
    } catch {
      // The lifecycle publishes the failure to the active dialog.
    }
  }

  async function retrySubsetPlan() {
    try {
      await deleteActions.retryPlan();
    } catch {
      // The lifecycle publishes the planning failure to the active dialog.
    }
  }

  function subsetDeleteActions(
    kind: SubsetDeleteKind,
    option: ChecklistOption,
  ): MultiSelectDropdownOptionAction[] | undefined {
    if (!logDeletionEnabled || !presetTargetExperiment) {
      return undefined;
    }
    const label = `Delete ${kind} ${option.label} from experiment ${presetTargetExperiment}`;
    return [
      {
        label,
        tooltip: label,
        icon: <Trash2 className="h-4 w-4" aria-hidden />,
        onAction: () => deleteActions.openPreset(option),
      },
    ];
  }

  const presetDeleteActions = presetTargetExperiment && logDeletionEnabled
    ? function renderPresetDeleteAction(option: ChecklistOption) {
        return subsetDeleteActions("preset", option);
      }
    : undefined;

  return (
    <div className="grid min-w-0 content-start gap-region">
      <WorkbenchSidebarHeader
        icon={<LineChart aria-hidden />}
        title="Logs"
        actions={
          <IconButton
            label="Refresh log runs"
            size="sm"
            variant="ghost"
            className="rounded-control active:translate-y-px"
            onClick={() => {
              void actions.refresh();
            }}
            disabled={isRefreshing}
            icon={
              <RefreshCw
                className={cn("h-4 w-4", isRefreshing && "animate-spin")}
                aria-hidden
              />
            }
          />
        }
      />

      {Boolean(status.runsError) && (
        <ErrorPanel title="Log scan failed" message={errorMessage(status.runsError)} />
      )}
      {Boolean(status.experimentsError) && (
        <ErrorPanel
          title="Experiment scan failed"
          message={errorMessage(status.experimentsError)}
        />
      )}
      {!logDeletionEnabled && (
        <InlineStatus tone="warning" compact>
          Log deletion is disabled by this backend.
        </InlineStatus>
      )}

      {isScanning ? (
        <SidebarStatus
          title="Scanning logs…"
          detail="Reading historical version folders from logs/."
          busy
        />
      ) : !results.hasExperiments ? (
        <SidebarStatus
          title="No log runs"
          detail="No version_* folders were found under logs/."
        />
      ) : (
        <>
          <LogFilterSection
            title="Experiments"
            icon={<FolderTree className="h-4 w-4" aria-hidden />}
            options={experimentOptions}
            selectedValues={selectedExperimentValues}
            onToggle={toggleExperiment}
            onAll={selectAllExperiments}
            onNone={selectNoExperiments}
            beforeDropdown={
              <div
                className="grid grid-cols-2 gap-2 rounded-panel border border-line-soft bg-panel-2/70 p-2"
                role="group"
                aria-label="Log scope"
              >
                <Button
                  type="button"
                  variant={scope.mode === "target" ? "secondary" : "ghost"}
                  className="h-touch type-compact md:h-control"
                  aria-pressed={scope.mode === "target"}
                  onClick={scope.useCurrentTarget}
                  disabled={!scope.canUseCurrentTarget}
                >
                  Current Target
                </Button>
                <Button
                  type="button"
                  variant={scope.allRunsSelected ? "secondary" : "ghost"}
                  className="h-touch type-compact md:h-control"
                  aria-pressed={scope.allRunsSelected}
                  onClick={scope.showAllRuns}
                >
                  All Runs
                </Button>
              </div>
            }
            optionCountDisplay="hover"
            optionActions={
              logDeletionEnabled
                ? (option) => {
                    const label = `Delete experiment ${option.label}`;
                    return [
                      {
                        label,
                        tooltip: label,
                        icon: <Trash2 className="h-4 w-4" aria-hidden />,
                        onAction: () => deleteActions.openExperiment(option),
                      },
                    ];
                  }
                : undefined
            }
          />
          {!results.hasRuns ? (
            <SidebarStatus
              title="No runs yet"
              detail="Experiment folders without version_* runs stay selectable."
            />
          ) : (
            <>
              {pagination.runs.canLoadMore && (
                <div className="grid gap-2 rounded-panel border border-line-soft bg-panel-2/70 p-panel">
                  <div className="text-xs text-ink-faint">
                    Loaded {pagination.runs.loaded} of {pagination.runs.total} matching runs
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    className="h-touch border border-line bg-panel-2/80 type-compact md:h-control"
                    onClick={actions.loadMoreRuns}
                    disabled={pagination.runs.isLoadingMore}
                  >
                    {pagination.runs.isLoadingMore && (
                      <Loader2
                        className="mr-2 h-3.5 w-3.5 animate-spin"
                        aria-hidden
                      />
                    )}
                Load More Runs
                  </Button>
                </div>
              )}
              <LogFilterSection
                title="Datasets"
                icon={<Database className="h-4 w-4" aria-hidden />}
                options={datasetOptions}
                selectedValues={filters.datasets.selectedValues}
                onToggle={toggleDataset}
                onAll={selectAllDatasets}
                onNone={selectNoDatasets}
                divided
              />
              <LogFilterSection
                title="Models"
                icon={<Cpu className="h-4 w-4" aria-hidden />}
                options={modelOptions}
                selectedValues={filters.models.selectedValues}
                onToggle={toggleModel}
                onAll={selectAllModels}
                onNone={selectNoModels}
                divided
              />
              <LogFilterSection
                title="Presets"
                icon={<Layers className="h-4 w-4" aria-hidden />}
                options={presetOptions}
                selectedValues={filters.presets.selectedValues}
                onToggle={togglePreset}
                onAll={selectAllPresets}
                onNone={selectNoPresets}
                optionActions={presetDeleteActions}
                divided
              />
              {status.tagsError && (
                <ErrorPanel title="Tag read failed" message={errorMessage(status.tagsError)} />
              )}
              <LogFilterSection
                title="Scalar Tags"
                icon={<Tag className="h-4 w-4" aria-hidden />}
                options={tagOptions}
                selectedValues={filters.tags.selectedValues}
                onToggle={toggleTag}
                onAll={selectAllTags}
                onNone={selectNoTags}
                beforeDropdown={
                  pagination.scalarTags.canLoadMore ? (
                    <div className="grid gap-2 rounded-panel border border-line-soft bg-panel-2/70 p-panel">
                      <div className="text-xs text-ink-faint">
                        Scalar tags scanned for {pagination.scalarTags.loadedRuns} of{" "}
                        {pagination.scalarTags.totalRuns} visible runs
                      </div>
                      <Button
                        type="button"
                        variant="ghost"
                        className="h-touch border border-line bg-panel-2/80 type-compact md:h-control"
                        onClick={actions.loadMoreScalarTags}
                        disabled={pagination.scalarTags.isLoadingMore}
                      >
                        {pagination.scalarTags.isLoadingMore && (
                          <Loader2
                            className="mr-2 h-3.5 w-3.5 animate-spin"
                            aria-hidden
                          />
                        )}
                Load More Scalar Tags
                      </Button>
                    </div>
                  ) : undefined
                }
                divided
              />
            </>
          )}
        </>
      )}

      {deleteOperation?.kind === "experiment" && (
        <DeleteExperimentDialog
          option={deleteOperation.option}
          error={deleteOperation.error}
          isDeleting={deleteOperation.phase === "mutating"}
          onClose={deleteActions.cancel}
          onConfirm={confirmDeletion}
        />
      )}
      {deleteOperation?.kind === "preset" && (
        <DeleteSubsetRunsDialog
          key={deleteOperation.target.key}
          target={deleteOperation.target}
          plan={deleteOperation.plan}
          error={deleteOperation.error}
          isPlanning={deleteOperation.phase === "planning"}
          isDeleting={deleteOperation.phase === "mutating"}
          onClose={deleteActions.cancel}
          onRetry={retrySubsetPlan}
          onConfirm={confirmDeletion}
        />
      )}
    </div>
  );
}
