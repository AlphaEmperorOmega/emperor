import { type ReactNode, useEffect, useMemo, useState } from "react";
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
import { ErrorPanel } from "@/features/viewer/components/error-panel";
import {
  DeleteExperimentDialog,
  DeleteSubsetRunsDialog,
  type SubsetDeleteKind,
  type SubsetDeleteTarget,
} from "@/features/viewer/components/logs/delete-dialogs";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { SectionHeading } from "@/features/viewer/components/shared/section-heading";
import { StatChip } from "@/features/viewer/components/shared/stat-chip";
import {
  MultiSelectDropdown,
  type MultiSelectDropdownOption,
  type MultiSelectDropdownOptionAction,
} from "@/features/viewer/components/screen/multi-select-dropdown";
import { type LogsWorkspaceState } from "@/features/viewer/state/logs/use-logs-workspace-state";
import { type LogRun, type ModelIdentity } from "@/lib/api";
import {
  type ChecklistOption,
  selectedOptionsSet,
} from "@/features/viewer/state/logs/logs-selectors";
import { cn, errorMessage } from "@/lib/utils";

export type LogsSidebarProps = {
  runs: LogRun[];
  runsQuery: LogsWorkspaceState["runsQuery"];
  experimentsQuery: LogsWorkspaceState["experimentsQuery"];
  tagsQuery: LogsWorkspaceState["tagsQuery"];
  logDeletionEnabled: boolean;
  experimentOptions: ChecklistOption[];
  datasetOptions: ChecklistOption[];
  modelOptions: ChecklistOption[];
  presetOptions: ChecklistOption[];
  tagOptions: ChecklistOption[];
  selectedExperiments: Set<string>;
  selectedDatasets: Set<string>;
  selectedModels: Set<string>;
  selectedPresets: Set<string>;
  selectedTags: Set<string>;
  toggleExperiment: LogsWorkspaceState["toggleExperiment"];
  toggleDataset: LogsWorkspaceState["toggleDataset"];
  toggleModel: LogsWorkspaceState["toggleModel"];
  togglePreset: LogsWorkspaceState["togglePreset"];
  toggleTag: LogsWorkspaceState["toggleTag"];
  selectAllExperiments: LogsWorkspaceState["selectAllExperiments"];
  selectNoExperiments: LogsWorkspaceState["selectNoExperiments"];
  selectAllDatasets: LogsWorkspaceState["selectAllDatasets"];
  selectNoDatasets: LogsWorkspaceState["selectNoDatasets"];
  selectAllModels: LogsWorkspaceState["selectAllModels"];
  selectNoModels: LogsWorkspaceState["selectNoModels"];
  selectAllPresets: LogsWorkspaceState["selectAllPresets"];
  selectNoPresets: LogsWorkspaceState["selectNoPresets"];
  selectAllTags: LogsWorkspaceState["selectAllTags"];
  selectNoTags: LogsWorkspaceState["selectNoTags"];
  refreshLogLists: LogsWorkspaceState["refreshLogLists"];
  resetDeleteExperiment: LogsWorkspaceState["resetDeleteExperiment"];
  deleteExperiment: LogsWorkspaceState["deleteExperiment"];
  deleteExperimentError: LogsWorkspaceState["deleteExperimentError"];
  isDeletingExperiment: LogsWorkspaceState["isDeletingExperiment"];
  resetRunDelete: LogsWorkspaceState["resetRunDelete"];
  createRunDeletePlan: LogsWorkspaceState["createRunDeletePlan"];
  runDeletePlan: LogsWorkspaceState["runDeletePlan"];
  runDeletePlanError: LogsWorkspaceState["runDeletePlanError"];
  isPlanningRunDelete: LogsWorkspaceState["isPlanningRunDelete"];
  deleteRuns: LogsWorkspaceState["deleteRuns"];
  runDeleteError: LogsWorkspaceState["runDeleteError"];
  isDeletingRunDelete: LogsWorkspaceState["isDeletingRunDelete"];
  loadedScalarTagRunCount: LogsWorkspaceState["loadedScalarTagRunCount"];
  totalScalarTagRunCount: LogsWorkspaceState["totalScalarTagRunCount"];
  canLoadMoreScalarTags: LogsWorkspaceState["canLoadMoreScalarTags"];
  isLoadingMoreScalarTags: LogsWorkspaceState["isLoadingMoreScalarTags"];
  loadMoreScalarTags: LogsWorkspaceState["loadMoreScalarTags"];
};

function uniqueSorted(values: string[]) {
  return Array.from(new Set(values)).sort((a, b) => a.localeCompare(b));
}

function uniqueModelIdentities(runs: LogRun[]): ModelIdentity[] {
  const models = new Map<string, ModelIdentity>();
  for (const run of runs) {
    const key = `${run.modelType}/${run.model}`;
    if (!models.has(key)) {
      models.set(key, { modelType: run.modelType, model: run.model });
    }
  }
  return Array.from(models.values()).sort((left, right) =>
    `${left.modelType}/${left.model}`.localeCompare(`${right.modelType}/${right.model}`),
  );
}

function buildSubsetDeleteTarget({
  kind,
  value,
  experiment,
  runs,
}: {
  kind: SubsetDeleteKind;
  value: string;
  experiment: string;
  runs: LogRun[];
}): SubsetDeleteTarget | null {
  const targetRuns = runs.filter((run) => {
    if (run.experiment !== experiment) {
      return false;
    }
    return run.preset === value;
  });

  if (targetRuns.length === 0) {
    return null;
  }

  const filters = {
    experiments: uniqueSorted(targetRuns.map((run) => run.experiment)),
    datasets: uniqueSorted(targetRuns.map((run) => run.dataset)),
    models: uniqueModelIdentities(targetRuns),
    presets: uniqueSorted(targetRuns.map((run) => run.preset)),
    runIds: uniqueSorted(targetRuns.map((run) => run.id)),
  };

  return {
    kind,
    value,
    experiment,
    filters,
    key: JSON.stringify({ kind, value, experiment, filters }),
  };
}

function selectedValuesForOptions(selected: Set<string>, options: ChecklistOption[]) {
  return options
    .filter((option) => selected.has(option.value))
    .map((option) => option.value);
}

function runCountLabel(count: number) {
  return `${count} ${count === 1 ? "run" : "runs"}`;
}

function LogFilterSection({
  title,
  icon,
  options,
  selected,
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
  selected: Set<string>;
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
  const selectedValues = useMemo(
    () => selectedValuesForOptions(selected, options),
    [options, selected],
  );
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
    <section className={cn("grid gap-2", divided && "border-t border-line-soft pt-3")}>
      <div className="flex min-h-[28px] flex-wrap items-center justify-between gap-2">
        <SectionHeading icon={icon} title={title} />
        <StatChip>
          {selectedValues.length} / {options.length}
        </StatChip>
      </div>
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
        <Button variant="secondary" className="h-8 text-xs" onClick={onAll}>
          All
        </Button>
        <Button
          variant="ghost"
          className="h-8 border border-line bg-white/[0.025] text-xs"
          onClick={onNone}
        >
          None
        </Button>
      </div>
    </section>
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
    <div className="rounded-[13px] border border-line-soft bg-white/[0.018] px-3 py-4 text-center">
      <div className="grid justify-items-center gap-2">
        {busy && <Loader2 className="h-5 w-5 animate-spin text-violet" aria-hidden />}
        <div className="text-sm font-semibold text-ink">{title}</div>
        <div className="text-xs leading-5 text-ink-faint">{detail}</div>
      </div>
    </div>
  );
}

export function LogsSidebar({
  runs,
  runsQuery,
  experimentsQuery,
  tagsQuery,
  logDeletionEnabled,
  experimentOptions,
  datasetOptions,
  modelOptions,
  presetOptions,
  tagOptions,
  selectedExperiments,
  selectedDatasets,
  selectedModels,
  selectedPresets,
  selectedTags,
  toggleExperiment,
  toggleDataset,
  toggleModel,
  togglePreset,
  toggleTag,
  selectAllExperiments,
  selectNoExperiments,
  selectAllDatasets,
  selectNoDatasets,
  selectAllModels,
  selectNoModels,
  selectAllPresets,
  selectNoPresets,
  selectAllTags,
  selectNoTags,
  refreshLogLists,
  resetDeleteExperiment,
  deleteExperiment,
  deleteExperimentError,
  isDeletingExperiment,
  resetRunDelete,
  createRunDeletePlan,
  runDeletePlan,
  runDeletePlanError,
  isPlanningRunDelete,
  deleteRuns,
  runDeleteError,
  isDeletingRunDelete,
  loadedScalarTagRunCount,
  totalScalarTagRunCount,
  canLoadMoreScalarTags,
  isLoadingMoreScalarTags,
  loadMoreScalarTags,
}: LogsSidebarProps) {
  const [deleteOption, setDeleteOption] = useState<ChecklistOption | null>(null);
  const [subsetDeleteTarget, setSubsetDeleteTarget] =
    useState<SubsetDeleteTarget | null>(null);
  const isScanning = runsQuery.isLoading || experimentsQuery.isLoading;
  const isRefreshing = runsQuery.isFetching || experimentsQuery.isFetching;
  const selectedExperimentOptions = useMemo(
    () => selectedOptionsSet(selectedExperiments, experimentOptions),
    [experimentOptions, selectedExperiments],
  );
  const singleSelectedExperiment =
    selectedExperimentOptions.size === 1
      ? Array.from(selectedExperimentOptions)[0]
      : null;

  useEffect(() => {
    if (!subsetDeleteTarget || singleSelectedExperiment === subsetDeleteTarget.experiment) {
      return;
    }
    resetRunDelete();
    setSubsetDeleteTarget(null);
  }, [resetRunDelete, singleSelectedExperiment, subsetDeleteTarget]);

  useEffect(() => {
    if (logDeletionEnabled || (!deleteOption && !subsetDeleteTarget)) {
      return;
    }
    resetDeleteExperiment();
    resetRunDelete();
    setDeleteOption(null);
    setSubsetDeleteTarget(null);
  }, [
    deleteOption,
    logDeletionEnabled,
    resetDeleteExperiment,
    resetRunDelete,
    subsetDeleteTarget,
  ]);

  function openDeleteDialog(option: ChecklistOption) {
    if (!logDeletionEnabled) {
      return;
    }
    resetDeleteExperiment();
    setDeleteOption(option);
  }

  function openSubsetDeleteDialog(kind: SubsetDeleteKind, option: ChecklistOption) {
    if (!logDeletionEnabled || !singleSelectedExperiment) {
      return;
    }
    const target = buildSubsetDeleteTarget({
      kind,
      value: option.value,
      experiment: singleSelectedExperiment,
      runs,
    });
    if (!target) {
      return;
    }
    resetRunDelete();
    setSubsetDeleteTarget(target);
    void createRunDeletePlan(target.filters);
  }

  function closeSubsetDeleteDialog() {
    resetRunDelete();
    setSubsetDeleteTarget(null);
  }

  function closeDeleteDialog() {
    resetDeleteExperiment();
    setDeleteOption(null);
  }

  async function confirmDeleteExperiment() {
    if (!logDeletionEnabled || !deleteOption) {
      return;
    }
    try {
      await deleteExperiment(deleteOption.value);
      setDeleteOption(null);
    } catch {
      // The mutation error is shown in the dialog.
    }
  }

  async function confirmDeleteSubsetRuns() {
    if (!logDeletionEnabled || !subsetDeleteTarget) {
      return;
    }
    try {
      await deleteRuns(subsetDeleteTarget.filters);
      setSubsetDeleteTarget(null);
    } catch {
      // The mutation error is shown in the dialog.
    }
  }

  function subsetDeleteActions(
    kind: SubsetDeleteKind,
    option: ChecklistOption,
  ): MultiSelectDropdownOptionAction[] | undefined {
    if (!logDeletionEnabled || !singleSelectedExperiment) {
      return undefined;
    }
    const label = `Delete ${kind} ${option.label} from experiment ${singleSelectedExperiment}`;
    return [
      {
        label,
        tooltip: label,
        icon: <Trash2 className="h-4 w-4" aria-hidden />,
        onAction: () => openSubsetDeleteDialog(kind, option),
      },
    ];
  }

  const presetDeleteActions = singleSelectedExperiment && logDeletionEnabled
    ? function renderPresetDeleteAction(option: ChecklistOption) {
        return subsetDeleteActions("preset", option);
      }
    : undefined;

  return (
    <div className="grid gap-3">
      <section className="grid gap-2">
        <div className="flex items-center justify-between gap-3">
          <SectionHeading
            icon={<LineChart className="h-[15px] w-[15px] text-violet" aria-hidden />}
            title="Logs"
          />
          <IconButton
            label="Refresh log runs"
            size="sm"
            variant="ghost"
            className="rounded-[10px] active:translate-y-px"
            onClick={() => {
              void refreshLogLists();
            }}
            disabled={isRefreshing}
            icon={
              <RefreshCw
                className={cn("h-4 w-4", isRefreshing && "animate-spin")}
                aria-hidden
              />
            }
          />
        </div>
      </section>

      {runsQuery.isError && (
        <ErrorPanel title="Log scan failed" message={errorMessage(runsQuery.error)} />
      )}
      {experimentsQuery.isError && (
        <ErrorPanel
          title="Experiment scan failed"
          message={errorMessage(experimentsQuery.error)}
        />
      )}
      {!logDeletionEnabled && (
        <InlineStatus tone="warning" compact>
          Log deletion is disabled by this backend.
        </InlineStatus>
      )}

      {isScanning ? (
        <SidebarStatus
          title="Scanning logs"
          detail="Reading historical version folders from logs/."
          busy
        />
      ) : experimentOptions.length === 0 ? (
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
            selected={selectedExperimentOptions}
            onToggle={toggleExperiment}
            onAll={selectAllExperiments}
            onNone={selectNoExperiments}
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
                        onAction: () => openDeleteDialog(option),
                      },
                    ];
                  }
                : undefined
            }
          />
          {runs.length === 0 ? (
            <SidebarStatus
              title="No runs yet"
              detail="Experiment folders without version_* runs stay selectable."
            />
          ) : (
            <>
              <LogFilterSection
                title="Datasets"
                icon={<Database className="h-4 w-4" aria-hidden />}
                options={datasetOptions}
                selected={selectedOptionsSet(selectedDatasets, datasetOptions)}
                onToggle={toggleDataset}
                onAll={selectAllDatasets}
                onNone={selectNoDatasets}
                divided
              />
              <LogFilterSection
                title="Models"
                icon={<Cpu className="h-4 w-4" aria-hidden />}
                options={modelOptions}
                selected={selectedOptionsSet(selectedModels, modelOptions)}
                onToggle={toggleModel}
                onAll={selectAllModels}
                onNone={selectNoModels}
                divided
              />
              <LogFilterSection
                title="Presets"
                icon={<Layers className="h-4 w-4" aria-hidden />}
                options={presetOptions}
                selected={selectedOptionsSet(selectedPresets, presetOptions)}
                onToggle={togglePreset}
                onAll={selectAllPresets}
                onNone={selectNoPresets}
                optionActions={presetDeleteActions}
                divided
              />
              {tagsQuery.isError && (
                <ErrorPanel title="Tag read failed" message={errorMessage(tagsQuery.error)} />
              )}
              <LogFilterSection
                title="Scalar Tags"
                icon={<Tag className="h-4 w-4" aria-hidden />}
                options={tagOptions}
                selected={selectedOptionsSet(selectedTags, tagOptions)}
                onToggle={toggleTag}
                onAll={selectAllTags}
                onNone={selectNoTags}
                beforeDropdown={
                  canLoadMoreScalarTags ? (
                    <div className="grid gap-2 rounded-[12px] border border-line-soft bg-white/[0.018] p-3">
                      <div className="text-xs text-ink-faint">
                        Scalar tags scanned for {loadedScalarTagRunCount} of{" "}
                        {totalScalarTagRunCount} visible runs
                      </div>
                      <Button
                        type="button"
                        variant="ghost"
                        className="h-8 border border-line bg-white/[0.025] text-xs"
                        onClick={loadMoreScalarTags}
                        disabled={isLoadingMoreScalarTags}
                      >
                        {isLoadingMoreScalarTags && (
                          <Loader2
                            className="mr-2 h-3.5 w-3.5 animate-spin"
                            aria-hidden
                          />
                        )}
                        Load more scalar tags
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

      {deleteOption && (
        <DeleteExperimentDialog
          option={deleteOption}
          error={deleteExperimentError}
          isDeleting={isDeletingExperiment}
          onClose={closeDeleteDialog}
          onConfirm={confirmDeleteExperiment}
        />
      )}
      {subsetDeleteTarget && (
        <DeleteSubsetRunsDialog
          key={subsetDeleteTarget.key}
          target={subsetDeleteTarget}
          plan={runDeletePlan}
          error={runDeletePlanError ?? runDeleteError}
          isPlanning={isPlanningRunDelete}
          isDeleting={isDeletingRunDelete}
          onClose={closeSubsetDeleteDialog}
          onConfirm={confirmDeleteSubsetRuns}
        />
      )}
    </div>
  );
}
