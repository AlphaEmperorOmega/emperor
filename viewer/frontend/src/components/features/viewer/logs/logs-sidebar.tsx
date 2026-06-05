import { type ReactNode, useEffect, useMemo, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Cpu,
  Database,
  FileText,
  FolderTree,
  Layers,
  LineChart,
  Loader2,
  RefreshCw,
  Search,
  Tag,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { IconButton } from "@/components/ui/icon-button";
import { ErrorPanel } from "@/components/features/viewer/error-panel";
import {
  DeleteExperimentDialog,
  DeleteSubsetRunsDialog,
  type SubsetDeleteKind,
  type SubsetDeleteTarget,
} from "@/components/features/viewer/logs/delete-dialogs";
import { InlineStatus } from "@/components/features/viewer/shared/inline-status";
import { MetricCard } from "@/components/features/viewer/shared/metric-card";
import { SectionHeading } from "@/components/features/viewer/shared/section-heading";
import { StatChip } from "@/components/features/viewer/shared/stat-chip";
import { type LogsWorkspaceState } from "@/components/features/viewer/state/use-logs-workspace-state";
import { type LogRun } from "@/lib/api";
import { type ChecklistOption, selectedOptionsSet } from "@/lib/logs/helpers";
import { cn, errorMessage } from "@/lib/utils";

export type LogsSidebarProps = {
  runs: LogRun[];
  visibleRuns: LogRun[];
  runsQuery: LogsWorkspaceState["runsQuery"];
  experimentsQuery: LogsWorkspaceState["experimentsQuery"];
  tagsQuery: LogsWorkspaceState["tagsQuery"];
  experimentOptions: ChecklistOption[];
  datasetOptions: ChecklistOption[];
  modelOptions: ChecklistOption[];
  presetOptions: ChecklistOption[];
  runOptions: ChecklistOption[];
  tagOptions: ChecklistOption[];
  selectedExperiments: Set<string>;
  selectedDatasets: Set<string>;
  selectedModels: Set<string>;
  selectedPresets: Set<string>;
  selectedRunIds: Set<string>;
  selectedTags: Set<string>;
  toggleExperiment: LogsWorkspaceState["toggleExperiment"];
  toggleDataset: LogsWorkspaceState["toggleDataset"];
  toggleModel: LogsWorkspaceState["toggleModel"];
  togglePreset: LogsWorkspaceState["togglePreset"];
  toggleRun: LogsWorkspaceState["toggleRun"];
  toggleTag: LogsWorkspaceState["toggleTag"];
  selectAllExperiments: LogsWorkspaceState["selectAllExperiments"];
  selectNoExperiments: LogsWorkspaceState["selectNoExperiments"];
  selectAllDatasets: LogsWorkspaceState["selectAllDatasets"];
  selectNoDatasets: LogsWorkspaceState["selectNoDatasets"];
  selectAllModels: LogsWorkspaceState["selectAllModels"];
  selectNoModels: LogsWorkspaceState["selectNoModels"];
  selectAllPresets: LogsWorkspaceState["selectAllPresets"];
  selectNoPresets: LogsWorkspaceState["selectNoPresets"];
  selectAllRuns: LogsWorkspaceState["selectAllRuns"];
  selectNoRuns: LogsWorkspaceState["selectNoRuns"];
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
};

function uniqueSorted(values: string[]) {
  return Array.from(new Set(values)).sort((a, b) => a.localeCompare(b));
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
    return kind === "dataset" ? run.dataset === value : run.preset === value;
  });

  if (targetRuns.length === 0) {
    return null;
  }

  const filters = {
    experiments: uniqueSorted(targetRuns.map((run) => run.experiment)),
    datasets: uniqueSorted(targetRuns.map((run) => run.dataset)),
    models: uniqueSorted(targetRuns.map((run) => run.model)),
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

function ChecklistSection({
  title,
  icon,
  options,
  selected,
  onToggle,
  onAll,
  onNone,
  search,
  defaultOpen = true,
  renderOptionAction,
}: {
  title: string;
  icon: ReactNode;
  options: ChecklistOption[];
  selected: Set<string>;
  onToggle: (value: string) => void;
  onAll: () => void;
  onNone: () => void;
  search?: boolean;
  defaultOpen?: boolean;
  renderOptionAction?: (option: ChecklistOption) => ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const [query, setQuery] = useState("");
  const filteredOptions = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) {
      return options;
    }
    return options.filter(
      (option) =>
        option.label.toLowerCase().includes(needle) ||
        option.detail?.toLowerCase().includes(needle),
    );
  }, [options, query]);

  return (
    <section className="rounded-[13px] border border-line-soft bg-white/[0.018]">
      <button
        type="button"
        className="grid min-h-[44px] w-full grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 px-3 text-left transition hover:bg-white/[0.03] focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        aria-expanded={isOpen}
        onClick={() => setIsOpen((open) => !open)}
      >
        <span className="grid h-7 w-7 place-items-center rounded-[8px] border border-line bg-white/[0.035] text-violet">
          {icon}
        </span>
        <span className="min-w-0">
          <span className="block truncate text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
            {title}
          </span>
          <span className="mt-0.5 block font-mono text-[11px] text-ink-faint">
            {selected.size} / {options.length}
          </span>
        </span>
        {isOpen ? (
          <ChevronDown className="h-4 w-4 text-ink-faint" aria-hidden />
        ) : (
          <ChevronRight className="h-4 w-4 text-ink-faint" aria-hidden />
        )}
      </button>

      {isOpen && (
        <div className="grid gap-2 border-t border-line-soft p-2.5">
          {search && (
            <label className="relative block">
              <Search
                className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint"
                aria-hidden
              />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search tags"
                aria-label={`Search ${title.toLowerCase()}`}
                className="h-9 w-full rounded-[10px] border border-line bg-black/25 pl-8 pr-2.5 font-mono text-xs text-ink outline-none transition placeholder:text-ink-faint focus-visible:border-violet/60 focus-visible:ring-2 focus-visible:ring-focus"
              />
            </label>
          )}

          <div className="grid max-h-64 auto-rows-max content-start gap-1.5 overflow-y-auto pr-1">
            {filteredOptions.length === 0 ? (
              <InlineStatus className="border-line-soft bg-transparent p-0 px-3 py-4 text-center text-xs">
                No options
              </InlineStatus>
            ) : (
              filteredOptions.map((option) => {
                const checked = selected.has(option.value);
                return (
                  <div
                    key={option.value}
                    className={cn(
                      "grid min-h-[44px] grid-cols-[minmax(0,1fr)_auto] items-center overflow-hidden rounded-[10px] border transition",
                      checked
                        ? "border-violet/40 bg-[linear-gradient(135deg,rgba(146,113,255,0.1),rgba(111,168,255,0.05))]"
                        : "border-line-soft bg-white/[0.012] hover:border-line hover:bg-white/[0.03]",
                    )}
                  >
                    <label className="grid min-h-[44px] min-w-0 cursor-pointer grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 px-2.5 py-2">
                      <Checkbox
                        checked={checked}
                        onCheckedChange={() => onToggle(option.value)}
                        aria-label={`${title} ${option.label}`}
                      />
                      <span className="min-w-0">
                        <span className="block truncate text-[13px] font-semibold text-ink">
                          {option.label}
                        </span>
                        {option.detail && (
                          <span className="mt-0.5 block truncate font-mono text-[11px] text-ink-faint">
                            {option.detail}
                          </span>
                        )}
                      </span>
                      {option.count !== undefined && (
                        <StatChip size="xs" className="shrink-0">
                          {option.count} runs
                        </StatChip>
                      )}
                    </label>
                    {renderOptionAction && (
                      <div className="pr-1.5">{renderOptionAction(option)}</div>
                    )}
                  </div>
                );
              })
            )}
          </div>

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
        </div>
      )}
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
  visibleRuns,
  runsQuery,
  experimentsQuery,
  tagsQuery,
  experimentOptions,
  datasetOptions,
  modelOptions,
  presetOptions,
  runOptions,
  tagOptions,
  selectedExperiments,
  selectedDatasets,
  selectedModels,
  selectedPresets,
  selectedRunIds,
  selectedTags,
  toggleExperiment,
  toggleDataset,
  toggleModel,
  togglePreset,
  toggleRun,
  toggleTag,
  selectAllExperiments,
  selectNoExperiments,
  selectAllDatasets,
  selectNoDatasets,
  selectAllModels,
  selectNoModels,
  selectAllPresets,
  selectNoPresets,
  selectAllRuns,
  selectNoRuns,
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

  function openDeleteDialog(option: ChecklistOption) {
    resetDeleteExperiment();
    setDeleteOption(option);
  }

  function openSubsetDeleteDialog(kind: SubsetDeleteKind, option: ChecklistOption) {
    if (!singleSelectedExperiment) {
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
    if (!deleteOption) {
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
    if (!subsetDeleteTarget) {
      return;
    }
    try {
      await deleteRuns(subsetDeleteTarget.filters);
      setSubsetDeleteTarget(null);
    } catch {
      // The mutation error is shown in the dialog.
    }
  }

  function renderSubsetDeleteAction(kind: SubsetDeleteKind, option: ChecklistOption) {
    if (!singleSelectedExperiment) {
      return null;
    }
    return (
      <IconButton
        label={`Delete ${kind} ${option.label} from experiment ${singleSelectedExperiment}`}
        size="sm"
        variant="danger"
        className="rounded-[10px] active:translate-y-px"
        onClick={() => openSubsetDeleteDialog(kind, option)}
        icon={<Trash2 className="h-4 w-4" aria-hidden />}
      />
    );
  }

  const renderDatasetDeleteAction = singleSelectedExperiment
    ? function renderDatasetDeleteAction(option: ChecklistOption) {
        return renderSubsetDeleteAction("dataset", option);
      }
    : undefined;
  const renderPresetDeleteAction = singleSelectedExperiment
    ? function renderPresetDeleteAction(option: ChecklistOption) {
        return renderSubsetDeleteAction("preset", option);
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
        <div className="grid grid-cols-2 gap-2">
          <MetricCard
            label="Runs"
            value={visibleRuns.length}
            className="py-2.5"
          />
          <MetricCard
            label="Tags"
            value={selectedOptionsSet(selectedTags, tagOptions).size}
            className="py-2.5"
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
          <ChecklistSection
            title="Experiments"
            icon={<FolderTree className="h-4 w-4" aria-hidden />}
            options={experimentOptions}
            selected={selectedExperimentOptions}
            onToggle={toggleExperiment}
            onAll={selectAllExperiments}
            onNone={selectNoExperiments}
            renderOptionAction={(option) => (
              <IconButton
                label={`Delete experiment ${option.label}`}
                size="sm"
                variant="danger"
                className="rounded-[10px] active:translate-y-px"
                onClick={() => openDeleteDialog(option)}
                icon={<Trash2 className="h-4 w-4" aria-hidden />}
              />
            )}
          />
          {runs.length === 0 ? (
            <SidebarStatus
              title="No runs yet"
              detail="Experiment folders without version_* runs stay selectable."
            />
          ) : (
            <>
              <ChecklistSection
                title="Datasets"
                icon={<Database className="h-4 w-4" aria-hidden />}
                options={datasetOptions}
                selected={selectedOptionsSet(selectedDatasets, datasetOptions)}
                onToggle={toggleDataset}
                onAll={selectAllDatasets}
                onNone={selectNoDatasets}
                renderOptionAction={renderDatasetDeleteAction}
              />
              <ChecklistSection
                title="Models"
                icon={<Cpu className="h-4 w-4" aria-hidden />}
                options={modelOptions}
                selected={selectedOptionsSet(selectedModels, modelOptions)}
                onToggle={toggleModel}
                onAll={selectAllModels}
                onNone={selectNoModels}
              />
              <ChecklistSection
                title="Presets"
                icon={<Layers className="h-4 w-4" aria-hidden />}
                options={presetOptions}
                selected={selectedOptionsSet(selectedPresets, presetOptions)}
                onToggle={togglePreset}
                onAll={selectAllPresets}
                onNone={selectNoPresets}
                defaultOpen={false}
                renderOptionAction={renderPresetDeleteAction}
              />
              <ChecklistSection
                title="Runs"
                icon={<FileText className="h-4 w-4" aria-hidden />}
                options={runOptions}
                selected={selectedOptionsSet(selectedRunIds, runOptions)}
                onToggle={toggleRun}
                onAll={selectAllRuns}
                onNone={selectNoRuns}
                defaultOpen={false}
              />
              {tagsQuery.isError && (
                <ErrorPanel title="Tag read failed" message={errorMessage(tagsQuery.error)} />
              )}
              <ChecklistSection
                title="Scalar Tags"
                icon={<Tag className="h-4 w-4" aria-hidden />}
                options={tagOptions}
                selected={selectedOptionsSet(selectedTags, tagOptions)}
                onToggle={toggleTag}
                onAll={selectAllTags}
                onNone={selectNoTags}
                search
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
