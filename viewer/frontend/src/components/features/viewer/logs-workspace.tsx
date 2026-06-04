import { type ReactNode, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Columns2,
  Columns3,
  Cpu,
  Database,
  FileText,
  FolderTree,
  Layers,
  LineChart,
  Loader2,
  RefreshCw,
  RectangleHorizontal,
  Search,
  Tag,
  Trash2,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { EdgeCard } from "@/components/ui/edge-card";
import { Input } from "@/components/ui/input";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { ErrorPanel } from "@/components/features/viewer/error-panel";
import { ViewModeButton } from "@/components/features/viewer/view-mode-button";
import {
  type LogRun,
  type LogRunDeleteFilters,
  type LogRunDeletePlan,
  type LogScalarSeries,
} from "@/lib/api";
import { useLogScalarsQuery } from "@/hooks/use-log-queries";
import { cn, errorMessage } from "@/lib/utils";
import {
  type ChecklistOption,
  formatMetricValue,
  formatNumber,
  formatRunLabel,
  selectedOptionsSet,
} from "@/lib/logs/helpers";
import { type LogsWorkspaceState } from "@/components/features/viewer/state/use-logs-workspace-state";

const SERIES_COLORS = [
  "#7c6dff",
  "#22d3ee",
  "#f59e0b",
  "#34d399",
  "#f472b6",
  "#a78bfa",
  "#fb7185",
  "#60a5fa",
  "#facc15",
  "#2dd4bf",
];

type ScalarChartGridMode = "full" | "two" | "three";

const SCALAR_CHART_GRID_CLASSES: Record<ScalarChartGridMode, string> = {
  full: "grid gap-4",
  two: "grid gap-4 xl:grid-cols-2",
  three: "grid gap-4 xl:grid-cols-2 2xl:grid-cols-3",
};

type SubsetDeleteKind = "dataset" | "preset";

type SubsetDeleteTarget = {
  kind: SubsetDeleteKind;
  value: string;
  experiment: string;
  filters: LogRunDeleteFilters;
  key: string;
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
              <div className="rounded-[10px] border border-dashed border-line-soft px-3 py-4 text-center text-xs text-ink-faint">
                No options
              </div>
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
                        <span className="shrink-0 rounded-[7px] border border-line bg-white/[0.04] px-1.5 py-0.5 font-mono text-[11px] text-ink-dim">
                          {option.count} runs
                        </span>
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

function DeleteExperimentDialog({
  option,
  error,
  isDeleting,
  onClose,
  onConfirm,
}: {
  option: ChecklistOption;
  error: unknown;
  isDeleting: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
}) {
  const [confirmation, setConfirmation] = useState("");
  const canDelete = confirmation === option.value && !isDeleting;
  const runCount = option.count ?? 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-3 backdrop-blur-sm sm:p-6">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-experiment-title"
        className="edge grid w-full max-w-lg gap-4 rounded-card p-4 shadow-[0_24px_80px_rgba(0,0,0,0.58)] sm:p-5"
      >
        <header className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 id="delete-experiment-title" className="text-base font-semibold text-ink">
              Delete Experiment
            </h2>
            <div className="mt-1 truncate font-mono text-xs text-ink-faint">
              logs/{option.value}
            </div>
          </div>
          <button
            type="button"
            aria-label="Close delete experiment"
            onClick={onClose}
            disabled={isDeleting}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-[10px] border border-line bg-white/[0.035] text-ink-faint transition hover:bg-white/[0.07] hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        </header>

        <div className="grid gap-3 text-sm leading-6 text-ink-dim">
          <p>
            This permanently deletes {runCount} {runCount === 1 ? "run" : "runs"} from this
            experiment and removes the whole experiment folder from disk.
          </p>
          <label className="grid gap-2">
            <span className="text-xs font-bold uppercase tracking-[0.09em] text-ink-faint">
              Type experiment name
            </span>
            <Input
              value={confirmation}
              onChange={(event) => setConfirmation(event.target.value)}
              aria-label="Type experiment name"
              autoComplete="off"
              disabled={isDeleting}
              placeholder={option.value}
            />
          </label>
        </div>

        {error ? (
          <ErrorPanel title="Delete failed" message={errorMessage(error)} />
        ) : null}

        <footer className="flex flex-wrap items-center justify-end gap-2">
          <Button variant="ghost" onClick={onClose} disabled={isDeleting}>
            Cancel
          </Button>
          <Button
            variant="danger"
            onClick={onConfirm}
            disabled={!canDelete}
          >
            {isDeleting && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
            Delete Experiment
          </Button>
        </footer>
      </section>
    </div>
  );
}

function AffectedValueGroup({
  label,
  values,
}: {
  label: string;
  values: string[];
}) {
  return (
    <div className="grid gap-1.5">
      <div className="text-[11px] font-bold uppercase tracking-[0.09em] text-ink-faint">
        {label}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {values.length === 0 ? (
          <span className="font-mono text-xs text-ink-faint">None</span>
        ) : (
          values.slice(0, 8).map((value) => (
            <Badge key={value} className="max-w-full truncate">
              {value}
            </Badge>
          ))
        )}
        {values.length > 8 && <Badge>+{values.length - 8}</Badge>}
      </div>
    </div>
  );
}

function DeleteSubsetRunsDialog({
  target,
  plan,
  error,
  isPlanning,
  isDeleting,
  onClose,
  onConfirm,
}: {
  target: SubsetDeleteTarget;
  plan: LogRunDeletePlan | undefined;
  error: unknown;
  isPlanning: boolean;
  isDeleting: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
}) {
  const targetLabel = target.kind === "dataset" ? "Dataset" : "Preset";
  const title = `Delete ${targetLabel}`;
  const runCount = plan?.candidateCount ?? 0;
  const previewCandidates = plan?.candidates.slice(0, 8) ?? [];
  const overflowCount = Math.max(0, runCount - previewCandidates.length);
  const blockers = plan?.blockedByActiveJobs ?? [];
  const canDelete =
    Boolean(plan?.canDelete) &&
    blockers.length === 0 &&
    !isPlanning &&
    !isDeleting;
  const dialogTitleId = `delete-${target.kind}-title`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-3 backdrop-blur-sm sm:p-6">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby={dialogTitleId}
        className="edge grid max-h-[min(760px,calc(100vh-32px))] w-full max-w-2xl gap-4 overflow-y-auto rounded-card p-4 shadow-[0_24px_80px_rgba(0,0,0,0.58)] sm:p-5"
      >
        <header className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2
              id={dialogTitleId}
              className="text-base font-semibold text-ink"
            >
              {title}
            </h2>
            <div className="mt-1 font-mono text-xs text-ink-faint">
              {isPlanning ? "Planning matched version folders" : `${runCount} matched runs`}
            </div>
          </div>
          <button
            type="button"
            aria-label={`Close delete ${target.kind}`}
            onClick={onClose}
            disabled={isDeleting}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-[10px] border border-line bg-white/[0.035] text-ink-faint transition hover:bg-white/[0.07] hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        </header>

        {isPlanning ? (
          <SidebarStatus
            title="Planning delete"
            detail="Matching the selected experiment and row."
            busy
          />
        ) : plan ? (
          <div className="grid gap-4">
            <div className="grid gap-3 text-sm leading-6 text-ink-dim">
              <p>
                This permanently removes only the matched <code>version_*</code> run
                folders. Other runs and non-empty parent folders are left on disk.
              </p>
              <div className="grid gap-2 rounded-[12px] border border-danger-line/70 bg-danger-soft p-3 sm:grid-cols-3">
                <div className="min-w-0">
                  <div className="text-[11px] font-bold uppercase tracking-[0.09em] text-[#fda4af]">
                    Experiment
                  </div>
                  <div className="mt-1 truncate font-mono text-xs text-ink">
                    {target.experiment}
                  </div>
                </div>
                <div className="min-w-0">
                  <div className="text-[11px] font-bold uppercase tracking-[0.09em] text-[#fda4af]">
                    {targetLabel}
                  </div>
                  <div className="mt-1 truncate font-mono text-xs text-ink">
                    {target.value}
                  </div>
                </div>
                <div className="min-w-0">
                  <div className="text-[11px] font-bold uppercase tracking-[0.09em] text-[#fda4af]">
                    Matched Runs
                  </div>
                  <div className="mt-1 font-mono text-xs text-ink">{runCount}</div>
                </div>
              </div>
              <div className="grid gap-3 rounded-[12px] border border-line-soft bg-black/20 p-3 sm:grid-cols-2">
                <AffectedValueGroup label="Experiments" values={plan.affected.experiments} />
                <AffectedValueGroup label="Datasets" values={plan.affected.datasets} />
                <AffectedValueGroup label="Models" values={plan.affected.models} />
                <AffectedValueGroup label="Presets" values={plan.affected.presets} />
              </div>
            </div>

            {blockers.length > 0 && (
              <div className="grid gap-2 rounded-[12px] border border-danger-line bg-danger-soft p-3 text-sm text-[#fecdd3]">
                <div className="flex items-start gap-2 font-semibold text-[#fda4af]">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                  <span>A training job is still writing to this log folder.</span>
                </div>
                <div className="grid gap-1.5">
                  {blockers.map((blocker) => (
                    <div
                      key={`${blocker.id}-${blocker.logFolder}`}
                      className="grid gap-1 rounded-[9px] border border-danger-line/70 bg-black/20 px-2.5 py-2 font-mono text-xs sm:grid-cols-[minmax(0,1fr)_auto]"
                    >
                      <span className="min-w-0 truncate">
                        {blocker.id} · logs/{blocker.logFolder}
                      </span>
                      <span>{blocker.status}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="grid gap-2">
              <div className="text-xs font-bold uppercase tracking-[0.09em] text-ink-faint">
                Path Preview
              </div>
              <div className="grid max-h-56 gap-1.5 overflow-y-auto rounded-[12px] border border-line-soft bg-black/20 p-2">
                {previewCandidates.length === 0 ? (
                  <div className="px-2 py-4 text-center text-sm text-ink-faint">
                    No matched run folders
                  </div>
                ) : (
                  previewCandidates.map((candidate) => (
                    <div
                      key={candidate.id}
                      className="truncate rounded-[8px] border border-line-soft bg-white/[0.025] px-2.5 py-1.5 font-mono text-xs text-ink-dim"
                      title={candidate.relativePath}
                    >
                      {candidate.relativePath}
                    </div>
                  ))
                )}
                {overflowCount > 0 && (
                  <div className="px-2 py-1 font-mono text-xs text-ink-faint">
                    +{overflowCount} more
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : null}

        {error ? (
          <ErrorPanel title="Delete failed" message={errorMessage(error)} />
        ) : null}

        <footer className="flex flex-wrap items-center justify-end gap-2">
          <Button variant="ghost" onClick={onClose} disabled={isDeleting}>
            Cancel
          </Button>
          <Button variant="danger" onClick={onConfirm} disabled={!canDelete}>
            {isDeleting && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
            {title}
          </Button>
        </footer>
      </section>
    </div>
  );
}

export function LogsSidebarPanel({ state }: { state: LogsWorkspaceState }) {
  const [deleteOption, setDeleteOption] = useState<ChecklistOption | null>(null);
  const [subsetDeleteTarget, setSubsetDeleteTarget] =
    useState<SubsetDeleteTarget | null>(null);
  const isScanning = state.runsQuery.isLoading || state.experimentsQuery.isLoading;
  const isRefreshing = state.runsQuery.isFetching || state.experimentsQuery.isFetching;
  const selectedExperimentOptions = useMemo(
    () => selectedOptionsSet(state.selectedExperiments, state.experimentOptions),
    [state.experimentOptions, state.selectedExperiments],
  );
  const singleSelectedExperiment =
    selectedExperimentOptions.size === 1
      ? Array.from(selectedExperimentOptions)[0]
      : null;

  useEffect(() => {
    if (!subsetDeleteTarget || singleSelectedExperiment === subsetDeleteTarget.experiment) {
      return;
    }
    state.resetRunDelete();
    setSubsetDeleteTarget(null);
  }, [singleSelectedExperiment, state, subsetDeleteTarget]);

  function openDeleteDialog(option: ChecklistOption) {
    state.resetDeleteExperiment();
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
      runs: state.runs,
    });
    if (!target) {
      return;
    }
    state.resetRunDelete();
    setSubsetDeleteTarget(target);
    void state.createRunDeletePlan(target.filters);
  }

  function closeSubsetDeleteDialog() {
    state.resetRunDelete();
    setSubsetDeleteTarget(null);
  }

  function closeDeleteDialog() {
    state.resetDeleteExperiment();
    setDeleteOption(null);
  }

  async function confirmDeleteExperiment() {
    if (!deleteOption) {
      return;
    }
    try {
      await state.deleteExperiment(deleteOption.value);
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
      await state.deleteRuns(subsetDeleteTarget.filters);
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
      <Button
        variant="ghost"
        className="h-8 w-8 px-0 text-ink-faint hover:border-danger-line hover:bg-danger-soft hover:text-[#fda4af]"
        onClick={() => openSubsetDeleteDialog(kind, option)}
        aria-label={`Delete ${kind} ${option.label} from experiment ${singleSelectedExperiment}`}
      >
        <Trash2 className="h-4 w-4" aria-hidden />
      </Button>
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
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
            <LineChart className="h-[15px] w-[15px] text-violet" aria-hidden />
            Logs
          </div>
          <Button
            variant="ghost"
            className="h-8 px-2"
            onClick={() => {
              void state.experimentsQuery.refetch();
              void state.runsQuery.refetch();
            }}
            disabled={isRefreshing}
            aria-label="Refresh log runs"
          >
            <RefreshCw
              className={cn("h-4 w-4", isRefreshing && "animate-spin")}
              aria-hidden
            />
          </Button>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <EdgeCard className="rounded-[12px] px-3 py-2.5">
            <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
              Runs
            </div>
            <div className="mt-1 font-mono text-xl font-extrabold text-ink">
              {state.visibleRuns.length}
            </div>
          </EdgeCard>
          <EdgeCard className="rounded-[12px] px-3 py-2.5">
            <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
              Tags
            </div>
            <div className="mt-1 font-mono text-xl font-extrabold text-ink">
              {selectedOptionsSet(state.selectedTags, state.tagOptions).size}
            </div>
          </EdgeCard>
        </div>
      </section>

      {state.runsQuery.isError && (
        <ErrorPanel title="Log scan failed" message={errorMessage(state.runsQuery.error)} />
      )}
      {state.experimentsQuery.isError && (
        <ErrorPanel
          title="Experiment scan failed"
          message={errorMessage(state.experimentsQuery.error)}
        />
      )}

      {isScanning ? (
        <SidebarStatus
          title="Scanning logs"
          detail="Reading historical version folders from logs/."
          busy
        />
      ) : state.experimentOptions.length === 0 ? (
        <SidebarStatus
          title="No log runs"
          detail="No version_* folders were found under logs/."
        />
      ) : (
        <>
          <ChecklistSection
            title="Experiments"
            icon={<FolderTree className="h-4 w-4" aria-hidden />}
            options={state.experimentOptions}
            selected={selectedOptionsSet(state.selectedExperiments, state.experimentOptions)}
            onToggle={state.toggleExperiment}
            onAll={state.selectAllExperiments}
            onNone={state.selectNoExperiments}
            renderOptionAction={(option) => (
              <Button
                variant="ghost"
                className="h-8 w-8 px-0 text-ink-faint hover:border-danger-line hover:bg-danger-soft hover:text-[#fda4af]"
                onClick={() => openDeleteDialog(option)}
                aria-label={`Delete experiment ${option.label}`}
              >
                <Trash2 className="h-4 w-4" aria-hidden />
              </Button>
            )}
          />
          {state.runs.length === 0 ? (
            <SidebarStatus
              title="No runs yet"
              detail="Experiment folders without version_* runs stay selectable."
            />
          ) : (
            <>
              <ChecklistSection
                title="Datasets"
                icon={<Database className="h-4 w-4" aria-hidden />}
                options={state.datasetOptions}
                selected={selectedOptionsSet(state.selectedDatasets, state.datasetOptions)}
                onToggle={state.toggleDataset}
                onAll={state.selectAllDatasets}
                onNone={state.selectNoDatasets}
                renderOptionAction={renderDatasetDeleteAction}
              />
              <ChecklistSection
                title="Models"
                icon={<Cpu className="h-4 w-4" aria-hidden />}
                options={state.modelOptions}
                selected={selectedOptionsSet(state.selectedModels, state.modelOptions)}
                onToggle={state.toggleModel}
                onAll={state.selectAllModels}
                onNone={state.selectNoModels}
              />
              <ChecklistSection
                title="Presets"
                icon={<Layers className="h-4 w-4" aria-hidden />}
                options={state.presetOptions}
                selected={selectedOptionsSet(state.selectedPresets, state.presetOptions)}
                onToggle={state.togglePreset}
                onAll={state.selectAllPresets}
                onNone={state.selectNoPresets}
                defaultOpen={false}
                renderOptionAction={renderPresetDeleteAction}
              />
              <ChecklistSection
                title="Runs"
                icon={<FileText className="h-4 w-4" aria-hidden />}
                options={state.runOptions}
                selected={selectedOptionsSet(state.selectedRunIds, state.runOptions)}
                onToggle={state.toggleRun}
                onAll={state.selectAllRuns}
                onNone={state.selectNoRuns}
                defaultOpen={false}
              />
              {state.tagsQuery.isError && (
                <ErrorPanel title="Tag read failed" message={errorMessage(state.tagsQuery.error)} />
              )}
              <ChecklistSection
                title="Scalar Tags"
                icon={<Tag className="h-4 w-4" aria-hidden />}
                options={state.tagOptions}
                selected={selectedOptionsSet(state.selectedTags, state.tagOptions)}
                onToggle={state.toggleTag}
                onAll={state.selectAllTags}
                onNone={state.selectNoTags}
                search
              />
            </>
          )}
        </>
      )}

      {deleteOption && (
        <DeleteExperimentDialog
          option={deleteOption}
          error={state.deleteExperimentError}
          isDeleting={state.isDeletingExperiment}
          onClose={closeDeleteDialog}
          onConfirm={confirmDeleteExperiment}
        />
      )}
      {subsetDeleteTarget && (
        <DeleteSubsetRunsDialog
          key={subsetDeleteTarget.key}
          target={subsetDeleteTarget}
          plan={state.runDeletePlan}
          error={state.runDeletePlanError ?? state.runDeleteError}
          isPlanning={state.isPlanningRunDelete}
          isDeleting={state.isDeletingRunDelete}
          onClose={closeSubsetDeleteDialog}
          onConfirm={confirmDeleteSubsetRuns}
        />
      )}
    </div>
  );
}

function ChartEmptyState({
  title,
  detail,
  busy,
}: {
  title: string;
  detail: string;
  busy?: boolean;
}) {
  return (
    <div className="grid h-full min-h-[360px] place-items-center p-6">
      <div className="edge grid max-w-md justify-items-center gap-3 rounded-card p-6 text-center shadow-panel">
        {busy && <Loader2 className="h-5 w-5 animate-spin text-violet" aria-hidden />}
        <div className="flex h-10 w-10 items-center justify-center rounded-[10px] border border-line bg-white/[0.04] text-violet">
          <LineChart className="h-5 w-5" aria-hidden />
        </div>
        <div>
          <div className="text-sm font-semibold text-ink">{title}</div>
          <div className="mt-1 text-xs leading-5 text-ink-faint">{detail}</div>
        </div>
      </div>
    </div>
  );
}

function linePath(
  points: LogScalarSeries["points"],
  domain: { minStep: number; maxStep: number; minValue: number; maxValue: number },
) {
  const width = 760;
  const height = 188;
  const paddingX = 34;
  const paddingY = 22;
  const stepSpan = domain.maxStep - domain.minStep || 1;
  const valueSpan = domain.maxValue - domain.minValue || 1;
  return points
    .map((point, index) => {
      const x =
        paddingX + ((point.step - domain.minStep) / stepSpan) * (width - paddingX * 2);
      const y =
        height -
        paddingY -
        ((point.value - domain.minValue) / valueSpan) * (height - paddingY * 2);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function LogScalarChart({
  tag,
  series,
  runsById,
  runOrder,
  onSelectRun,
}: {
  tag: string;
  series: LogScalarSeries[];
  runsById: Map<string, LogRun>;
  runOrder: string[];
  onSelectRun: (runId: string) => void;
}) {
  const width = 760;
  const height = 188;
  const paddingX = 34;
  const paddingY = 22;
  const allPoints = series.flatMap((entry) => entry.points);
  const minStep = Math.min(...allPoints.map((point) => point.step));
  const maxStep = Math.max(...allPoints.map((point) => point.step));
  const minValue = Math.min(...allPoints.map((point) => point.value));
  const maxValue = Math.max(...allPoints.map((point) => point.value));
  const domain = { minStep, maxStep, minValue, maxValue };

  return (
    <section className="edge grid gap-3 rounded-card p-4">
      <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-bold text-ink">{tag}</h2>
          <div className="mt-0.5 font-mono text-xs text-ink-faint">
            {series.length} lines · step {minStep} to {maxStep}
          </div>
        </div>
        <Badge>{formatNumber(minValue)} to {formatNumber(maxValue)}</Badge>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="h-52 w-full overflow-visible text-violet"
        role="img"
        aria-label={`${tag} scalar chart`}
      >
        <line
          x1={paddingX}
          y1={height - paddingY}
          x2={width - paddingX}
          y2={height - paddingY}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth="1"
        />
        <line
          x1={paddingX}
          y1={paddingY}
          x2={paddingX}
          y2={height - paddingY}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth="1"
        />
        <text x={paddingX} y={height - 4} fill="rgba(230,232,255,0.45)" fontSize="10">
          {minStep}
        </text>
        <text
          x={width - paddingX}
          y={height - 4}
          fill="rgba(230,232,255,0.45)"
          fontSize="10"
          textAnchor="end"
        >
          {maxStep}
        </text>
        <text x="4" y={paddingY + 4} fill="rgba(230,232,255,0.45)" fontSize="10">
          {formatNumber(maxValue)}
        </text>
        <text x="4" y={height - paddingY} fill="rgba(230,232,255,0.45)" fontSize="10">
          {formatNumber(minValue)}
        </text>
        {series.map((entry) => {
          const color =
            SERIES_COLORS[Math.max(runOrder.indexOf(entry.runId), 0) % SERIES_COLORS.length];
          if (entry.points.length === 1) {
            const point = entry.points[0];
            const stepSpan = maxStep - minStep || 1;
            const valueSpan = maxValue - minValue || 1;
            const x = paddingX + ((point.step - minStep) / stepSpan) * (width - paddingX * 2);
            const y =
              height -
              paddingY -
              ((point.value - minValue) / valueSpan) * (height - paddingY * 2);
            return (
              <circle
                key={entry.runId}
                cx={x}
                cy={y}
                r="3.5"
                fill={color}
                aria-label={runsById.get(entry.runId)?.runName}
              />
            );
          }
          return (
            <path
              key={entry.runId}
              d={linePath(entry.points, domain)}
              fill="none"
              stroke={color}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity="0.9"
            />
          );
        })}
      </svg>

      <div className="grid gap-1.5 sm:grid-cols-2 xl:grid-cols-3">
        {series.map((entry) => {
          const run = runsById.get(entry.runId);
          if (!run) {
            return null;
          }
          const color =
            SERIES_COLORS[Math.max(runOrder.indexOf(entry.runId), 0) % SERIES_COLORS.length];
          const latest = entry.points.at(-1);
          return (
            <button
              key={entry.runId}
              type="button"
              className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 rounded-[9px] border border-line-soft bg-black/20 px-2 py-1.5 text-left text-xs transition hover:border-line hover:bg-white/[0.035] focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              onClick={() => onSelectRun(entry.runId)}
            >
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: color }}
                aria-hidden
              />
              <span className="truncate text-ink-dim">{formatRunLabel(run)}</span>
              {latest && (
                <span className="font-mono text-ink-faint">{formatNumber(latest.value)}</span>
              )}
            </button>
          );
        })}
      </div>
    </section>
  );
}

export function LogsGraphPreviewPanel({ state }: { state: LogsWorkspaceState }) {
  const [scalarChartGridMode, setScalarChartGridMode] =
    useState<ScalarChartGridMode>("full");
  const scalarQuery = useLogScalarsQuery({
    runIds: state.visibleRunIds,
    tags: state.selectedTagList,
    enabled: state.enabled,
    queryKey: ["log-scalars", state.visibleRunIds, state.selectedTagList],
  });

  const runsById = useMemo(
    () => new Map(state.visibleRuns.map((run) => [run.id, run])),
    [state.visibleRuns],
  );
  const seriesByTag = useMemo(() => {
    const byTag = new Map<string, LogScalarSeries[]>();
    for (const series of scalarQuery.data?.series ?? []) {
      if (series.points.length === 0) {
        continue;
      }
      byTag.set(series.tag, [...(byTag.get(series.tag) ?? []), series]);
    }
    return byTag;
  }, [scalarQuery.data]);
  const selectedSeriesCount = Array.from(seriesByTag.values()).reduce(
    (total, series) => total + series.length,
    0,
  );
  const hasEventFiles = state.visibleRuns.some((run) => run.eventFileCount > 0);

  let emptyTitle = "";
  let emptyDetail = "";
  if (state.runsQuery.isLoading) {
    emptyTitle = "Scanning logs";
    emptyDetail = "Reading historical run folders.";
  } else if (state.visibleRuns.length === 0) {
    emptyTitle = "No runs selected";
    emptyDetail = "Use the sidebar filters to include at least one historical run.";
  } else if (state.tagsQuery.isLoading) {
    emptyTitle = "Reading TensorBoard tags";
    emptyDetail = "Collecting scalar tags from the selected runs.";
  } else if (state.tagOptions.length === 0) {
    emptyTitle = "No TensorBoard scalars";
    emptyDetail = "The selected runs do not contain scalar event data.";
  } else if (state.selectedTagList.length === 0) {
    emptyTitle = "No scalar tags selected";
    emptyDetail = "Select one or more scalar tags to draw historical charts.";
  } else if (scalarQuery.isLoading) {
    emptyTitle = "Loading scalar points";
    emptyDetail = "Reading TensorBoard scalar series for the selected runs.";
  } else if (selectedSeriesCount === 0 && hasEventFiles) {
    emptyTitle = "No scalar points for selection";
    emptyDetail = "The selected runs have event files, but none contain the checked scalar tags.";
  } else if (selectedSeriesCount === 0) {
    emptyTitle = "No TensorBoard scalars";
    emptyDetail = "The selected runs do not contain scalar event data.";
  }

  return (
    <div className="grid min-h-0 grid-rows-[56px_minmax(0,1fr)]">
      <div className="flex min-w-0 items-center justify-between gap-3 border-b border-line bg-panel/45 px-4">
        <div className="min-w-0">
          <div className="text-sm font-bold text-ink">Historical Scalars</div>
          <div className="truncate font-mono text-xs text-ink-faint">
            {state.visibleRuns.length} runs · {state.selectedTagList.length} selected tags
          </div>
        </div>
        <div className="flex min-w-0 items-center justify-end gap-2 overflow-x-auto">
          <SegmentedControl aria-label="Scalar chart layout" className="shrink-0">
            <ViewModeButton
              active={scalarChartGridMode === "full"}
              onClick={() => setScalarChartGridMode("full")}
            >
              <RectangleHorizontal className="h-3.5 w-3.5" aria-hidden />
              Full
            </ViewModeButton>
            <ViewModeButton
              active={scalarChartGridMode === "two"}
              onClick={() => setScalarChartGridMode("two")}
            >
              <Columns2 className="h-3.5 w-3.5" aria-hidden />
              2 Col
            </ViewModeButton>
            <ViewModeButton
              active={scalarChartGridMode === "three"}
              onClick={() => setScalarChartGridMode("three")}
            >
              <Columns3 className="h-3.5 w-3.5" aria-hidden />
              3 Col
            </ViewModeButton>
          </SegmentedControl>
          <Button
            variant="secondary"
            className="h-8 shrink-0 px-2"
            onClick={() => scalarQuery.refetch()}
            disabled={!scalarQuery.isSuccess && !scalarQuery.isError}
            aria-label="Refresh scalar charts"
          >
            <RefreshCw
              className={cn("h-4 w-4", scalarQuery.isFetching && "animate-spin")}
              aria-hidden
            />
          </Button>
        </div>
      </div>

      <div className="min-h-0 overflow-y-auto p-4">
        {scalarQuery.isError && (
          <div className="mb-4">
            <ErrorPanel title="Scalar read failed" message={errorMessage(scalarQuery.error)} />
          </div>
        )}

        {emptyTitle ? (
          <ChartEmptyState
            title={emptyTitle}
            detail={emptyDetail}
            busy={state.runsQuery.isLoading || state.tagsQuery.isLoading || scalarQuery.isLoading}
          />
        ) : (
          <div className={SCALAR_CHART_GRID_CLASSES[scalarChartGridMode]}>
            {state.selectedTagList.map((tag) => {
              const series = seriesByTag.get(tag) ?? [];
              if (series.length === 0) {
                return null;
              }
              return (
                <LogScalarChart
                  key={tag}
                  tag={tag}
                  series={series}
                  runsById={runsById}
                  runOrder={state.visibleRunIds}
                  onSelectRun={state.setSelectedDetailRunId}
                />
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function FilePresenceRow({
  label,
  value,
  present,
}: {
  label: string;
  value: string | number;
  present: boolean;
}) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2 rounded-[9px] border border-line-soft bg-black/20 px-3 py-2 text-xs">
      <span className="truncate text-ink-dim">{label}</span>
      <Badge className={present ? "text-ink" : "text-ink-faint"}>{value}</Badge>
    </div>
  );
}

export function LogRunDetailsPanel({ state }: { state: LogsWorkspaceState }) {
  const run = state.selectedRun;
  const metrics = Object.entries(run?.metrics ?? {});

  return (
    <aside className="min-h-0 overflow-y-auto border-t border-line bg-[linear-gradient(180deg,rgba(13,12,22,0.6),rgba(8,8,13,0.4))] px-[18px] pb-8 pt-5 backdrop-blur lg:border-l lg:border-t-0">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h2 className="text-base font-bold text-ink">Run Details</h2>
        {run && <Badge>{run.hasResult ? "result.json" : "No result.json"}</Badge>}
      </div>

      {!run ? (
        <div className="edge rounded-card p-4 text-sm text-ink-faint">
          Select a visible run to inspect its metadata.
        </div>
      ) : (
        <div className="grid gap-4">
          <EdgeCard className="rounded-[12px] px-3 py-3">
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold text-ink" title={run.runName}>
                {run.runName}
              </div>
              <div className="mt-1 break-words font-mono text-xs leading-5 text-ink-faint">
                {run.relativePath}
              </div>
            </div>
          </EdgeCard>

          <div className="grid grid-cols-2 gap-[9px]">
            <EdgeCard className="rounded-[12px] px-3 py-3">
              <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
                Experiment
              </div>
              <div className="mt-1.5 truncate font-mono text-sm font-bold text-ink">
                {run.experiment}
              </div>
            </EdgeCard>
            <EdgeCard className="rounded-[12px] px-3 py-3">
              <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
                Dataset
              </div>
              <div className="mt-1.5 truncate font-mono text-sm font-bold text-ink">
                {run.dataset}
              </div>
            </EdgeCard>
            <EdgeCard className="rounded-[12px] px-3 py-3">
              <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
                Model
              </div>
              <div className="mt-1.5 truncate font-mono text-sm font-bold text-ink">
                {run.model}
              </div>
            </EdgeCard>
            <EdgeCard className="rounded-[12px] px-3 py-3">
              <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
                Preset
              </div>
              <div className="mt-1.5 truncate font-mono text-sm font-bold text-ink">
                {run.preset}
              </div>
            </EdgeCard>
            <EdgeCard className="rounded-[12px] px-3 py-3">
              <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
                Version
              </div>
              <div className="mt-1.5 truncate font-mono text-sm font-bold text-ink">
                {run.version}
              </div>
            </EdgeCard>
          </div>

          <section className="grid gap-2">
            <h3 className="text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
              Files
            </h3>
            <div className="grid gap-2">
              <FilePresenceRow
                label="Event files"
                value={run.eventFileCount}
                present={run.eventFileCount > 0}
              />
              <FilePresenceRow
                label="hparams.yaml"
                value={run.hasHparams ? "present" : "missing"}
                present={run.hasHparams}
              />
              <FilePresenceRow
                label="result.json"
                value={run.hasResult ? "present" : "missing"}
                present={run.hasResult}
              />
              <FilePresenceRow
                label="Checkpoints"
                value={run.checkpointCount}
                present={run.checkpointCount > 0}
              />
            </div>
          </section>

          <section className="grid gap-2">
            <h3 className="text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
              Metrics
            </h3>
            {!run.hasResult ? (
              <div className="rounded-[10px] border border-dashed border-line-soft px-3 py-4 text-sm text-ink-faint">
                No result.json
              </div>
            ) : metrics.length === 0 ? (
              <div className="rounded-[10px] border border-dashed border-line-soft px-3 py-4 text-sm text-ink-faint">
                No metrics found
              </div>
            ) : (
              <div className="grid gap-2">
                {metrics.map(([key, value]) => (
                  <div
                    key={key}
                    className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2 rounded-[9px] border border-line-soft bg-black/20 px-3 py-2 text-xs"
                  >
                    <span className="truncate font-mono text-ink-dim">{key}</span>
                    <span className="font-mono font-semibold text-ink">
                      {formatMetricValue(value)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      )}
    </aside>
  );
}
