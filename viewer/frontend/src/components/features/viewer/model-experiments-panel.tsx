import { CheckCircle2, FileText, FlaskConical, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { EdgeCard } from "@/components/ui/edge-card";
import { Select } from "@/components/ui/select";
import { type LogRun } from "@/lib/api";
import { groupModelLogRunsByExperiment } from "@/lib/historical-monitor-runs";
import { cn, errorMessage } from "@/lib/utils";
import { useHistoricalRuns } from "@/components/features/viewer/providers/viewer-providers";

function pluralize(count: number, singular: string, plural: string) {
  return `${count} ${count === 1 ? singular : plural}`;
}

function runTimestamp(run: LogRun) {
  return run.timestamp ?? run.version;
}

export function ModelExperimentsPanel() {
  const {
    filteredHistoricalRuns: runs,
    historicalExperimentOptions: experimentOptions,
    historicalDatasetOptions: datasetOptions,
    selectedHistoricalExperiment: selectedExperiment,
    selectedHistoricalDataset: selectedDataset,
    selectedLogRunId: selectedRunId,
    setSelectedHistoricalExperiment: onSelectExperiment,
    setSelectedHistoricalDataset: onSelectDataset,
    selectLogRun: onSelectRun,
    logRunsQuery,
  } = useHistoricalRuns();
  const isLoading = logRunsQuery.isLoading;
  const isError = logRunsQuery.isError;
  const error = logRunsQuery.error;
  const groups = groupModelLogRunsByExperiment(runs);

  return (
    <EdgeCard className="rounded-card p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <FlaskConical className="h-4 w-4 shrink-0 text-violet" aria-hidden />
          <h2 className="truncate text-sm font-bold text-ink">Experiments</h2>
        </div>
        <Badge>{runs.length}</Badge>
      </div>

      <div className="mt-3 grid gap-2 rounded-[10px] border border-line-soft bg-black/18 p-2">
        <label className="grid gap-1 text-[11px] font-bold uppercase text-ink-dim">
          Experiment
          <Select
            value={selectedExperiment}
            onChange={(event) => onSelectExperiment(event.target.value)}
            disabled={isLoading || experimentOptions.length === 0}
            className="h-9 text-xs"
          >
            {experimentOptions.length === 0 ? (
              <option value="">No experiments</option>
            ) : (
              experimentOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label} ({option.count})
                </option>
              ))
            )}
          </Select>
        </label>
        <label className="grid gap-1 text-[11px] font-bold uppercase text-ink-dim">
          Dataset
          <Select
            value={selectedDataset}
            onChange={(event) => onSelectDataset(event.target.value)}
            disabled={isLoading || datasetOptions.length === 0}
            className="h-9 text-xs"
          >
            {datasetOptions.length === 0 ? (
              <option value="">No datasets</option>
            ) : (
              datasetOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label} ({option.count})
                </option>
              ))
            )}
          </Select>
        </label>
      </div>

      <div className="mt-3 grid gap-2">
        {isLoading && (
          <div className="flex min-h-16 items-center justify-center gap-2 rounded-[10px] border border-line-soft bg-black/20 text-sm text-ink-dim">
            <Loader2 className="h-4 w-4 animate-spin text-violet" aria-hidden />
            Loading runs
          </div>
        )}

        {isError && (
          <div className="rounded-[10px] border border-red-400/25 bg-red-500/10 p-3 text-sm text-red-100">
            {errorMessage(error)}
          </div>
        )}

        {!isLoading && !isError && runs.length === 0 && (
          <div className="rounded-[10px] border border-dashed border-faint bg-white/[0.018] p-3 text-sm text-ink-faint">
            {experimentOptions.length === 0 ? "No runs for this model" : "No matching runs"}
          </div>
        )}

        {!isLoading &&
          !isError &&
          groups.map((group) => (
            <section key={group.experiment} className="grid gap-1.5">
              <div className="flex items-center justify-between gap-2 px-1 text-[11px] font-bold uppercase text-ink-dim">
                <span className="truncate">{group.experiment}</span>
                <span className="font-mono">{group.runs.length}</span>
              </div>
              <div className="grid gap-1.5">
                {group.runs.map((run) => {
                  const selected = run.id === selectedRunId;
                  return (
                    <button
                      key={run.id}
                      type="button"
                      aria-pressed={selected}
                      aria-label={`Select experiment run ${run.experiment} ${run.preset} ${run.dataset} ${runTimestamp(run)}`}
                      onClick={() => onSelectRun(run.id)}
                      className={cn(
                        "grid min-h-[72px] gap-1 rounded-[10px] border px-3 py-2 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
                        selected
                          ? "border-violet/45 bg-violet/15 shadow-[inset_0_0_0_1px_rgba(167,139,250,0.16)]"
                          : "border-line-soft bg-black/18 hover:border-line hover:bg-white/[0.04]",
                      )}
                    >
                      <div className="flex min-w-0 items-center justify-between gap-2">
                        <span className="min-w-0 truncate text-xs font-semibold text-ink">
                          {run.preset}
                        </span>
                        <span
                          className={cn(
                            "inline-flex shrink-0 items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-bold",
                            selected
                              ? "border-violet/30 bg-violet/20 text-ink"
                              : "border-line-soft bg-white/[0.025] text-ink-dim",
                          )}
                        >
                          {selected && <CheckCircle2 className="h-3 w-3" aria-hidden />}
                          {selected ? "selected" : run.hasResult ? "result" : "started"}
                        </span>
                      </div>
                      <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-ink-dim">
                        <span className="truncate font-mono">{run.dataset}</span>
                        <span className="font-mono">{runTimestamp(run)}</span>
                      </div>
                      <div className="flex min-w-0 items-center gap-1 text-[11px] text-ink-faint">
                        <FileText className="h-3.5 w-3.5 shrink-0" aria-hidden />
                        <span>{pluralize(run.eventFileCount, "event file", "event files")}</span>
                        <span>&middot;</span>
                        <span>{run.version}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </section>
          ))}
      </div>
    </EdgeCard>
  );
}
