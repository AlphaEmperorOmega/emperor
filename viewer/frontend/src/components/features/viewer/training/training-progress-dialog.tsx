import { AlertTriangle, RefreshCw, Terminal, Trash2, X } from "lucide-react";
import { useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TrainingCommandDialog } from "@/components/features/viewer/config/training-command-dialog";
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";
import {
  type TrainingRun,
  type TrainingRunPlan,
} from "@/lib/api";

function statusClass(status: TrainingRun["status"]) {
  if (status === "Completed") {
    return "border-ok/30 bg-ok/10 text-ok";
  }
  if (status === "Running") {
    return "border-violet/30 bg-violet/15 text-violet";
  }
  if (status === "Failed" || status === "Cancelled") {
    return "border-danger-line bg-danger-soft text-[#fda4af]";
  }
  if (status === "Skipped") {
    return "border-amber/40 bg-amber/[0.12] text-amber";
  }
  return "";
}

function metricValue(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Number.parseFloat(value.toPrecision(4)).toString();
  }
  return String(value);
}

function metricsText(metrics: Record<string, unknown>) {
  const entries = Object.entries(metrics);
  if (entries.length === 0) {
    return "-";
  }
  return entries
    .slice(0, 2)
    .map(([key, value]) => `${key}=${metricValue(value)}`)
    .join("  ");
}

function epochText(run: TrainingRun) {
  return `${run.currentEpoch} / ${run.totalEpochs || "-"}`;
}

const bodyCellClass = "border-b border-line-soft px-3 py-3";
const monoCellClass = `${bodyCellClass} font-mono text-xs text-ink`;
const emptyCell = <span className="font-mono text-xs text-ink-dim">-</span>;

function RunStatusCell({
  run,
  onFullError,
}: {
  run: TrainingRun;
  onFullError: (run: TrainingRun) => void;
}) {
  const fullError = run.errorTraceback || run.error;
  return (
    <td className={bodyCellClass}>
      <Badge className={statusClass(run.status)}>{run.status}</Badge>
      {run.error && (
        <div className="mt-1 grid max-w-48 gap-1.5 text-xs text-[#fda4af]">
          <span>{run.error}</span>
          {fullError && (
            <Button
              variant="ghost"
              className="h-7 justify-start border border-danger-line bg-danger-soft px-2 text-xs text-[#fda4af] hover:bg-[#7f1d2d]/40 hover:text-white"
              onClick={() => onFullError(run)}
              aria-label={`Full error for run ${run.index}`}
            >
              <AlertTriangle className="h-3.5 w-3.5" aria-hidden />
              Full Error
            </Button>
          )}
        </div>
      )}
    </td>
  );
}

function RunChangesCell({ run }: { run: TrainingRun }) {
  if (run.changes.length === 0) {
    return <td className={bodyCellClass}>{emptyCell}</td>;
  }

  return (
    <td className={bodyCellClass}>
      <div className="flex max-w-[20rem] flex-wrap gap-1.5">
        {run.changes.map((change) => (
          <span
            key={`${change.source}-${change.key}-${String(change.value)}`}
            className={
              change.source === "search"
                ? "rounded-[7px] border border-violet/30 bg-violet/10 px-2 py-0.5 font-mono text-xs text-violet"
                : "rounded-[7px] border border-line bg-white/[0.04] px-2 py-0.5 font-mono text-xs text-ink-dim"
            }
            title={`${change.label}: ${String(change.value)}`}
          >
            {change.key}={String(change.value)}
          </span>
        ))}
      </div>
    </td>
  );
}

function RunSnapshotCell({
  run,
  canRemoveSnapshots,
  onRemoveSnapshot,
}: {
  run: TrainingRun;
  canRemoveSnapshots: boolean;
  onRemoveSnapshot?: (snapshotId: string) => void;
}) {
  const snapshotName = run.snapshotName ?? "";
  const snapshotId = run.snapshotId ?? "";
  if (!snapshotName) {
    return <td className={bodyCellClass}>{emptyCell}</td>;
  }

  return (
    <td className={bodyCellClass}>
      <div className="grid max-w-[14rem] gap-1.5">
        <span className="truncate font-mono text-xs text-ink" title={snapshotName}>
          {snapshotName}
        </span>
        {canRemoveSnapshots && snapshotId && (
          <Button
            variant="ghost"
            className="h-7 justify-start border border-danger-line bg-danger-soft px-2 text-xs text-[#fda4af] hover:bg-[#7f1d2d]/40 hover:text-white"
            onClick={() => onRemoveSnapshot?.(snapshotId)}
            aria-label={`Remove snapshot "${snapshotName}"`}
          >
            <Trash2 className="h-3.5 w-3.5" aria-hidden />
            Remove snapshot &quot;{snapshotName}&quot;
          </Button>
        )}
      </div>
    </td>
  );
}

function RunMetricsCell({ run }: { run: TrainingRun }) {
  const metrics = metricsText(run.metrics);

  return (
    <td className="max-w-[16rem] border-b border-line-soft px-3 py-3 font-mono text-xs text-ink-dim">
      <div className="truncate" title={metrics}>
        {metrics}
      </div>
    </td>
  );
}

function RunArtifactsCell({ run }: { run: TrainingRun }) {
  return (
    <td className={bodyCellClass}>
      {run.logDir ? (
        <button
          type="button"
          aria-label={`Copy log path for run ${run.index}`}
          onClick={() => {
            void navigator.clipboard?.writeText(run.logDir ?? "");
          }}
          className="rounded-[7px] border border-ok/30 bg-ok/10 px-2 py-0.5 text-xs font-semibold text-ok transition hover:bg-ok/15 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          title={run.logDir}
        >
          Log
        </button>
      ) : (
        emptyCell
      )}
    </td>
  );
}

function TrainingRunProgressRow({
  run,
  canRemoveSnapshots,
  onRemoveSnapshot,
  onCommand,
  onFullError,
}: {
  run: TrainingRun;
  canRemoveSnapshots: boolean;
  onRemoveSnapshot?: (snapshotId: string) => void;
  onCommand: (run: TrainingRun) => void;
  onFullError: (run: TrainingRun) => void;
}) {
  return (
    <tr className="align-top">
      <td className="border-b border-line-soft px-3 py-3 font-mono text-xs text-ink-dim">
        {run.index}
      </td>
      <RunStatusCell run={run} onFullError={onFullError} />
      <td className={monoCellClass}>{run.preset}</td>
      <RunSnapshotCell
        run={run}
        canRemoveSnapshots={canRemoveSnapshots}
        onRemoveSnapshot={onRemoveSnapshot}
      />
      <td className={monoCellClass}>{run.dataset}</td>
      <RunChangesCell run={run} />
      <td className={monoCellClass}>{epochText(run)}</td>
      <RunMetricsCell run={run} />
      <RunArtifactsCell run={run} />
      <td className={bodyCellClass}>
        <Button
          variant="secondary"
          className="h-8 px-2.5 text-xs"
          onClick={() => onCommand(run)}
          aria-label={`Command for run ${run.index}`}
        >
          <Terminal className="h-3.5 w-3.5" aria-hidden />
          Command
        </Button>
      </td>
    </tr>
  );
}

export function TrainingProgressDialog({
  plan,
  isLoading,
  error,
  canResample,
  isResampling,
  onResample,
  canRemoveSnapshots = false,
  onRemoveSnapshot,
  onClose,
}: {
  plan: TrainingRunPlan | undefined;
  isLoading: boolean;
  error: string;
  canResample: boolean;
  isResampling: boolean;
  onResample: () => void;
  canRemoveSnapshots?: boolean;
  onRemoveSnapshot?: (snapshotId: string) => void;
  onClose: () => void;
}) {
  const [commandRun, setCommandRun] = useState<TrainingRun | null>(null);
  const [errorRun, setErrorRun] = useState<TrainingRun | null>(null);
  const command = commandRun?.command ?? "";
  const fullErrorText = errorRun?.errorTraceback || errorRun?.error || "";
  const { status: copyStatus, copy } = useCopyToClipboard(command);
  const summary = plan?.summary;
  const summaryText = useMemo(() => {
    if (isLoading) {
      return "Planning...";
    }
    if (error) {
      return "Plan error";
    }
    if (!summary) {
      return "No run plan";
    }
    return `${summary.completedRuns} / ${summary.totalRuns} runs · ${summary.remainingEpochs} epochs left`;
  }, [error, isLoading, summary]);

  const dialog = (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-3 backdrop-blur-sm sm:p-6">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="training-progress-title"
        className="edge full-config-dialog-shell relative flex max-h-[calc(100vh-1.5rem)] w-full max-w-[92rem] flex-col overflow-hidden rounded-card shadow-[0_24px_80px_rgba(0,0,0,0.58)] sm:max-h-[calc(100vh-3rem)]"
      >
        <header className="full-config-dialog-chrome full-config-dialog-header sticky top-0 z-10 border-b border-line-soft px-4 py-3 backdrop-blur sm:px-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <h2 id="training-progress-title" className="text-base font-semibold text-ink">
                Training Progress
              </h2>
              <div className="mt-1 flex min-w-0 flex-wrap items-center gap-1.5 text-xs text-ink-faint">
                <span className="max-w-full truncate font-mono">
                  {plan?.model ?? "No model"}
                </span>
                {plan?.preset && <span aria-hidden>/</span>}
                {plan?.preset && (
                  <span className="max-w-full truncate font-mono">
                    {plan.preset}
                  </span>
                )}
                <span aria-hidden>·</span>
                <span>{summaryText}</span>
              </div>
            </div>
            <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
              {summary && (
                <>
                  <Badge>{summary.totalRuns} runs</Badge>
                  <Badge>{summary.remainingEpochs} epochs left</Badge>
                </>
              )}
              {canResample && (
                <Button
                  variant="secondary"
                  onClick={onResample}
                  disabled={isResampling}
                >
                  <RefreshCw
                    className={isResampling ? "h-4 w-4 animate-spin" : "h-4 w-4"}
                    aria-hidden
                  />
                  Resample
                </Button>
              )}
              <button
                type="button"
                aria-label="Close training progress"
                onClick={onClose}
                className="flex h-9 w-9 items-center justify-center rounded-[10px] border border-line-soft bg-white/[0.025] text-ink-faint transition hover:bg-white/[0.055] hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                <X className="h-4 w-4" aria-hidden />
              </button>
            </div>
          </div>
        </header>

        <div className="full-config-dialog-body min-h-0 flex-1 overflow-auto px-4 py-4 sm:px-5">
          {error ? (
            <div
              role="alert"
              className="rounded-[10px] border border-danger-line bg-danger-soft p-3 text-sm text-[#fda4af]"
            >
              {error}
            </div>
          ) : isLoading ? (
            <div className="rounded-[10px] border border-dashed border-faint bg-white/[0.018] p-4 text-sm text-ink-faint">
              Planning training runs
            </div>
          ) : !plan || plan.runs.length === 0 ? (
            <div className="rounded-[10px] border border-dashed border-faint bg-white/[0.018] p-4 text-sm text-ink-faint">
              No training runs planned
            </div>
          ) : (
            <table className="min-w-[1080px] w-full border-separate border-spacing-0 text-left text-sm">
              <thead className="sticky top-0 z-10 bg-bg-2/95 text-xs uppercase tracking-[0.08em] text-ink-faint">
                <tr>
                  {[
                    "#",
                    "Status",
                    "Preset",
                    "Snapshot",
                    "Dataset",
                    "Search / Config",
                    "Epochs",
                    "Metrics",
                    "Artifacts",
                    "Command",
                  ].map((heading) => (
                    <th
                      key={heading}
                      scope="col"
                      className="border-b border-line-soft px-3 py-2 font-bold"
                    >
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {plan.runs.map((run) => (
                  <TrainingRunProgressRow
                    key={run.id}
                    run={run}
                    canRemoveSnapshots={canRemoveSnapshots}
                    onRemoveSnapshot={onRemoveSnapshot}
                    onCommand={setCommandRun}
                    onFullError={setErrorRun}
                  />
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>
      {commandRun && (
        <TrainingCommandDialog
          model={plan?.model ?? ""}
          preset={`${commandRun.preset} / ${commandRun.dataset}`}
          trainingCommand={command}
          copyStatus={copyStatus}
          onCopy={copy}
          onClose={() => setCommandRun(null)}
        />
      )}
      {errorRun && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 p-3 backdrop-blur-sm sm:p-6">
          <section
            role="dialog"
            aria-modal="true"
            aria-labelledby="training-run-error-title"
            className="edge grid max-h-[calc(100vh-1.5rem)] w-full max-w-5xl grid-rows-[auto_minmax(0,1fr)] overflow-hidden rounded-card shadow-[0_24px_80px_rgba(0,0,0,0.58)] sm:max-h-[calc(100vh-3rem)]"
          >
            <header className="flex items-start justify-between gap-3 border-b border-line-soft px-4 py-3 sm:px-5">
              <div className="min-w-0">
                <h2
                  id="training-run-error-title"
                  className="text-base font-semibold text-ink"
                >
                  Training Error
                </h2>
                <div className="mt-1 flex min-w-0 flex-wrap items-center gap-1.5 text-xs text-ink-faint">
                  <span className="font-mono">run {errorRun.index}</span>
                  <span aria-hidden>/</span>
                  <span className="font-mono">{errorRun.preset}</span>
                  <span aria-hidden>/</span>
                  <span className="font-mono">{errorRun.dataset}</span>
                </div>
              </div>
              <button
                type="button"
                aria-label="Close training error"
                onClick={() => setErrorRun(null)}
                className="flex h-9 w-9 items-center justify-center rounded-[10px] border border-line-soft bg-white/[0.025] text-ink-faint transition hover:bg-white/[0.055] hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                <X className="h-4 w-4" aria-hidden />
              </button>
            </header>
            <div className="min-h-0 overflow-auto p-4 sm:p-5">
              <pre className="min-h-[18rem] whitespace-pre-wrap rounded-[10px] border border-danger-line bg-black/35 p-3 font-mono text-xs leading-5 text-[#fda4af]">
                {fullErrorText}
              </pre>
            </div>
          </section>
        </div>
      )}
    </div>
  );

  if (typeof document === "undefined") {
    return dialog;
  }

  return createPortal(dialog, document.body);
}
