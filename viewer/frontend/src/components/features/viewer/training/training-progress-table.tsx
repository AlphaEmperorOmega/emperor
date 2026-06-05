import { AlertTriangle, Terminal, Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { type TrainingRun } from "@/lib/api";

type TrainingProgressTableProps = {
  runs: TrainingRun[];
  canRemoveSnapshots: boolean;
  onRemoveSnapshot?: (snapshotId: string) => void;
  onCommand: (run: TrainingRun) => void;
  onFullError: (run: TrainingRun) => void;
};

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

export function TrainingProgressTable({
  runs,
  canRemoveSnapshots,
  onRemoveSnapshot,
  onCommand,
  onFullError,
}: TrainingProgressTableProps) {
  return (
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
        {runs.map((run) => (
          <TrainingRunProgressRow
            key={run.id}
            run={run}
            canRemoveSnapshots={canRemoveSnapshots}
            onRemoveSnapshot={onRemoveSnapshot}
            onCommand={onCommand}
            onFullError={onFullError}
          />
        ))}
      </tbody>
    </table>
  );
}
