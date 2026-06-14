import { type ReactNode, useId, useState } from "react";
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  CircleSlash,
  Clock,
  LoaderCircle,
  Terminal,
  X,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { cn } from "@/lib/utils";
import { type TrainingRun } from "@/lib/api";

type TrainingProgressTableProps = {
  runs: TrainingRun[];
  onCommand: (run: TrainingRun) => void;
  onFullError: (run: TrainingRun) => void;
  canManageDraftRuns?: boolean;
  onExcludePreset?: (preset: string) => void;
  onExcludeSnapshot?: (snapshotId: string) => void;
};

type StatusMeta = {
  className: string;
  icon: LucideIcon;
  iconClassName?: string;
  tooltip: string;
};

const statusMeta: Record<TrainingRun["status"], StatusMeta> = {
  Pending: {
    className: "border-line bg-control text-ink-dim",
    icon: Clock,
    tooltip: "Pending: this run has not started",
  },
  Running: {
    className: "border-violet/30 bg-violet/15 text-violet",
    icon: LoaderCircle,
    iconClassName: "animate-spin motion-reduce:animate-none",
    tooltip: "Running: this run is currently training",
  },
  Completed: {
    className: "border-ok/30 bg-ok/10 text-ok",
    icon: CheckCircle2,
    tooltip: "Completed: this run finished successfully",
  },
  Failed: {
    className: "border-danger-line bg-danger-soft text-danger-text",
    icon: XCircle,
    tooltip: "Failed: this run stopped with an error",
  },
  Cancelled: {
    className: "border-danger-line bg-danger-soft text-danger-text",
    icon: Ban,
    tooltip: "Cancelled: this run was cancelled",
  },
  Skipped: {
    className: "border-amber/40 bg-amber/[0.12] text-amber",
    icon: CircleSlash,
    tooltip: "Skipped: this run was skipped",
  },
};

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

const bodyCellClass = "border-b border-line-soft px-3 py-3 align-middle";
const monoCellClass = `${bodyCellClass} font-mono text-xs text-ink`;
const emptyCell = <span className="font-mono text-xs text-ink-dim">-</span>;

function HoverTooltip({
  children,
  tooltip,
  tooltipClassName,
}: {
  children: (props: {
    "aria-describedby"?: string;
    onBlur: () => void;
    onFocus: () => void;
    onMouseEnter: () => void;
    onMouseLeave: () => void;
  }) => ReactNode;
  tooltip: string;
  tooltipClassName?: string;
}) {
  const [isTooltipVisible, setIsTooltipVisible] = useState(false);
  const tooltipId = useId();
  const triggerProps = {
    "aria-describedby": isTooltipVisible ? tooltipId : undefined,
    onBlur: () => setIsTooltipVisible(false),
    onFocus: () => setIsTooltipVisible(true),
    onMouseEnter: () => setIsTooltipVisible(true),
    onMouseLeave: () => setIsTooltipVisible(false),
  };

  return (
    <span className="relative inline-flex">
      {children(triggerProps)}
      {isTooltipVisible && (
        <span
          id={tooltipId}
          role="tooltip"
          className={cn(
            "pointer-events-none absolute top-[calc(100%+6px)] z-30 whitespace-nowrap rounded-[7px] border border-line-soft bg-panel px-2 py-1 font-sans text-[11px] font-bold leading-none text-ink shadow-panel",
            tooltipClassName,
          )}
        >
          {tooltip}
        </span>
      )}
    </span>
  );
}

function TooltipIconButton({
  className,
  icon,
  label,
  onClick,
  tooltip,
}: {
  className?: string;
  icon: ReactNode;
  label: string;
  onClick: () => void;
  tooltip: string;
}) {
  return (
    <HoverTooltip tooltip={tooltip} tooltipClassName="right-0">
      {(triggerProps) => (
        <IconButton
          label={label}
          icon={icon}
          size="sm"
          variant="edge"
          className={cn("h-8 w-8 rounded-[7px] active:translate-y-px", className)}
          onClick={onClick}
          {...triggerProps}
        />
      )}
    </HoverTooltip>
  );
}

function TooltipIcon({
  className,
  icon,
  label,
  tooltip,
}: {
  className?: string;
  icon: ReactNode;
  label: string;
  tooltip: string;
}) {
  return (
    <HoverTooltip tooltip={tooltip} tooltipClassName="left-0">
      {(triggerProps) => (
        <span
          aria-label={label}
          role="img"
          tabIndex={0}
          className={cn(
            "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[7px] border transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
            className,
          )}
          {...triggerProps}
        >
          {icon}
        </span>
      )}
    </HoverTooltip>
  );
}

function RunStatusIcon({
  runIndex,
  status,
}: {
  runIndex: number;
  status: TrainingRun["status"];
}) {
  const meta = statusMeta[status];
  const Icon = meta.icon;

  return (
    <TooltipIcon
      label={`Run ${runIndex} status: ${status}`}
      tooltip={meta.tooltip}
      className={meta.className}
      icon={
        <Icon
          className={cn("h-3.5 w-3.5", meta.iconClassName)}
          aria-hidden
        />
      }
    />
  );
}

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
      <RunStatusIcon runIndex={run.index} status={run.status} />
      {run.error && (
        <div className="mt-1 grid max-w-48 gap-1.5 text-xs text-danger-text">
          <span>{run.error}</span>
          {fullError && (
            <Button
              variant="ghost"
              className="h-7 justify-start border border-danger-line bg-danger-soft px-2 text-xs text-danger-text hover:bg-danger-hover/40 hover:text-white"
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

function RunSnapshotCell({ run }: { run: TrainingRun }) {
  const snapshotName = run.snapshotName ?? "";
  if (!snapshotName) {
    return <td className={bodyCellClass}>{emptyCell}</td>;
  }

  return (
    <td className={bodyCellClass}>
      <div className="flex max-w-[14rem] items-center">
        <span className="truncate font-mono text-xs text-ink" title={snapshotName}>
          {snapshotName}
        </span>
      </div>
    </td>
  );
}

function RunMetricsCell({ run }: { run: TrainingRun }) {
  const metrics = metricsText(run.metrics);

  return (
    <td className="max-w-[16rem] border-b border-line-soft px-3 py-3 align-middle font-mono text-xs text-ink-dim">
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
  onCommand,
  onFullError,
  canManageDraftRuns,
  onExcludePreset,
  onExcludeSnapshot,
}: {
  run: TrainingRun;
  onCommand: (run: TrainingRun) => void;
  onFullError: (run: TrainingRun) => void;
  canManageDraftRuns: boolean;
  onExcludePreset?: (preset: string) => void;
  onExcludeSnapshot?: (snapshotId: string) => void;
}) {
  const snapshotId = run.snapshotId ?? "";
  const canExcludeSnapshot = Boolean(snapshotId && onExcludeSnapshot);
  const canExcludePreset = Boolean(!snapshotId && onExcludePreset);
  const canExcludeRun =
    canManageDraftRuns && (canExcludeSnapshot || canExcludePreset);
  const removeLabel = snapshotId
    ? `Remove snapshot ${run.snapshotName ?? snapshotId} from this run plan`
    : `Remove preset ${run.preset} from this run plan`;
  const excludeRun = () => {
    if (snapshotId) {
      onExcludeSnapshot?.(snapshotId);
      return;
    }
    onExcludePreset?.(run.preset);
  };

  return (
    <tr className="align-middle">
      <td className="border-b border-line-soft px-3 py-3 align-middle font-mono text-xs text-ink-dim">
        {run.index}
      </td>
      <RunStatusCell run={run} onFullError={onFullError} />
      <td className={monoCellClass}>{run.preset}</td>
      <RunSnapshotCell run={run} />
      <td className={monoCellClass}>{run.dataset}</td>
      <RunChangesCell run={run} />
      <td className={monoCellClass}>{epochText(run)}</td>
      <RunMetricsCell run={run} />
      <RunArtifactsCell run={run} />
      <td className={bodyCellClass}>
        <div className="flex items-center gap-1.5">
          <TooltipIconButton
            label={`Command for run ${run.index}`}
            tooltip="Show command for this run"
            icon={<Terminal className="h-3.5 w-3.5" aria-hidden />}
            onClick={() => onCommand(run)}
          />
          {canExcludeRun && (
            <TooltipIconButton
              label={removeLabel}
              tooltip="Remove from this run plan"
              icon={<X className="h-3.5 w-3.5" aria-hidden />}
              className="bg-transparent text-ink-faint hover:border-danger-line hover:bg-danger-soft hover:text-danger-text"
              onClick={excludeRun}
            />
          )}
        </div>
      </td>
    </tr>
  );
}

export function TrainingProgressTable({
  runs,
  onCommand,
  onFullError,
  canManageDraftRuns = false,
  onExcludePreset,
  onExcludeSnapshot,
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
            "Actions",
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
            onCommand={onCommand}
            onFullError={onFullError}
            canManageDraftRuns={canManageDraftRuns}
            onExcludePreset={onExcludePreset}
            onExcludeSnapshot={onExcludeSnapshot}
          />
        ))}
      </tbody>
    </table>
  );
}
