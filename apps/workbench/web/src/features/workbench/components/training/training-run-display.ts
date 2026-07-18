import {
  Ban,
  CheckCircle2,
  CircleSlash,
  Clock,
  LoaderCircle,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import type {
  TrainingJob,
  TrainingRun,
  TrainingRunPlan,
} from "@/lib/api/training-jobs";
import { formatSignificantNumber } from "@/lib/format";

type StatusMeta = {
  className: string;
  icon: LucideIcon;
  iconClassName?: string;
  tooltip: string;
};

const TERMINAL_JOB_STATUSES = new Set(["completed", "failed", "cancelled"]);

export const trainingRunStatusMeta: Record<TrainingRun["status"], StatusMeta> = {
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

export function isTerminalTrainingJob(job?: Pick<TrainingJob, "status">) {
  return Boolean(job && TERMINAL_JOB_STATUSES.has(job.status));
}

export function selectTrainingRunForDisplay({
  job,
  plan,
}: {
  job?: Pick<TrainingJob, "status">;
  plan?: TrainingRunPlan;
}) {
  const runs = plan?.runs ?? [];
  const runningRun = runs.find((run) => run.status === "Running");
  if (runningRun) {
    return runningRun;
  }
  if (isTerminalTrainingJob(job)) {
    const terminalErrorRun = runs.find(
      (run) => run.status === "Failed" || run.status === "Cancelled",
    );
    if (terminalErrorRun) {
      return terminalErrorRun;
    }
  }
  const pendingRun = runs.find((run) => run.status === "Pending");
  if (pendingRun) {
    return pendingRun;
  }
  const completedRuns = runs.filter((run) => run.status === "Completed");
  return completedRuns[completedRuns.length - 1] ?? runs[runs.length - 1];
}

export function trainingRunDisplayLabel(
  run: TrainingRun,
  job?: Pick<TrainingJob, "status">,
) {
  if (run.status === "Running") {
    return "Active run";
  }
  if (run.status === "Pending" || job?.status === "queued") {
    return "Next run";
  }
  return "Result run";
}

export function getTrainingRunDraftRemoval({
  run,
  canManageDraftRuns,
  onExcludePreset,
  onExcludeSnapshot,
}: {
  run: TrainingRun;
  canManageDraftRuns: boolean;
  onExcludePreset?: (preset: string) => void;
  onExcludeSnapshot?: (snapshotId: string) => void;
}) {
  const snapshotId = run.snapshotId ?? "";
  const canExcludeSnapshot = Boolean(snapshotId && onExcludeSnapshot);
  const canExcludePreset = Boolean(!snapshotId && onExcludePreset);
  const canExcludeRun =
    canManageDraftRuns && (canExcludeSnapshot || canExcludePreset);

  if (!canExcludeRun) {
    return null;
  }

  return {
    label: snapshotId
      ? `Remove snapshot ${run.snapshotName ?? snapshotId} from this run plan`
      : `Remove preset ${run.preset} from this run plan`,
    remove: () => {
      if (snapshotId) {
        onExcludeSnapshot?.(snapshotId);
        return;
      }
      onExcludePreset?.(run.preset);
    },
  };
}

export function formatTrainingMetricValue(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return formatSignificantNumber(value);
  }
  return String(value);
}

export function formatTrainingMetricsText(
  metrics: Record<string, unknown>,
  limit = 2,
) {
  const entries = Object.entries(metrics);
  if (entries.length === 0) {
    return "-";
  }
  return entries
    .slice(0, limit)
    .map(([key, value]) => `${key}=${formatTrainingMetricValue(value)}`)
    .join("  ");
}
