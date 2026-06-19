import { AlertTriangle, ListChecks, Loader2, Play, Timer } from "lucide-react";
import { StatusPill } from "@/features/viewer/components/status-pill";
import {
  selectTrainingRunForDisplay,
  trainingRunDisplayLabel,
} from "@/features/viewer/components/training/training-run-display";
import { type TrainingJob, type TrainingRun, type TrainingRunPlan } from "@/lib/api";
import { cn } from "@/lib/utils";

export type TrainingFooterRunSummaryProps = {
  plan?: TrainingRunPlan;
  job?: TrainingJob;
  isLoading?: boolean;
  error?: string;
  className?: string;
};

function runTone(status: TrainingRun["status"]): "neutral" | "good" | "warn" | "danger" {
  if (status === "Completed") {
    return "good";
  }
  if (status === "Failed" || status === "Cancelled") {
    return "danger";
  }
  if (status === "Skipped") {
    return "warn";
  }
  return "neutral";
}

function summaryLabel({
  error,
  isLoading,
  job,
  plan,
}: Required<Pick<TrainingFooterRunSummaryProps, "isLoading" | "error">> &
  Pick<TrainingFooterRunSummaryProps, "job" | "plan">) {
  if (isLoading) {
    return "Training run summary: planning training runs";
  }
  if (error) {
    return `Training run summary: plan error: ${error}`;
  }
  if (!plan) {
    return "Training run summary: no run plan";
  }

  const run = selectTrainingRunForDisplay({ plan, job });
  const runText = run
    ? `${trainingRunDisplayLabel(run, job)} #${run.index} ${run.currentEpoch} / ${
        run.totalEpochs || "-"
      } epochs`
    : "No current run";

  return `Training run summary: ${plan.summary.completedRuns} / ${plan.summary.totalRuns} runs; ${plan.summary.completedEpochs} / ${plan.summary.totalEpochs} epochs; ${runText}`;
}

export function TrainingFooterRunSummary({
  className,
  error = "",
  isLoading = false,
  job,
  plan,
}: TrainingFooterRunSummaryProps) {
  const run = selectTrainingRunForDisplay({ plan, job });
  const ariaLabel = summaryLabel({ error, isLoading, job, plan });

  return (
    <div
      role="status"
      aria-label={ariaLabel}
      title={error || undefined}
      className={cn(
        "flex min-w-[13rem] max-w-[34rem] flex-wrap items-center gap-1.5",
        className,
      )}
    >
      {isLoading ? (
        <StatusPill
          icon={<Loader2 className="h-4 w-4 animate-spin" />}
          label="runs"
          value="planning"
          tone="warn"
        />
      ) : error ? (
        <StatusPill
          icon={<AlertTriangle className="h-4 w-4" />}
          label="plan"
          value="error"
          tone="danger"
        />
      ) : plan ? (
        <>
          <StatusPill
            icon={<ListChecks className="h-4 w-4" />}
            label="runs"
            value={`${plan.summary.completedRuns} / ${plan.summary.totalRuns}`}
            tone={
              plan.summary.completedRuns === plan.summary.totalRuns ? "good" : "neutral"
            }
          />
          <StatusPill
            icon={<Timer className="h-4 w-4" />}
            label="epochs"
            value={`${plan.summary.completedEpochs} / ${plan.summary.totalEpochs}`}
            tone={
              plan.summary.completedEpochs === plan.summary.totalEpochs
                ? "good"
                : "neutral"
            }
          />
          {run && (
            <StatusPill
              icon={<Play className="h-4 w-4" />}
              label={trainingRunDisplayLabel(run, job).toLowerCase()}
              value={`#${run.index} ${run.currentEpoch} / ${
                run.totalEpochs || "-"
              } epochs`}
              tone={runTone(run.status)}
              className="max-w-full"
            />
          )}
        </>
      ) : (
        <StatusPill
          icon={<ListChecks className="h-4 w-4" />}
          label="runs"
          value="no plan"
        />
      )}
    </div>
  );
}
