import { AlertTriangle, ListChecks, Loader2 } from "lucide-react";
import { StatChip, type StatChipTone } from "@/features/viewer/components/shared/stat-chip";
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

function runTone(status: TrainingRun["status"]): StatChipTone {
  if (status === "Completed") {
    return "success";
  }
  if (status === "Failed" || status === "Cancelled") {
    return "danger";
  }
  if (status === "Running") {
    return "violet";
  }
  if (status === "Skipped") {
    return "warning";
  }
  return "default";
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
        "flex min-w-[13rem] max-w-[34rem] flex-wrap items-center gap-1.5 rounded-[10px] border border-line bg-white/[0.025] px-2 py-1.5",
        className,
      )}
    >
      {isLoading ? (
        <>
          <Loader2 className="h-3.5 w-3.5 animate-spin text-violet" aria-hidden />
          <StatChip tone="violet" size="xs">
            Planning runs
          </StatChip>
        </>
      ) : error ? (
        <>
          <AlertTriangle className="h-3.5 w-3.5 text-danger-text" aria-hidden />
          <StatChip tone="danger" size="xs">
            Plan error
          </StatChip>
        </>
      ) : plan ? (
        <>
          <ListChecks className="h-3.5 w-3.5 text-violet" aria-hidden />
          <StatChip
            tone={plan.summary.completedRuns === plan.summary.totalRuns ? "success" : "default"}
            size="xs"
          >
            Runs {plan.summary.completedRuns} / {plan.summary.totalRuns}
          </StatChip>
          <StatChip
            tone={
              plan.summary.completedEpochs === plan.summary.totalEpochs
                ? "success"
                : "default"
            }
            size="xs"
          >
            Epochs {plan.summary.completedEpochs} / {plan.summary.totalEpochs}
          </StatChip>
          {run && (
            <StatChip tone={runTone(run.status)} size="xs" className="max-w-full">
              <span className="whitespace-nowrap">
                {trainingRunDisplayLabel(run, job)} #{run.index}{" "}
                {run.currentEpoch} / {run.totalEpochs || "-"} epochs
              </span>
            </StatChip>
          )}
        </>
      ) : (
        <>
          <ListChecks className="h-3.5 w-3.5 text-ink-faint" aria-hidden />
          <StatChip size="xs">No run plan</StatChip>
        </>
      )}
    </div>
  );
}
