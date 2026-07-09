import { useState } from "react";
import { Info, ListChecks } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { IconButton } from "@/components/ui/icon-button";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { StatChip, type StatChipTone } from "@/features/workbench/components/shared/stat-chip";
import { SurfacePanel } from "@/features/workbench/components/shared/surface-panel";
import { TrainingMetricInfoDialog } from "@/features/workbench/components/shared/training-metric-info-dialog";
import {
  formatTrainingMetricValue,
  formatTrainingMetricsText,
  isTerminalTrainingJob,
  selectTrainingRunForDisplay,
  trainingRunDisplayLabel,
} from "@/features/workbench/components/training/training-run-display";
import { type TrainingJob, type TrainingRun, type TrainingRunPlan } from "@/lib/api";
import {
  type TrainingSearchLockSummary,
  type TrainingSearchState,
} from "@/lib/training-search";
import { cn } from "@/lib/utils";

export type TrainingRunPlanCardProps = {
  plan?: TrainingRunPlan;
  job?: TrainingJob;
  isPlanning?: boolean;
  planError?: string;
  trainingError?: string;
  effectiveTrainingSearch: TrainingSearchState;
  searchModeLabel: string;
  activeSearchAxisCount: number;
  searchConflictCount: number;
  searchLockSummary?: TrainingSearchLockSummary;
  trainingSearchValidation: { ready: boolean; message: string };
  displayedRunCount: number;
  requiresLargeGridConfirmation: boolean;
  selectedMonitorCount: number;
  presetCountLabel?: string;
  datasetCountLabel?: string;
};

const footerIconClass = "h-[15px] w-[15px] text-violet";

function cardTitle(job?: TrainingJob) {
  if (!job) {
    return "Run Plan";
  }
  if (job.status === "running" || job.status === "queued") {
    return "Active Run";
  }
  if (isTerminalTrainingJob(job)) {
    return "Results";
  }
  return "Run Plan";
}

function jobTone(status?: string): StatChipTone {
  if (status === "completed") {
    return "success";
  }
  if (status === "failed" || status === "cancelled") {
    return "danger";
  }
  if (status === "running" || status === "queued") {
    return "violet";
  }
  return "default";
}

function runTone(status: TrainingRun["status"]) {
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

function headerDetail({
  isPlanning,
  job,
  plan,
}: {
  isPlanning: boolean;
  job?: TrainingJob;
  plan?: TrainingRunPlan;
}) {
  if (isPlanning) {
    return <StatChip tone="violet">planning</StatChip>;
  }
  if (job) {
    return <StatChip tone={jobTone(job.status)}>{job.status}</StatChip>;
  }
  if (plan) {
    return (
      <StatChip>
        {plan.summary.totalRuns} run
        {plan.summary.totalRuns === 1 ? "" : "s"}
      </StatChip>
    );
  }
  return <StatChip>No plan</StatChip>;
}

function epochStepText(run?: TrainingRun, job?: TrainingJob) {
  if (job?.epoch !== null && job?.epoch !== undefined) {
    return `epoch ${job.epoch}${
      job.step !== null && job.step !== undefined ? ` / step ${job.step}` : ""
    }`;
  }
  if (job?.step !== null && job?.step !== undefined) {
    return `step ${job.step}`;
  }
  if (!run) {
    return "waiting";
  }
  return `${run.currentEpoch} / ${run.totalEpochs || "-"} epochs`;
}

function latestMetricEntries(job?: TrainingJob, run?: TrainingRun) {
  const jobEntries = Object.entries(job?.metrics ?? {});
  if (jobEntries.length > 0) {
    return jobEntries;
  }
  return Object.entries(run?.metrics ?? {});
}

function logDirectory(job?: TrainingJob, run?: TrainingRun) {
  return job?.logDir ?? run?.logDir ?? "";
}

function SummaryChips({
  plan,
  job,
  presetCountLabel,
  datasetCountLabel,
}: {
  plan: TrainingRunPlan;
  job?: TrainingJob;
  presetCountLabel?: string;
  datasetCountLabel?: string;
}) {
  const summary = plan.summary;
  const totalRunLabel = job
    ? `${summary.totalRuns} run${summary.totalRuns === 1 ? "" : "s"}`
    : `${summary.totalRuns} planned run${summary.totalRuns === 1 ? "" : "s"}`;
  const failedCount = summary.failedRuns;

  return (
    <div className="flex flex-wrap gap-1.5">
      <StatChip>{totalRunLabel}</StatChip>
      {presetCountLabel && <StatChip>{presetCountLabel}</StatChip>}
      {datasetCountLabel && <StatChip>{datasetCountLabel}</StatChip>}
      <StatChip tone={summary.completedRuns > 0 ? "success" : "default"}>
        {summary.completedRuns} completed
      </StatChip>
      <StatChip tone={summary.runningRuns > 0 ? "violet" : "default"}>
        {summary.runningRuns} running
      </StatChip>
      <StatChip>{summary.pendingRuns} pending</StatChip>
      <StatChip tone={failedCount > 0 ? "danger" : "default"}>
        {failedCount} failed
      </StatChip>
      {summary.cancelledRuns > 0 && (
        <StatChip tone="danger">{summary.cancelledRuns} cancelled</StatChip>
      )}
      {summary.skippedRuns > 0 && (
        <StatChip tone="warning">{summary.skippedRuns} skipped</StatChip>
      )}
      <StatChip>{summary.totalEpochs} epochs</StatChip>
      <StatChip tone={summary.remainingEpochs === 0 ? "success" : "default"}>
        {summary.remainingEpochs} left
      </StatChip>
    </div>
  );
}

function RunRow({ run, job }: { run: TrainingRun; job?: TrainingJob }) {
  return (
    <div className="grid gap-1 rounded-[8px] border border-line-soft bg-black/15 px-2.5 py-2 text-xs">
      <div className="flex min-w-0 flex-wrap items-center gap-1.5">
        <span className="font-semibold text-ink-dim">
          {trainingRunDisplayLabel(run, job)}
        </span>
        <Badge>#{run.index}</Badge>
        <Badge variant={runTone(run.status)}>{run.status}</Badge>
      </div>
      <div className="grid min-w-0 gap-1 font-mono text-ink-dim sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]">
        <span className="truncate" title={run.preset}>
          {run.preset}
        </span>
        <span className="truncate" title={run.dataset}>
          {run.dataset}
        </span>
        <span className="whitespace-nowrap text-violet-muted">
          {epochStepText(run, job)}
        </span>
      </div>
      {run.error && (
        <div
          role="alert"
          className="rounded-[7px] border border-danger-line bg-danger-soft px-2 py-1 text-danger-text"
          title={run.errorTraceback ?? run.error}
        >
          {run.error}
        </div>
      )}
    </div>
  );
}

function MetricSummary({ job, run }: { job?: TrainingJob; run?: TrainingRun }) {
  const [selectedMetric, setSelectedMetric] = useState<{
    key: string;
    value: unknown;
  } | null>(null);
  const entries = latestMetricEntries(job, run);
  if (entries.length === 0) {
    return (
      <div className="rounded-[8px] border border-line-soft bg-white/[0.018] px-2.5 py-2 text-xs text-ink-faint">
        No metrics yet
      </div>
    );
  }
  const visibleEntries = entries.slice(0, 3);
  const text = formatTrainingMetricsText(Object.fromEntries(entries), 3);

  return (
    <>
      <div className="grid gap-1 rounded-[8px] border border-line-soft bg-white/[0.018] px-2.5 py-2 text-xs">
        <span className="font-semibold text-ink-dim">Latest metrics</span>
        <div className="flex min-w-0 flex-wrap gap-1.5" title={text}>
          {visibleEntries.map(([key, value]) => (
            <span
              key={key}
              className="inline-flex max-w-full min-w-0 items-center gap-0.5"
            >
              <Badge className="max-w-[calc(100%-1.375rem)]">
                <span className="truncate">
                  {key}={formatTrainingMetricValue(value)}
                </span>
              </Badge>
              <IconButton
                label={`Explain metric ${key}`}
                title={`Explain ${key}`}
                size="sm"
                variant="ghost"
                aria-haspopup="dialog"
                aria-expanded={selectedMetric?.key === key ? true : undefined}
                className="h-5 w-5 shrink-0 rounded-[6px] border border-line-soft bg-white/[0.018] text-ink-faint hover:border-violet/40 hover:bg-violet/10 hover:text-violet focus-visible:ring-2 focus-visible:ring-focus"
                icon={<Info className="h-3.5 w-3.5" aria-hidden />}
                onClick={() => setSelectedMetric({ key, value })}
              />
            </span>
          ))}
        </div>
      </div>
      {selectedMetric && (
        <TrainingMetricInfoDialog
          metricKey={selectedMetric.key}
          valueLabel={formatTrainingMetricValue(selectedMetric.value)}
          onClose={() => setSelectedMetric(null)}
        />
      )}
    </>
  );
}

function LogDirectory({ value }: { value: string }) {
  if (!value) {
    return null;
  }
  return (
    <div className="truncate rounded-[8px] border border-ok/30 bg-ok/10 px-2.5 py-2 font-mono text-xs text-ok" title={value}>
      {value}
    </div>
  );
}

function Notices({
  effectiveTrainingSearch,
  searchConflictCount,
  searchLockSummary,
  trainingSearchValidation,
  displayedRunCount,
  requiresLargeGridConfirmation,
  selectedMonitorCount,
  planError,
  trainingError,
}: Pick<
  TrainingRunPlanCardProps,
  | "effectiveTrainingSearch"
  | "searchConflictCount"
  | "searchLockSummary"
  | "trainingSearchValidation"
  | "displayedRunCount"
  | "requiresLargeGridConfirmation"
  | "selectedMonitorCount"
  | "planError"
  | "trainingError"
>) {
  const searchLockNotice =
    effectiveTrainingSearch.mode !== "off"
      ? searchLockSummary?.skippedSelectedAxisMessage ||
        searchLockSummary?.lockedAxesMessage ||
        ""
      : "";
  return (
    <div className="grid gap-1.5">
      {planError && (
        <InlineStatus tone="danger" compact role="alert" className="px-2.5 py-2 text-xs">
          {planError}
        </InlineStatus>
      )}
      {trainingError && (
        <InlineStatus tone="danger" compact role="alert" className="px-2.5 py-2 text-xs">
          {trainingError}
        </InlineStatus>
      )}
      {effectiveTrainingSearch.mode !== "off" &&
        !trainingSearchValidation.ready && (
          <InlineStatus tone="danger" compact role="alert" className="px-2.5 py-2 text-xs">
            {trainingSearchValidation.message}
          </InlineStatus>
        )}
      {searchConflictCount > 0 && (
        <InlineStatus tone="warning" compact className="px-2.5 py-2 text-xs">
          {searchConflictCount} override
          {searchConflictCount === 1 ? "" : "s"} replaced by search.
        </InlineStatus>
      )}
      {searchLockNotice && (
        <InlineStatus tone="warning" compact className="px-2.5 py-2 text-xs">
          {searchLockNotice}
        </InlineStatus>
      )}
      {requiresLargeGridConfirmation && (
        <InlineStatus tone="warning" compact className="px-2.5 py-2 text-xs">
          {displayedRunCount} planned runs require confirmation before start.
        </InlineStatus>
      )}
      {selectedMonitorCount === 0 && (
        <InlineStatus compact className="px-2.5 py-2 text-xs">
          No monitors selected.
        </InlineStatus>
      )}
    </div>
  );
}

export function TrainingRunPlanCard({
  plan,
  job,
  isPlanning = false,
  planError = "",
  trainingError = "",
  effectiveTrainingSearch,
  searchModeLabel,
  activeSearchAxisCount,
  searchConflictCount,
  searchLockSummary,
  trainingSearchValidation,
  displayedRunCount,
  requiresLargeGridConfirmation,
  selectedMonitorCount,
  presetCountLabel,
  datasetCountLabel,
}: TrainingRunPlanCardProps) {
  const selectedRun = selectTrainingRunForDisplay({ plan, job });
  const metrics = latestMetricEntries(job, selectedRun);
  const directory = logDirectory(job, selectedRun);

  return (
    <SurfacePanel
      icon={<ListChecks className={footerIconClass} aria-hidden />}
      title={cardTitle(job)}
      detail={headerDetail({ isPlanning, job, plan })}
    >
      <div className={cn("grid gap-2", isPlanning && "opacity-90")}>
        {plan ? (
          <>
            <SummaryChips
              plan={plan}
              job={job}
              presetCountLabel={presetCountLabel}
              datasetCountLabel={datasetCountLabel}
            />
            <div className="flex flex-wrap gap-1.5">
              {effectiveTrainingSearch.mode !== "off" && (
                <>
                  <Badge variant="violet">{searchModeLabel} search</Badge>
                  <Badge>{activeSearchAxisCount} axes</Badge>
                </>
              )}
            </div>
            {selectedRun && <RunRow run={selectedRun} job={job} />}
            <MetricSummary job={job} run={selectedRun} />
            <LogDirectory value={directory} />
          </>
        ) : (
          <InlineStatus busy={isPlanning} compact className="px-2.5 py-2 text-xs">
            {isPlanning
              ? "Building run plan..."
              : "No run plan yet. Select a trainable target to preview runs."}
          </InlineStatus>
        )}
        {metrics.length > 3 && (
          <span className="text-xs text-ink-faint">
            +{metrics.length - 3} more metrics in the run list
          </span>
        )}
        <Notices
          effectiveTrainingSearch={effectiveTrainingSearch}
          searchConflictCount={searchConflictCount}
          searchLockSummary={searchLockSummary}
          trainingSearchValidation={trainingSearchValidation}
          displayedRunCount={displayedRunCount}
          requiresLargeGridConfirmation={requiresLargeGridConfirmation}
          selectedMonitorCount={selectedMonitorCount}
          planError={planError}
          trainingError={trainingError}
        />
      </div>
    </SurfacePanel>
  );
}
