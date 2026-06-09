import {
  AlertTriangle,
  CheckCircle2,
  EyeOff,
  FileText,
  FlaskConical,
  Loader2,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { EdgeCard } from "@/components/ui/edge-card";
import { Select } from "@/components/ui/select";
import { formatRunTimestamp } from "@/lib/format";
import { groupModelLogRunsByExperiment } from "@/lib/historical-monitor-runs";
import {
  type HistoricalParameterSummaryState,
  type ParameterSummary,
  type ParameterSummaryCounts,
} from "@/lib/parameter-summary";
import { cn, errorMessage } from "@/lib/utils";
import { useHistoricalRuns } from "@/components/features/viewer/providers/viewer-providers";
import { InlineStatus } from "@/components/features/viewer/shared/inline-status";
import { LabeledField } from "@/components/features/viewer/shared/labeled-field";
import { SectionHeading } from "@/components/features/viewer/shared/section-heading";

function pluralize(count: number, singular: string, plural: string) {
  return `${count} ${count === 1 ? singular : plural}`;
}

type SummaryCounterKind = keyof ParameterSummaryCounts;

type SummaryCounterMeta = {
  label: string;
  icon: LucideIcon;
  className: string;
};

const summaryCounterMeta: Record<SummaryCounterKind, SummaryCounterMeta> = {
  updated: {
    label: "Updated",
    icon: CheckCircle2,
    className: "border-ok/35 bg-ok/10 text-ok",
  },
  unchanged: {
    label: "Unchanged",
    icon: XCircle,
    className: "border-danger-line bg-danger-soft text-[#fda4af]",
  },
  mixed: {
    label: "Mixed",
    icon: AlertTriangle,
    className: "border-amber/40 bg-amber/[0.12] text-amber",
  },
  notTracked: {
    label: "Not tracked",
    icon: EyeOff,
    className: "border-line bg-white/[0.03] text-ink-faint",
  },
};

const summarySeverityClassNames: Record<ParameterSummary["severity"], string> = {
  danger: "border-danger-line bg-danger-soft",
  warning: "border-amber/35 bg-amber/[0.08]",
  "not-tracked": "border-line-soft bg-white/[0.035]",
  success: "border-ok/35 bg-ok/[0.08]",
};

function summaryBreakdown(summary: ParameterSummary) {
  return `${summary.counts.updated} updated, ${summary.counts.unchanged} unchanged, ${summary.counts.mixed} mixed, ${summary.counts.notTracked} not tracked (${summary.breakdown.missing} missing, ${summary.breakdown.unknown} unknown)`;
}

function summaryCounterValue(
  state: HistoricalParameterSummaryState | undefined,
  kind: SummaryCounterKind,
) {
  if (state?.summary) {
    return String(state.summary.counts[kind]);
  }
  if (state?.isLoading) {
    return "...";
  }
  return "-";
}

function summaryCounterLabel(
  state: HistoricalParameterSummaryState | undefined,
  kind: SummaryCounterKind,
) {
  const label = summaryCounterMeta[kind].label;
  if (state?.summary) {
    const count = state.summary.counts[kind];
    return `${label} parameters: ${count} of ${state.summary.total}. Breakdown: ${summaryBreakdown(state.summary)}.`;
  }
  if (state?.isLoading) {
    return `${label} parameters: summary loading.`;
  }
  return `${label} parameters: summary unavailable.`;
}

function SummaryCounter({
  kind,
  state,
  onSelect,
}: {
  kind: SummaryCounterKind;
  state: HistoricalParameterSummaryState | undefined;
  onSelect: () => void;
}) {
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const meta = summaryCounterMeta[kind];
  const Icon = meta.icon;
  const summary = state?.summary;
  const label = summaryCounterLabel(state, kind);
  const tooltipText = summary
    ? `${meta.label}: ${summary.counts[kind]} of ${summary.total}. ${summaryBreakdown(summary)}.`
    : state?.isLoading
      ? `${meta.label}: summary loading.`
      : `${meta.label}: summary unavailable.`;

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        aria-label={label}
        onBlur={() => setTooltipVisible(false)}
        onClick={(event) => {
          event.stopPropagation();
          onSelect();
        }}
        onFocus={() => setTooltipVisible(true)}
        onMouseEnter={() => setTooltipVisible(true)}
        onMouseLeave={() => setTooltipVisible(false)}
        className={cn(
          "inline-flex h-7 min-w-11 flex-row items-center justify-center gap-1 whitespace-nowrap rounded-[8px] border px-1.5 text-[11px] font-bold transition focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
          meta.className,
        )}
      >
        <Icon className="h-3.5 w-3.5 shrink-0" aria-hidden />
        <span className="inline-block font-mono leading-none">
          {summaryCounterValue(state, kind)}
        </span>
      </button>
      {tooltipVisible && (
        <span
          role="tooltip"
          className="pointer-events-none absolute bottom-[calc(100%+6px)] right-0 z-30 w-64 rounded-[8px] border border-line-soft bg-panel px-2.5 py-2 text-left text-[11px] font-semibold leading-4 text-ink shadow-panel"
        >
          {tooltipText}
        </span>
      )}
    </span>
  );
}

export function ModelExperimentsPanel() {
  const {
    visibleHistoricalRuns: runs,
    historicalPresetOptions: presetOptions,
    selectedHistoricalPreset: selectedPreset,
    setSelectedHistoricalPreset: onSelectPreset,
    selectedLogRunId: selectedRunId,
    selectLogRun: onSelectRun,
    historicalParameterSummariesByRunId,
    experimentsLoading,
    experimentsError,
  } = useHistoricalRuns();
  const isLoading = experimentsLoading;
  const isError = Boolean(experimentsError);
  const error = experimentsError;
  const groups = groupModelLogRunsByExperiment(runs);

  return (
    <EdgeCard className="rounded-card p-4">
      <div className="flex items-center justify-between gap-3">
        <SectionHeading
          as="h2"
          className="min-w-0 normal-case tracking-normal text-ink"
          icon={<FlaskConical className="h-4 w-4 shrink-0 text-violet" aria-hidden />}
          title={<span className="min-w-0 truncate text-sm font-bold text-ink">Experiments</span>}
        />
        <Badge>{runs.length}</Badge>
      </div>

      <div className="mt-3 grid gap-2 rounded-[10px] border border-line-soft bg-black/18 p-2">
        <LabeledField label="Run preset">
          <Select
            value={selectedPreset}
            onChange={(event) => onSelectPreset(event.target.value)}
            disabled={isLoading || presetOptions.length === 0}
            className="h-9 text-xs"
          >
            <option value="">All presets</option>
            {presetOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label} ({option.count})
              </option>
            ))}
          </Select>
        </LabeledField>
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
          <InlineStatus compact>
            {presetOptions.length === 0
              ? "No experiments with layer monitor data"
              : "No runs for this preset"}
          </InlineStatus>
        )}

        {!isLoading &&
          !isError &&
          groups.map((group) => (
            <section key={group.experiment} className="grid gap-1.5">
              <SectionHeading
                className="justify-between px-1 text-[11px] font-bold uppercase tracking-normal text-ink-dim"
                title={<span className="truncate">{group.experiment}</span>}
                count={<span className="font-mono">{group.runs.length}</span>}
              />
              <div className="grid gap-1.5">
                {group.runs.map((run) => {
                  const selected = run.id === selectedRunId;
                  const timestamp = formatRunTimestamp(run.timestamp ?? run.version);
                  const summaryState = historicalParameterSummariesByRunId.get(run.id);
                  const severity = summaryState?.isError
                    ? undefined
                    : summaryState?.summary?.severity;
                  return (
                    <div
                      key={run.id}
                      data-testid={`experiment-run-card-${run.id}`}
                      className={cn(
                        "relative grid min-h-[98px] gap-2 rounded-[10px] border px-3 py-2 text-left transition",
                        severity
                          ? summarySeverityClassNames[severity]
                          : "border-line-soft bg-black/18",
                        selected
                          ? "ring-2 ring-violet/45 shadow-[inset_0_0_0_1px_rgba(167,139,250,0.18)]"
                          : "hover:border-line hover:bg-white/[0.04]",
                      )}
                    >
                      <button
                        type="button"
                        aria-pressed={selected}
                        aria-label={`Select experiment run ${run.experiment} ${run.preset} ${run.dataset} ${timestamp}`}
                        onClick={() => onSelectRun(run.id)}
                        className="absolute inset-0 z-0 rounded-[10px] border-0 bg-transparent p-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                      />
                      <div className="pointer-events-none relative z-10 grid min-w-0 gap-1.5">
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
                          <span className="font-mono">{timestamp}</span>
                        </div>
                        <div className="flex min-w-0 items-center gap-1 text-[11px] text-ink-faint">
                          <FileText className="h-3.5 w-3.5 shrink-0" aria-hidden />
                          <span>{pluralize(run.eventFileCount, "event file", "event files")}</span>
                          <span>&middot;</span>
                          <span>{run.version}</span>
                        </div>
                      </div>
                      <div className="relative z-20 flex min-w-0 flex-wrap items-center gap-1">
                        {(Object.keys(summaryCounterMeta) as SummaryCounterKind[]).map(
                          (kind) => (
                            <SummaryCounter
                              key={kind}
                              kind={kind}
                              state={summaryState}
                              onSelect={() => onSelectRun(run.id)}
                            />
                          ),
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          ))}
      </div>
    </EdgeCard>
  );
}
