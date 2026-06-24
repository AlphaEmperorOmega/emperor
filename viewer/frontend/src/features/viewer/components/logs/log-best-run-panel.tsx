"use client";

import { AlertTriangle, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { ErrorPanel } from "@/features/viewer/components/error-panel";
import {
  SelectOnlyDropdown,
  type SelectOnlyDropdownOption,
} from "@/features/viewer/components/screen/select-only-dropdown";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { SurfacePanel } from "@/features/viewer/components/shared/surface-panel";
import { ViewModeButton } from "@/features/viewer/components/view-mode-button";
import {
  type LogBestRunViewModel,
  type LogMetricDatasetRankingRow,
  type LogMetricDirection,
  type LogMetricPointPolicy,
} from "@/features/viewer/state/logs/logs-chart-view-model";
import {
  formatNumber,
  formatRunLabel,
} from "@/features/viewer/state/logs/logs-selectors";
import { formatRunTimestamp } from "@/lib/format";
import { errorMessage } from "@/lib/utils";

function countLabel(count: number, noun: string) {
  return `${count} ${count === 1 ? noun : `${noun}s`}`;
}

function directionLabel(direction: LogMetricDirection) {
  return direction === "higher" ? "higher is better" : "lower is better";
}

function pointPolicyLabel(policy: LogMetricPointPolicy) {
  return policy === "best" ? "best point" : "latest point";
}

function MetadataCell({ value }: { value: string }) {
  return (
    <td className="max-w-40 border-b border-line-soft px-3 py-2.5 text-ink-dim">
      <span className="block truncate" title={value}>
        {value}
      </span>
    </td>
  );
}

function BestRunRowsTable({
  metricTag,
  onSelectRun,
  rows,
}: {
  metricTag: string;
  onSelectRun: (runId: string) => void;
  rows: LogMetricDatasetRankingRow[];
}) {
  return (
    <div className="min-w-0 overflow-auto">
      <table
        className="w-full min-w-[980px] border-separate border-spacing-0 text-left text-xs"
        aria-label={`${metricTag} best run leaderboard`}
      >
        <thead className="sticky top-0 z-10 bg-bg-2/95 text-[11px] uppercase tracking-[0.08em] text-ink-faint">
          <tr>
            {[
              "Dataset",
              "Score",
              "Step",
              "Experiment",
              "Model",
              "Preset",
              "Run",
              "Timestamp",
              "Details",
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
          {rows.map((row) => {
            const best = row.best;
            const runLabel = best ? formatRunLabel(best.run) : row.dataset;
            return (
              <tr
                key={row.dataset}
                className="transition hover:bg-white/[0.025] data-[missing=true]:opacity-70"
                data-missing={!best}
              >
                <td className="max-w-44 border-b border-line-soft px-3 py-2.5 text-ink">
                  <span className="block truncate font-semibold" title={row.dataset}>
                    {row.dataset}
                  </span>
                  <span className="font-mono text-[11px] text-ink-faint">
                    {countLabel(row.runCount, "run")}
                  </span>
                </td>
                <td className="border-b border-line-soft px-3 py-2.5 font-mono tabular-nums text-ink">
                  {best ? formatNumber(best.score) : "No points"}
                </td>
                <td className="w-20 border-b border-line-soft px-3 py-2.5 font-mono tabular-nums text-ink-faint">
                  {best ? best.step : "-"}
                </td>
                <MetadataCell value={best?.run.experiment ?? "-"} />
                <MetadataCell
                  value={best ? `${best.run.model} · ${best.run.modelType}` : "-"}
                />
                <MetadataCell value={best?.run.preset ?? "-"} />
                <MetadataCell value={best?.run.runName ?? "-"} />
                <MetadataCell
                  value={best ? formatRunTimestamp(best.run.timestamp) : "-"}
                />
                <td className="border-b border-line-soft px-3 py-2.5">
                  <Button
                    variant="ghost"
                    className="h-7 rounded-[8px] px-2 text-[11px]"
                    aria-label={
                      best
                        ? `Open details for ${runLabel}`
                        : `No details for ${row.dataset}`
                    }
                    disabled={!best}
                    onClick={() => {
                      if (best) {
                        onSelectRun(best.runId);
                      }
                    }}
                  >
                    Details
                  </Button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function LogBestRunPanel({
  bestRun,
  onSelectRun,
}: {
  bestRun: LogBestRunViewModel;
  onSelectRun: (runId: string) => void;
}) {
  const metricTag = bestRun.selectedMetricTag ?? "metric";
  const hasAnyBestRun = bestRun.rows.some((row) => row.best !== null);
  const controlsDisabled =
    bestRun.visibleRunCount === 0 || bestRun.metricTagOptions.length === 0;
  const metricOptions: SelectOnlyDropdownOption[] = bestRun.metricTagOptions.map(
    (option) => ({
      value: option.value,
      label: option.label,
      description:
        option.count === undefined ? undefined : countLabel(option.count, "run"),
    }),
  );

  return (
    <SurfacePanel
      as="section"
      padding="spacious"
      className="min-w-0"
      aria-labelledby="logs-best-run-title"
    >
      <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 id="logs-best-run-title" className="text-sm font-bold text-ink">
            Best Run
          </h2>
          <div className="mt-0.5 font-mono text-xs text-ink-faint">
            Best run per visible dataset · {countLabel(bestRun.visibleRunCount, "run")} ·{" "}
            {countLabel(bestRun.rows.length, "dataset")}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {bestRun.isFetching && !bestRun.isLoading && (
            <Loader2 className="h-4 w-4 animate-spin text-violet" aria-hidden />
          )}
          <Badge>{pointPolicyLabel(bestRun.selectedPointPolicy)}</Badge>
          <Badge>{directionLabel(bestRun.selectedDirection)}</Badge>
        </div>
      </div>

      {bestRun.hasMoreRuns && (
        <div className="flex items-start gap-2 rounded-[10px] border border-amber-400/25 bg-amber-400/10 px-3 py-2 text-xs leading-5 text-amber-100">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          <span>
            Results exclude unloaded runs because the current log query has more pages.
          </span>
        </div>
      )}

      <div className="grid gap-3 md:grid-cols-[minmax(240px,1fr)_auto_auto]">
        <div className="grid min-w-0 gap-1.5">
          <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-ink-faint">
            Metric
          </span>
          <SelectOnlyDropdown
            label="Best run metric"
            options={metricOptions}
            value={bestRun.selectedMetricTag ?? ""}
            disabled={controlsDisabled}
            onChange={bestRun.onMetricTagChange}
            placeholder="None"
          />
        </div>
        <div className="grid gap-1.5">
          <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-ink-faint">
            Point
          </span>
          <SegmentedControl aria-label="Best run point policy" className="h-9">
            <ViewModeButton
              active={bestRun.selectedPointPolicy === "best"}
              onClick={() => bestRun.onPointPolicyChange("best")}
              disabled={controlsDisabled}
            >
              Best
            </ViewModeButton>
            <ViewModeButton
              active={bestRun.selectedPointPolicy === "latest"}
              onClick={() => bestRun.onPointPolicyChange("latest")}
              disabled={controlsDisabled}
            >
              Latest
            </ViewModeButton>
          </SegmentedControl>
        </div>
        <div className="grid gap-1.5">
          <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-ink-faint">
            Direction
          </span>
          <SegmentedControl aria-label="Best run direction" className="h-9">
            <ViewModeButton
              active={bestRun.selectedDirection === "higher"}
              onClick={() => bestRun.onDirectionChange("higher")}
              disabled={controlsDisabled}
            >
              Higher
            </ViewModeButton>
            <ViewModeButton
              active={bestRun.selectedDirection === "lower"}
              onClick={() => bestRun.onDirectionChange("lower")}
              disabled={controlsDisabled}
            >
              Lower
            </ViewModeButton>
          </SegmentedControl>
        </div>
      </div>

      {bestRun.visibleRunCount === 0 ? (
        <InlineStatus compact>No visible runs to rank.</InlineStatus>
      ) : !bestRun.selectedMetricTag ? (
        <InlineStatus compact>No scalar metric tags found for the visible runs.</InlineStatus>
      ) : bestRun.isError ? (
        <ErrorPanel
          title="Best Run scalar read failed"
          message={errorMessage(bestRun.error)}
        />
      ) : bestRun.isLoading && !hasAnyBestRun ? (
        <InlineStatus busy compact role="status">
          Loading best run scalar points
        </InlineStatus>
      ) : hasAnyBestRun ? (
        <BestRunRowsTable
          metricTag={metricTag}
          rows={bestRun.rows}
          onSelectRun={onSelectRun}
        />
      ) : (
        <InlineStatus compact>
          No selected metric points for any visible dataset.
        </InlineStatus>
      )}
    </SurfacePanel>
  );
}
