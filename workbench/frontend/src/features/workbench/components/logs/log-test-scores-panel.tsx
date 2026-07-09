"use client";

import { useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { ErrorPanel } from "@/features/workbench/components/error-panel";
import { LogTestLeaderboardSection } from "@/features/workbench/components/logs/log-test-leaderboard-table";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { SurfacePanel } from "@/features/workbench/components/shared/surface-panel";
import { type LogMetricGroupScalarQueryState } from "@/features/workbench/state/logs/logs-chart-view-model";
import { type RenderableLogMetric } from "@/features/workbench/state/logs/logs-selectors";
import { type LogRun, type LogScalarSeries } from "@/lib/api";
import { errorMessage } from "@/lib/utils";

const TEST_SCORE_METRIC_RENDER_LIMIT = 100;

function countLabel(count: number, noun: string) {
  return `${count} ${count === 1 ? noun : `${noun}s`}`;
}

type SplitTestLeaderboard = {
  key: string;
  experiment: string;
  tag: string;
  series: LogScalarSeries[];
  runOrder: string[];
};

function splitTestLeaderboardsByExperiment({
  metrics,
  runsById,
  runOrder,
}: {
  metrics: RenderableLogMetric[];
  runsById: Map<string, LogRun>;
  runOrder: string[];
}) {
  const orderedExperiments: string[] = [];
  const runIdsByExperiment = new Map<string, string[]>();

  for (const runId of runOrder) {
    const experiment = runsById.get(runId)?.experiment;
    if (!experiment) {
      continue;
    }
    const experimentRunIds = runIdsByExperiment.get(experiment);
    if (experimentRunIds) {
      experimentRunIds.push(runId);
      continue;
    }
    orderedExperiments.push(experiment);
    runIdsByExperiment.set(experiment, [runId]);
  }

  return metrics.flatMap<SplitTestLeaderboard>((metric) => {
    const seriesRunIds = new Set(
      metric.series
        .filter((series) => series.points.length > 0)
        .map((series) => series.runId),
    );

    return orderedExperiments.flatMap((experiment) => {
      const experimentRunOrder = (runIdsByExperiment.get(experiment) ?? []).filter(
        (runId) => seriesRunIds.has(runId),
      );
      if (experimentRunOrder.length === 0) {
        return [];
      }

      const experimentRunIds = new Set(experimentRunOrder);
      const experimentSeries = metric.series.filter((series) =>
        experimentRunIds.has(series.runId),
      );

      return [
        {
          key: `${metric.tag}::${experiment}`,
          experiment,
          tag: metric.tag,
          series: experimentSeries,
          runOrder: experimentRunOrder,
        },
      ];
    });
  });
}

export function LogTestScoresPanel({
  metrics,
  selectedTags,
  queryState,
  runsById,
  runOrder,
  onSelectRun,
}: {
  metrics: RenderableLogMetric[];
  selectedTags: string[];
  queryState: LogMetricGroupScalarQueryState;
  runsById: Map<string, LogRun>;
  runOrder: string[];
  onSelectRun: (runId: string) => void;
}) {
  const [splitByExperiment, setSplitByExperiment] = useState(false);
  const visibleMetrics = useMemo(
    () => metrics.slice(0, TEST_SCORE_METRIC_RENDER_LIMIT),
    [metrics],
  );
  const splitLeaderboards = useMemo(
    () =>
      splitTestLeaderboardsByExperiment({
        metrics: visibleMetrics,
        runsById,
        runOrder,
      }),
    [visibleMetrics, runsById, runOrder],
  );
  const hiddenMetricCount = Math.max(0, metrics.length - visibleMetrics.length);

  if ((metrics.length === 0 && selectedTags.length === 0) || runOrder.length === 0) {
    return null;
  }

  return (
    <SurfacePanel
      as="section"
      padding="spacious"
      className="min-w-0"
      aria-labelledby="logs-test-scores-title"
    >
      <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 id="logs-test-scores-title" className="text-sm font-bold text-ink">
            Test Metric Leaderboards
          </h2>
          <div className="mt-0.5 font-mono text-xs text-ink-faint">
            {countLabel(selectedTags.length, "selected tag")} · latest point ranking
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {queryState.isFetching && !queryState.isInitialLoading && (
            <Loader2 className="h-4 w-4 animate-spin text-violet" aria-hidden />
          )}
          <label className="inline-flex h-8 cursor-pointer items-center gap-2 rounded-control-sm border border-line bg-white/[0.025] px-2.5 text-xs font-semibold text-ink-dim transition hover:border-white/15 hover:bg-white/[0.055] hover:text-ink">
            <Checkbox
              checked={splitByExperiment}
              onCheckedChange={setSplitByExperiment}
            />
            <span>Split by experiment</span>
          </label>
          <Badge>{countLabel(metrics.length, "leaderboard")}</Badge>
        </div>
      </div>

      {queryState.isError ? (
        <ErrorPanel
          title="Test scalar read failed"
          message={errorMessage(queryState.error)}
        />
      ) : queryState.isInitialLoading && metrics.length === 0 ? (
        <InlineStatus busy compact role="status">
          Loading Test scalar points
        </InlineStatus>
      ) : metrics.length === 0 ? (
        <InlineStatus compact>
          No test metric points for the selected runs.
        </InlineStatus>
      ) : splitByExperiment && splitLeaderboards.length === 0 ? (
        <InlineStatus compact>
          No test metric points for the selected runs.
        </InlineStatus>
      ) : splitByExperiment ? (
        <div className="grid min-w-0 gap-5 xl:grid-cols-2">
          {splitLeaderboards.map(({ key, experiment, tag, series, runOrder }) => (
            <LogTestLeaderboardSection
              key={key}
              heading={`${experiment} · ${tag}`}
              tableAriaLabel={`${experiment} · ${tag} test leaderboard`}
              tag={tag}
              series={series}
              runsById={runsById}
              runOrder={runOrder}
              onSelectRun={onSelectRun}
            />
          ))}
        </div>
      ) : (
        <div className="grid min-w-0 gap-5 xl:grid-cols-2">
          {visibleMetrics.map(({ tag, series }) => (
            <LogTestLeaderboardSection
              key={tag}
              tag={tag}
              series={series}
              runsById={runsById}
              runOrder={runOrder}
              onSelectRun={onSelectRun}
            />
          ))}
        </div>
      )}

      {hiddenMetricCount > 0 && (
        <div className="rounded-[10px] border border-line-soft bg-white/[0.018] px-3 py-3 text-center text-xs text-ink-faint">
          Showing {visibleMetrics.length} of {metrics.length} test score leaderboards.
          Narrow selected tags to inspect the rest.
        </div>
      )}
    </SurfacePanel>
  );
}
