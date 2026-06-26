import { BarChart3 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { EChart } from "@/features/viewer/components/charts/echart";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { SurfacePanel } from "@/features/viewer/components/shared/surface-panel";
import { type ExperimentCompareWorkspaceState } from "./use-experiment-compare-workspace-state";
import { scalarSeriesColors } from "@/lib/charts";
import {
  buildScalarLineOption,
  type ScalarLine,
} from "@/lib/echarts/scalar-options";
import { formatNumber } from "@/features/viewer/state/logs/logs-selectors";

export function CompareGraphsView({
  comparison,
}: {
  comparison: ExperimentCompareWorkspaceState;
}) {
  if (comparison.selectedRuns.length === 0) {
    return (
      <InlineStatus compact>
        Select at least one scalar-capable Training Run to compare graphs.
      </InlineStatus>
    );
  }

  if (comparison.selectedMetricTags.length === 0) {
    return (
      <InlineStatus compact>
        Select one or more scalar metrics to render comparison graphs.
      </InlineStatus>
    );
  }

  return (
    <div className="grid gap-3 xl:grid-cols-2">
      {comparison.selectedMetricTags.map((tag) => {
        const series = comparison.seriesByTag.get(tag) ?? [];
        const runOrder = new Map(
          comparison.selectedRuns.map((run, index) => [run.id, index]),
        );
        const lines: ScalarLine[] = series.map((entry) => {
          const index = runOrder.get(entry.runId) ?? 0;
          return {
            id: entry.runId,
            name:
              comparison.selectedRunFullLabels[index] ??
              comparison.selectedRunLabels[index] ??
              entry.runId,
            color: scalarSeriesColors[index % scalarSeriesColors.length],
            points: entry.points,
          };
        });
        const option = buildScalarLineOption(lines, { xMode: "step" });
        const pointCount = series.reduce(
          (total, entry) => total + entry.points.length,
          0,
        );

        return (
          <SurfacePanel
            key={tag}
            as="section"
            padding="spacious"
            className="min-w-0"
          >
            <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <h3 className="truncate text-sm font-bold text-ink">{tag}</h3>
                <div className="mt-0.5 font-mono text-xs text-ink-faint">
                  {series.length} lines · {pointCount} points
                </div>
              </div>
              <Badge>
                <BarChart3 className="mr-1 h-3.5 w-3.5" aria-hidden />
                step
              </Badge>
            </div>

            {series.length === 0 ? (
              <InlineStatus compact>No selected run has this metric.</InlineStatus>
            ) : (
              <div
                className="h-52 w-full min-w-0"
                role="img"
                aria-label={`${tag} Training Run comparison chart`}
              >
                <EChart option={option} group="compare-runs" />
              </div>
            )}

            <div className="grid max-h-36 gap-1.5 overflow-y-auto pr-1">
              {comparison.selectedRuns.map((run, index) => {
                const entry = series.find((candidate) => candidate.runId === run.id);
                const latest = entry?.points.at(-1);
                const label = comparison.selectedRunLabels[index] ?? run.runName;
                const fullLabel =
                  comparison.selectedRunFullLabels[index] ?? run.runName;
                return (
                  <div
                    key={run.id}
                    className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 rounded-[8px] border border-line-soft bg-black/16 px-2 py-1.5 text-xs"
                    title={fullLabel}
                  >
                    <span
                      className="h-2.5 w-2.5 rounded-full"
                      style={{
                        backgroundColor:
                          scalarSeriesColors[index % scalarSeriesColors.length],
                      }}
                      aria-hidden
                    />
                    <span className="truncate text-ink-dim">{label}</span>
                    <span className="font-mono text-ink-faint">
                      {latest ? formatNumber(latest.value) : "—"}
                    </span>
                  </div>
                );
              })}
            </div>
          </SurfacePanel>
        );
      })}
    </div>
  );
}
