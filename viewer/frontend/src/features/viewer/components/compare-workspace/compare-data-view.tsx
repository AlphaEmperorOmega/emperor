import { Badge } from "@/components/ui/badge";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { SurfacePanel } from "@/features/viewer/components/shared/surface-panel";
import {
  buildCompareMetricSummaryRows,
} from "@/features/viewer/components/compare-workspace/compare-run-derive";
import { type ExperimentCompareWorkspaceState } from "./use-experiment-compare-workspace-state";
import { cn } from "@/lib/utils";

export function CompareDataView({
  comparison,
}: {
  comparison: ExperimentCompareWorkspaceState;
}) {
  if (comparison.selectedRuns.length === 0) {
    return (
      <InlineStatus compact>
        Select at least one scalar-capable Training Run to compare data.
      </InlineStatus>
    );
  }

  if (comparison.selectedMetricTags.length === 0) {
    return (
      <InlineStatus compact>
        Select one or more scalar metrics to build the summary table.
      </InlineStatus>
    );
  }

  const rows = buildCompareMetricSummaryRows({
    runIds: comparison.selectedRunIds,
    selectedTags: comparison.selectedMetricTags,
    series: comparison.scalarSeries,
  });

  if (rows.length === 0) {
    return (
      <InlineStatus compact>
        No scalar points are available for the selected Training Runs and metrics.
      </InlineStatus>
    );
  }

  return (
    <SurfacePanel as="section" padding="none" className="overflow-hidden">
      <div className="flex items-center justify-between gap-3 border-b border-line-soft px-3 py-2">
        <h3 className="text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
          Metric Summary
        </h3>
        <Badge>{rows.length}</Badge>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[860px] border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-line-soft text-xs uppercase tracking-[0.08em] text-ink-faint">
              <th className="w-[260px] px-3 py-2 font-bold">Metric</th>
              {comparison.selectedRuns.map((run, index) => (
                <th key={run.id} className="px-3 py-2 font-bold">
                  <span
                    className="block max-w-[220px] truncate"
                    title={comparison.selectedRunFullLabels[index] ?? run.runName}
                  >
                    {comparison.selectedRunLabels[index] ?? run.runName}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key} className="border-b border-line-soft last:border-b-0">
                <th className="px-3 py-2 align-top">
                  <span className="block max-w-[240px] truncate text-xs font-semibold text-ink">
                    {row.tag}
                  </span>
                  <span className="block font-mono text-[11px] text-ink-faint">
                    {row.label}
                    {row.summary === "best" ? ` · ${row.direction} is better` : ""}
                  </span>
                </th>
                {row.values.map((value, index) => (
                  <td
                    key={`${row.key}-${comparison.selectedRunIds[index] ?? index}`}
                    className={cn(
                      "px-3 py-2 align-top font-mono text-xs",
                      value.highlighted
                        ? "bg-violet/10 text-violet"
                        : "text-ink-dim",
                    )}
                  >
                    {value.text}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SurfacePanel>
  );
}
