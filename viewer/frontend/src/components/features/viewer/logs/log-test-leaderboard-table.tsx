"use client";

import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { type LogRun, type LogScalarSeries } from "@/lib/api";
import { formatNumber, formatRunLabel } from "@/lib/logs/helpers";

type TestLeaderboardSortDirection = "ascending" | "descending";

type TestLeaderboardRow = {
  runId: string;
  run: LogRun;
  value: number;
  step: number;
  visibleIndex: number;
  sourceIndex: number;
};

function testMetricSortDirection(tag: string): TestLeaderboardSortDirection {
  return tag.toLowerCase().includes("loss") ? "ascending" : "descending";
}

export function buildTestLeaderboardRows({
  tag,
  series,
  runsById,
  runOrder,
}: {
  tag: string;
  series: LogScalarSeries[];
  runsById: Map<string, LogRun>;
  runOrder: string[];
}) {
  const runPositions = new Map(runOrder.map((runId, index) => [runId, index]));
  const direction = testMetricSortDirection(tag);

  return series
    .flatMap<TestLeaderboardRow>((entry, sourceIndex) => {
      if (entry.tag !== tag || entry.points.length === 0) {
        return [];
      }

      const run = runsById.get(entry.runId);
      const latest = entry.points.at(-1);
      if (!run || !latest) {
        return [];
      }

      return [
        {
          runId: entry.runId,
          run,
          value: latest.value,
          step: latest.step,
          visibleIndex: runPositions.get(entry.runId) ?? runOrder.length + sourceIndex,
          sourceIndex,
        },
      ];
    })
    .sort((left, right) => {
      const valueDelta =
        direction === "ascending" ? left.value - right.value : right.value - left.value;
      if (valueDelta !== 0) {
        return valueDelta;
      }
      return left.visibleIndex - right.visibleIndex || left.sourceIndex - right.sourceIndex;
    });
}

function directionLabel(direction: TestLeaderboardSortDirection) {
  return direction === "ascending" ? "ascending (lower first)" : "descending (higher first)";
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

export function LogTestLeaderboardTable({
  tag,
  series,
  runsById,
  runOrder,
  onSelectRun,
}: {
  tag: string;
  series: LogScalarSeries[];
  runsById: Map<string, LogRun>;
  runOrder: string[];
  onSelectRun: (runId: string) => void;
}) {
  const direction = testMetricSortDirection(tag);
  const rows = useMemo(
    () => buildTestLeaderboardRows({ tag, series, runsById, runOrder }),
    [tag, series, runsById, runOrder],
  );

  return (
    <section className="edge grid min-w-0 gap-3 rounded-card p-4">
      <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-bold text-ink">{tag}</h2>
          <div className="mt-0.5 font-mono text-xs text-ink-faint">
            {rows.length} {rows.length === 1 ? "run" : "runs"} ·{" "}
            {directionLabel(direction)}
          </div>
        </div>
        <Badge>latest point</Badge>
      </div>

      <div className="max-h-[420px] min-w-0 overflow-auto">
        <table
          className="w-full min-w-[920px] border-separate border-spacing-0 text-left text-xs"
          aria-label={`${tag} test leaderboard`}
        >
          <thead className="sticky top-0 z-10 bg-bg-2/95 text-[11px] uppercase tracking-[0.08em] text-ink-faint">
            <tr>
              {["Rank", tag, "Experiment", "Dataset", "Model", "Preset", "Run", "Step"].map(
                (heading) => (
                  <th
                    key={heading}
                    scope="col"
                    className="border-b border-line-soft px-3 py-2 font-bold"
                  >
                    {heading}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td
                  className="border-b border-line-soft px-3 py-6 text-center text-ink-faint"
                  colSpan={8}
                >
                  No test metric points for the selected runs.
                </td>
              </tr>
            ) : (
              rows.map((row, index) => {
                const runLabel = formatRunLabel(row.run);
                return (
                  <tr
                    key={row.runId}
                    className="transition hover:bg-white/[0.025]"
                  >
                    <td className="w-16 border-b border-line-soft px-3 py-2.5 font-mono text-ink-faint">
                      {index + 1}
                    </td>
                    <td className="border-b border-line-soft px-3 py-2.5 font-mono tabular-nums text-ink">
                      {formatNumber(row.value)}
                    </td>
                    <MetadataCell value={row.run.experiment} />
                    <MetadataCell value={row.run.dataset} />
                    <MetadataCell value={row.run.model} />
                    <MetadataCell value={row.run.preset} />
                    <td className="border-b border-line-soft px-3 py-2.5">
                      <div className="flex min-w-0 items-center gap-2">
                        <button
                          type="button"
                          className="min-w-0 truncate text-left font-semibold text-ink-dim transition hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                          title={runLabel}
                          aria-label={`Open run details for ${runLabel}`}
                          onClick={() => onSelectRun(row.runId)}
                        >
                          {row.run.runName}
                        </button>
                        <Button
                          variant="ghost"
                          className="h-7 shrink-0 rounded-[8px] px-2 text-[11px]"
                          aria-label={`Open details for ${runLabel}`}
                          onClick={() => onSelectRun(row.runId)}
                        >
                          Details
                        </Button>
                      </div>
                    </td>
                    <td className="w-20 border-b border-line-soft px-3 py-2.5 font-mono tabular-nums text-ink-faint">
                      {row.step}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
