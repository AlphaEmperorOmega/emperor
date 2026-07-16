"use client";

import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SurfacePanel } from "@/components/ui/surface-panel";
import type { LogRun, LogScalarSeries } from "@/lib/api/logs";
import { formatRunLabel } from "@/features/workbench/state/logs/logs-selectors";
import { formatNumber } from "@/lib/format";
import {
  buildLogMetricRankingRows,
  inferLogMetricDirection,
  type LogMetricDirection,
} from "@/features/workbench/state/logs/log-metric-ranking";

const TEST_LEADERBOARD_ROW_LIMIT = 100;

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
  return buildLogMetricRankingRows({
    direction: inferLogMetricDirection(tag),
    pointPolicy: "latest",
    runOrder,
    runs: Array.from(runsById.values()),
    series,
    tag,
  });
}

function directionLabel(direction: LogMetricDirection) {
  return direction === "lower" ? "ascending (lower first)" : "descending (higher first)";
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

function LogTestLeaderboardContent({
  heading,
  headingLevel = "h2",
  tableAriaLabel,
  tag,
  series,
  runsById,
  runOrder,
  onSelectRun,
}: {
  heading?: string;
  headingLevel?: "h2" | "h3";
  tableAriaLabel?: string;
  tag: string;
  series: LogScalarSeries[];
  runsById: Map<string, LogRun>;
  runOrder: string[];
  onSelectRun: (runId: string) => void;
}) {
  const Heading = headingLevel;
  const headingText = heading ?? tag;
  const direction = inferLogMetricDirection(tag);
  const rows = useMemo(
    () => buildTestLeaderboardRows({ tag, series, runsById, runOrder }),
    [tag, series, runsById, runOrder],
  );
  const visibleRows = rows.slice(0, TEST_LEADERBOARD_ROW_LIMIT);
  const hiddenRowCount = Math.max(0, rows.length - visibleRows.length);

  return (
    <>
      <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <Heading className="truncate text-sm font-bold text-ink" title={headingText}>
            {headingText}
          </Heading>
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
          aria-label={tableAriaLabel ?? `${tag} test leaderboard`}
        >
          <thead className="sticky top-0 z-10 bg-bg-2/95 type-meta uppercase tracking-label text-ink-faint">
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
            {visibleRows.length === 0 ? (
              <tr>
                <td
                  className="border-b border-line-soft px-3 py-6 text-center text-ink-faint"
                  colSpan={8}
                >
                  No test metric points for the selected runs.
                </td>
              </tr>
            ) : (
              visibleRows.map((row, index) => {
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
                      {formatNumber(row.score)}
                    </td>
                    <MetadataCell value={row.run.experiment} />
                    <MetadataCell value={row.run.dataset} />
                    <MetadataCell value={row.run.model} />
                    <MetadataCell value={row.run.preset} />
                    <td className="border-b border-line-soft px-3 py-2.5">
                      <div className="flex min-w-0 items-center gap-2">
                        <button
                          type="button"
                          className="min-h-touch min-w-0 truncate text-left font-semibold text-ink-dim transition hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:min-h-control-sm"
                          title={runLabel}
                          aria-label={`Open run details for ${runLabel}`}
                          onClick={() => onSelectRun(row.runId)}
                        >
                          {row.run.runName}
                        </button>
                        <Button
                          variant="ghost"
                          className="h-touch shrink-0 rounded-control-md px-2 type-meta md:h-control-sm"
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
      {hiddenRowCount > 0 && (
        <div className="rounded-control border border-line-soft bg-white/[0.018] px-3 py-2 text-center text-xs text-ink-faint">
          Showing top {visibleRows.length} of {rows.length} runs.
        </div>
      )}
    </>
  );
}

export function LogTestLeaderboardSection({
  heading,
  tableAriaLabel,
  tag,
  series,
  runsById,
  runOrder,
  onSelectRun,
}: {
  heading?: string;
  tableAriaLabel?: string;
  tag: string;
  series: LogScalarSeries[];
  runsById: Map<string, LogRun>;
  runOrder: string[];
  onSelectRun: (runId: string) => void;
}) {
  return (
    <section className="grid min-w-0 gap-3">
      <LogTestLeaderboardContent
        heading={heading}
        headingLevel="h3"
        tableAriaLabel={tableAriaLabel}
        tag={tag}
        series={series}
        runsById={runsById}
        runOrder={runOrder}
        onSelectRun={onSelectRun}
      />
    </section>
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
  return (
    <SurfacePanel as="section" padding="spacious" className="min-w-0">
      <LogTestLeaderboardContent
        tag={tag}
        series={series}
        runsById={runsById}
        runOrder={runOrder}
        onSelectRun={onSelectRun}
      />
    </SurfacePanel>
  );
}
