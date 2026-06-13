"use client";

import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { EChart } from "@/features/viewer/components/charts/echart";
import { type LogCheckpoint, type LogRun, type LogScalarSeries } from "@/lib/api";
import { scalarSeriesColors } from "@/lib/charts";
import {
  buildScalarLineOption,
  type ScalarCheckpointMarker,
  type ScalarLine,
  type ScalarXMode,
  type ScalarYScale,
} from "@/lib/echarts/scalar-options";
import {
  formatNumber,
  formatRunLabel,
} from "@/features/viewer/state/logs/logs-selectors";

export function LogScalarChart({
  tag,
  series,
  runsById,
  checkpointsByRunId,
  runOrder,
  onSelectRun,
  xMode = "step",
  yScale = "linear",
  smoothing = 0,
  group,
}: {
  tag: string;
  series: LogScalarSeries[];
  runsById: Map<string, LogRun>;
  checkpointsByRunId: Map<string, LogCheckpoint[]>;
  runOrder: string[];
  onSelectRun: (runId: string) => void;
  xMode?: ScalarXMode;
  yScale?: ScalarYScale;
  smoothing?: number;
  group?: string;
}) {
  const colorFor = (runId: string) =>
    scalarSeriesColors[Math.max(runOrder.indexOf(runId), 0) % scalarSeriesColors.length];

  const allPoints = series.flatMap((entry) => entry.points);
  const steps = allPoints.map((point) => point.step);
  const values = allPoints.map((point) => point.value);
  const minStep = steps.length ? Math.min(...steps) : 0;
  const maxStep = steps.length ? Math.max(...steps) : 0;
  const minValue = values.length ? Math.min(...values) : 0;
  const maxValue = values.length ? Math.max(...values) : 0;

  const option = useMemo(() => {
    const lines: ScalarLine[] = series.map((entry) => {
      const run = runsById.get(entry.runId);
      return {
        id: entry.runId,
        name: run ? formatRunLabel(run) : entry.runId,
        color: colorFor(entry.runId),
        points: entry.points,
      };
    });
    const checkpointMarkers: ScalarCheckpointMarker[] = series.flatMap((entry) => {
      const run = runsById.get(entry.runId);
      const runLabel = run ? formatRunLabel(run) : entry.runId;
      return (checkpointsByRunId.get(entry.runId) ?? []).map((checkpoint) => ({
        runId: entry.runId,
        runLabel,
        filename: checkpoint.filename,
        epoch: checkpoint.epoch,
        step: checkpoint.step,
      }));
    });
    return buildScalarLineOption(lines, {
      xMode,
      yScale,
      smoothing,
      dataZoom: true,
      checkpointMarkers,
    });
    // colorFor depends on runOrder; runsById/series cover the rest.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series, runsById, checkpointsByRunId, runOrder, xMode, yScale, smoothing]);

  return (
    <section className="edge grid min-w-0 gap-3 rounded-card p-4">
      <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-bold text-ink">{tag}</h2>
          <div className="mt-0.5 font-mono text-xs text-ink-faint">
            {series.length} lines · step {minStep} to {maxStep}
          </div>
        </div>
        <Badge>
          {formatNumber(minValue)} to {formatNumber(maxValue)}
        </Badge>
      </div>

      <div className="h-56 w-full min-w-0" role="img" aria-label={`${tag} scalar chart`}>
        <EChart option={option} group={group} />
      </div>

      <div className="grid gap-1.5 sm:grid-cols-2 xl:grid-cols-3">
        {series.map((entry) => {
          const run = runsById.get(entry.runId);
          if (!run) {
            return null;
          }
          const color = colorFor(entry.runId);
          const latest = entry.points.at(-1);
          return (
            <button
              key={entry.runId}
              type="button"
              className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 rounded-[9px] border border-line-soft bg-black/20 px-2 py-1.5 text-left text-xs transition hover:border-line hover:bg-white/[0.035] focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              onClick={() => onSelectRun(entry.runId)}
            >
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: color }}
                aria-hidden
              />
              <span className="truncate text-ink-dim">{formatRunLabel(run)}</span>
              {latest && (
                <span className="font-mono text-ink-faint">{formatNumber(latest.value)}</span>
              )}
            </button>
          );
        })}
      </div>
    </section>
  );
}
