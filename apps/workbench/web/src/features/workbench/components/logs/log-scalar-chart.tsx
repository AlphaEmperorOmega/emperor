"use client";

import { useMemo } from "react";
import type { LogCheckpoint, LogRun, LogScalarSeries } from "@/lib/api/logs";
import { scalarSeriesColors } from "@/lib/charts";
import {
  type ScalarXMode,
  type ScalarYScale,
} from "@/lib/echarts/scalar-options";
import { formatRunLabel } from "@/features/workbench/state/logs/logs-selectors";
import {
  LazyLogScalarChartFrame,
  LogScalarChartCard,
  type LogScalarCardLine,
  type LogScalarLegendEntry,
} from "@/features/workbench/components/logs/_log-scalar-chart-card";

type SharedScalarChartProps = {
  series: LogScalarSeries[];
  runsById: Map<string, LogRun>;
  checkpointsByRunId: Map<string, LogCheckpoint[]>;
  runOrder: string[];
  onSelectRun: (runId: string) => void;
  highlightedRunId?: string | null;
  onHoverRunChange?: (runId: string | null) => void;
  xMode?: ScalarXMode;
  yScale?: ScalarYScale;
  smoothing?: number;
  group?: string;
  hasRequested?: boolean;
  isLoading?: boolean;
  isError?: boolean;
  error?: unknown;
};

type LogScalarChartProps = SharedScalarChartProps & {
  tag: string;
  onVisible?: (tag: string) => void;
};

type LogTrainValidationScalarChartProps = SharedScalarChartProps & {
  suffix: string;
  trainTag: string;
  validationTag: string;
  onVisible?: (suffix: string) => void;
};

export type TrainValidationScalarLine = LogScalarCardLine & {
  tag: string;
  phase: "Train" | "Validation";
  latest: LogScalarSeries["points"][number] | undefined;
};

function runIndexForOrder(runOrder: readonly string[]) {
  return new Map(runOrder.map((runId, index) => [runId, index]));
}

function colorForRunIndex(runIndex: Map<string, number>, runId: string) {
  return scalarSeriesColors[
    Math.max(runIndex.get(runId) ?? 0, 0) % scalarSeriesColors.length
  ];
}

export function buildTrainValidationScalarLines({
  series,
  runsById,
  runOrder,
  trainTag,
  validationTag,
}: {
  series: readonly LogScalarSeries[];
  runsById: Map<string, LogRun>;
  runOrder: readonly string[];
  trainTag: string;
  validationTag: string;
}): TrainValidationScalarLine[] {
  const runIndex = runIndexForOrder(runOrder);
  const orderedRunIds = [...runOrder];
  const orderedRunIdSet = new Set(orderedRunIds);
  const seriesByRunAndTag = new Map<string, LogScalarSeries>();
  for (const entry of series) {
    seriesByRunAndTag.set(`${entry.runId}::${entry.tag}`, entry);
    if (!orderedRunIdSet.has(entry.runId)) {
      orderedRunIdSet.add(entry.runId);
      orderedRunIds.push(entry.runId);
      runIndex.set(entry.runId, runIndex.size);
    }
  }

  const phases = [
    { tag: trainTag, label: "Train" as const, lineStyle: { type: "solid" as const } },
    {
      tag: validationTag,
      label: "Validation" as const,
      lineStyle: { type: "dashed" as const },
    },
  ];

  return orderedRunIds.flatMap((runId) => {
    const run = runsById.get(runId);
    const runLabel = run ? formatRunLabel(run) : runId;
    return phases.flatMap(({ tag, label, lineStyle }) => {
      const entry = seriesByRunAndTag.get(`${runId}::${tag}`);
      if (!entry || entry.points.length === 0) {
        return [];
      }
      return [
        {
          id: `${runId}::${tag}`,
          runId,
          tag,
          phase: label,
          name: `${runLabel} · ${label}`,
          color: colorForRunIndex(runIndex, runId),
          lineStyle,
          points: entry.points,
          latest: entry.points.at(-1),
        },
      ];
    });
  });
}

export function LazyLogScalarChart(props: LogScalarChartProps) {
  const chartLabel = `${props.tag} scalar chart`;
  return (
    <LazyLogScalarChartFrame
      visibilityKey={props.tag}
      chartLabel={chartLabel}
      loadingLabel={`${props.tag} scalar points`}
      errorTitle={`${props.tag} scalar read failed`}
      hasContent={props.series.length > 0}
      hasRequested={props.hasRequested}
      isLoading={props.isLoading}
      isError={props.isError}
      error={props.error}
      onVisible={props.onVisible}
    >
      <LogScalarChart {...props} />
    </LazyLogScalarChartFrame>
  );
}

export function LazyLogTrainValidationScalarChart(
  props: LogTrainValidationScalarChartProps,
) {
  const chartLabel = `${props.suffix} train vs validation scalar chart`;
  return (
    <LazyLogScalarChartFrame
      visibilityKey={props.suffix}
      chartLabel={chartLabel}
      loadingLabel={`${props.suffix} train vs validation scalar points`}
      errorTitle={`${props.suffix} train vs validation scalar read failed`}
      hasContent={props.series.length > 0}
      hasRequested={props.hasRequested}
      isLoading={props.isLoading}
      isError={props.isError}
      error={props.error}
      onVisible={props.onVisible}
    >
      <LogTrainValidationScalarChart {...props} />
    </LazyLogScalarChartFrame>
  );
}

export function LogScalarChart({
  tag,
  series,
  runsById,
  checkpointsByRunId,
  runOrder,
  onSelectRun,
  highlightedRunId = null,
  onHoverRunChange,
  xMode = "step",
  yScale = "linear",
  smoothing = 0,
  group,
}: LogScalarChartProps) {
  const runIndex = useMemo(() => runIndexForOrder(runOrder), [runOrder]);
  const lines = useMemo<LogScalarCardLine[]>(
    () =>
      series.map((entry) => {
        const run = runsById.get(entry.runId);
        return {
          id: entry.runId,
          runId: entry.runId,
          name: run ? formatRunLabel(run) : entry.runId,
          color: colorForRunIndex(runIndex, entry.runId),
          points: entry.points,
        };
      }),
    [runIndex, runsById, series],
  );
  const legendEntries = useMemo<LogScalarLegendEntry[]>(
    () =>
      series.flatMap((entry) => {
        const run = runsById.get(entry.runId);
        if (!run) {
          return [];
        }
        return [
          {
            id: entry.runId,
            runId: entry.runId,
            label: formatRunLabel(run),
            color: colorForRunIndex(runIndex, entry.runId),
            latest: entry.points.at(-1),
            marker: { kind: "dot" as const },
          },
        ];
      }),
    [runIndex, runsById, series],
  );

  return (
    <LogScalarChartCard
      title={tag}
      infoMetricKey={tag}
      chartLabel={`${tag} scalar chart`}
      summarySeries={series}
      lines={lines}
      legendEntries={legendEntries}
      runsById={runsById}
      checkpointsByRunId={checkpointsByRunId}
      runOrder={runOrder}
      onSelectRun={onSelectRun}
      highlightedRunId={highlightedRunId}
      onHoverRunChange={onHoverRunChange}
      xMode={xMode}
      yScale={yScale}
      smoothing={smoothing}
      group={group}
    />
  );
}

export function LogTrainValidationScalarChart({
  suffix,
  trainTag,
  validationTag,
  series,
  runsById,
  checkpointsByRunId,
  runOrder,
  onSelectRun,
  highlightedRunId = null,
  onHoverRunChange,
  xMode = "step",
  yScale = "linear",
  smoothing = 0,
  group,
}: LogTrainValidationScalarChartProps) {
  const lines = useMemo(
    () =>
      buildTrainValidationScalarLines({
        series,
        runsById,
        runOrder,
        trainTag,
        validationTag,
      }),
    [runOrder, runsById, series, trainTag, validationTag],
  );
  const legendEntries = useMemo<LogScalarLegendEntry[]>(
    () =>
      lines.map((line) => ({
        id: line.id,
        runId: line.runId,
        label: line.name,
        color: line.color,
        latest: line.latest,
        marker: {
          kind: "line" as const,
          style: line.phase === "Validation" ? "dashed" : "solid",
        },
      })),
    [lines],
  );

  return (
    <LogScalarChartCard
      title={suffix}
      infoMetricKey={validationTag}
      chartLabel={`${suffix} train vs validation scalar chart`}
      summarySeries={series}
      lines={lines}
      legendEntries={legendEntries}
      runsById={runsById}
      checkpointsByRunId={checkpointsByRunId}
      runOrder={runOrder}
      onSelectRun={onSelectRun}
      highlightedRunId={highlightedRunId}
      onHoverRunChange={onHoverRunChange}
      xMode={xMode}
      yScale={yScale}
      smoothing={smoothing}
      group={group}
    />
  );
}
