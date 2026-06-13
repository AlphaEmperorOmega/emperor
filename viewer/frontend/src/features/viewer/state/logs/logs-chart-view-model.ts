import { useMemo, useState } from "react";
import { type LogCheckpoint, type LogRun, type LogScalarSeries } from "@/lib/api";
import { type ScalarXMode, type ScalarYScale } from "@/lib/echarts/scalar-options";
import {
  useLogCheckpointsQuery,
  useLogScalarsQuery,
} from "@/features/viewer/state/logs/use-log-queries";
import { logQueryKeys } from "@/lib/query-keys";
import { type LogsWorkspaceState } from "@/features/viewer/state/logs/use-logs-workspace-state";

export type LogsChartEmptyState = {
  title: string;
  detail: string;
  busy?: boolean;
};

export type ScalarChartGridMode = "full" | "two" | "three";

export type LogScalarQueryInput = {
  runIds: string[];
  tags: string[];
  enabled: boolean;
  queryKey: ReturnType<typeof logQueryKeys.scalarsForRunsAndTags>;
};

export function buildLogScalarQueryInput({
  enabled,
  selectedTagList,
  visibleRunIds,
}: {
  enabled: boolean;
  selectedTagList: string[];
  visibleRunIds: string[];
}): LogScalarQueryInput {
  return {
    runIds: visibleRunIds,
    tags: selectedTagList,
    enabled: enabled && visibleRunIds.length > 0 && selectedTagList.length > 0,
    queryKey: logQueryKeys.scalarsForRunsAndTags(visibleRunIds, selectedTagList),
  };
}

export function groupLogScalarSeriesByTag(seriesList: LogScalarSeries[]) {
  const byTag = new Map<string, LogScalarSeries[]>();
  for (const series of seriesList) {
    if (series.points.length === 0) {
      continue;
    }
    byTag.set(series.tag, [...(byTag.get(series.tag) ?? []), series]);
  }
  return byTag;
}

export function countGroupedLogScalarSeries(
  seriesByTag: Map<string, LogScalarSeries[]>,
) {
  return Array.from(seriesByTag.values()).reduce(
    (total, series) => total + series.length,
    0,
  );
}

export function deriveLogsChartEmptyState({
  hasEventFiles,
  runsLoading,
  scalarLoading,
  selectedSeriesCount,
  selectedTagCount,
  tagOptionCount,
  tagsLoading,
  visibleRunCount,
}: {
  hasEventFiles: boolean;
  runsLoading: boolean;
  scalarLoading: boolean;
  selectedSeriesCount: number;
  selectedTagCount: number;
  tagOptionCount: number;
  tagsLoading: boolean;
  visibleRunCount: number;
}): LogsChartEmptyState | null {
  let title = "";
  let detail = "";

  if (runsLoading) {
    title = "Scanning logs";
    detail = "Reading historical run folders.";
  } else if (visibleRunCount === 0) {
    title = "No runs selected";
    detail = "Use the sidebar filters to include at least one historical run.";
  } else if (tagsLoading) {
    title = "Reading TensorBoard tags";
    detail = "Collecting scalar tags from the selected runs.";
  } else if (tagOptionCount === 0) {
    title = "No TensorBoard scalars";
    detail = "The selected runs do not contain scalar event data.";
  } else if (selectedTagCount === 0) {
    title = "No scalar tags selected";
    detail = "Select one or more scalar tags to draw historical charts.";
  } else if (scalarLoading) {
    title = "Loading scalar points";
    detail = "Reading TensorBoard scalar series for the selected runs.";
  } else if (selectedSeriesCount === 0 && hasEventFiles) {
    title = "No scalar points for selection";
    detail = "The selected runs have event files, but none contain the checked scalar tags.";
  } else if (selectedSeriesCount === 0) {
    title = "No TensorBoard scalars";
    detail = "The selected runs do not contain scalar event data.";
  }

  if (!title) {
    return null;
  }

  return {
    title,
    detail,
    busy: runsLoading || tagsLoading || scalarLoading,
  };
}

function runsById(runs: LogRun[]) {
  return new Map(runs.map((run) => [run.id, run]));
}

function checkpointsByRunId(checkpoints: LogCheckpoint[]) {
  const byRunId = new Map<string, LogCheckpoint[]>();
  for (const checkpoint of checkpoints) {
    byRunId.set(checkpoint.runId, [
      ...(byRunId.get(checkpoint.runId) ?? []),
      checkpoint,
    ]);
  }
  return byRunId;
}

export function useLogsChartViewModel(state: LogsWorkspaceState) {
  const [gridMode, setGridMode] = useState<ScalarChartGridMode>("two");
  const [smoothing, setSmoothing] = useState(0);
  const [xMode, setXMode] = useState<ScalarXMode>("step");
  const [yScale, setYScale] = useState<ScalarYScale>("linear");
  const scalarQueryInput = buildLogScalarQueryInput({
    enabled: state.enabled,
    selectedTagList: state.selectedTagList,
    visibleRunIds: state.visibleRunIds,
  });
  const scalarQuery = useLogScalarsQuery(scalarQueryInput);
  const checkpointQuery = useLogCheckpointsQuery({
    runIds: state.visibleRunIds,
    enabled: state.enabled,
    queryKey: logQueryKeys.checkpointsForRuns(state.visibleRunIds),
  });
  const scalarSeries = scalarQuery.data?.series;
  const checkpoints = checkpointQuery.data?.checkpoints;

  const seriesByTag = useMemo(
    () => groupLogScalarSeriesByTag(scalarSeries ?? []),
    [scalarSeries],
  );
  const visibleRunsById = useMemo(
    () => runsById(state.visibleRuns),
    [state.visibleRuns],
  );
  const visibleCheckpointsByRunId = useMemo(
    () => checkpointsByRunId(checkpoints ?? []),
    [checkpoints],
  );
  const selectedSeriesCount = countGroupedLogScalarSeries(seriesByTag);
  const emptyState = deriveLogsChartEmptyState({
    hasEventFiles: state.visibleRuns.some((run) => run.eventFileCount > 0),
    runsLoading: state.runsQuery.isLoading,
    scalarLoading: scalarQuery.isLoading,
    selectedSeriesCount,
    selectedTagCount: state.selectedTagList.length,
    tagOptionCount: state.tagOptions.length,
    tagsLoading: state.tagsQuery.isLoading,
    visibleRunCount: state.visibleRuns.length,
  });

  return {
    selectedTagList: state.selectedTagList,
    seriesByTag,
    runsById: visibleRunsById,
    checkpointsByRunId: visibleCheckpointsByRunId,
    runOrder: state.visibleRunIds,
    visibleRunCount: state.visibleRuns.length,
    selectedTagCount: state.selectedTagList.length,
    collapsedMetricGroups: state.collapsedMetricGroups,
    onToggleMetricGroup: state.toggleMetricGroup,
    gridMode,
    onGridModeChange: setGridMode,
    smoothing,
    onSmoothingChange: setSmoothing,
    xMode,
    onXModeChange: setXMode,
    yScale,
    onYScaleChange: setYScale,
    isFetching: scalarQuery.isFetching || checkpointQuery.isFetching,
    isRefreshDisabled: !scalarQuery.isSuccess && !scalarQuery.isError,
    onRefresh: () => {
      void scalarQuery.refetch();
      void checkpointQuery.refetch();
    },
    isError: scalarQuery.isError,
    error: scalarQuery.error,
    emptyState,
    onSelectRun: state.setSelectedDetailRunId,
  };
}
