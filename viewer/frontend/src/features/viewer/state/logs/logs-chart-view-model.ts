import { useCallback, useMemo, useState } from "react";
import {
  DEFAULT_LOG_SCALAR_MAX_POINTS,
  LOG_SCALAR_SAMPLING,
  type LogCheckpoint,
  type LogRun,
  type LogScalarSeries,
} from "@/lib/api";
import { type ScalarXMode, type ScalarYScale } from "@/lib/echarts/scalar-options";
import {
  useLogCheckpointsQuery,
  useLogMediaQuery,
  useLogScalarsQuery,
} from "@/features/viewer/state/logs/use-log-queries";
import { logQueryKeys } from "@/lib/query-keys";
import { type LogsWorkspaceState } from "@/features/viewer/state/logs/use-logs-workspace-state";
import {
  LOG_METRIC_GROUPS,
  groupLogMetricTags,
  groupRenderableLogMetrics,
  isTestMetricTag,
} from "@/features/viewer/state/logs/logs-selectors";
import {
  buildConfusionMatrixHeatmaps,
  pairValidationExampleMedia,
  selectValidationExampleMediaTags,
} from "@/features/viewer/state/logs/log-diagnostics";

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
  group?: string;
  queryKey: readonly unknown[];
};

export function buildLogScalarQueryInput({
  enabled,
  group,
  selectedTagList,
  visibleRunIds,
}: {
  enabled: boolean;
  group?: string;
  selectedTagList: string[];
  visibleRunIds: string[];
}): LogScalarQueryInput {
  return {
    runIds: visibleRunIds,
    tags: selectedTagList,
    enabled: enabled && visibleRunIds.length > 0 && selectedTagList.length > 0,
    group,
    queryKey: logQueryKeys.scalarsForRunsAndTags(visibleRunIds, selectedTagList, {
      group,
      maxPoints: DEFAULT_LOG_SCALAR_MAX_POINTS,
      sampling: LOG_SCALAR_SAMPLING,
    }),
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
  expandedSelectedTagCount,
  hasEventFiles,
  runsLoading,
  scalarLoading,
  selectedSeriesCount,
  selectedTagCount,
  tagOptionCount,
  tagsLoading,
  visibleRunCount,
}: {
  expandedSelectedTagCount: number;
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
  } else if (expandedSelectedTagCount === 0) {
    return null;
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
  const [validationExamplesVisible, setValidationExamplesVisible] = useState(false);
  const markValidationExamplesVisible = useCallback(
    () => setValidationExamplesVisible(true),
    [],
  );
  const selectedTagsByGroup = useMemo(
    () => groupLogMetricTags(state.selectedTagList),
    [state.selectedTagList],
  );
  const trainScalarQuery = useLogScalarsQuery(
    buildLogScalarQueryInput({
      enabled: state.enabled && !state.collapsedMetricGroups.has("train"),
      group: "train",
      selectedTagList: selectedTagsByGroup.train,
      visibleRunIds: state.visibleRunIds,
    }),
  );
  const validationScalarQuery = useLogScalarsQuery(
    buildLogScalarQueryInput({
      enabled: state.enabled && !state.collapsedMetricGroups.has("validation"),
      group: "validation",
      selectedTagList: selectedTagsByGroup.validation,
      visibleRunIds: state.visibleRunIds,
    }),
  );
  const testScalarQuery = useLogScalarsQuery(
    buildLogScalarQueryInput({
      enabled: state.enabled && !state.collapsedMetricGroups.has("test"),
      group: "test",
      selectedTagList: selectedTagsByGroup.test,
      visibleRunIds: state.visibleRunIds,
    }),
  );
  const otherScalarQuery = useLogScalarsQuery(
    buildLogScalarQueryInput({
      enabled: state.enabled && !state.collapsedMetricGroups.has("other"),
      group: "other",
      selectedTagList: selectedTagsByGroup.other,
      visibleRunIds: state.visibleRunIds,
    }),
  );
  const scalarQueryEntries = [
    {
      query: trainScalarQuery,
      active:
        state.enabled &&
        !state.collapsedMetricGroups.has("train") &&
        selectedTagsByGroup.train.length > 0 &&
        state.visibleRunIds.length > 0,
    },
    {
      query: validationScalarQuery,
      active:
        state.enabled &&
        !state.collapsedMetricGroups.has("validation") &&
        selectedTagsByGroup.validation.length > 0 &&
        state.visibleRunIds.length > 0,
    },
    {
      query: testScalarQuery,
      active:
        state.enabled &&
        !state.collapsedMetricGroups.has("test") &&
        selectedTagsByGroup.test.length > 0 &&
        state.visibleRunIds.length > 0,
    },
    {
      query: otherScalarQuery,
      active:
        state.enabled &&
        !state.collapsedMetricGroups.has("other") &&
        selectedTagsByGroup.other.length > 0 &&
        state.visibleRunIds.length > 0,
    },
  ];
  const scalarQueries = scalarQueryEntries.map((entry) => entry.query);
  const activeScalarQueries = scalarQueryEntries
    .filter((entry) => entry.active)
    .map((entry) => entry.query);
  const mediaTags = useMemo(
    () => selectValidationExampleMediaTags(state.tagsQuery.data),
    [state.tagsQuery.data],
  );
  const hasValidationExampleMedia =
    mediaTags.imageTags.length > 0 || mediaTags.textTags.length > 0;
  const mediaQuery = useLogMediaQuery({
    runIds: state.visibleRunIds,
    imageTags: mediaTags.imageTags,
    textTags: mediaTags.textTags,
    enabled: state.enabled && validationExamplesVisible,
    queryKey: logQueryKeys.mediaForRunsAndTags(
      state.visibleRunIds,
      mediaTags.imageTags,
      mediaTags.textTags,
    ),
  });
  const hasExpandedCheckpointChart =
    LOG_METRIC_GROUPS.some((group) => {
      if (state.collapsedMetricGroups.has(group.key)) {
        return false;
      }
      return selectedTagsByGroup[group.key].some((tag) => !isTestMetricTag(tag));
    }) && state.visibleRuns.some((run) => run.checkpointCount > 0);
  const checkpointQuery = useLogCheckpointsQuery({
    runIds: state.visibleRunIds,
    enabled: state.enabled && hasExpandedCheckpointChart,
    queryKey: logQueryKeys.checkpointsForRuns(state.visibleRunIds),
  });
  const scalarSeries = scalarQueries.flatMap((query) => query.data?.series ?? []);
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
  const expandedSelectedTagCount = LOG_METRIC_GROUPS.reduce((total, group) => {
    if (state.collapsedMetricGroups.has(group.key)) {
      return total;
    }
    return total + selectedTagsByGroup[group.key].length;
  }, 0);
  const metricsByGroup = useMemo(
    () =>
      groupRenderableLogMetrics({
        selectedTagList: state.selectedTagList,
        seriesByTag,
      }),
    [seriesByTag, state.selectedTagList],
  );
  const confusionHeatmaps = useMemo(
    () =>
      buildConfusionMatrixHeatmaps({
        selectedTagList: state.selectedTagList,
        seriesByTag,
        runsById: visibleRunsById,
        runOrder: state.visibleRunIds,
      }),
    [seriesByTag, state.selectedTagList, state.visibleRunIds, visibleRunsById],
  );
  const validationMedia = useMemo(
    () =>
      pairValidationExampleMedia({
        images: mediaQuery.data?.images ?? [],
        texts: mediaQuery.data?.texts ?? [],
      }),
    [mediaQuery.data?.images, mediaQuery.data?.texts],
  );
  const emptyState = deriveLogsChartEmptyState({
    expandedSelectedTagCount,
    hasEventFiles: state.visibleRuns.some((run) => run.eventFileCount > 0),
    runsLoading: state.runsQuery.isLoading,
    scalarLoading: scalarQueries.some((query) => query.isLoading),
    selectedSeriesCount,
    selectedTagCount: state.selectedTagList.length,
    tagOptionCount: state.tagOptions.length,
    tagsLoading: state.tagsQuery.isLoading,
    visibleRunCount: state.visibleRuns.length,
  });

  return {
    metricsByGroup,
    confusionHeatmaps,
    runsById: visibleRunsById,
    checkpointsByRunId: visibleCheckpointsByRunId,
    mediaImages: validationMedia.images,
    mediaTexts: validationMedia.texts,
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
    isFetching:
      activeScalarQueries.some((query) => query.isFetching) ||
      checkpointQuery.isFetching ||
      mediaQuery.isFetching,
    isRefreshDisabled: !activeScalarQueries.some(
      (query) => query.isSuccess || query.isError,
    ),
    onRefresh: () => {
      activeScalarQueries.forEach((query) => {
        void query.refetch();
      });
      if (hasExpandedCheckpointChart) {
        void checkpointQuery.refetch();
      }
      if (validationExamplesVisible) {
        void mediaQuery.refetch();
      }
    },
    isError: activeScalarQueries.some((query) => query.isError),
    error: activeScalarQueries.find((query) => query.error)?.error,
    emptyState,
    selectedTagsByGroup,
    hasValidationExampleMedia,
    isValidationExampleMediaLoading: mediaQuery.isLoading,
    onValidationExamplesVisible: markValidationExamplesVisible,
    onSelectRun: state.setSelectedDetailRunId,
  };
}
