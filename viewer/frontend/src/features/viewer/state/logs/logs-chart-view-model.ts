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
  type ChecklistOption,
  type LogMetricGroupKey,
  groupLogMetricTags,
  groupRenderableLogMetrics,
  isTestMetricTag,
} from "@/features/viewer/state/logs/logs-selectors";
import {
  buildConfusionMatrixHeatmaps,
  pairValidationExampleMedia,
  selectValidationExampleMediaTags,
} from "@/features/viewer/state/logs/log-diagnostics";
import {
  DEFAULT_LOG_METRIC_DIRECTION,
  DEFAULT_LOG_METRIC_POINT_POLICY,
  buildLogMetricDatasetRankingRows,
  inferLogMetricDirection,
  type LogMetricDatasetRankingRow,
  type LogMetricDirection,
  type LogMetricPointPolicy,
} from "@/features/viewer/state/logs/log-metric-ranking";

export type LogsChartEmptyState = {
  title: string;
  detail: string;
  busy?: boolean;
};

export type ScalarChartGridMode = "full" | "two" | "three";
export type LogMetricChartLayoutGroupKey = Exclude<LogMetricGroupKey, "test">;
export type {
  LogMetricDatasetRankingRow,
  LogMetricDirection,
  LogMetricPointPolicy,
};

export type LogScalarQueryInput = {
  runIds: string[];
  tags: string[];
  enabled: boolean;
  group?: string;
  queryKey: readonly unknown[];
};

export type LogMetricGroupScalarQueryState = {
  isInitialLoading: boolean;
  isFetching: boolean;
  isError: boolean;
  error: unknown;
};

export type LogMetricGroupScalarQueryStates = Record<
  LogMetricGroupKey,
  LogMetricGroupScalarQueryState
>;

export type LogBestRunViewModel = {
  metricTagOptions: ChecklistOption[];
  selectedMetricTag: string | null;
  selectedDirection: LogMetricDirection;
  selectedPointPolicy: LogMetricPointPolicy;
  rows: LogMetricDatasetRankingRow[];
  visibleRunCount: number;
  hasMoreRuns: boolean;
  isLoading: boolean;
  isFetching: boolean;
  isError: boolean;
  error: unknown;
  onMetricTagChange: (tag: string) => void;
  onDirectionChange: (direction: LogMetricDirection) => void;
  onPointPolicyChange: (policy: LogMetricPointPolicy) => void;
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

export type LogMetricGroupScalarQuerySnapshot =
  LogMetricGroupScalarQueryState & {
    active: boolean;
  };

function scopedScalarQueryState({
  active,
  error,
  isError,
  isFetching,
  isInitialLoading,
}: LogMetricGroupScalarQuerySnapshot): LogMetricGroupScalarQueryState {
  return {
    isInitialLoading: active && isInitialLoading,
    isFetching: active && isFetching,
    isError: active && isError,
    error: active && isError ? error : null,
  };
}

export function deriveLogMetricGroupScalarQueryStates(
  snapshots: Record<LogMetricGroupKey, LogMetricGroupScalarQuerySnapshot>,
): LogMetricGroupScalarQueryStates {
  return {
    train: scopedScalarQueryState(snapshots.train),
    validation: scopedScalarQueryState(snapshots.validation),
    test: scopedScalarQueryState(snapshots.test),
    other: scopedScalarQueryState(snapshots.other),
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

const BEST_RUN_DEFAULT_TAGS = [
  "validation/accuracy",
  "test/accuracy",
  "train/accuracy",
];

const DEFAULT_LOG_METRIC_GRID_MODES: Record<
  LogMetricChartLayoutGroupKey,
  ScalarChartGridMode
> = {
  train: "full",
  validation: "full",
  other: "full",
};

export function defaultLogBestRunMetricTag(tagOptions: ChecklistOption[]) {
  const availableTags = new Set(tagOptions.map((option) => option.value));
  return (
    BEST_RUN_DEFAULT_TAGS.find((tag) => availableTags.has(tag)) ??
    tagOptions[0]?.value ??
    null
  );
}

export function deriveLogsChartEmptyState({
  expandedSelectedTagCount,
  confusionMatrixTagCount,
  hasEventFiles,
  runsLoading,
  scalarLoading,
  selectedSeriesCount,
  selectedTagCount,
  tagOptionCount,
  tagsLoading,
  tagsRefreshing,
  visibleRunCount,
}: {
  expandedSelectedTagCount: number;
  confusionMatrixTagCount: number;
  hasEventFiles: boolean;
  runsLoading: boolean;
  scalarLoading: boolean;
  selectedSeriesCount: number;
  selectedTagCount: number;
  tagOptionCount: number;
  tagsLoading: boolean;
  tagsRefreshing: boolean;
  visibleRunCount: number;
}): LogsChartEmptyState | null {
  let title = "";
  let detail = "";
  const hasConfusionMatrixTags = confusionMatrixTagCount > 0;

  if (runsLoading) {
    title = "Scanning logs";
    detail = "Reading historical run folders.";
  } else if (visibleRunCount === 0) {
    title = "No runs selected";
    detail = "Use the sidebar filters to include at least one historical run.";
  } else if (tagsLoading) {
    title = "Reading TensorBoard tags";
    detail = "Collecting scalar tags from the selected runs.";
  } else if (tagsRefreshing && selectedSeriesCount === 0) {
    title = "Refreshing TensorBoard tags";
    detail = "Collecting scalar tags from the selected runs.";
  } else if (tagOptionCount === 0 && !hasConfusionMatrixTags) {
    title = "No TensorBoard scalars";
    detail = "The selected runs do not contain scalar event data.";
  } else if (selectedTagCount === 0 && !hasConfusionMatrixTags) {
    title = "No scalar tags selected";
    detail = "Select one or more scalar tags to draw historical charts.";
  } else if (expandedSelectedTagCount === 0) {
    return null;
  } else if (scalarLoading && selectedSeriesCount === 0) {
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
    busy: runsLoading || tagsLoading || tagsRefreshing || scalarLoading,
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

function scalarQuerySnapshot(
  query: ReturnType<typeof useLogScalarsQuery>,
  active: boolean,
): LogMetricGroupScalarQuerySnapshot {
  return {
    active,
    isInitialLoading: query.isLoading,
    isFetching: query.isFetching,
    isError: query.isError,
    error: query.error,
  };
}

export function useLogsChartViewModel(state: LogsWorkspaceState) {
  const [accordionGridMode, setAccordionGridMode] =
    useState<ScalarChartGridMode>("two");
  const [metricGridModes, setMetricGridModes] = useState<
    Record<LogMetricChartLayoutGroupKey, ScalarChartGridMode>
  >(() => ({ ...DEFAULT_LOG_METRIC_GRID_MODES }));
  const [smoothing, setSmoothing] = useState(0);
  const [xMode, setXMode] = useState<ScalarXMode>("step");
  const [yScale, setYScale] = useState<ScalarYScale>("linear");
  const [selectedBestRunMetricTag, setSelectedBestRunMetricTag] = useState<
    string | null
  >(null);
  const [selectedBestRunDirection, setSelectedBestRunDirection] =
    useState<LogMetricDirection | null>(null);
  const [selectedBestRunPointPolicy, setSelectedBestRunPointPolicy] =
    useState<LogMetricPointPolicy>(DEFAULT_LOG_METRIC_POINT_POLICY);
  const [isValidationExamplesCollapsed, setIsValidationExamplesCollapsed] =
    useState(true);
  const [isConfusionMatrixCollapsed, setIsConfusionMatrixCollapsed] =
    useState(true);
  const [validationExamplesVisible, setValidationExamplesVisible] = useState(false);
  const toggleValidationExamples = useCallback(() => {
    if (isValidationExamplesCollapsed) {
      setValidationExamplesVisible(true);
    }
    setIsValidationExamplesCollapsed((previous) => !previous);
  }, [isValidationExamplesCollapsed]);
  const markValidationExamplesVisible = useCallback(
    () => setValidationExamplesVisible(true),
    [],
  );
  const toggleConfusionMatrix = useCallback(
    () => setIsConfusionMatrixCollapsed((previous) => !previous),
    [],
  );
  const handleMetricGridModeChange = useCallback(
    (group: LogMetricChartLayoutGroupKey, mode: ScalarChartGridMode) => {
      setMetricGridModes((current) =>
        current[group] === mode ? current : { ...current, [group]: mode },
      );
    },
    [],
  );
  const selectedTagsByGroup = useMemo(
    () => groupLogMetricTags(state.selectedTagList),
    [state.selectedTagList],
  );
  const bestRunMetricTagValues = useMemo(
    () => new Set(state.tagOptions.map((option) => option.value)),
    [state.tagOptions],
  );
  const defaultBestRunMetricTag = useMemo(
    () => defaultLogBestRunMetricTag(state.tagOptions),
    [state.tagOptions],
  );
  const effectiveBestRunMetricTag =
    selectedBestRunMetricTag !== null &&
    bestRunMetricTagValues.has(selectedBestRunMetricTag)
      ? selectedBestRunMetricTag
      : defaultBestRunMetricTag;
  const inferredBestRunDirection = effectiveBestRunMetricTag
    ? inferLogMetricDirection(effectiveBestRunMetricTag)
    : DEFAULT_LOG_METRIC_DIRECTION;
  const effectiveBestRunDirection =
    selectedBestRunDirection ?? inferredBestRunDirection;
  const tagsAreRefreshing = Boolean(state.tagsQuery.isPlaceholderData);
  const trainScalarQueryActive =
    state.enabled &&
    !tagsAreRefreshing &&
    !state.collapsedMetricGroups.has("train") &&
    selectedTagsByGroup.train.length > 0 &&
    state.visibleRunIds.length > 0;
  const validationScalarQueryActive =
    state.enabled &&
    !tagsAreRefreshing &&
    !state.collapsedMetricGroups.has("validation") &&
    selectedTagsByGroup.validation.length > 0 &&
    state.visibleRunIds.length > 0;
  const testScalarQueryActive =
    state.enabled &&
    !tagsAreRefreshing &&
    selectedTagsByGroup.test.length > 0 &&
    state.visibleRunIds.length > 0;
  const otherScalarQueryActive =
    state.enabled &&
    !tagsAreRefreshing &&
    !state.collapsedMetricGroups.has("other") &&
    selectedTagsByGroup.other.length > 0 &&
    state.visibleRunIds.length > 0;
  const trainScalarQuery = useLogScalarsQuery(
    buildLogScalarQueryInput({
      enabled: trainScalarQueryActive,
      group: "train",
      selectedTagList: selectedTagsByGroup.train,
      visibleRunIds: state.visibleRunIds,
    }),
  );
  const validationScalarQuery = useLogScalarsQuery(
    buildLogScalarQueryInput({
      enabled: validationScalarQueryActive,
      group: "validation",
      selectedTagList: selectedTagsByGroup.validation,
      visibleRunIds: state.visibleRunIds,
    }),
  );
  const testScalarQuery = useLogScalarsQuery(
    buildLogScalarQueryInput({
      enabled: testScalarQueryActive,
      group: "test",
      selectedTagList: selectedTagsByGroup.test,
      visibleRunIds: state.visibleRunIds,
    }),
  );
  const otherScalarQuery = useLogScalarsQuery(
    buildLogScalarQueryInput({
      enabled: otherScalarQueryActive,
      group: "other",
      selectedTagList: selectedTagsByGroup.other,
      visibleRunIds: state.visibleRunIds,
    }),
  );
  const confusionMatrixScalarQueryActive =
    state.enabled &&
    !tagsAreRefreshing &&
    !isConfusionMatrixCollapsed &&
    state.visibleRunIds.length > 0 &&
    state.confusionMatrixRateTags.length > 0;
  const confusionMatrixScalarQuery = useLogScalarsQuery(
    buildLogScalarQueryInput({
      enabled: confusionMatrixScalarQueryActive,
      group: "confusion-matrix",
      selectedTagList: state.confusionMatrixRateTags,
      visibleRunIds: state.visibleRunIds,
    }),
  );
  const bestRunScalarQueryActive =
    state.enabled &&
    !tagsAreRefreshing &&
    state.visibleRunIds.length > 0 &&
    Boolean(effectiveBestRunMetricTag);
  const bestRunScalarQuery = useLogScalarsQuery(
    buildLogScalarQueryInput({
      enabled: bestRunScalarQueryActive,
      group: "best-run",
      selectedTagList: effectiveBestRunMetricTag ? [effectiveBestRunMetricTag] : [],
      visibleRunIds: state.visibleRunIds,
    }),
  );
  const scalarQueryEntries = [
    {
      query: trainScalarQuery,
      active: trainScalarQueryActive,
    },
    {
      query: validationScalarQuery,
      active: validationScalarQueryActive,
    },
    {
      query: testScalarQuery,
      active: testScalarQueryActive,
    },
    {
      query: otherScalarQuery,
      active: otherScalarQueryActive,
    },
  ];
  const scalarQueryStates = deriveLogMetricGroupScalarQueryStates({
    train: scalarQuerySnapshot(trainScalarQuery, trainScalarQueryActive),
    validation: scalarQuerySnapshot(
      validationScalarQuery,
      validationScalarQueryActive,
    ),
    test: scalarQuerySnapshot(testScalarQuery, testScalarQueryActive),
    other: scalarQuerySnapshot(otherScalarQuery, otherScalarQueryActive),
  });
  const scalarQueries = scalarQueryEntries.map((entry) => entry.query);
  const activeScalarQueries = scalarQueryEntries
    .filter((entry) => entry.active)
    .map((entry) => entry.query);
  const confusionMatrixQueryState = scopedScalarQueryState(
    scalarQuerySnapshot(
      confusionMatrixScalarQuery,
      confusionMatrixScalarQueryActive,
    ),
  );
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
    enabled:
      state.enabled &&
      !isValidationExamplesCollapsed &&
      validationExamplesVisible &&
      !tagsAreRefreshing,
    queryKey: logQueryKeys.mediaForRunsAndTags(
      state.visibleRunIds,
      mediaTags.imageTags,
      mediaTags.textTags,
    ),
  });
  const hasExpandedCheckpointChart =
    LOG_METRIC_GROUPS.some((group) => {
      if (group.key !== "test" && state.collapsedMetricGroups.has(group.key)) {
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
  const confusionMatrixSeriesByTag = useMemo(
    () => groupLogScalarSeriesByTag(confusionMatrixScalarQuery.data?.series ?? []),
    [confusionMatrixScalarQuery.data?.series],
  );
  const visibleRunsById = useMemo(
    () => runsById(state.visibleRuns),
    [state.visibleRuns],
  );
  const visibleCheckpointsByRunId = useMemo(
    () => checkpointsByRunId(checkpoints ?? []),
    [checkpoints],
  );
  const bestRunRows = useMemo(
    () =>
      buildLogMetricDatasetRankingRows({
        direction: effectiveBestRunDirection,
        pointPolicy: selectedBestRunPointPolicy,
        runOrder: state.visibleRunIds,
        runs: state.visibleRuns,
        series: bestRunScalarQuery.data?.series ?? [],
        tag: effectiveBestRunMetricTag,
      }),
    [
      bestRunScalarQuery.data?.series,
      effectiveBestRunDirection,
      effectiveBestRunMetricTag,
      selectedBestRunPointPolicy,
      state.visibleRunIds,
      state.visibleRuns,
    ],
  );
  const bestRun = useMemo<LogBestRunViewModel>(
    () => ({
      metricTagOptions: state.tagOptions,
      selectedMetricTag: effectiveBestRunMetricTag,
      selectedDirection: effectiveBestRunDirection,
      selectedPointPolicy: selectedBestRunPointPolicy,
      rows: bestRunRows,
      visibleRunCount: state.visibleRuns.length,
      hasMoreRuns: Boolean(state.runsQuery.data?.hasMore),
      isLoading: bestRunScalarQueryActive && bestRunScalarQuery.isLoading,
      isFetching: bestRunScalarQueryActive && bestRunScalarQuery.isFetching,
      isError: bestRunScalarQueryActive && bestRunScalarQuery.isError,
      error:
        bestRunScalarQueryActive && bestRunScalarQuery.isError
          ? bestRunScalarQuery.error
          : null,
      onMetricTagChange: (tag) => {
        setSelectedBestRunMetricTag(tag);
        setSelectedBestRunDirection(null);
      },
      onDirectionChange: setSelectedBestRunDirection,
      onPointPolicyChange: setSelectedBestRunPointPolicy,
    }),
    [
      bestRunRows,
      bestRunScalarQuery.error,
      bestRunScalarQuery.isError,
      bestRunScalarQuery.isFetching,
      bestRunScalarQuery.isLoading,
      bestRunScalarQueryActive,
      effectiveBestRunDirection,
      effectiveBestRunMetricTag,
      selectedBestRunPointPolicy,
      state.runsQuery.data?.hasMore,
      state.tagOptions,
      state.visibleRuns.length,
    ],
  );
  const expandedSelectedTagCount = LOG_METRIC_GROUPS.reduce((total, group) => {
    if (group.key === "test" || state.collapsedMetricGroups.has(group.key)) {
      return total;
    }
    return total + selectedTagsByGroup[group.key].length;
  }, 0);
  const expandedSelectedSeriesCount = LOG_METRIC_GROUPS.reduce((total, group) => {
    if (group.key === "test" || state.collapsedMetricGroups.has(group.key)) {
      return total;
    }
    return (
      total +
      selectedTagsByGroup[group.key].reduce(
        (groupTotal, tag) => groupTotal + (seriesByTag.get(tag)?.length ?? 0),
        0,
      )
    );
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
        matrixTagList: state.confusionMatrixRateTags,
        seriesByTag: confusionMatrixSeriesByTag,
        runsById: visibleRunsById,
        runOrder: state.visibleRunIds,
      }),
    [
      confusionMatrixSeriesByTag,
      state.confusionMatrixRateTags,
      state.visibleRunIds,
      visibleRunsById,
    ],
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
    confusionMatrixTagCount: state.confusionMatrixRateTags.length,
    hasEventFiles: state.visibleRuns.some((run) => run.eventFileCount > 0),
    runsLoading: state.runsQuery.isLoading,
    scalarLoading: activeScalarQueries.some((query) => query.isLoading),
    selectedSeriesCount: expandedSelectedSeriesCount,
    selectedTagCount: state.selectedTagList.length,
    tagOptionCount: state.tagOptions.length,
    tagsLoading:
      state.tagsQuery.isLoading &&
      state.tagOptions.length === 0 &&
      state.confusionMatrixRateTags.length === 0,
    tagsRefreshing: tagsAreRefreshing,
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
    scalarQueryStates,
    hasConfusionMatrixTags: state.confusionMatrixRateTags.length > 0,
    isConfusionMatrixCollapsed,
    isConfusionMatrixLoaded:
      confusionMatrixScalarQueryActive && confusionMatrixScalarQuery.isSuccess,
    isConfusionMatrixLoading: confusionMatrixQueryState.isInitialLoading,
    isConfusionMatrixError: confusionMatrixQueryState.isError,
    confusionMatrixError: confusionMatrixQueryState.error,
    onToggleConfusionMatrix: toggleConfusionMatrix,
    isTagRefreshLoading:
      state.tagsQuery.isFetching &&
      (state.tagOptions.length > 0 || state.confusionMatrixRateTags.length > 0) &&
      state.visibleRunIds.length > 0,
    collapsedMetricGroups: state.collapsedMetricGroups,
    onToggleMetricGroup: state.toggleMetricGroup,
    accordionGridMode,
    onAccordionGridModeChange: setAccordionGridMode,
    metricGridModes,
    onMetricGridModeChange: handleMetricGridModeChange,
    smoothing,
    onSmoothingChange: setSmoothing,
    xMode,
    onXModeChange: setXMode,
    yScale,
    onYScaleChange: setYScale,
    isFetching:
      activeScalarQueries.some((query) => query.isFetching) ||
      (bestRunScalarQueryActive && bestRunScalarQuery.isFetching) ||
      confusionMatrixQueryState.isFetching ||
      checkpointQuery.isFetching ||
      mediaQuery.isFetching,
    isRefreshDisabled:
      !activeScalarQueries.some((query) => query.isSuccess || query.isError) &&
      !(
        bestRunScalarQueryActive &&
        (bestRunScalarQuery.isSuccess || bestRunScalarQuery.isError)
      ) &&
      !(
        confusionMatrixScalarQueryActive &&
        (confusionMatrixScalarQuery.isSuccess || confusionMatrixScalarQuery.isError)
      ),
    onRefresh: () => {
      activeScalarQueries.forEach((query) => {
        void query.refetch();
      });
      if (bestRunScalarQueryActive) {
        void bestRunScalarQuery.refetch();
      }
      if (confusionMatrixScalarQueryActive) {
        void confusionMatrixScalarQuery.refetch();
      }
      if (hasExpandedCheckpointChart) {
        void checkpointQuery.refetch();
      }
      if (!isValidationExamplesCollapsed && validationExamplesVisible) {
        void mediaQuery.refetch();
      }
    },
    emptyState,
    selectedTagsByGroup,
    hasValidationExampleMedia,
    isValidationExampleMediaLoading: mediaQuery.isLoading,
    isValidationExamplesCollapsed,
    onToggleValidationExamples: toggleValidationExamples,
    onValidationExamplesVisible: markValidationExamplesVisible,
    bestRun,
    onSelectRun: state.setSelectedDetailRunId,
  };
}
