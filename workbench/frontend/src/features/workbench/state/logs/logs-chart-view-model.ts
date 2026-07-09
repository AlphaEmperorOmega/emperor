import { useCallback, useEffect, useMemo, useState } from "react";
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
  useLogScalarQueries,
  useLogScalarsQuery,
  type LogScalarQueryInput,
} from "@/features/workbench/state/logs/use-log-queries";
import { logQueryKeys } from "@/lib/query-keys";
import { type LogsWorkspaceState } from "@/features/workbench/state/logs/use-logs-workspace-state";
import {
  LOG_METRIC_GROUPS,
  type ChecklistOption,
  type LogMetricGroupKey,
  type LogMetricsByGroup,
  type LogMetricTagsByGroup,
  type TrainValidationScalarPair,
  buildTrainValidationScalarPairs,
  defaultTrainValidationScalarPairSuffixes,
  groupLogMetricTags,
  groupLogPlotSelectorTags,
  isTestMetricTag,
  metricGroupForTag,
} from "@/features/workbench/state/logs/logs-selectors";
import {
  buildConfusionMatrixHeatmaps,
  pairValidationExampleMedia,
  selectValidationExampleMediaTags,
} from "@/features/workbench/state/logs/log-diagnostics";
import {
  DEFAULT_LOG_METRIC_DIRECTION,
  DEFAULT_LOG_METRIC_POINT_POLICY,
  buildLogMetricDatasetRankingRows,
  inferLogMetricDirection,
  type LogMetricDatasetRankingRow,
  type LogMetricDirection,
  type LogMetricPointPolicy,
} from "@/features/workbench/state/logs/log-metric-ranking";

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

export type LogMetricGroupScalarQueryState = {
  isInitialLoading: boolean;
  isFetching: boolean;
  isError: boolean;
  error: unknown;
};

export type LogScalarTagQueryState = LogMetricGroupScalarQueryState & {
  hasRequested: boolean;
};

export type LogMetricGroupScalarQueryStates = Record<
  LogMetricGroupKey,
  LogMetricGroupScalarQueryState
>;

export type LogTrainValidationComparisonMetric = TrainValidationScalarPair & {
  series: LogScalarSeries[];
};

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

export function bestRunMetricGroupForActiveScalarQuery({
  activeGroups,
  selectedTagsByGroup,
  tag,
}: {
  activeGroups: Record<LogMetricGroupKey, boolean>;
  selectedTagsByGroup: LogMetricTagsByGroup;
  tag: string | null;
}): LogMetricGroupKey | null {
  if (!tag) {
    return null;
  }
  const group = metricGroupForTag(tag);
  return activeGroups[group] && selectedTagsByGroup[group].includes(tag)
    ? group
    : null;
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
export const LOG_SCALAR_TAG_CHUNK_SIZE = 6;
export const LOG_SCALAR_RUN_CHUNK_SIZE = 2;

export function defaultLogBestRunMetricTag(tagOptions: ChecklistOption[]) {
  const availableTags = new Set(tagOptions.map((option) => option.value));
  return (
    BEST_RUN_DEFAULT_TAGS.find((tag) => availableTags.has(tag)) ??
    tagOptions[0]?.value ??
    null
  );
}

export function chunkScalarTagsForQueries(
  tags: string[],
  chunkSize = LOG_SCALAR_TAG_CHUNK_SIZE,
) {
  const size = Math.max(1, Math.floor(chunkSize));
  const uniqueTags = Array.from(new Set(tags));
  const chunks: string[][] = [];
  for (let index = 0; index < uniqueTags.length; index += size) {
    chunks.push(uniqueTags.slice(index, index + size));
  }
  return chunks;
}

export function chunkScalarRunIdsForQueries(
  runIds: string[],
  chunkSize = LOG_SCALAR_RUN_CHUNK_SIZE,
) {
  const size = Math.max(1, Math.floor(chunkSize));
  const uniqueRunIds = Array.from(new Set(runIds));
  const chunks: string[][] = [];
  for (let index = 0; index < uniqueRunIds.length; index += size) {
    chunks.push(uniqueRunIds.slice(index, index + size));
  }
  return chunks;
}

export function groupSelectedLogMetrics({
  selectedTagList,
  seriesByTag,
}: {
  selectedTagList: string[];
  seriesByTag: Map<string, LogScalarSeries[]>;
}): LogMetricsByGroup {
  const groups: LogMetricsByGroup = {
    train: [],
    validation: [],
    test: [],
    other: [],
  };

  for (const tag of selectedTagList) {
    groups[metricGroupForTag(tag)].push({
      tag,
      series: seriesByTag.get(tag) ?? [],
    });
  }

  return groups;
}

export function buildLogScalarChunkQueryInputs({
  enabled,
  group,
  requestedTags,
  selectedTagList,
  visibleRunIds,
}: {
  enabled: boolean;
  group: string;
  requestedTags: Set<string>;
  selectedTagList: string[];
  visibleRunIds: string[];
}) {
  const requestedSelectedTags = selectedTagList.filter((tag) =>
    requestedTags.has(tag),
  );
  const tagChunks = chunkScalarTagsForQueries(requestedSelectedTags);
  const runChunks = chunkScalarRunIdsForQueries(visibleRunIds);
  return runChunks.flatMap((runIds) =>
    tagChunks.map((tags) =>
      buildLogScalarQueryInput({
        enabled,
        group,
        selectedTagList: tags,
        visibleRunIds: runIds,
      }),
    ),
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
  } else if (selectedSeriesCount === 0 && !scalarLoading && !hasEventFiles) {
    return null;
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
  query: Pick<
    ReturnType<typeof useLogScalarsQuery>,
    "error" | "isError" | "isFetching" | "isLoading" | "isPlaceholderData"
  >,
  active: boolean,
): LogMetricGroupScalarQuerySnapshot {
  return {
    active,
    isInitialLoading: active && (query.isLoading || query.isPlaceholderData),
    isFetching: active && query.isFetching,
    isError: active && query.isError,
    error: active && query.isError ? query.error : null,
  };
}

function combinedScalarQuerySnapshot(
  queries: Array<
    Pick<
      ReturnType<typeof useLogScalarsQuery>,
      "error" | "isError" | "isFetching" | "isLoading" | "isPlaceholderData"
    >
  >,
  active: boolean,
): LogMetricGroupScalarQuerySnapshot {
  const errorQuery = queries.find((query) => query.isError);
  return {
    active,
    isInitialLoading:
      active && queries.some((query) => query.isLoading || query.isPlaceholderData),
    isFetching: active && queries.some((query) => query.isFetching),
    isError: active && Boolean(errorQuery),
    error: active && errorQuery ? errorQuery.error : null,
  };
}

function visibleScalarSeries(
  data: { series: LogScalarSeries[] } | undefined,
  isPlaceholderData: boolean,
  active: boolean,
): LogScalarSeries[] {
  if (!active || isPlaceholderData) {
    return [];
  }
  return data?.series ?? [];
}

function mergeLogScalarTagQueryState(
  states: Map<string, LogScalarTagQueryState>,
  tag: string,
  snapshot: LogMetricGroupScalarQueryState,
  hasRequested: boolean,
) {
  const current = states.get(tag);
  states.set(tag, {
    hasRequested: (current?.hasRequested ?? false) || hasRequested,
    isInitialLoading:
      (current?.isInitialLoading ?? false) || snapshot.isInitialLoading,
    isFetching: (current?.isFetching ?? false) || snapshot.isFetching,
    isError: (current?.isError ?? false) || snapshot.isError,
    error: current?.error ?? snapshot.error,
  });
}

export function useLogsChartViewModel(state: LogsWorkspaceState) {
  const [accordionGridMode, setAccordionGridMode] =
    useState<ScalarChartGridMode>("two");
  const [trainValidationComparisonGridMode, setTrainValidationComparisonGridMode] =
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
  const [
    isTrainValidationComparisonCollapsed,
    setIsTrainValidationComparisonCollapsed,
  ] = useState(true);
  const [
    selectedTrainValidationPairSuffixes,
    setSelectedTrainValidationPairSuffixes,
  ] = useState<Set<string> | null>(null);
  const [validationExamplesVisible, setValidationExamplesVisible] = useState(false);
  const [requestedScalarTags, setRequestedScalarTags] = useState<Set<string>>(
    () => new Set(),
  );
  const [
    requestedTrainValidationPairSuffixes,
    setRequestedTrainValidationPairSuffixes,
  ] = useState<Set<string>>(() => new Set());
  const toggleTrainValidationComparison = useCallback(
    () => setIsTrainValidationComparisonCollapsed((previous) => !previous),
    [],
  );
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
  const selectedScalarTagSet = useMemo(
    () => new Set(state.selectedTagList),
    [state.selectedTagList],
  );
  const trainValidationPairs = useMemo(
    () => buildTrainValidationScalarPairs(state.tagOptions),
    [state.tagOptions],
  );
  const defaultTrainValidationPairSuffixes = useMemo(
    () => defaultTrainValidationScalarPairSuffixes(trainValidationPairs),
    [trainValidationPairs],
  );
  const selectedTrainValidationPairSuffixList = useMemo(() => {
    const selectedSuffixes =
      selectedTrainValidationPairSuffixes ??
      new Set(defaultTrainValidationPairSuffixes);
    return trainValidationPairs
      .map((pair) => pair.suffix)
      .filter((suffix) => selectedSuffixes.has(suffix));
  }, [
    defaultTrainValidationPairSuffixes,
    selectedTrainValidationPairSuffixes,
    trainValidationPairs,
  ]);
  const selectedTrainValidationPairSuffixSet = useMemo(
    () => new Set(selectedTrainValidationPairSuffixList),
    [selectedTrainValidationPairSuffixList],
  );
  const selectedTrainValidationPairs = useMemo(
    () =>
      trainValidationPairs.filter((pair) =>
        selectedTrainValidationPairSuffixSet.has(pair.suffix),
      ),
    [selectedTrainValidationPairSuffixSet, trainValidationPairs],
  );
  const selectedTrainValidationScalarTags = useMemo(
    () =>
      selectedTrainValidationPairs.flatMap((pair) => [
        pair.trainTag,
        pair.validationTag,
      ]),
    [selectedTrainValidationPairs],
  );
  useEffect(() => {
    setRequestedScalarTags((current) => {
      const next = new Set(
        Array.from(current).filter((tag) => selectedScalarTagSet.has(tag)),
      );
      return next.size === current.size ? current : next;
    });
  }, [selectedScalarTagSet]);
  useEffect(() => {
    setRequestedTrainValidationPairSuffixes((current) => {
      const next = new Set(
        Array.from(current).filter((suffix) =>
          selectedTrainValidationPairSuffixSet.has(suffix),
        ),
      );
      return next.size === current.size ? current : next;
    });
  }, [selectedTrainValidationPairSuffixSet]);
  const markScalarChartVisible = useCallback((tag: string) => {
    setRequestedScalarTags((current) => {
      if (current.has(tag)) {
        return current;
      }
      const next = new Set(current);
      next.add(tag);
      return next;
    });
  }, []);
  const markTrainValidationComparisonChartVisible = useCallback((suffix: string) => {
    setRequestedTrainValidationPairSuffixes((current) => {
      if (current.has(suffix)) {
        return current;
      }
      const next = new Set(current);
      next.add(suffix);
      return next;
    });
  }, []);
  const handleTrainValidationPairSelectionChange = useCallback((suffixes: string[]) => {
    setSelectedTrainValidationPairSuffixes(new Set(suffixes));
  }, []);
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
  const effectiveBestRunMetricGroup = effectiveBestRunMetricTag
    ? metricGroupForTag(effectiveBestRunMetricTag)
    : null;
  const requestedChartScalarTags = useMemo(() => {
    const next = new Set(requestedScalarTags);
    if (effectiveBestRunMetricTag && effectiveBestRunMetricGroup !== "test") {
      next.delete(effectiveBestRunMetricTag);
    }
    return next;
  }, [effectiveBestRunMetricGroup, effectiveBestRunMetricTag, requestedScalarTags]);
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
  const trainValidationComparisonQueryActive =
    state.enabled &&
    !tagsAreRefreshing &&
    !isTrainValidationComparisonCollapsed &&
    selectedTrainValidationPairs.length > 0 &&
    state.visibleRunIds.length > 0;
  const requestedTrainValidationScalarTags = useMemo(() => {
    const requestedTags = new Set<string>();
    for (const pair of selectedTrainValidationPairs) {
      if (!requestedTrainValidationPairSuffixes.has(pair.suffix)) {
        continue;
      }
      requestedTags.add(pair.trainTag);
      requestedTags.add(pair.validationTag);
    }
    return requestedTags;
  }, [requestedTrainValidationPairSuffixes, selectedTrainValidationPairs]);
  const trainScalarQueryInputs = useMemo(
    () =>
      buildLogScalarChunkQueryInputs({
        enabled: trainScalarQueryActive,
        group: "train",
        requestedTags: requestedChartScalarTags,
        selectedTagList: selectedTagsByGroup.train,
        visibleRunIds: state.visibleRunIds,
      }),
    [
      requestedChartScalarTags,
      selectedTagsByGroup.train,
      state.visibleRunIds,
      trainScalarQueryActive,
    ],
  );
  const validationScalarQueryInputs = useMemo(
    () =>
      buildLogScalarChunkQueryInputs({
        enabled: validationScalarQueryActive,
        group: "validation",
        requestedTags: requestedChartScalarTags,
        selectedTagList: selectedTagsByGroup.validation,
        visibleRunIds: state.visibleRunIds,
      }),
    [
      requestedChartScalarTags,
      selectedTagsByGroup.validation,
      state.visibleRunIds,
      validationScalarQueryActive,
    ],
  );
  const otherScalarQueryInputs = useMemo(
    () =>
      buildLogScalarChunkQueryInputs({
        enabled: otherScalarQueryActive,
        group: "other",
        requestedTags: requestedChartScalarTags,
        selectedTagList: selectedTagsByGroup.other,
        visibleRunIds: state.visibleRunIds,
      }),
    [
      otherScalarQueryActive,
      requestedChartScalarTags,
      selectedTagsByGroup.other,
      state.visibleRunIds,
    ],
  );
  const trainValidationScalarQueryInputs = useMemo(
    () =>
      buildLogScalarChunkQueryInputs({
        enabled: trainValidationComparisonQueryActive,
        group: "train-validation",
        requestedTags: requestedTrainValidationScalarTags,
        selectedTagList: selectedTrainValidationScalarTags,
        visibleRunIds: state.visibleRunIds,
      }),
    [
      requestedTrainValidationScalarTags,
      selectedTrainValidationScalarTags,
      state.visibleRunIds,
      trainValidationComparisonQueryActive,
    ],
  );
  const chartScalarQueryInputs = useMemo(
    () => [
      ...trainScalarQueryInputs,
      ...validationScalarQueryInputs,
      ...otherScalarQueryInputs,
    ],
    [otherScalarQueryInputs, trainScalarQueryInputs, validationScalarQueryInputs],
  );
  const chartScalarQueries = useLogScalarQueries(chartScalarQueryInputs);
  const trainValidationScalarQueries = useLogScalarQueries(
    trainValidationScalarQueryInputs,
  );
  const testScalarQuery = useLogScalarsQuery(
    buildLogScalarQueryInput({
      enabled: testScalarQueryActive,
      group: "test",
      selectedTagList: selectedTagsByGroup.test,
      visibleRunIds: state.visibleRunIds,
    }),
  );
  const chartScalarQueryEntries = chartScalarQueryInputs.map((input, index) => ({
    input,
    query: chartScalarQueries[index],
  }));
  const trainValidationScalarQueryEntries = trainValidationScalarQueryInputs.map(
    (input, index) => ({
      input,
      query: trainValidationScalarQueries[index],
    }),
  );
  const bestRunCoveredChartQuery =
    effectiveBestRunMetricTag && effectiveBestRunMetricGroup !== "test"
      ? chartScalarQueryEntries.find((entry) =>
          entry.input.tags.includes(effectiveBestRunMetricTag),
        )?.query ?? null
      : null;
  const bestRunCoveredTestQuery =
    effectiveBestRunMetricTag &&
    selectedTagsByGroup.test.includes(effectiveBestRunMetricTag)
      ? testScalarQuery
      : null;
  const bestRunCoveredMetricQuery =
    bestRunCoveredChartQuery ?? bestRunCoveredTestQuery;
  const bestRunScalarQueryActive =
    state.enabled &&
    !tagsAreRefreshing &&
    state.visibleRunIds.length > 0 &&
    Boolean(effectiveBestRunMetricTag) &&
    bestRunCoveredMetricQuery === null;
  const bestRunScalarQuery = useLogScalarsQuery(
    buildLogScalarQueryInput({
      enabled: bestRunScalarQueryActive,
      group: effectiveBestRunMetricGroup ?? "best-run",
      selectedTagList: effectiveBestRunMetricTag ? [effectiveBestRunMetricTag] : [],
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
  const trainScalarQueries = chartScalarQueryEntries
    .filter((entry) => entry.input.group === "train")
    .map((entry) => entry.query);
  const validationScalarQueries = chartScalarQueryEntries
    .filter((entry) => entry.input.group === "validation")
    .map((entry) => entry.query);
  const otherScalarQueries = chartScalarQueryEntries
    .filter((entry) => entry.input.group === "other")
    .map((entry) => entry.query);
  const scalarQueryStates = deriveLogMetricGroupScalarQueryStates({
    train: combinedScalarQuerySnapshot(trainScalarQueries, trainScalarQueryActive),
    validation: combinedScalarQuerySnapshot(
      validationScalarQueries,
      validationScalarQueryActive,
    ),
    test: scalarQuerySnapshot(testScalarQuery, testScalarQueryActive),
    other: combinedScalarQuerySnapshot(otherScalarQueries, otherScalarQueryActive),
  });
  const trainValidationComparisonQueryState = scopedScalarQueryState(
    combinedScalarQuerySnapshot(
      trainValidationScalarQueryEntries.map((entry) => entry.query),
      trainValidationComparisonQueryActive,
    ),
  );
  const activeScalarQueries = [
    ...chartScalarQueryEntries
      .filter((entry) => entry.input.enabled)
      .map((entry) => entry.query),
    ...trainValidationScalarQueryEntries
      .filter((entry) => entry.input.enabled)
      .map((entry) => entry.query),
    ...(testScalarQueryActive ? [testScalarQuery] : []),
  ];
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
  const hasExpandedMetricGroupCheckpointChart = LOG_METRIC_GROUPS.some((group) => {
    if (group.key !== "test" && state.collapsedMetricGroups.has(group.key)) {
      return false;
    }
    return selectedTagsByGroup[group.key].some(
      (tag) => !isTestMetricTag(tag) && requestedScalarTags.has(tag),
    );
  });
  const hasExpandedTrainValidationComparisonCheckpointChart =
    !isTrainValidationComparisonCollapsed &&
    selectedTrainValidationPairs.some((pair) =>
      requestedTrainValidationPairSuffixes.has(pair.suffix),
    );
  const hasExpandedCheckpointChart =
    (hasExpandedMetricGroupCheckpointChart ||
      hasExpandedTrainValidationComparisonCheckpointChart) &&
    state.visibleRuns.some((run) => run.checkpointCount > 0);
  const checkpointQuery = useLogCheckpointsQuery({
    runIds: state.visibleRunIds,
    enabled: state.enabled && hasExpandedCheckpointChart,
    queryKey: logQueryKeys.checkpointsForRuns(state.visibleRunIds),
  });
  const bestRunStandaloneScalarSeries = useMemo(
    () =>
      visibleScalarSeries(
        bestRunScalarQuery.data,
        bestRunScalarQuery.isPlaceholderData,
        bestRunScalarQueryActive,
      ),
    [
      bestRunScalarQuery.data,
      bestRunScalarQuery.isPlaceholderData,
      bestRunScalarQueryActive,
    ],
  );
  const scalarSeries = useMemo(
    () => [
      ...chartScalarQueryEntries.flatMap((entry) =>
        visibleScalarSeries(
          entry.query.data,
          entry.query.isPlaceholderData,
          entry.input.enabled,
        ),
      ),
      ...visibleScalarSeries(
        testScalarQuery.data,
        testScalarQuery.isPlaceholderData,
        testScalarQueryActive,
      ),
      ...bestRunStandaloneScalarSeries,
    ],
    [
      bestRunStandaloneScalarSeries,
      chartScalarQueryEntries,
      testScalarQuery.data,
      testScalarQuery.isPlaceholderData,
      testScalarQueryActive,
    ],
  );
  const trainValidationScalarSeries = useMemo(
    () =>
      trainValidationScalarQueryEntries.flatMap((entry) =>
        visibleScalarSeries(
          entry.query.data,
          entry.query.isPlaceholderData,
          entry.input.enabled,
        ),
      ),
    [trainValidationScalarQueryEntries],
  );
  const confusionMatrixScalarSeries = useMemo(
    () =>
      visibleScalarSeries(
        confusionMatrixScalarQuery.data,
        confusionMatrixScalarQuery.isPlaceholderData,
        confusionMatrixScalarQueryActive,
      ),
    [
      confusionMatrixScalarQuery.data,
      confusionMatrixScalarQuery.isPlaceholderData,
      confusionMatrixScalarQueryActive,
    ],
  );
  const checkpoints = checkpointQuery.data?.checkpoints;

  const seriesByTag = useMemo(
    () => groupLogScalarSeriesByTag(scalarSeries ?? []),
    [scalarSeries],
  );
  const trainValidationSeriesByTag = useMemo(
    () => groupLogScalarSeriesByTag(trainValidationScalarSeries),
    [trainValidationScalarSeries],
  );
  const scalarTagQueryStates = useMemo(() => {
    const states = new Map<string, LogScalarTagQueryState>();

    for (const entry of chartScalarQueryEntries) {
      const snapshot = scalarQuerySnapshot(entry.query, entry.input.enabled);
      for (const tag of entry.input.tags) {
        mergeLogScalarTagQueryState(states, tag, snapshot, true);
      }
    }

    if (effectiveBestRunMetricTag && bestRunScalarQueryActive) {
      mergeLogScalarTagQueryState(
        states,
        effectiveBestRunMetricTag,
        scalarQuerySnapshot(bestRunScalarQuery, true),
        true,
      );
    }

    if (testScalarQueryActive) {
      const snapshot = scalarQuerySnapshot(testScalarQuery, true);
      for (const tag of selectedTagsByGroup.test) {
        mergeLogScalarTagQueryState(states, tag, snapshot, true);
      }
    }

    return states;
  }, [
    bestRunScalarQuery,
    bestRunScalarQueryActive,
    chartScalarQueryEntries,
    effectiveBestRunMetricTag,
    selectedTagsByGroup.test,
    testScalarQuery,
    testScalarQueryActive,
  ]);
  const trainValidationPairQueryStates = useMemo(() => {
    const tagStates = new Map<string, LogScalarTagQueryState>();
    for (const entry of trainValidationScalarQueryEntries) {
      const snapshot = scalarQuerySnapshot(entry.query, entry.input.enabled);
      for (const tag of entry.input.tags) {
        mergeLogScalarTagQueryState(tagStates, tag, snapshot, true);
      }
    }

    const pairStates = new Map<string, LogScalarTagQueryState>();
    for (const pair of selectedTrainValidationPairs) {
      const trainState = tagStates.get(pair.trainTag);
      const validationState = tagStates.get(pair.validationTag);
      pairStates.set(pair.suffix, {
        hasRequested: requestedTrainValidationPairSuffixes.has(pair.suffix),
        isInitialLoading:
          (trainState?.isInitialLoading ?? false) ||
          (validationState?.isInitialLoading ?? false),
        isFetching:
          (trainState?.isFetching ?? false) ||
          (validationState?.isFetching ?? false),
        isError:
          (trainState?.isError ?? false) || (validationState?.isError ?? false),
        error: trainState?.error ?? validationState?.error ?? null,
      });
    }
    return pairStates;
  }, [
    requestedTrainValidationPairSuffixes,
    selectedTrainValidationPairs,
    trainValidationScalarQueryEntries,
  ]);
  const confusionMatrixSeriesByTag = useMemo(
    () => groupLogScalarSeriesByTag(confusionMatrixScalarSeries),
    [confusionMatrixScalarSeries],
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
        series:
          bestRunCoveredMetricQuery && effectiveBestRunMetricTag
            ? seriesByTag.get(effectiveBestRunMetricTag) ?? []
            : bestRunStandaloneScalarSeries,
        tag: effectiveBestRunMetricTag,
      }),
    [
      bestRunCoveredMetricQuery,
      bestRunStandaloneScalarSeries,
      effectiveBestRunDirection,
      effectiveBestRunMetricTag,
      seriesByTag,
      selectedBestRunPointPolicy,
      state.visibleRunIds,
      state.visibleRuns,
    ],
  );
  const bestRunIsLoading = bestRunCoveredMetricQuery
    ? bestRunCoveredMetricQuery.isLoading ||
      bestRunCoveredMetricQuery.isPlaceholderData
    : bestRunScalarQueryActive &&
      (bestRunScalarQuery.isLoading || bestRunScalarQuery.isPlaceholderData);
  const bestRunIsFetching = bestRunCoveredMetricQuery
    ? bestRunCoveredMetricQuery.isFetching
    : bestRunScalarQueryActive && bestRunScalarQuery.isFetching;
  const bestRunIsError = bestRunCoveredMetricQuery
    ? bestRunCoveredMetricQuery.isError
    : bestRunScalarQueryActive && bestRunScalarQuery.isError;
  const bestRunError = bestRunCoveredMetricQuery
    ? bestRunCoveredMetricQuery.error
    : bestRunScalarQueryActive && bestRunScalarQuery.isError
      ? bestRunScalarQuery.error
      : null;
  const bestRun = useMemo<LogBestRunViewModel>(
    () => ({
      metricTagOptions: state.tagOptions,
      selectedMetricTag: effectiveBestRunMetricTag,
      selectedDirection: effectiveBestRunDirection,
      selectedPointPolicy: selectedBestRunPointPolicy,
      rows: bestRunRows,
      visibleRunCount: state.visibleRuns.length,
      hasMoreRuns: Boolean(state.runsQuery.data?.hasMore),
      isLoading: bestRunIsLoading,
      isFetching: bestRunIsFetching,
      isError: bestRunIsError,
      error: bestRunError,
      onMetricTagChange: (tag) => {
        setSelectedBestRunMetricTag(tag);
        setSelectedBestRunDirection(null);
      },
      onDirectionChange: setSelectedBestRunDirection,
      onPointPolicyChange: setSelectedBestRunPointPolicy,
    }),
    [
      bestRunRows,
      bestRunError,
      bestRunIsError,
      bestRunIsFetching,
      bestRunIsLoading,
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
      groupSelectedLogMetrics({
        selectedTagList: state.selectedTagList,
        seriesByTag,
      }),
    [seriesByTag, state.selectedTagList],
  );
  const trainValidationComparisonMetrics = useMemo(
    () =>
      selectedTrainValidationPairs.map((pair) => ({
        ...pair,
        series: [
          ...(trainValidationSeriesByTag.get(pair.trainTag) ?? []),
          ...(trainValidationSeriesByTag.get(pair.validationTag) ?? []),
        ],
      })),
    [selectedTrainValidationPairs, trainValidationSeriesByTag],
  );
  const availableMetricTagsByGroup = useMemo(
    () => groupLogPlotSelectorTags(state.tagOptions.map((option) => option.value)),
    [state.tagOptions],
  );
  const handleMetricGroupTagSelectionChange = useCallback(
    (group: LogMetricGroupKey, selectedValues: string[]) => {
      const selectedValueSet = new Set(selectedValues);
      const currentSelectedTagSet = new Set(state.selectedTagList);
      for (const tag of availableMetricTagsByGroup[group]) {
        const isSelected = currentSelectedTagSet.has(tag);
        const shouldBeSelected = selectedValueSet.has(tag);
        if (isSelected !== shouldBeSelected) {
          state.toggleTag(tag);
        }
      }
    },
    [availableMetricTagsByGroup, state],
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
    scalarLoading: activeScalarQueries.some(
      (query) => query.isLoading || query.isPlaceholderData,
    ),
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
    availableMetricTagsByGroup,
    onMetricGroupTagSelectionChange: handleMetricGroupTagSelectionChange,
    trainValidationPairs,
    trainValidationComparisonMetrics,
    selectedTrainValidationPairSuffixes: selectedTrainValidationPairSuffixList,
    trainValidationPairQueryStates,
    trainValidationComparisonQueryState,
    isTrainValidationComparisonCollapsed,
    onToggleTrainValidationComparison: toggleTrainValidationComparison,
    trainValidationComparisonGridMode,
    onTrainValidationComparisonGridModeChange: setTrainValidationComparisonGridMode,
    onTrainValidationPairSelectionChange:
      handleTrainValidationPairSelectionChange,
    onTrainValidationComparisonChartVisible:
      markTrainValidationComparisonChartVisible,
    confusionHeatmaps,
    runsById: visibleRunsById,
    checkpointsByRunId: visibleCheckpointsByRunId,
    mediaImages: validationMedia.images,
    mediaTexts: validationMedia.texts,
    runOrder: state.visibleRunIds,
    visibleRunCount: state.visibleRuns.length,
    selectedTagCount: state.selectedTagList.length,
    scalarQueryStates,
    scalarTagQueryStates,
    onScalarChartVisible: markScalarChartVisible,
    hasConfusionMatrixTags: state.confusionMatrixRateTags.length > 0,
    isConfusionMatrixCollapsed,
    isConfusionMatrixLoaded:
      confusionMatrixScalarQueryActive &&
      confusionMatrixScalarQuery.isSuccess &&
      !confusionMatrixScalarQuery.isPlaceholderData,
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
