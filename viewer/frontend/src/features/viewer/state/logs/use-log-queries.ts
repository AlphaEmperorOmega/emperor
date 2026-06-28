import {
  keepPreviousData,
  useQueries,
  useQuery,
  type UseQueryResult,
} from "@tanstack/react-query";
import {
  DEFAULT_LOG_SCALAR_MAX_POINTS,
  LOG_SCALAR_SAMPLING,
  fetchLogCheckpoints,
  fetchLogExperiments,
  fetchLogMedia,
  fetchLogRuns,
  fetchLogRunArtifacts,
  fetchLogScalars,
  fetchLogTags,
  type FetchLogRunsInput,
} from "@/lib/api";
import {
  LOG_EXPERIMENTS_QUERY_KEY,
  logQueryKeys,
} from "@/lib/query-keys";

export {
  LOG_ARTIFACTS_QUERY_KEY,
  LOG_CHECKPOINTS_QUERY_KEY,
  LOG_EXPERIMENTS_QUERY_KEY,
  LOG_MEDIA_QUERY_KEY,
  LOG_RUNS_QUERY_KEY,
  LOG_SCALARS_QUERY_KEY,
  LOG_TAGS_QUERY_KEY,
} from "@/lib/query-keys";

type QueryOptions = {
  enabled?: boolean;
};

export type LogScalarQueryInput = {
  runIds: string[];
  tags: string[];
  enabled: boolean;
  group?: string;
  queryKey: readonly unknown[];
};

export type LogRunsQueryInput = FetchLogRunsInput & {
  enabled: boolean;
  queryKey: readonly unknown[];
  keepPreviousData?: boolean;
};

const LOG_RUNS_STALE_TIME_MS = 30_000;
const LOG_TAGS_STALE_TIME_MS = 5 * 60_000;
const LOG_SERIES_STALE_TIME_MS = 60_000;
type LogRunsQueryData = Awaited<ReturnType<typeof fetchLogRuns>>;
type LogScalarsQueryData = Awaited<ReturnType<typeof fetchLogScalars>>;

export function useLogRunsQuery({
  enabled = true,
  filters,
  pagination,
  includeAllPages,
  keepPreviousData: shouldKeepPreviousData = true,
}: QueryOptions & FetchLogRunsInput & { keepPreviousData?: boolean } = {}) {
  return useQuery({
    queryKey: logQueryKeys.runs({ filters, pagination, includeAllPages }),
    queryFn: ({ signal }) =>
      fetchLogRuns({ filters, pagination, includeAllPages }, { signal }),
    enabled,
    placeholderData: shouldKeepPreviousData ? keepPreviousData : undefined,
    retry: false,
    staleTime: LOG_RUNS_STALE_TIME_MS,
  });
}

export function useLogRunQueries(
  inputs: LogRunsQueryInput[],
): Array<UseQueryResult<LogRunsQueryData>> {
  return useQueries({
    queries: inputs.map((input) => ({
      queryKey: input.queryKey,
      queryFn: ({ signal }) =>
        fetchLogRuns(
          {
            filters: input.filters,
            pagination: input.pagination,
            includeAllPages: input.includeAllPages,
          },
          { signal },
        ),
      enabled: input.enabled,
      placeholderData: input.keepPreviousData === false ? undefined : keepPreviousData,
      retry: false,
      staleTime: LOG_RUNS_STALE_TIME_MS,
    })),
  }) as Array<UseQueryResult<LogRunsQueryData>>;
}

export function useLogExperimentsQuery({ enabled = true }: QueryOptions = {}) {
  return useQuery({
    queryKey: LOG_EXPERIMENTS_QUERY_KEY,
    queryFn: ({ signal }) => fetchLogExperiments({ signal }),
    enabled,
    retry: false,
    staleTime: LOG_RUNS_STALE_TIME_MS,
  });
}

export function useLogTagsQuery({
  runIds,
  enabled = true,
  queryKey = logQueryKeys.tagsForRuns(runIds),
}: QueryOptions & {
  runIds: string[];
  queryKey?: readonly unknown[];
}) {
  return useQuery({
    queryKey,
    queryFn: ({ signal }) => fetchLogTags({ runIds }, { signal }),
    enabled: enabled && runIds.length > 0,
    placeholderData: keepPreviousData,
    retry: false,
    staleTime: LOG_TAGS_STALE_TIME_MS,
  });
}

export function useLogScalarsQuery({
  runIds,
  tags,
  maxPoints = DEFAULT_LOG_SCALAR_MAX_POINTS,
  sampling = LOG_SCALAR_SAMPLING,
  group,
  enabled = true,
  queryKey = logQueryKeys.scalarsForRunsAndTags(runIds, tags, {
    maxPoints,
    sampling,
    group,
  }),
}: QueryOptions & {
  runIds: string[];
  tags: string[];
  maxPoints?: number;
  sampling?: typeof LOG_SCALAR_SAMPLING;
  group?: string;
  queryKey?: readonly unknown[];
}) {
  return useQuery({
    queryKey,
    queryFn: ({ signal }) =>
      fetchLogScalars({ runIds, tags, maxPoints, sampling }, { signal }),
    enabled: enabled && runIds.length > 0 && tags.length > 0,
    placeholderData: keepPreviousData,
    retry: false,
    staleTime: LOG_SERIES_STALE_TIME_MS,
  });
}

export function useLogScalarQueries(
  inputs: LogScalarQueryInput[],
): Array<UseQueryResult<LogScalarsQueryData>> {
  return useQueries({
    queries: inputs.map((input) => ({
      queryKey: input.queryKey,
      queryFn: ({ signal }) =>
        fetchLogScalars(
          {
            runIds: input.runIds,
            tags: input.tags,
            maxPoints: DEFAULT_LOG_SCALAR_MAX_POINTS,
            sampling: LOG_SCALAR_SAMPLING,
          },
          { signal },
        ),
      enabled: input.enabled && input.runIds.length > 0 && input.tags.length > 0,
      placeholderData: keepPreviousData,
      retry: false,
      staleTime: LOG_SERIES_STALE_TIME_MS,
    })),
  }) as Array<UseQueryResult<LogScalarsQueryData>>;
}

export function useLogMediaQuery({
  runIds,
  imageTags,
  textTags,
  enabled = true,
  queryKey = logQueryKeys.mediaForRunsAndTags(runIds, imageTags, textTags),
}: QueryOptions & {
  runIds: string[];
  imageTags: string[];
  textTags: string[];
  queryKey?: readonly unknown[];
}) {
  return useQuery({
    queryKey,
    queryFn: ({ signal }) =>
      fetchLogMedia({ runIds, imageTags, textTags }, { signal }),
    enabled:
      enabled && runIds.length > 0 && (imageTags.length > 0 || textTags.length > 0),
    placeholderData: keepPreviousData,
    retry: false,
    staleTime: LOG_SERIES_STALE_TIME_MS,
  });
}

export function useLogCheckpointsQuery({
  runIds,
  enabled = true,
  queryKey = logQueryKeys.checkpointsForRuns(runIds),
}: QueryOptions & {
  runIds: string[];
  queryKey?: readonly unknown[];
}) {
  return useQuery({
    queryKey,
    queryFn: ({ signal }) => fetchLogCheckpoints({ runIds }, { signal }),
    enabled: enabled && runIds.length > 0,
    retry: false,
    staleTime: LOG_RUNS_STALE_TIME_MS,
  });
}

export function useLogRunArtifactsQuery({
  runId,
  enabled = true,
}: QueryOptions & {
  runId: string | null | undefined;
}) {
  return useQuery({
    queryKey: logQueryKeys.artifactsForRun(runId),
    queryFn: ({ signal }) => fetchLogRunArtifacts(runId ?? "", { signal }),
    enabled: enabled && Boolean(runId),
    retry: false,
    staleTime: LOG_RUNS_STALE_TIME_MS,
  });
}
