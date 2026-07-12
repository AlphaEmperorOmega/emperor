import {
  keepPreviousData,
  useInfiniteQuery,
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

export type LogTagsQueryInput = {
  runIds: string[];
  enabled: boolean;
  queryKey: readonly unknown[];
};

const LOG_RUNS_STALE_TIME_MS = 30_000;
const LOG_TAGS_STALE_TIME_MS = 5 * 60_000;
const LOG_SERIES_STALE_TIME_MS = 60_000;
type LogTagsQueryData = Awaited<ReturnType<typeof fetchLogTags>>;
type LogScalarsQueryData = Awaited<ReturnType<typeof fetchLogScalars>>;

export function useLogRunsQuery({
  enabled = true,
  filters,
  pagination,
  includeAllPages,
  projection,
  keepPreviousData: shouldKeepPreviousData = true,
}: QueryOptions & FetchLogRunsInput & { keepPreviousData?: boolean } = {}) {
  return useQuery({
    queryKey: logQueryKeys.runs({
      filters,
      pagination,
      includeAllPages,
      projection,
    }),
    queryFn: ({ signal }) =>
      fetchLogRuns(
        { filters, pagination, includeAllPages, projection },
        { signal },
      ),
    enabled,
    placeholderData: shouldKeepPreviousData ? keepPreviousData : undefined,
    retry: false,
    staleTime: LOG_RUNS_STALE_TIME_MS,
  });
}

export function useInfiniteLogRunsQuery({
  enabled = true,
  filters,
  pageSize,
  projection,
  keepPreviousData: shouldKeepPreviousData = true,
}: QueryOptions &
  Pick<FetchLogRunsInput, "filters" | "projection"> & {
    pageSize: number;
    keepPreviousData?: boolean;
  }) {
  return useInfiniteQuery({
    queryKey: logQueryKeys.runs({
      filters,
      pagination: { limit: pageSize },
      projection,
      mode: "pages",
    }),
    queryFn: ({ pageParam, signal }) =>
      fetchLogRuns(
        {
          filters,
          pagination: { limit: pageSize, offset: pageParam },
          projection,
        },
        { signal },
      ),
    enabled,
    initialPageParam: 0,
    getNextPageParam: (lastPage) =>
      lastPage.hasMore
        ? (lastPage.offset ?? 0) + (lastPage.limit ?? pageSize)
        : undefined,
    placeholderData: shouldKeepPreviousData ? keepPreviousData : undefined,
    retry: false,
    staleTime: LOG_RUNS_STALE_TIME_MS,
  });
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

export function useLogTagQueries(
  inputs: LogTagsQueryInput[],
): Array<UseQueryResult<LogTagsQueryData>> {
  return useQueries({
    queries: inputs.map((input) => ({
      queryKey: input.queryKey,
      queryFn: ({ signal }) => fetchLogTags({ runIds: input.runIds }, { signal }),
      enabled: input.enabled && input.runIds.length > 0,
      placeholderData: keepPreviousData,
      retry: false,
      staleTime: LOG_TAGS_STALE_TIME_MS,
    })),
  }) as Array<UseQueryResult<LogTagsQueryData>>;
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
