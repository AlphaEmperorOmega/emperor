import { useQuery } from "@tanstack/react-query";
import {
  fetchLogCheckpoints,
  fetchLogExperiments,
  fetchLogRuns,
  fetchLogRunArtifacts,
  fetchLogScalars,
  fetchLogTags,
} from "@/lib/api";
import {
  LOG_EXPERIMENTS_QUERY_KEY,
  LOG_RUNS_QUERY_KEY,
  logQueryKeys,
} from "@/lib/query-keys";

export {
  LOG_ARTIFACTS_QUERY_KEY,
  LOG_CHECKPOINTS_QUERY_KEY,
  LOG_EXPERIMENTS_QUERY_KEY,
  LOG_RUNS_QUERY_KEY,
  LOG_SCALARS_QUERY_KEY,
  LOG_TAGS_QUERY_KEY,
} from "@/lib/query-keys";

type QueryOptions = {
  enabled?: boolean;
};

export function useLogRunsQuery({ enabled = true }: QueryOptions = {}) {
  return useQuery({
    queryKey: LOG_RUNS_QUERY_KEY,
    queryFn: fetchLogRuns,
    enabled,
    retry: false,
  });
}

export function useLogExperimentsQuery({ enabled = true }: QueryOptions = {}) {
  return useQuery({
    queryKey: LOG_EXPERIMENTS_QUERY_KEY,
    queryFn: fetchLogExperiments,
    enabled,
    retry: false,
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
    queryFn: () => fetchLogTags({ runIds }),
    enabled: enabled && runIds.length > 0,
    retry: false,
  });
}

export function useLogScalarsQuery({
  runIds,
  tags,
  enabled = true,
  queryKey = logQueryKeys.scalarsForRunsAndTags(runIds, tags),
}: QueryOptions & {
  runIds: string[];
  tags: string[];
  queryKey?: readonly unknown[];
}) {
  return useQuery({
    queryKey,
    queryFn: () => fetchLogScalars({ runIds, tags }),
    enabled: enabled && runIds.length > 0 && tags.length > 0,
    retry: false,
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
    queryFn: () => fetchLogCheckpoints({ runIds }),
    enabled: enabled && runIds.length > 0,
    retry: false,
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
    queryFn: () => fetchLogRunArtifacts(runId ?? ""),
    enabled: enabled && Boolean(runId),
    retry: false,
  });
}
