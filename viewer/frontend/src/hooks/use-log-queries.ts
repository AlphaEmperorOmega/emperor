import { useQuery } from "@tanstack/react-query";
import {
  fetchLogExperiments,
  fetchLogRuns,
  fetchLogScalars,
  fetchLogTags,
} from "@/lib/api";

export const LOG_RUNS_QUERY_KEY = ["log-runs"] as const;
export const LOG_EXPERIMENTS_QUERY_KEY = ["log-experiments"] as const;
export const LOG_TAGS_QUERY_KEY = ["log-tags"] as const;
export const LOG_SCALARS_QUERY_KEY = ["log-scalars"] as const;

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
  queryKey = [...LOG_TAGS_QUERY_KEY, runIds],
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
  queryKey = [...LOG_SCALARS_QUERY_KEY, runIds, tags],
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
