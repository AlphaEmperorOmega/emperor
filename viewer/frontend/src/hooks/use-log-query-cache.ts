import { useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  LOG_EXPERIMENTS_QUERY_KEY,
  LOG_RUNS_QUERY_KEY,
  LOG_SCALARS_QUERY_KEY,
  LOG_TAGS_QUERY_KEY,
} from "@/lib/query-keys";

type RefreshAfterMutationArgs = {
  runId?: string;
};

export function useLogQueryCache() {
  const queryClient = useQueryClient();

  const invalidateLogLists = useCallback(
    () =>
      Promise.all([
        queryClient.invalidateQueries({ queryKey: LOG_EXPERIMENTS_QUERY_KEY }),
        queryClient.invalidateQueries({ queryKey: LOG_RUNS_QUERY_KEY }),
      ]).then(() => undefined),
    [queryClient],
  );

  const invalidateRunDetails = useCallback(
    (_runId?: string) => {
      void _runId;
      queryClient.removeQueries({ queryKey: LOG_TAGS_QUERY_KEY });
      return Promise.resolve();
    },
    [queryClient],
  );

  const removeRunScalars = useCallback(
    (_runId?: string) => {
      void _runId;
      queryClient.removeQueries({ queryKey: LOG_SCALARS_QUERY_KEY });
    },
    [queryClient],
  );

  const refreshAfterMutation = useCallback(
    async ({ runId }: RefreshAfterMutationArgs = {}) => {
      const listInvalidation = invalidateLogLists();
      void invalidateRunDetails(runId);
      removeRunScalars(runId);
      await listInvalidation;
    },
    [invalidateLogLists, invalidateRunDetails, removeRunScalars],
  );

  return {
    invalidateLogLists,
    invalidateRunDetails,
    removeRunScalars,
    refreshAfterMutation,
  };
}
