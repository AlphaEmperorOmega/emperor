import { useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  LOG_ARTIFACTS_QUERY_KEY,
  LOG_CHECKPOINTS_QUERY_KEY,
  LOG_EXPERIMENTS_QUERY_KEY,
  LOG_RUNS_QUERY_KEY,
  LOG_SCALARS_QUERY_KEY,
  LOG_TAGS_QUERY_KEY,
} from "@/lib/query-keys";

type RefreshAfterMutationArgs = {
  runId?: string;
  runIds?: string[];
};

type RunIdInput = string | string[] | undefined;

function normalizeRunIds(runIds: RunIdInput) {
  if (Array.isArray(runIds)) {
    return runIds;
  }
  return runIds ? [runIds] : undefined;
}

function queryKeyRoot(queryKey: readonly unknown[]) {
  return typeof queryKey[0] === "string" ? queryKey[0] : "";
}

function queryKeyContainsRunId(value: unknown, runIds: Set<string>): boolean {
  if (typeof value === "string") {
    return runIds.has(value);
  }
  if (Array.isArray(value)) {
    return value.some((item) => queryKeyContainsRunId(item, runIds));
  }
  if (value && typeof value === "object") {
    return Object.values(value).some((item) => queryKeyContainsRunId(item, runIds));
  }
  return false;
}

function runScopedRemovalFilter({
  root,
  runIds,
}: {
  root: string;
  runIds?: RunIdInput;
}) {
  const runIdSet = new Set((normalizeRunIds(runIds) ?? []).filter(Boolean));
  if (runIdSet.size === 0) {
    return { queryKey: [root] };
  }
  return {
    predicate: (query: { queryKey: readonly unknown[] }) =>
      queryKeyRoot(query.queryKey) === root &&
      queryKeyContainsRunId(query.queryKey.slice(1), runIdSet),
  };
}

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
    (runIds?: RunIdInput) => {
      queryClient.removeQueries(
        runScopedRemovalFilter({ root: LOG_TAGS_QUERY_KEY[0], runIds }),
      );
      queryClient.removeQueries(
        runScopedRemovalFilter({ root: LOG_CHECKPOINTS_QUERY_KEY[0], runIds }),
      );
      queryClient.removeQueries(
        runScopedRemovalFilter({ root: LOG_ARTIFACTS_QUERY_KEY[0], runIds }),
      );
      return Promise.resolve();
    },
    [queryClient],
  );

  const removeRunScalars = useCallback(
    (runIds?: RunIdInput) => {
      queryClient.removeQueries(
        runScopedRemovalFilter({ root: LOG_SCALARS_QUERY_KEY[0], runIds }),
      );
    },
    [queryClient],
  );

  const refreshAfterMutation = useCallback(
    async ({ runId, runIds }: RefreshAfterMutationArgs = {}) => {
      const affectedRunIds = runIds ?? runId;
      const listInvalidation = invalidateLogLists();
      void invalidateRunDetails(affectedRunIds);
      removeRunScalars(affectedRunIds);
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
