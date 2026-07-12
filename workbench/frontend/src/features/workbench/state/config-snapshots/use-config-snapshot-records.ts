import { useCallback, useEffect, useMemo, useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  type ConfigSnapshotCreateInput,
  type ConfigSnapshotRecord,
  type ConfigSnapshotUpdateInput,
  type ModelIdentity,
  fetchConfigSnapshots,
} from "@/lib/api";
import { type ConfigSnapshotMutationCommand } from "@/features/workbench/state/config-snapshots/_config-snapshot-mutation";
import { workbenchQueryKeys } from "@/lib/query-keys";
import { errorMessage } from "@/lib/utils";

const EMPTY_CONFIG_SNAPSHOTS: ConfigSnapshotRecord[] = [];
const CONFIG_SNAPSHOTS_STALE_TIME_MS = 5 * 60_000;

export type ConfigSnapshotMutationKind =
  ConfigSnapshotMutationCommand["kind"];
export type ConfigSnapshotMutationPhase =
  | "idle"
  | "pending"
  | "succeeded"
  | "failed";
export type ConfigSnapshotMutationStatus = Readonly<{
  phase: ConfigSnapshotMutationPhase;
  kind: ConfigSnapshotMutationKind | null;
  snapshotId: string | null;
  error: string;
  canRetry: boolean;
}>;
export type ConfigSnapshotMutationOutcome =
  | {
      ok: true;
      kind: ConfigSnapshotMutationKind;
      snapshotId: string | null;
      record: ConfigSnapshotRecord | null;
    }
  | {
      ok: false;
      kind: ConfigSnapshotMutationKind;
      snapshotId: string | null;
      error: string;
      retryable: boolean;
    };

const IDLE_MUTATION_STATUS: ConfigSnapshotMutationStatus = {
  phase: "idle",
  kind: null,
  snapshotId: null,
  error: "",
  canRetry: false,
};

function commandSnapshotId(command: ConfigSnapshotMutationCommand) {
  return command.kind === "create" ? null : command.id;
}

function failedOutcome(
  command: ConfigSnapshotMutationCommand,
  error: unknown,
  retryable: boolean,
): ConfigSnapshotMutationOutcome {
  return {
    ok: false,
    kind: command.kind,
    snapshotId: commandSnapshotId(command),
    error: errorMessage(error),
    retryable,
  };
}

/**
 * Shared Config Snapshot records Module. The backend remains the source of
 * truth. One mutation lifecycle owns persistence, invalidation, pending and
 * failure state, exact-command retry, and connection-change quarantine.
 */
export function useConfigSnapshotRecords(
  identity: ModelIdentity,
  options: { enabled?: boolean } = {},
) {
  const queryClient = useQueryClient();
  const enabled =
    (options.enabled ?? true) && Boolean(identity.modelType && identity.model);
  const generationRef = useRef(0);
  const query = useQuery({
    queryKey: workbenchQueryKeys.configSnapshots(identity.modelType, identity.model),
    queryFn: ({ signal }) => fetchConfigSnapshots(identity, { signal }),
    enabled,
    retry: false,
    staleTime: CONFIG_SNAPSHOTS_STALE_TIME_MS,
  });
  const mutation = useMutation({
    mutationFn: async ({
      command,
      generation,
    }: {
      command: ConfigSnapshotMutationCommand;
      generation: number;
    }) => {
      const { persistConfigSnapshotMutation } = await import(
        "@/features/workbench/state/config-snapshots/_config-snapshot-mutation"
      );
      return persistConfigSnapshotMutation(
        queryClient,
        command,
        () => generationRef.current === generation,
      );
    },
  });
  const mutateAsync = mutation.mutateAsync;
  const resetMutation = mutation.reset;
  const mutationData = mutation.data;
  const mutationError = mutation.error;
  const mutationIsError = mutation.isError;
  const mutationIsPending = mutation.isPending;
  const mutationIsSuccess = mutation.isSuccess;
  const mutationRequest = mutation.variables;
  const clearMutationLifecycle = useCallback(() => {
    generationRef.current += 1;
    resetMutation();
  }, [resetMutation]);

  useEffect(clearMutationLifecycle, [
    clearMutationLifecycle,
    enabled,
    identity.model,
    identity.modelType,
  ]);

  const run = useCallback(
    async (
      command: ConfigSnapshotMutationCommand,
    ): Promise<ConfigSnapshotMutationOutcome> => {
      if (!enabled) {
        return failedOutcome(
          command,
          "Unavailable.",
          false,
        );
      }
      if (mutationIsPending) {
        return failedOutcome(
          command,
          "Pending.",
          false,
        );
      }
      const generation = generationRef.current;
      try {
        const execution = await mutateAsync({ command, generation });
        if (generationRef.current !== generation) {
          return failedOutcome(
            command,
            "Connection changed.",
            false,
          );
        }
        return { ok: true, ...execution };
      } catch (error) {
        return failedOutcome(
          command,
          error,
          generationRef.current === generation,
        );
      }
    },
    [enabled, mutateAsync, mutationIsPending],
  );
  const retry = useCallback(() => {
    return mutationIsError &&
      mutationRequest?.generation === generationRef.current
      ? run(mutationRequest.command)
      : Promise.resolve(null);
  }, [mutationIsError, mutationRequest, run]);
  const dismissMutation = useCallback(() => {
    if (!mutationIsPending) {
      resetMutation();
    }
  }, [mutationIsPending, resetMutation]);
  const mutationStatus = useMemo<ConfigSnapshotMutationStatus>(() => {
    if (
      !mutationRequest ||
      mutationRequest.generation !== generationRef.current
    ) {
      return IDLE_MUTATION_STATUS;
    }
    return {
      phase: mutationIsPending
        ? "pending"
        : mutationIsError
          ? "failed"
          : mutationIsSuccess
            ? "succeeded"
            : "idle",
      kind: mutationRequest.command.kind,
      snapshotId:
        mutationData?.snapshotId ?? commandSnapshotId(mutationRequest.command),
      error: mutationIsError ? errorMessage(mutationError) : "",
      canRetry: mutationIsError,
    };
  }, [
    mutationData?.snapshotId,
    mutationError,
    mutationIsError,
    mutationIsPending,
    mutationIsSuccess,
    mutationRequest,
  ]);
  const status = useMemo(
    () => ({
      isLoading: query.isLoading,
      isReady: query.isSuccess,
      isError: query.isError,
      error: query.error ?? null,
      mutation: mutationStatus,
    }),
    [
      mutationStatus,
      query.error,
      query.isError,
      query.isLoading,
      query.isSuccess,
    ],
  );
  const actions = useMemo(
    () => ({
      create: (input: ConfigSnapshotCreateInput) =>
        run({ kind: "create", input }),
      rename: (input: { id: string; name: string }) =>
        run({ kind: "rename", ...input }),
      update: (input: { id: string; input: ConfigSnapshotUpdateInput }) =>
        run({ kind: "update", ...input }),
      remove: (id: string) => run({ kind: "remove", id }),
      retry,
      dismissMutation,
      clearForConnectionChange: clearMutationLifecycle,
    }),
    [clearMutationLifecycle, dismissMutation, retry, run],
  );

  return useMemo(
    () => ({
      records: query.data?.snapshots ?? EMPTY_CONFIG_SNAPSHOTS,
      status,
      actions,
    }),
    [actions, query.data?.snapshots, status],
  );
}
