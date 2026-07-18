import { useCallback, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  type ConfigSnapshotCreateInput,
  type ConfigSnapshotRecord,
  type ConfigSnapshotUpdateInput,
} from "@/lib/api/config-snapshots";
import {
  createMutationRequestOptions,
  type MutationRequestOptions,
} from "@/lib/api/client";
import { type ModelIdentity } from "@/lib/api/model-catalog";
import { type ConfigSnapshotMutationCommand } from "@/features/workbench/state/config-snapshots/_config-snapshot-mutation";
import { workbenchQueryKeys } from "@/lib/query-keys";
import { errorMessage } from "@/lib/utils";
import { createLazyFunction } from "@/lib/lazy-value";

const EMPTY_CONFIG_SNAPSHOTS: ConfigSnapshotRecord[] = [];
const CONFIG_SNAPSHOTS_STALE_TIME_MS = 5 * 60_000;
type FetchConfigSnapshots =
  typeof import("@/lib/api/config-snapshots").fetchConfigSnapshots;
const fetchConfigSnapshots: FetchConfigSnapshots = createLazyFunction(() =>
  import("@/lib/api/config-snapshots").then(
    (module) => module.fetchConfigSnapshots,
  ),
);

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

type ConfigSnapshotMutationLifecycleToken = Readonly<{
  revision: number;
  identityKey: string;
}>;

class ConfigSnapshotMutationLifecycle {
  #revision = 0;
  #identityKey: string;

  constructor(identityKey: string) {
    this.#identityKey = identityKey;
  }

  transition(identityKey: string) {
    if (identityKey === this.#identityKey) {
      return;
    }
    this.#identityKey = identityKey;
    this.#revision += 1;
  }

  quarantine() {
    this.#revision += 1;
  }

  capture(): ConfigSnapshotMutationLifecycleToken {
    return {
      revision: this.#revision,
      identityKey: this.#identityKey,
    };
  }

  isCurrent(token: ConfigSnapshotMutationLifecycleToken) {
    return (
      token.revision === this.#revision &&
      token.identityKey === this.#identityKey
    );
  }
}

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
  const identityKey = JSON.stringify([
    enabled,
    identity.modelType,
    identity.model,
  ]);
  const [lifecycle] = useState(
    () => new ConfigSnapshotMutationLifecycle(identityKey),
  );
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
      lifecycleToken,
      mutation: mutationOptions,
    }: {
      command: ConfigSnapshotMutationCommand;
      lifecycleToken: ConfigSnapshotMutationLifecycleToken;
      mutation: MutationRequestOptions;
    }) => {
      const { persistConfigSnapshotMutation } = await import(
        "@/features/workbench/state/config-snapshots/_config-snapshot-mutation"
      );
      return persistConfigSnapshotMutation(
        queryClient,
        command,
        mutationOptions,
        () => lifecycle.isCurrent(lifecycleToken),
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
    lifecycle.quarantine();
    resetMutation();
  }, [lifecycle, resetMutation]);

  useEffect(() => {
    lifecycle.transition(identityKey);
    resetMutation();
  }, [identityKey, lifecycle, resetMutation]);

  const executeMutation = useCallback(
    async (
      command: ConfigSnapshotMutationCommand,
      mutationOptions: MutationRequestOptions = createMutationRequestOptions(),
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
      const lifecycleToken = lifecycle.capture();
      try {
        const execution = await mutateAsync({
          command,
          lifecycleToken,
          mutation: mutationOptions,
        });
        if (!lifecycle.isCurrent(lifecycleToken)) {
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
          lifecycle.isCurrent(lifecycleToken),
        );
      }
    },
    [enabled, lifecycle, mutateAsync, mutationIsPending],
  );
  const retry = useCallback(() => {
    return mutationIsError &&
      mutationRequest &&
      lifecycle.isCurrent(mutationRequest.lifecycleToken)
      ? executeMutation(mutationRequest.command, mutationRequest.mutation)
      : Promise.resolve(null);
  }, [executeMutation, lifecycle, mutationIsError, mutationRequest]);
  const dismissMutation = useCallback(() => {
    if (!mutationIsPending) {
      resetMutation();
    }
  }, [mutationIsPending, resetMutation]);
  const mutationStatus = useMemo<ConfigSnapshotMutationStatus>(() => {
    if (
      !mutationRequest ||
      mutationRequest.lifecycleToken.identityKey !== identityKey
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
    identityKey,
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
        executeMutation({ kind: "create", input }),
      rename: (input: { id: string; name: string }) =>
        executeMutation({ kind: "rename", ...input }),
      update: (input: { id: string; input: ConfigSnapshotUpdateInput }) =>
        executeMutation({ kind: "update", ...input }),
      remove: (id: string) => executeMutation({ kind: "remove", id }),
      retry,
      dismissMutation,
      clearForConnectionChange: clearMutationLifecycle,
    }),
    [clearMutationLifecycle, dismissMutation, executeMutation, retry],
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
