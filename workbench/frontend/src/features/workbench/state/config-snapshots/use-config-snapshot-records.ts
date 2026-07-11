import { useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  type ConfigSnapshotCreateInput,
  type ConfigSnapshotRecord,
  type ConfigSnapshotUpdateInput,
  type ModelIdentity,
  createConfigSnapshot,
  deleteConfigSnapshot,
  fetchConfigSnapshots,
  renameConfigSnapshot,
  updateConfigSnapshot,
} from "@/lib/api";
import { workbenchQueryKeys } from "@/lib/query-keys";

const EMPTY_CONFIG_SNAPSHOTS: ConfigSnapshotRecord[] = [];
const CONFIG_SNAPSHOTS_STALE_TIME_MS = 5 * 60_000;

/**
 * Shared Config Snapshot records Module. The backend remains the source of
 * truth; callers receive records, semantic mutation commands, and a small
 * status projection rather than React Query objects or invalidation details.
 */
export function useConfigSnapshotRecords(
  identity: ModelIdentity,
  options: { enabled?: boolean } = {},
) {
  const queryClient = useQueryClient();
  const enabled =
    (options.enabled ?? true) &&
    identity.modelType.length > 0 &&
    identity.model.length > 0;
  const query = useQuery({
    queryKey: workbenchQueryKeys.configSnapshots(identity.modelType, identity.model),
    queryFn: ({ signal }) => fetchConfigSnapshots(identity, { signal }),
    enabled,
    retry: false,
    staleTime: CONFIG_SNAPSHOTS_STALE_TIME_MS,
  });

  function invalidateModel(snapshot: ModelIdentity) {
    return queryClient.invalidateQueries({
      queryKey: workbenchQueryKeys.configSnapshots(
        snapshot.modelType,
        snapshot.model,
      ),
    });
  }

  const createMutation = useMutation({
    mutationFn: (input: ConfigSnapshotCreateInput) => createConfigSnapshot(input),
    onSuccess: invalidateModel,
  });
  const renameMutation = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) =>
      renameConfigSnapshot(id, name),
    onSuccess: invalidateModel,
  });
  const updateMutation = useMutation({
    mutationFn: ({
      id,
      input,
    }: {
      id: string;
      input: ConfigSnapshotUpdateInput;
    }) => updateConfigSnapshot(id, input),
    onSuccess: invalidateModel,
  });
  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteConfigSnapshot(id),
    onSuccess: invalidateModel,
  });
  const createSnapshot = createMutation.mutate;
  const renameSnapshot = renameMutation.mutate;
  const updateSnapshot = updateMutation.mutate;
  const removeSnapshot = deleteMutation.mutate;

  const status = useMemo(
    () => ({
      isLoading: query.isLoading,
      isReady: query.isSuccess,
      isError: query.isError,
    }),
    [query.isError, query.isLoading, query.isSuccess],
  );
  const actions = useMemo(
    () => ({
      create: (input: ConfigSnapshotCreateInput) => {
        if (enabled) {
          createSnapshot(input);
        }
      },
      rename: (input: { id: string; name: string }) => {
        if (enabled) {
          renameSnapshot(input);
        }
      },
      update: (input: { id: string; input: ConfigSnapshotUpdateInput }) => {
        if (enabled) {
          updateSnapshot(input);
        }
      },
      remove: (id: string) => {
        if (enabled) {
          removeSnapshot(id);
        }
      },
    }),
    [
      createSnapshot,
      enabled,
      removeSnapshot,
      renameSnapshot,
      updateSnapshot,
    ],
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
