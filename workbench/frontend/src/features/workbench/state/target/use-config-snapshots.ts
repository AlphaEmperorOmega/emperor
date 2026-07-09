import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  type ConfigSnapshotCreateInput,
  type ConfigSnapshotRecord,
  type ConfigSnapshotUpdateInput,
  type ModelIdentity,
  createConfigSnapshot,
  deleteConfigSnapshot,
  fetchConfigSnapshotLibrary,
  fetchConfigSnapshots,
  renameConfigSnapshot,
  updateConfigSnapshot,
} from "@/lib/api";
import { workbenchQueryKeys } from "@/lib/query-keys";

const EMPTY_CONFIG_SNAPSHOTS: ConfigSnapshotRecord[] = [];

/**
 * Server-backed config snapshot library for a model. The backend is the single
 * source of truth (persisted JSON, reached only through the typed API); this hook
 * reads the per-model list and exposes create/rename/delete mutations that
 * invalidate the cache so the list stays consistent after every change.
 */
export function useConfigSnapshots(identity: ModelIdentity) {
  const queryClient = useQueryClient();
  const enabled = identity.modelType.length > 0 && identity.model.length > 0;

  const query = useQuery({
    queryKey: workbenchQueryKeys.configSnapshots(identity.modelType, identity.model),
    queryFn: () => fetchConfigSnapshots(identity),
    enabled,
    retry: false,
  });

  function invalidateModel(snapshot: ModelIdentity) {
    return queryClient.invalidateQueries({
      queryKey: workbenchQueryKeys.configSnapshots(
        snapshot.modelType,
        snapshot.model,
      ),
    });
  }

  function invalidateLibrary() {
    return queryClient.invalidateQueries({
      queryKey: workbenchQueryKeys.configSnapshotLibrary(),
    });
  }

  const createMutation = useMutation({
    mutationFn: (input: ConfigSnapshotCreateInput) => createConfigSnapshot(input),
    onSuccess: (snapshot) =>
      Promise.all([invalidateModel(snapshot), invalidateLibrary()]),
  });

  const renameMutation = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) =>
      renameConfigSnapshot(id, name),
    onSuccess: (snapshot) =>
      Promise.all([invalidateModel(snapshot), invalidateLibrary()]),
  });

  const updateMutation = useMutation({
    mutationFn: ({
      id,
      input,
    }: {
      id: string;
      input: ConfigSnapshotUpdateInput;
    }) => updateConfigSnapshot(id, input),
    onSuccess: (snapshot) =>
      Promise.all([invalidateModel(snapshot), invalidateLibrary()]),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteConfigSnapshot(id),
    onSuccess: (result) =>
      Promise.all([invalidateModel(result), invalidateLibrary()]),
  });

  const snapshots = query.data?.snapshots ?? EMPTY_CONFIG_SNAPSHOTS;

  return {
    query,
    snapshots,
    createMutation,
    renameMutation,
    updateMutation,
    deleteMutation,
  };
}

export function useConfigSnapshotLibrary() {
  const query = useQuery({
    queryKey: workbenchQueryKeys.configSnapshotLibrary(),
    queryFn: fetchConfigSnapshotLibrary,
    retry: false,
  });

  const snapshots = query.data?.snapshots ?? EMPTY_CONFIG_SNAPSHOTS;

  return { query, snapshots };
}
