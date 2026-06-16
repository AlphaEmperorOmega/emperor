import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  type ConfigSnapshotCreateInput,
  type ConfigSnapshotRecord,
  type ConfigSnapshotUpdateInput,
  createConfigSnapshot,
  deleteConfigSnapshot,
  fetchConfigSnapshotLibrary,
  fetchConfigSnapshots,
  renameConfigSnapshot,
  updateConfigSnapshot,
} from "@/lib/api";
import { viewerQueryKeys } from "@/lib/query-keys";

const EMPTY_CONFIG_SNAPSHOTS: ConfigSnapshotRecord[] = [];

/**
 * Server-backed config snapshot library for a model. The backend is the single
 * source of truth (persisted JSON, reached only through the typed API); this hook
 * reads the per-model list and exposes create/rename/delete mutations that
 * invalidate the cache so the list stays consistent after every change.
 */
export function useConfigSnapshots(model: string) {
  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: viewerQueryKeys.configSnapshots(model),
    queryFn: () => fetchConfigSnapshots(model),
    enabled: model.length > 0,
    retry: false,
  });

  function invalidateModel(snapshotModel: string) {
    return queryClient.invalidateQueries({
      queryKey: viewerQueryKeys.configSnapshots(snapshotModel),
    });
  }

  function invalidateLibrary() {
    return queryClient.invalidateQueries({
      queryKey: viewerQueryKeys.configSnapshotLibrary(),
    });
  }

  const createMutation = useMutation({
    mutationFn: (input: ConfigSnapshotCreateInput) => createConfigSnapshot(input),
    onSuccess: (snapshot) =>
      Promise.all([invalidateModel(snapshot.model), invalidateLibrary()]),
  });

  const renameMutation = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) =>
      renameConfigSnapshot(id, name),
    onSuccess: (snapshot) =>
      Promise.all([invalidateModel(snapshot.model), invalidateLibrary()]),
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
      Promise.all([invalidateModel(snapshot.model), invalidateLibrary()]),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteConfigSnapshot(id),
    onSuccess: (result) =>
      Promise.all([invalidateModel(result.model), invalidateLibrary()]),
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
    queryKey: viewerQueryKeys.configSnapshotLibrary(),
    queryFn: fetchConfigSnapshotLibrary,
    retry: false,
  });

  const snapshots = query.data?.snapshots ?? EMPTY_CONFIG_SNAPSHOTS;

  return { query, snapshots };
}
