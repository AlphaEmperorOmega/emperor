import { type QueryClient } from "@tanstack/react-query";
import {
  type ConfigSnapshotCreateInput,
  type ConfigSnapshotRecord,
  type ConfigSnapshotUpdateInput,
  type ModelIdentity,
  type MutationRequestOptions,
  createConfigSnapshot,
  deleteConfigSnapshot,
  renameConfigSnapshot,
  updateConfigSnapshot,
} from "@/lib/api";
import { workbenchQueryKeys } from "@/lib/query-keys";

export type ConfigSnapshotMutationCommand =
  | { kind: "create"; input: ConfigSnapshotCreateInput }
  | { kind: "rename"; id: string; name: string }
  | { kind: "update"; id: string; input: ConfigSnapshotUpdateInput }
  | { kind: "remove"; id: string };

export type ConfigSnapshotMutationExecution = {
  kind: ConfigSnapshotMutationCommand["kind"];
  snapshotId: string | null;
  record: ConfigSnapshotRecord | null;
  records: ConfigSnapshotRecord[] | null;
  identity: ModelIdentity;
};

async function persist(
  command: ConfigSnapshotMutationCommand,
  mutation: MutationRequestOptions,
) {
  if (command.kind === "remove") {
    const library = await deleteConfigSnapshot(command.id, mutation);
    return {
      kind: command.kind,
      snapshotId: command.id,
      record: null,
      records: library.snapshots,
      identity: library,
    } satisfies ConfigSnapshotMutationExecution;
  }
  const record =
    command.kind === "create"
      ? await createConfigSnapshot(command.input, mutation)
      : command.kind === "rename"
        ? await renameConfigSnapshot(command.id, command.name, mutation)
        : await updateConfigSnapshot(command.id, command.input, mutation);
  return {
    kind: command.kind,
    snapshotId: record.id,
    record,
    records: null,
    identity: record,
  } satisfies ConfigSnapshotMutationExecution;
}

export async function persistConfigSnapshotMutation(
  queryClient: QueryClient,
  command: ConfigSnapshotMutationCommand,
  mutation: MutationRequestOptions,
  isCurrent: () => boolean,
) {
  const execution = await persist(command, mutation);
  if (isCurrent()) {
    const modelQueryKey = workbenchQueryKeys.configSnapshots(
      execution.identity.modelType,
      execution.identity.model,
    );
    if (execution.records) {
      queryClient.setQueryData(modelQueryKey, {
        ...execution.identity,
        snapshots: execution.records,
      });
      await queryClient.invalidateQueries({
        queryKey: workbenchQueryKeys.configSnapshotLibrary(),
      });
    } else {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: modelQueryKey }),
        queryClient.invalidateQueries({
          queryKey: workbenchQueryKeys.configSnapshotLibrary(),
        }),
      ]);
    }
  }
  return execution;
}
