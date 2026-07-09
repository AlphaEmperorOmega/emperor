import { z } from "zod";

import { requestJson } from "@/lib/api/client";
import { type ModelIdentity } from "@/lib/api/models";

export const configSnapshotSchema = z.object({
  id: z.string(),
  modelType: z.string(),
  model: z.string(),
  preset: z.string(),
  name: z.string(),
  overrides: z.record(z.string(), z.string()),
  createdAt: z.string(),
  updatedAt: z.string(),
});

export const configSnapshotsSchema = z.object({
  modelType: z.string(),
  model: z.string(),
  snapshots: z.array(configSnapshotSchema),
});

export const configSnapshotLibrarySchema = z.object({
  snapshots: z.array(configSnapshotSchema),
});

export type ConfigSnapshotRecord = z.infer<typeof configSnapshotSchema>;

export type ConfigSnapshotCreateInput = {
  modelType: string;
  model: string;
  preset: string;
  name: string;
  overrides: Record<string, string>;
};

export type ConfigSnapshotUpdateInput = {
  name?: string;
  overrides?: Record<string, string>;
};

export function fetchConfigSnapshots(identity: ModelIdentity) {
  const params = new URLSearchParams({
    modelType: identity.modelType,
    model: identity.model,
  });
  return requestJson(
    `/config-snapshots?${params.toString()}`,
    configSnapshotsSchema,
  );
}

export function fetchConfigSnapshotLibrary() {
  return requestJson("/config-snapshots/library", configSnapshotLibrarySchema);
}

export function createConfigSnapshot(input: ConfigSnapshotCreateInput) {
  return requestJson("/config-snapshots", configSnapshotSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function renameConfigSnapshot(snapshotId: string, name: string) {
  return updateConfigSnapshot(snapshotId, { name });
}

export function updateConfigSnapshot(
  snapshotId: string,
  input: ConfigSnapshotUpdateInput,
) {
  return requestJson(
    `/config-snapshots/${encodeURIComponent(snapshotId)}`,
    configSnapshotSchema,
    {
      method: "PATCH",
      body: JSON.stringify(input),
    },
  );
}

export function deleteConfigSnapshot(snapshotId: string) {
  return requestJson(
    `/config-snapshots/${encodeURIComponent(snapshotId)}`,
    configSnapshotsSchema,
    {
      method: "DELETE",
    },
  );
}
