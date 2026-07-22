import { z } from "zod";

import {
  requestJson,
  type MutationRequestOptions,
} from "@/lib/api/client";
import { type ModelIdentity } from "@/lib/api/model-catalog";
import { modelIdentitySegmentSchema } from "@/lib/api/model-identity-schema";

export const configSnapshotSchema = z.object({
  id: z.string(),
  modelType: modelIdentitySegmentSchema,
  model: modelIdentitySegmentSchema,
  preset: z.string(),
  name: z.string(),
  overrides: z.record(z.string(), z.string()),
  createdAt: z.string(),
  updatedAt: z.string(),
});

export const configSnapshotsSchema = z.object({
  modelType: modelIdentitySegmentSchema,
  model: modelIdentitySegmentSchema,
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

type ApiRequestOptions = {
  signal?: AbortSignal;
};

export function fetchConfigSnapshots(
  identity: ModelIdentity,
  options: ApiRequestOptions = {},
) {
  const params = new URLSearchParams({
    modelType: identity.modelType,
    model: identity.model,
  });
  return requestJson(
    `/config-snapshots?${params.toString()}`,
    configSnapshotsSchema,
    options,
  );
}

export function fetchConfigSnapshotLibrary(options: ApiRequestOptions = {}) {
  return requestJson(
    "/config-snapshots/library",
    configSnapshotLibrarySchema,
    options,
  );
}

export function createConfigSnapshot(
  input: ConfigSnapshotCreateInput,
  mutation: MutationRequestOptions,
) {
  return requestJson(
    "/config-snapshots",
    configSnapshotSchema,
    {
      method: "POST",
      body: JSON.stringify(input),
    },
    { mutation },
  );
}

export function renameConfigSnapshot(
  snapshotId: string,
  name: string,
  mutation: MutationRequestOptions,
) {
  return updateConfigSnapshot(snapshotId, { name }, mutation);
}

export function updateConfigSnapshot(
  snapshotId: string,
  input: ConfigSnapshotUpdateInput,
  mutation: MutationRequestOptions,
) {
  return requestJson(
    `/config-snapshots/${encodeURIComponent(snapshotId)}`,
    configSnapshotSchema,
    {
      method: "PATCH",
      body: JSON.stringify(input),
    },
    { mutation },
  );
}

export function deleteConfigSnapshot(
  snapshotId: string,
  mutation: MutationRequestOptions,
) {
  return requestJson(
    `/config-snapshots/${encodeURIComponent(snapshotId)}`,
    configSnapshotsSchema,
    {
      method: "DELETE",
    },
    { mutation },
  );
}
