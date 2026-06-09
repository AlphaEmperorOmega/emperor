import { z } from "zod";

import { requestJson } from "@/lib/api/client";

const configSnapshotSchema = z.object({
  id: z.string(),
  model: z.string(),
  preset: z.string(),
  name: z.string(),
  overrides: z.record(z.string(), z.string()),
  createdAt: z.string(),
  updatedAt: z.string(),
});

const configSnapshotsSchema = z.object({
  model: z.string(),
  snapshots: z.array(configSnapshotSchema),
});

export type ConfigSnapshotRecord = z.infer<typeof configSnapshotSchema>;

export type ConfigSnapshotCreateInput = {
  model: string;
  preset: string;
  name: string;
  overrides: Record<string, string>;
};

export function fetchConfigSnapshots(model: string) {
  return requestJson(
    `/config-snapshots?model=${encodeURIComponent(model)}`,
    configSnapshotsSchema,
  );
}

export function createConfigSnapshot(input: ConfigSnapshotCreateInput) {
  return requestJson("/config-snapshots", configSnapshotSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function renameConfigSnapshot(snapshotId: string, name: string) {
  return requestJson(
    `/config-snapshots/${encodeURIComponent(snapshotId)}`,
    configSnapshotSchema,
    {
      method: "PATCH",
      body: JSON.stringify({ name }),
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
