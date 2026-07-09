import { z } from "zod";

import { requestJson } from "@/lib/api/client";

const healthSchema = z.object({ status: z.string() });

const dataSourceCapabilityPlaceholderSchema = z.object({}).strict();

const capabilitiesSchema = z.object({
  authMode: z.enum(["none", "bearer"]),
  trainingEnabled: z.boolean(),
  trainingCancellationCapability: z
    .enum(["strict-cgroup", "process-group", "unsupported"])
    .default("unsupported"),
  logDeletionEnabled: z.boolean(),
  configSnapshotsEnabled: z.boolean().default(true),
  historicalLogsEnabled: z.boolean(),
  liveMonitorDataEnabled: z.boolean(),
  historicalMonitorDataEnabled: z.boolean(),
  uploadsEnabled: z.boolean().default(false),
  maxUploadSize: z.number().int().nonnegative().nullable().default(null),
  dataSourcesEnabled: z.boolean().default(false),
  dataSources: z.array(dataSourceCapabilityPlaceholderSchema).default([]),
});

export type Capabilities = z.infer<typeof capabilitiesSchema>;

export function fetchHealth() {
  return requestJson("/health", healthSchema);
}

export function fetchCapabilities() {
  return requestJson("/capabilities", capabilitiesSchema);
}
