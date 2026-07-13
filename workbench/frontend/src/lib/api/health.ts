import { z } from "zod";

import { requestJson } from "@/lib/api/client";

const healthSchema = z.object({ status: z.string() });

const capabilitiesSchema = z.object({
  authMode: z.enum(["none", "bearer"]),
  trainingEnabled: z.boolean(),
  trainingCancellationCapability: z
    .enum([
      "strict-cgroup",
      "process-group",
      "windows-job-object",
      "unsupported",
    ])
    .default("unsupported"),
  trainingResourceLimitsEnforced: z.boolean().default(false),
  logDeletionEnabled: z.boolean(),
  configSnapshotsEnabled: z.boolean().default(true),
  historicalLogsEnabled: z.boolean(),
  liveMonitorDataEnabled: z.boolean(),
  historicalMonitorDataEnabled: z.boolean(),
  uploadsEnabled: z.boolean().default(false),
  maxUploadSize: z.number().int().nonnegative().nullable().default(null),
  maxActiveTrainingJobs: z.number().int().positive().default(2),
  trainingJobMemoryLimitBytes: z
    .number()
    .int()
    .positive()
    .default(16 * 1024 ** 3),
  trainingJobCpuLimit: z.number().int().positive().default(8),
  trainingJobProcessLimit: z.number().int().positive().default(512),
});

export type Capabilities = z.infer<typeof capabilitiesSchema>;

type ApiRequestOptions = {
  signal?: AbortSignal;
};

export function fetchHealth(options: ApiRequestOptions = {}) {
  return requestJson("/health", healthSchema, options);
}

export function fetchCapabilities(options: ApiRequestOptions = {}) {
  return requestJson("/capabilities", capabilitiesSchema, options);
}
