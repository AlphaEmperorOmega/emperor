import {
  boolean,
  enum as enumSchema,
  int,
  nonnegative,
  nullable,
  object,
  positive,
  string,
  type output,
} from "zod/v4-mini";

import { requestJson } from "@/lib/api/client";

const healthSchema = object({ status: string() });
const positiveIntegerSchema = int().check(positive());

const capabilitiesSchema = object({
  authMode: enumSchema(["none", "bearer"]),
  trainingEnabled: boolean(),
  trainingCancellationCapability: enumSchema([
    "strict-cgroup",
    "process-group",
    "windows-job-object",
    "unsupported",
  ]),
  trainingResourceLimitsEnforced: boolean(),
  logDeletionEnabled: boolean(),
  configSnapshotsEnabled: boolean(),
  historicalLogsEnabled: boolean(),
  liveMonitorDataEnabled: boolean(),
  historicalMonitorDataEnabled: boolean(),
  uploadsEnabled: boolean(),
  maxUploadSize: nullable(int().check(nonnegative())),
  maxActiveTrainingJobs: positiveIntegerSchema,
  trainingJobMemoryLimitBytes: positiveIntegerSchema,
  trainingJobCpuLimit: positiveIntegerSchema,
  trainingJobProcessLimit: positiveIntegerSchema,
});

export type Capabilities = output<typeof capabilitiesSchema>;

type ApiRequestOptions = {
  signal?: AbortSignal;
};

export function fetchHealth(options: ApiRequestOptions = {}) {
  return requestJson("/health", healthSchema, options);
}

export function fetchCapabilities(options: ApiRequestOptions = {}) {
  return requestJson("/capabilities", capabilitiesSchema, options);
}
