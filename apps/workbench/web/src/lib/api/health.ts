import {
  _default,
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
  trainingCancellationCapability: _default(
    enumSchema([
      "strict-cgroup",
      "process-group",
      "windows-job-object",
      "unsupported",
    ]),
    "unsupported",
  ),
  trainingResourceLimitsEnforced: _default(boolean(), false),
  logDeletionEnabled: boolean(),
  configSnapshotsEnabled: _default(boolean(), true),
  historicalLogsEnabled: boolean(),
  liveMonitorDataEnabled: boolean(),
  historicalMonitorDataEnabled: boolean(),
  uploadsEnabled: _default(boolean(), false),
  maxUploadSize: _default(
    nullable(int().check(nonnegative())),
    null,
  ),
  maxActiveTrainingJobs: _default(positiveIntegerSchema, 2),
  trainingJobMemoryLimitBytes: _default(
    positiveIntegerSchema,
    16 * 1024 ** 3,
  ),
  trainingJobCpuLimit: _default(positiveIntegerSchema, 8),
  trainingJobProcessLimit: _default(positiveIntegerSchema, 512),
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
