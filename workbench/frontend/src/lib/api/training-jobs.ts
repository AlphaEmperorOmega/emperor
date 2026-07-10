import { z } from "zod";

import { requestJson } from "@/lib/api/client";
import {
  configOverridesSchema,
  configValueSchema,
  jsonObjectSchema,
  type ConfigOverrides,
  type ConfigValue,
  type JsonObject,
} from "@/lib/api/schemas";

type ApiRequestOptions = {
  signal?: AbortSignal;
};

export const trainingRunChangeSchema = z.object({
  key: z.string(),
  label: z.string(),
  value: configValueSchema,
  source: z.enum(["override", "search"]),
});

export const trainingRunSchema = z.object({
  id: z.string(),
  index: z.number(),
  status: z.enum([
    "Pending",
    "Running",
    "Completed",
    "Failed",
    "Cancelled",
    "Skipped",
  ]),
  preset: z.string(),
  snapshotId: z.string().nullable().optional(),
  snapshotName: z.string().nullable().optional(),
  dataset: z.string(),
  experimentTask: z.string().optional(),
  changes: z.array(trainingRunChangeSchema),
  overrides: configOverridesSchema,
  command: z.string(),
  totalEpochs: z.number(),
  currentEpoch: z.number(),
  metrics: jsonObjectSchema,
  logDir: z.string().nullable(),
  error: z.string().nullable(),
  errorTraceback: z.string().nullable().optional(),
});

export const trainingRunPlanSummarySchema = z.object({
  totalRuns: z.number(),
  completedRuns: z.number(),
  runningRuns: z.number(),
  pendingRuns: z.number(),
  failedRuns: z.number(),
  cancelledRuns: z.number(),
  skippedRuns: z.number(),
  totalEpochs: z.number(),
  completedEpochs: z.number(),
  remainingEpochs: z.number(),
});

export const trainingRunPlanSchema = z.object({
  modelType: z.string(),
  model: z.string(),
  preset: z.string(),
  presets: z.array(z.string()),
  experimentTask: z.string().optional(),
  datasets: z.array(z.string()),
  overrides: configOverridesSchema,
  search: z
    .object({
      mode: z.enum(["grid", "random"]),
      values: z.record(z.array(configValueSchema)),
      randomSamples: z.number().nullable().optional(),
    })
    .nullable(),
  logFolder: z.string(),
  isRandomSearch: z.boolean(),
  runs: z.array(trainingRunSchema),
  summary: trainingRunPlanSummarySchema,
});

const trainingProgressStatusSchema = z.enum([
  "running",
  "completed",
  "failed",
  "cancelled",
]);

export const trainingJobStatusSchema = z.enum([
  "queued",
  "running",
  "unknown",
  "completed",
  "failed",
  "cancelled",
]);

const knownTrainingProgressEventTypes = [
  "job_started",
  "started",
  "completed",
  "cancelled",
  "error",
  "dataset_started",
  "dataset_completed",
  "epoch_started",
  "step",
  "validation",
  "fit_completed",
  "test_completed",
  "cluster_initialized",
  "neuron_added",
  "neurons_added",
] as const;
const knownTrainingProgressEventTypeSet: ReadonlySet<string> = new Set(
  knownTrainingProgressEventTypes,
);

const trainingProgressEventBaseSchema = z
  .object({
    type: z.string(),
    timestamp: z.string().nullable().optional(),
    status: trainingProgressStatusSchema.nullable().optional(),
    jobId: z.string().nullable().optional(),
    dataset: z.string().nullable().optional(),
    experimentTask: z.string().nullable().optional(),
    preset: z.string().nullable().optional(),
    presetKey: z.string().nullable().optional(),
    logDir: z.string().nullable().optional(),
    runId: z.string().nullable().optional(),
    runIndex: z.number().nullable().optional(),
    runTotal: z.number().nullable().optional(),
    totalEpochs: z.number().nullable().optional(),
  })
  .passthrough();

const trainingJobStartedEventSchema = trainingProgressEventBaseSchema.extend({
  type: z.literal("job_started"),
  status: z.literal("running").nullable().optional(),
});

const trainingWorkerStartedEventSchema = trainingProgressEventBaseSchema.extend({
  type: z.literal("started"),
  status: z.literal("running").nullable().optional(),
  modelType: z.string().nullable().optional(),
  model: z.string().nullable().optional(),
  presets: z.array(z.string()).nullable().optional(),
  datasets: z.array(z.string()).nullable().optional(),
  monitors: z.array(z.string()).nullable().optional(),
});

const trainingWorkerCompletedEventSchema = trainingProgressEventBaseSchema.extend({
  type: z.literal("completed"),
  status: z.literal("completed").nullable().optional(),
  presets: z.array(z.string()).nullable().optional(),
});

const trainingCancelledEventSchema = trainingProgressEventBaseSchema.extend({
  type: z.literal("cancelled"),
  status: z.literal("cancelled").nullable().optional(),
});

const trainingErrorEventSchema = trainingProgressEventBaseSchema.extend({
  type: z.literal("error"),
  status: z.literal("failed").nullable().optional(),
  error: z.string().nullable().optional(),
  traceback: z.string().nullable().optional(),
});

const trainingDatasetStartedEventSchema = trainingProgressEventBaseSchema.extend({
  type: z.literal("dataset_started"),
  status: z.literal("running").nullable().optional(),
  params: jsonObjectSchema.nullable().optional(),
});

const trainingDatasetCompletedEventSchema = trainingProgressEventBaseSchema.extend({
  type: z.literal("dataset_completed"),
  status: z.enum(["running", "completed"]).nullable().optional(),
  metrics: jsonObjectSchema.nullable().optional(),
});

const trainingRunProgressEventSchema = trainingProgressEventBaseSchema.extend({
  type: z.enum([
    "epoch_started",
    "step",
    "validation",
    "fit_completed",
    "test_completed",
  ]),
  status: z.literal("running").nullable().optional(),
  epoch: z.number().nullable().optional(),
  step: z.number().nullable().optional(),
  batch: z.number().nullable().optional(),
  metrics: jsonObjectSchema.nullable().optional(),
});

const trainingClusterInitializedEventSchema =
  trainingProgressEventBaseSchema.extend({
    type: z.literal("cluster_initialized"),
    node: z.string(),
    count: z.number(),
    capacity: z.array(z.number()),
    coordinates: z.array(z.array(z.number())),
    coordinateCount: z.number().nullable().optional(),
    coordinatesTruncated: z.boolean().nullable().optional(),
  });

const trainingNeuronAddedEventSchema = trainingProgressEventBaseSchema.extend({
  type: z.literal("neuron_added"),
  node: z.string(),
  coord: z.array(z.number()),
  count: z.number(),
  capacity: z.array(z.number()),
  epoch: z.number().nullable().optional(),
  step: z.number().nullable().optional(),
});

const trainingNeuronsAddedEventSchema = trainingProgressEventBaseSchema.extend({
  type: z.literal("neurons_added"),
  node: z.string(),
  coordinates: z.array(z.array(z.number())),
  coordinateCount: z.number(),
  coordinatesTruncated: z.boolean().nullable().optional(),
  count: z.number(),
  capacity: z.array(z.number()),
  epoch: z.number().nullable().optional(),
  step: z.number().nullable().optional(),
});

const futureTrainingProgressEventSchema = trainingProgressEventBaseSchema.refine(
  (event) => !knownTrainingProgressEventTypeSet.has(event.type),
  {
    path: ["type"],
    message: "Known training progress events must match their event schema",
  },
);

export const trainingProgressEventSchema = z.union([
  trainingJobStartedEventSchema,
  trainingWorkerStartedEventSchema,
  trainingWorkerCompletedEventSchema,
  trainingCancelledEventSchema,
  trainingErrorEventSchema,
  trainingDatasetStartedEventSchema,
  trainingDatasetCompletedEventSchema,
  trainingRunProgressEventSchema,
  trainingClusterInitializedEventSchema,
  trainingNeuronAddedEventSchema,
  trainingNeuronsAddedEventSchema,
  futureTrainingProgressEventSchema,
]);

export const trainingClusterGrowthAdditionSchema = z.object({
  coord: z.tuple([z.number(), z.number(), z.number()]),
  step: z.number().nullable(),
  epoch: z.number().nullable(),
});

export const trainingClusterGrowthSchema = z.object({
  node: z.string(),
  count: z.number(),
  capacityTotal: z.number(),
  additionCount: z.number(),
  additions: z.array(trainingClusterGrowthAdditionSchema),
});

export const trainingJobSchema = z.object({
  id: z.string(),
  status: trainingJobStatusSchema,
  modelType: z.string(),
  model: z.string(),
  preset: z.string(),
  presets: z.array(z.string()).optional(),
  experimentTask: z.string().optional(),
  datasets: z.array(z.string()),
  overrides: configOverridesSchema,
  search: z
    .object({
      mode: z.enum(["grid", "random"]),
      values: z.record(z.array(configValueSchema)),
      randomSamples: z.number().nullable().optional(),
    })
    .nullable()
    .optional(),
  plannedRunCount: z.number().optional(),
  runPlan: trainingRunPlanSchema.nullable().optional(),
  monitors: z.array(z.string()),
  logFolder: z.string(),
  createdAt: z.string(),
  updatedAt: z.string(),
  exitCode: z.number().nullable(),
  pid: z.number(),
  cancellationMode: z
    .enum(["strict-cgroup", "process-group", "unsupported"])
    .optional(),
  currentPreset: z.string().nullable().optional(),
  currentDataset: z.string().nullable(),
  epoch: z.number().nullable(),
  step: z.number().nullable(),
  metrics: jsonObjectSchema,
  logDir: z.string().nullable(),
  events: z.array(trainingProgressEventSchema).optional().default([]),
  eventCount: z.number().optional().default(0),
  eventCounts: z.record(z.number()).optional().default({}),
  eventsTruncated: z.boolean().optional().default(false),
  clusterGrowth: z.array(trainingClusterGrowthSchema).optional().default([]),
  logTail: z.array(z.string()),
  resultLinks: z.array(
    z.object({
      preset: z.string().nullable().optional(),
      dataset: z.string().nullable().optional(),
      logDir: z.string().nullable().optional(),
    }),
  ),
});

export const trainingJobEventsSchema = z.object({
  jobId: z.string(),
  offset: z.number(),
  limit: z.number(),
  totalCount: z.number(),
  nextOffset: z.number().nullable(),
  events: z.array(trainingProgressEventSchema),
});

export type TrainingRunChange = z.infer<typeof trainingRunChangeSchema>;
export type TrainingRun = z.infer<typeof trainingRunSchema>;
export type TrainingRunPlanSummary = z.infer<typeof trainingRunPlanSummarySchema>;
export type TrainingRunPlan = z.infer<typeof trainingRunPlanSchema>;
export type TrainingProgressEvent = z.infer<typeof trainingProgressEventSchema>;
export type TrainingClusterGrowth = z.infer<typeof trainingClusterGrowthSchema>;
export type TrainingJob = z.infer<typeof trainingJobSchema>;
export type TrainingJobEvents = z.infer<typeof trainingJobEventsSchema>;

export type TrainingSearchCreateInput = {
  mode: "grid" | "random";
  values: Record<string, ConfigValue[]>;
  randomSamples?: number;
};

export type TrainingSearchSubmitInput = {
  mode: "grid" | "random";
  values: Record<string, ConfigValue[]>;
  randomSamples?: number | null;
};

export type TrainingRunSubmitChangeInput = {
  key: string;
  label: string;
  value: ConfigValue;
  source: "override" | "search";
};

export type TrainingRunSubmitInput = {
  id: string;
  index: number;
  status: TrainingRun["status"];
  preset: string;
  snapshotId?: string | null;
  snapshotName?: string | null;
  dataset: string;
  experimentTask?: string;
  changes: TrainingRunSubmitChangeInput[];
  overrides: ConfigOverrides;
  command: string;
  totalEpochs: number;
  currentEpoch: number;
  metrics: JsonObject;
  logDir: string | null;
  error: string | null;
  errorTraceback?: string | null;
};

export type TrainingRunPlanSubmitSummaryInput = {
  totalRuns: number;
  completedRuns: number;
  runningRuns: number;
  pendingRuns: number;
  failedRuns: number;
  cancelledRuns: number;
  skippedRuns: number;
  totalEpochs: number;
  completedEpochs: number;
  remainingEpochs: number;
};

export type TrainingRunPlanSubmitInput = {
  modelType: string;
  model: string;
  preset: string;
  presets: string[];
  experimentTask?: string;
  datasets: string[];
  overrides: ConfigOverrides;
  search: TrainingSearchSubmitInput | null;
  logFolder: string;
  isRandomSearch: boolean;
  runs: TrainingRunSubmitInput[];
  summary: TrainingRunPlanSubmitSummaryInput;
};

export type TrainingJobCreateInput = {
  modelType: string;
  model: string;
  preset: string;
  presets?: string[];
  experimentTask?: string;
  datasets: string[];
  overrides: ConfigOverrides;
  logFolder: string;
  monitors: string[];
  search?: TrainingSearchCreateInput;
  runPlan?: TrainingRunPlanSubmitInput;
};

export type TrainingRunPlanCreateInput = {
  modelType: string;
  model: string;
  preset: string;
  presets?: string[];
  experimentTask?: string;
  datasets: string[];
  overrides: ConfigOverrides;
  logFolder?: string;
  monitors?: string[];
  search?: TrainingSearchCreateInput;
};

export function createTrainingJob(input: TrainingJobCreateInput) {
  return requestJson("/training/jobs", trainingJobSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function fetchTrainingRunPlan(
  input: TrainingRunPlanCreateInput,
  { signal }: ApiRequestOptions = {},
) {
  return requestJson("/training/run-plan", trainingRunPlanSchema, {
    method: "POST",
    body: JSON.stringify(input),
    signal,
  });
}

export function fetchTrainingJob(
  id: string,
  { signal }: ApiRequestOptions = {},
) {
  return requestJson(
    `/training/jobs/${encodeURIComponent(id)}`,
    trainingJobSchema,
    { signal },
  );
}

export function fetchTrainingJobEvents(
  id: string,
  input: { offset?: number; limit?: number } = {},
) {
  const params = new URLSearchParams();
  if (input.offset !== undefined) {
    params.set("offset", String(input.offset));
  }
  if (input.limit !== undefined) {
    params.set("limit", String(input.limit));
  }
  const query = params.toString();
  return requestJson(
    `/training/jobs/${encodeURIComponent(id)}/events${query ? `?${query}` : ""}`,
    trainingJobEventsSchema,
  );
}

export function cancelTrainingJob(id: string) {
  return requestJson(`/training/jobs/${encodeURIComponent(id)}/cancel`, trainingJobSchema, {
    method: "POST",
  });
}
