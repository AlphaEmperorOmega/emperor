import { z } from "zod";

import { requestJson } from "@/lib/api/client";
import { configValueSchema, type ConfigValue } from "@/lib/api/schemas";

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
  changes: z.array(trainingRunChangeSchema),
  overrides: z.record(configValueSchema),
  command: z.string(),
  totalEpochs: z.number(),
  currentEpoch: z.number(),
  metrics: z.record(z.unknown()),
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
  model: z.string(),
  preset: z.string(),
  presets: z.array(z.string()),
  datasets: z.array(z.string()),
  overrides: z.record(z.unknown()),
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

export const trainingJobSchema = z.object({
  id: z.string(),
  status: z.string(),
  model: z.string(),
  preset: z.string(),
  presets: z.array(z.string()).optional(),
  datasets: z.array(z.string()),
  overrides: z.record(z.unknown()),
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
  currentPreset: z.string().nullable().optional(),
  currentDataset: z.string().nullable(),
  epoch: z.number().nullable(),
  step: z.number().nullable(),
  metrics: z.record(z.unknown()),
  logDir: z.string().nullable(),
  events: z.array(z.record(z.unknown())),
  logTail: z.array(z.string()),
  resultLinks: z.array(
    z.object({
      preset: z.string().nullable().optional(),
      dataset: z.string().nullable().optional(),
      logDir: z.string().nullable().optional(),
    }),
  ),
});

export type TrainingRunChange = z.infer<typeof trainingRunChangeSchema>;
export type TrainingRun = z.infer<typeof trainingRunSchema>;
export type TrainingRunPlanSummary = z.infer<typeof trainingRunPlanSummarySchema>;
export type TrainingRunPlan = z.infer<typeof trainingRunPlanSchema>;
export type TrainingJob = z.infer<typeof trainingJobSchema>;

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
  changes: TrainingRunSubmitChangeInput[];
  overrides: Record<string, ConfigValue>;
  command: string;
  totalEpochs: number;
  currentEpoch: number;
  metrics: Record<string, unknown>;
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
  model: string;
  preset: string;
  presets: string[];
  datasets: string[];
  overrides: Record<string, unknown>;
  search: TrainingSearchSubmitInput | null;
  logFolder: string;
  isRandomSearch: boolean;
  runs: TrainingRunSubmitInput[];
  summary: TrainingRunPlanSubmitSummaryInput;
};

export type TrainingJobCreateInput = {
  model: string;
  preset: string;
  presets?: string[];
  datasets: string[];
  overrides: Record<string, string>;
  logFolder: string;
  monitors: string[];
  search?: TrainingSearchCreateInput;
  runPlan?: TrainingRunPlanSubmitInput;
};

export type TrainingRunPlanCreateInput = {
  model: string;
  preset: string;
  presets?: string[];
  datasets: string[];
  overrides: Record<string, string>;
  logFolder?: string;
  search?: TrainingSearchCreateInput;
};

export function createTrainingJob(input: TrainingJobCreateInput) {
  return requestJson("/training/jobs", trainingJobSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function fetchTrainingRunPlan(input: TrainingRunPlanCreateInput) {
  return requestJson("/training/run-plan", trainingRunPlanSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function fetchTrainingJob(id: string) {
  return requestJson(`/training/jobs/${id}`, trainingJobSchema);
}

export function cancelTrainingJob(id: string) {
  return requestJson(`/training/jobs/${id}/cancel`, trainingJobSchema, {
    method: "POST",
  });
}
