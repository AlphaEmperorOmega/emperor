import { z } from "zod";

import {
  requestJson,
  type MutationRequestOptions,
} from "@/lib/api/client";
import { modelIdentitySchema, type ModelIdentity } from "@/lib/api/models";

export const logExperimentDeleteSchema = z.object({
  experiment: z.string(),
  deletedRunIds: z.array(z.string()),
  deletedRunCount: z.number(),
  deletedRelativePath: z.string(),
});

export const logRunDeleteCandidateSchema = z.object({
  id: z.string(),
  experiment: z.string(),
  modelType: z.string(),
  model: z.string(),
  preset: z.string(),
  dataset: z.string(),
  runName: z.string(),
  version: z.string(),
  relativePath: z.string(),
});

export const logRunDeleteAffectedValuesSchema = z.object({
  experiments: z.array(z.string()),
  datasets: z.array(z.string()),
  models: z.array(modelIdentitySchema),
  presets: z.array(z.string()),
  runIds: z.array(z.string()),
});

export const logRunDeleteCountsSchema = z.object({
  runs: z.number(),
  experiments: z.number(),
  datasets: z.number(),
  models: z.number(),
  presets: z.number(),
});

export const logRunDeleteBlockerSchema = z.object({
  id: z.string(),
  logFolder: z.string(),
  status: z.string(),
});

export const logRunDeletePlanSchema = z.object({
  candidateCount: z.number(),
  sourceItemCount: z.number().optional().nullable(),
  returnedItemCount: z.number().optional().nullable(),
  truncated: z.boolean().optional().nullable(),
  truncationReason: z.string().optional().nullable(),
  counts: logRunDeleteCountsSchema,
  affected: logRunDeleteAffectedValuesSchema,
  candidates: z.array(logRunDeleteCandidateSchema),
  blockedByActiveJobs: z.array(logRunDeleteBlockerSchema),
  canDelete: z.boolean(),
});

export const logRunDeleteSchema = logRunDeletePlanSchema.extend({
  deletedRunIds: z.array(z.string()),
  deletedRunCount: z.number(),
  deletedRelativePaths: z.array(z.string()),
});

export type LogExperimentDeleteResponse = z.infer<typeof logExperimentDeleteSchema>;
export type LogRunDeleteCandidate = z.infer<typeof logRunDeleteCandidateSchema>;
export type LogRunDeletePlan = z.infer<typeof logRunDeletePlanSchema>;
export type LogRunDeleteResponse = z.infer<typeof logRunDeleteSchema>;

export type LogPresetDeleteTarget = {
  experiment: string;
  preset: string;
};

export type LogRunDeleteFilters = {
  experiments: string[];
  datasets: string[];
  models: ModelIdentity[];
  presets: string[];
  runIds: string[];
};

export function deleteLogExperiment(
  experiment: string,
  mutation: MutationRequestOptions,
) {
  return requestJson(
    `/logs/experiments/${encodeURIComponent(experiment)}`,
    logExperimentDeleteSchema,
    {
      method: "DELETE",
    },
    { mutation },
  );
}

export function createLogRunDeletePlan(filters: LogRunDeleteFilters) {
  return requestJson("/logs/runs/delete-plan", logRunDeletePlanSchema, {
    method: "POST",
    body: JSON.stringify(filters),
  });
}

export function createLogPresetDeletePlan(target: LogPresetDeleteTarget) {
  return requestJson(
    "/logs/runs/preset-delete-plan",
    logRunDeletePlanSchema,
    {
      method: "POST",
      body: JSON.stringify(target),
    },
  );
}

export function deleteLogPreset(
  target: LogPresetDeleteTarget,
  mutation: MutationRequestOptions,
) {
  return requestJson(
    "/logs/runs/preset-delete",
    logRunDeleteSchema,
    {
      method: "POST",
      body: JSON.stringify(target),
    },
    { mutation },
  );
}

export function deleteLogRuns(
  filters: LogRunDeleteFilters,
  mutation: MutationRequestOptions,
) {
  return requestJson(
    "/logs/runs/delete",
    logRunDeleteSchema,
    {
      method: "POST",
      body: JSON.stringify(filters),
    },
    { mutation },
  );
}
