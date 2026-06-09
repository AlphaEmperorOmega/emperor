import { z } from "zod";

import { requestJson } from "@/lib/api/client";
import { configValueSchema } from "@/lib/api/schemas";

export const presetSchema = z.object({
  name: z.string(),
  label: z.string(),
  description: z.string(),
});

export const datasetSchema = z.object({
  name: z.string(),
  label: z.string(),
  inputDim: z.number(),
  outputDim: z.number(),
});

export const monitorOptionSchema = z.object({
  name: z.string(),
  label: z.string(),
  description: z.string(),
  kinds: z.array(z.string()),
  defaultEnabled: z.boolean(),
});

export const configFieldSchema = z.object({
  key: z.string(),
  configKey: z.string(),
  flag: z.string(),
  label: z.string(),
  section: z.string(),
  type: z.string(),
  default: configValueSchema,
  nullable: z.boolean(),
  choices: z.array(configValueSchema),
  locked: z.boolean().optional(),
  lockedValue: configValueSchema.optional(),
  lockedReason: z.string().optional(),
});

export const searchAxisSchema = z.object({
  key: z.string(),
  configKey: z.string(),
  searchKey: z.string(),
  label: z.string(),
  section: z.string(),
  type: z.string(),
  values: z.array(configValueSchema),
  locked: z.boolean().optional(),
  lockedValue: configValueSchema.optional(),
  lockedReason: z.string().optional(),
});

const modelsSchema = z.object({ models: z.array(z.string()) });
const presetsSchema = z.object({ model: z.string(), presets: z.array(presetSchema) });
const datasetsSchema = z.object({ model: z.string(), datasets: z.array(datasetSchema) });
const monitorsSchema = z.object({
  model: z.string(),
  monitors: z.array(monitorOptionSchema),
});
const configSchema = z.object({ model: z.string(), fields: z.array(configFieldSchema) });
const searchSpaceSchema = z.object({
  model: z.string(),
  preset: z.string().nullable().optional(),
  axes: z.array(searchAxisSchema),
});

export type Preset = z.infer<typeof presetSchema>;
export type Dataset = z.infer<typeof datasetSchema>;
export type MonitorOption = z.infer<typeof monitorOptionSchema>;
export type ConfigField = z.infer<typeof configFieldSchema>;
export type SearchAxis = z.infer<typeof searchAxisSchema>;
export type SearchSpace = z.infer<typeof searchSpaceSchema>;

function modelPath(model: string) {
  return model.split("/").map(encodeURIComponent).join("/");
}

export function fetchModels() {
  return requestJson("/models", modelsSchema);
}

export function fetchPresets(model: string) {
  return requestJson(`/models/${modelPath(model)}/presets`, presetsSchema);
}

export function fetchDatasets(model: string) {
  return requestJson(`/models/${modelPath(model)}/datasets`, datasetsSchema);
}

export function fetchMonitors(model: string) {
  return requestJson(`/models/${modelPath(model)}/monitors`, monitorsSchema);
}

export function fetchConfigSchema(model: string, preset?: string) {
  const query = preset ? `?preset=${encodeURIComponent(preset)}` : "";
  return requestJson(`/models/${modelPath(model)}/config-schema${query}`, configSchema);
}

export function fetchSearchSpace(model: string, preset?: string) {
  const query = preset ? `?preset=${encodeURIComponent(preset)}` : "";
  return requestJson(`/models/${modelPath(model)}/search-space${query}`, searchSpaceSchema);
}
