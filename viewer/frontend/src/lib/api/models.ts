import { z } from "zod";

import { requestJson } from "@/lib/api/client";
import { configValueSchema } from "@/lib/api/schemas";

export const modelIdentitySchema = z.object({
  modelType: z.string(),
  model: z.string(),
});

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
  lockedByPresets: z.array(z.string()).optional(),
  lockReasons: z.array(z.string()).optional(),
});

const modelsSchema = z.object({ models: z.array(modelIdentitySchema) });
const presetsSchema = z.object({
  modelType: z.string(),
  model: z.string(),
  presets: z.array(presetSchema),
});
const datasetsSchema = z.object({
  modelType: z.string(),
  model: z.string(),
  datasets: z.array(datasetSchema),
});
const monitorsSchema = z.object({
  modelType: z.string(),
  model: z.string(),
  monitors: z.array(monitorOptionSchema),
});
const configSchema = z.object({
  modelType: z.string(),
  model: z.string(),
  fields: z.array(configFieldSchema),
});
const searchSpaceSchema = z.object({
  modelType: z.string(),
  model: z.string(),
  preset: z.string().nullable().optional(),
  axes: z.array(searchAxisSchema),
});

export type ModelIdentity = z.infer<typeof modelIdentitySchema>;
export type Preset = z.infer<typeof presetSchema>;
export type Dataset = z.infer<typeof datasetSchema>;
export type MonitorOption = z.infer<typeof monitorOptionSchema>;
export type ConfigField = z.infer<typeof configFieldSchema>;
export type SearchAxis = z.infer<typeof searchAxisSchema>;
export type SearchSpace = z.infer<typeof searchSpaceSchema>;

function modelPath({ modelType, model }: ModelIdentity) {
  return `${encodeURIComponent(modelType)}/${encodeURIComponent(model)}`;
}

export function fetchModels() {
  return requestJson("/models", modelsSchema);
}

export function fetchPresets(identity: ModelIdentity) {
  return requestJson(`/models/${modelPath(identity)}/presets`, presetsSchema);
}

export function fetchDatasets(identity: ModelIdentity) {
  return requestJson(`/models/${modelPath(identity)}/datasets`, datasetsSchema);
}

export function fetchMonitors(identity: ModelIdentity) {
  return requestJson(`/models/${modelPath(identity)}/monitors`, monitorsSchema);
}

export function fetchConfigSchema(identity: ModelIdentity, preset?: string) {
  const query = preset ? `?preset=${encodeURIComponent(preset)}` : "";
  return requestJson(
    `/models/${modelPath(identity)}/config-schema${query}`,
    configSchema,
  );
}

export function fetchSearchSpace(
  identity: ModelIdentity,
  preset?: string,
  presets?: readonly string[],
) {
  const params = new URLSearchParams();
  if (preset) {
    params.set("preset", preset);
  }
  const selectedPresets = Array.from(
    new Set((presets ?? []).filter((selectedPreset) => selectedPreset.length > 0)),
  );
  if (selectedPresets.length > 0) {
    params.set("presets", selectedPresets.join(","));
  }
  const query = params.toString() ? `?${params.toString()}` : "";
  return requestJson(
    `/models/${modelPath(identity)}/search-space${query}`,
    searchSpaceSchema,
  );
}
