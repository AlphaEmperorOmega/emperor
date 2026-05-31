import { z } from "zod";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_VIEWER_API_URL ?? "http://127.0.0.1:9999";

export const presetSchema = z.object({
  name: z.string(),
  label: z.string(),
  description: z.string(),
});

export const configFieldSchema = z.object({
  key: z.string(),
  configKey: z.string(),
  flag: z.string(),
  label: z.string(),
  section: z.string(),
  type: z.string(),
  default: z.union([z.string(), z.number(), z.boolean(), z.null()]),
  nullable: z.boolean(),
  choices: z.array(z.union([z.string(), z.number(), z.boolean(), z.null()])),
  searchChoices: z.array(z.union([z.string(), z.number(), z.boolean(), z.null()])),
});

export const graphNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  typeName: z.string(),
  path: z.string(),
  graphRole: z.enum(["architecture", "internal", "runtime"]),
  parameterCount: z.number(),
  details: z.record(z.unknown()),
});

export const graphEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
});

export const inspectResponseSchema = z.object({
  model: z.string(),
  preset: z.string(),
  parameterCount: z.number(),
  nodes: z.array(graphNodeSchema),
  edges: z.array(graphEdgeSchema),
});

const errorBodySchema = z.object({ detail: z.string() }).partial();

const healthSchema = z.object({ status: z.string() });
const modelsSchema = z.object({ models: z.array(z.string()) });
const presetsSchema = z.object({ model: z.string(), presets: z.array(presetSchema) });
const configSchema = z.object({ model: z.string(), fields: z.array(configFieldSchema) });

export type Preset = z.infer<typeof presetSchema>;
export type ConfigField = z.infer<typeof configFieldSchema>;
export type GraphNode = z.infer<typeof graphNodeSchema>;
export type GraphEdge = z.infer<typeof graphEdgeSchema>;
export type InspectResponse = z.infer<typeof inspectResponseSchema>;

async function requestJson<T>(path: string, schema: z.ZodType<T>, init?: RequestInit) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = errorBodySchema.safeParse(await response.json());
      if (payload.success && payload.data.detail) {
        detail = payload.data.detail;
      }
    } catch {
      // Response was not JSON; keep status text.
    }
    throw new Error(detail || `Request failed with ${response.status}`);
  }
  return schema.parse(await response.json());
}

export function fetchHealth() {
  return requestJson("/health", healthSchema);
}

export function fetchModels() {
  return requestJson("/models", modelsSchema);
}

export function fetchPresets(model: string) {
  return requestJson(`/models/${model}/presets`, presetsSchema);
}

export function fetchConfigSchema(model: string) {
  return requestJson(`/models/${model}/config-schema`, configSchema);
}

export function inspectModel(input: {
  model: string;
  preset: string;
  overrides: Record<string, string>;
}) {
  return requestJson("/inspect", inspectResponseSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}
