import { z } from "zod";

import { requestJson } from "@/lib/api/client";
import {
  jsonObjectSchema,
  jsonValueSchema,
  type ConfigOverrides,
} from "@/lib/api/schemas";

export const graphConfigSchema = z.object({
  typeName: z.string(),
  fields: z.array(
    z.object({
      key: z.string(),
      value: jsonValueSchema,
    }),
  ),
});

export const graphNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  typeName: z.string(),
  path: z.string(),
  graphRole: z.enum(["architecture", "internal", "runtime"]),
  parameterCount: z.number(),
  parameterSizeBytes: z.number().default(0),
  details: jsonObjectSchema,
  config: graphConfigSchema.nullable(),
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
  parameterSizeBytes: z.number().default(0),
  nodes: z.array(graphNodeSchema),
  edges: z.array(graphEdgeSchema),
});

export type GraphConfig = z.infer<typeof graphConfigSchema>;
export type GraphNode = z.infer<typeof graphNodeSchema>;
export type GraphEdge = z.infer<typeof graphEdgeSchema>;
export type InspectResponse = z.infer<typeof inspectResponseSchema>;

export function inspectModel(input: {
  model: string;
  preset: string;
  overrides: ConfigOverrides;
  dataset?: string;
}) {
  return requestJson("/inspect", inspectResponseSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}
