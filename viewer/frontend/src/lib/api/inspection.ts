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
  modelType: z.string(),
  model: z.string(),
  preset: z.string(),
  parameterCount: z.number(),
  parameterSizeBytes: z.number().default(0),
  nodes: z.array(graphNodeSchema),
  edges: z.array(graphEdgeSchema),
});

export const operationGraphNodeSchema = z
  .object({
    id: z.string(),
    label: z.string(),
    opKind: z.enum([
      "placeholder",
      "call_function",
      "call_module",
      "call_method",
      "get_attr",
      "output",
    ]),
    target: z.string(),
    modulePath: z.string().nullable(),
    groupId: z.string().nullable(),
    details: jsonObjectSchema,
  })
  .strict();

export const operationGraphEdgeSchema = z
  .object({
    id: z.string(),
    source: z.string(),
    target: z.string(),
  })
  .strict();

export const operationGraphResponseSchema = z
  .object({
    model: z.string(),
    modelType: z.string(),
    preset: z.string(),
    source: z.literal("torch-export"),
    status: z.enum(["ok", "unsupported"]),
    nodes: z.array(operationGraphNodeSchema),
    edges: z.array(operationGraphEdgeSchema),
    warnings: z.array(z.string()),
  })
  .strict();

export type GraphConfig = z.infer<typeof graphConfigSchema>;
export type GraphNode = z.infer<typeof graphNodeSchema>;
export type GraphEdge = z.infer<typeof graphEdgeSchema>;
export type InspectResponse = z.infer<typeof inspectResponseSchema>;
export type OperationGraphNode = z.infer<typeof operationGraphNodeSchema>;
export type OperationGraphEdge = z.infer<typeof operationGraphEdgeSchema>;
export type OperationGraphResponse = z.infer<typeof operationGraphResponseSchema>;

export function inspectModel(input: {
  modelType: string;
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

export function inspectOperationGraph(input: {
  modelType: string;
  model: string;
  preset: string;
  overrides: ConfigOverrides;
  dataset?: string;
}) {
  return requestJson("/inspect/operation-graph", operationGraphResponseSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}
