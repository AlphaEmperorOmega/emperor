import { z } from "zod";

import { requestJson } from "@/lib/api/client";

export const graphConfigSchema = z.object({
  typeName: z.string(),
  fields: z.array(
    z.object({
      key: z.string(),
      value: z.unknown(),
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
  details: z.record(z.unknown()),
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
  overrides: Record<string, string>;
  dataset?: string;
}) {
  return requestJson("/inspect", inspectResponseSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}
