import { z } from "zod";

export const configValueSchema = z.union([z.string(), z.number(), z.boolean(), z.null()]);
export const configOverridesSchema = z.record(configValueSchema);

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export const jsonValueSchema: z.ZodType<JsonValue> = z.lazy(() =>
  z.union([
    z.string(),
    z.number(),
    z.boolean(),
    z.null(),
    z.array(jsonValueSchema),
    z.record(jsonValueSchema),
  ]),
);

export const jsonObjectSchema = z.record(jsonValueSchema);

export type ConfigValue = z.infer<typeof configValueSchema>;
export type ConfigOverrides = z.infer<typeof configOverridesSchema>;
export type JsonObject = z.infer<typeof jsonObjectSchema>;
