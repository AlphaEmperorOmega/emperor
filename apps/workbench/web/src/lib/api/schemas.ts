import { z } from "zod";

export const configValueSchema = z.union([z.string(), z.number(), z.boolean(), z.null()]);
export const configOverridesSchema = z.record(configValueSchema);
export const imageMimeTypeSchema = z.enum([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
]);
export const imageDataUrlSchema = z.union([
  z.literal(""),
  z.string().regex(
    /^data:image\/(?:png|jpeg|webp|gif);base64,(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/,
    "Expected a base64 data URL for png, jpeg, webp, or gif image data",
  ),
]);

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
export const responseCompletenessSchema = z.object({
  sourceItemCount: z.number().optional().nullable(),
  returnedItemCount: z.number().optional().nullable(),
  truncated: z.boolean().optional().nullable(),
  truncationReason: z.string().optional().nullable(),
});
export const eventMetadataSchema = responseCompletenessSchema.extend({
  eventBytes: z.number().optional().nullable(),
  skippedEventFiles: z.number().optional().nullable(),
});

export type ConfigValue = z.infer<typeof configValueSchema>;
export type ConfigOverrides = z.infer<typeof configOverridesSchema>;
export type JsonObject = z.infer<typeof jsonObjectSchema>;
export type ResponseCompleteness = z.infer<typeof responseCompletenessSchema>;
export type EventMetadata = z.infer<typeof eventMetadataSchema>;
