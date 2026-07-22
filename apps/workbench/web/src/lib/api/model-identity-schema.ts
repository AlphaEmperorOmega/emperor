import { z } from "zod";

export const modelIdentitySegmentSchema = z
  .string()
  .regex(/^[A-Za-z_][A-Za-z0-9_]*$/);

export const modelIdentitySchema = z.object({
  modelType: modelIdentitySegmentSchema,
  model: modelIdentitySegmentSchema,
});

export type ModelIdentity = z.infer<typeof modelIdentitySchema>;
