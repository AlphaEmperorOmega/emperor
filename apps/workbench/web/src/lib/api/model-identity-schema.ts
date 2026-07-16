import { z } from "zod";

import { type ModelIdentity } from "@/lib/api/model-catalog";

export const modelIdentitySchema: z.ZodType<ModelIdentity> = z.object({
  modelType: z.string(),
  model: z.string(),
});
