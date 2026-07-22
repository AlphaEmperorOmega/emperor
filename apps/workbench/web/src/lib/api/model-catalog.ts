import { z } from "zod";

import { requestJson } from "@/lib/api/client";
import {
  modelIdentitySchema,
  type ModelIdentity,
} from "@/lib/api/model-identity-schema";

const modelsSchema = z.object({ models: z.array(modelIdentitySchema) });

export type { ModelIdentity };

type ApiRequestOptions = {
  signal?: AbortSignal;
};

export function fetchModels(options: ApiRequestOptions = {}) {
  return requestJson("/models", modelsSchema, options, {
    authenticationProbe: true,
  });
}
