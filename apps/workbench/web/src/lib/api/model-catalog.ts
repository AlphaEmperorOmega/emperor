import {
  array,
  object,
  string,
  type output,
} from "zod/v4-mini";

import { requestJson } from "@/lib/api/client";

const modelIdentitySchema = object({
  modelType: string(),
  model: string(),
});

const modelsSchema = object({ models: array(modelIdentitySchema) });

export type ModelIdentity = output<typeof modelIdentitySchema>;

type ApiRequestOptions = {
  signal?: AbortSignal;
};

export function fetchModels(options: ApiRequestOptions = {}) {
  return requestJson("/models", modelsSchema, options, {
    authenticationProbe: true,
  });
}
