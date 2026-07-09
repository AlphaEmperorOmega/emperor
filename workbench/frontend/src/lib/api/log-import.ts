import { z } from "zod";

import { requestMultipartJson } from "@/lib/api/client";

type ApiRequestOptions = {
  signal?: AbortSignal;
};

export const logArchiveImportSchema = z.object({
  extractedFileCount: z.number().int().nonnegative(),
  skippedFileCount: z.number().int().nonnegative(),
  destinationRoot: z.string(),
});

export type LogArchiveImportResponse = z.infer<typeof logArchiveImportSchema>;

export function importLogArchive(file: File, options: ApiRequestOptions = {}) {
  const formData = new FormData();
  formData.set("archive", file, file.name);
  return requestMultipartJson("/logs/import", logArchiveImportSchema, formData, {
    method: "POST",
    signal: options.signal,
  });
}
