import { z } from "zod";

import { requestJson } from "@/lib/api/client";
import { imageDataUrlSchema, imageMimeTypeSchema } from "@/lib/api/schemas";

type ApiRequestOptions = {
  signal?: AbortSignal;
};

const responseMetadataSchema = z.object({
  eventBytes: z.number().optional().nullable(),
  skippedEventFiles: z.number().optional().nullable(),
  sourceItemCount: z.number().optional().nullable(),
  returnedItemCount: z.number().optional().nullable(),
  truncated: z.boolean().optional().nullable(),
  truncationReason: z.string().optional().nullable(),
});

export const monitorDataSchema = z.object({
  jobId: z.string(),
  nodePath: z.string(),
  preset: z.string().nullable().optional(),
  dataset: z.string().nullable(),
  logDir: z.string().nullable(),
  ...responseMetadataSchema.shape,
  scalarSeries: z.array(
    z.object({
      tag: z.string(),
      label: z.string(),
      sourceItemCount: z.number().optional().nullable(),
      returnedItemCount: z.number().optional().nullable(),
      truncated: z.boolean().optional().nullable(),
      truncationReason: z.string().optional().nullable(),
      points: z.array(
        z.object({
          step: z.number(),
          wallTime: z.number(),
          value: z.number(),
        }),
      ),
    }),
  ),
  histograms: z.array(
    z.object({
      tag: z.string(),
      step: z.number(),
      wallTime: z.number(),
      sourceItemCount: z.number().optional().nullable(),
      returnedItemCount: z.number().optional().nullable(),
      truncated: z.boolean().optional().nullable(),
      truncationReason: z.string().optional().nullable(),
      buckets: z.array(
        z.object({
          left: z.number(),
          right: z.number(),
          count: z.number(),
        }),
      ),
    }),
  ),
  images: z.array(
    z.object({
      tag: z.string(),
      step: z.number(),
      wallTime: z.number(),
      mimeType: imageMimeTypeSchema,
      dataUrl: imageDataUrlSchema,
      eventBytes: z.number().optional().nullable(),
      sourceItemCount: z.number().optional().nullable(),
      returnedItemCount: z.number().optional().nullable(),
      truncated: z.boolean().optional().nullable(),
      truncationReason: z.string().optional().nullable(),
    }),
  ),
});

export const parameterChannelStatusSchema = z.object({
  status: z.enum(["updated", "unchanged", "missing", "unknown"]),
  metric: z.string().nullable().optional(),
  lastStep: z.number().nullable().optional(),
  observedPoints: z.number(),
});

export const parameterNodeStatusSchema = z.object({
  nodePath: z.string(),
  weights: parameterChannelStatusSchema,
  bias: parameterChannelStatusSchema,
});

export const parameterStatusSchema = z.object({
  sourceId: z.string(),
  preset: z.string().nullable().optional(),
  dataset: z.string().nullable().optional(),
  logDir: z.string().nullable().optional(),
  ...responseMetadataSchema.shape,
  nodes: z.array(parameterNodeStatusSchema),
});

export const logParameterStatusSchema = z.object({
  runs: z.array(parameterStatusSchema),
});

export type MonitorData = z.infer<typeof monitorDataSchema>;
export type ParameterActivityStatus = z.infer<
  typeof parameterChannelStatusSchema
>["status"];
export type ParameterChannelStatus = z.infer<typeof parameterChannelStatusSchema>;
export type ParameterNodeStatus = z.infer<typeof parameterNodeStatusSchema>;
export type ParameterStatus = z.infer<typeof parameterStatusSchema>;
export type LogParameterStatusResponse = z.infer<typeof logParameterStatusSchema>;

export function fetchMonitorData(input: {
  jobId: string;
  nodePath: string;
  preset?: string;
  dataset?: string;
}, options: ApiRequestOptions = {}) {
  const params = new URLSearchParams({ nodePath: input.nodePath });
  if (input.preset) {
    params.set("preset", input.preset);
  }
  if (input.dataset) {
    params.set("dataset", input.dataset);
  }
  return requestJson(
    `/training/jobs/${encodeURIComponent(input.jobId)}/monitor-data?${params.toString()}`,
    monitorDataSchema,
    { signal: options.signal },
  );
}

export function fetchMonitorParameterStatus(input: {
  jobId: string;
  preset?: string;
  dataset?: string;
}, options: ApiRequestOptions = {}) {
  const params = new URLSearchParams();
  if (input.preset) {
    params.set("preset", input.preset);
  }
  if (input.dataset) {
    params.set("dataset", input.dataset);
  }
  const query = params.toString();
  return requestJson(
    `/training/jobs/${encodeURIComponent(input.jobId)}/monitor-parameter-status${query ? `?${query}` : ""}`,
    parameterStatusSchema,
    { signal: options.signal },
  );
}

export function fetchLogRunMonitorData(input: {
  runId: string;
  nodePath: string;
}, options: ApiRequestOptions = {}) {
  const params = new URLSearchParams({ nodePath: input.nodePath });
  return requestJson(
    `/logs/runs/${encodeURIComponent(input.runId)}/monitor-data?${params.toString()}`,
    monitorDataSchema,
    { signal: options.signal },
  );
}

export function fetchLogParameterStatus(
  input: { runIds: string[] },
  options: ApiRequestOptions = {},
) {
  return requestJson("/logs/parameter-status", logParameterStatusSchema, {
    method: "POST",
    signal: options.signal,
    body: JSON.stringify(input),
  });
}
