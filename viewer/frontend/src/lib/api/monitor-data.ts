import { z } from "zod";

import { requestJson } from "@/lib/api/client";

export const monitorDataSchema = z.object({
  jobId: z.string(),
  nodePath: z.string(),
  preset: z.string().nullable().optional(),
  dataset: z.string().nullable(),
  logDir: z.string().nullable(),
  scalarSeries: z.array(
    z.object({
      tag: z.string(),
      label: z.string(),
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
      mimeType: z.string(),
      dataUrl: z.string(),
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
}) {
  const params = new URLSearchParams({ nodePath: input.nodePath });
  if (input.preset) {
    params.set("preset", input.preset);
  }
  if (input.dataset) {
    params.set("dataset", input.dataset);
  }
  return requestJson(
    `/training/jobs/${input.jobId}/monitor-data?${params.toString()}`,
    monitorDataSchema,
  );
}

export function fetchMonitorParameterStatus(input: {
  jobId: string;
  preset?: string;
  dataset?: string;
}) {
  const params = new URLSearchParams();
  if (input.preset) {
    params.set("preset", input.preset);
  }
  if (input.dataset) {
    params.set("dataset", input.dataset);
  }
  const query = params.toString();
  return requestJson(
    `/training/jobs/${input.jobId}/monitor-parameter-status${query ? `?${query}` : ""}`,
    parameterStatusSchema,
  );
}

export function fetchLogRunMonitorData(input: {
  runId: string;
  nodePath: string;
}) {
  const params = new URLSearchParams({ nodePath: input.nodePath });
  return requestJson(
    `/logs/runs/${encodeURIComponent(input.runId)}/monitor-data?${params.toString()}`,
    monitorDataSchema,
  );
}

export function fetchLogParameterStatus(input: { runIds: string[] }) {
  return requestJson("/logs/parameter-status", logParameterStatusSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}
