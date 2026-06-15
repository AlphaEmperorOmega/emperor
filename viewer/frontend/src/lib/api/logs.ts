import { z } from "zod";

import { requestJson } from "@/lib/api/client";
import { jsonObjectSchema } from "@/lib/api/schemas";

function experimentFromRelativePath(relativePath: string) {
  return relativePath.split("/").find(Boolean) ?? "unknown";
}

const logRunFields = {
  id: z.string(),
  group: z.string().nullable(),
  model: z.string(),
  preset: z.string(),
  dataset: z.string(),
  runName: z.string(),
  timestamp: z.string().nullable(),
  version: z.string(),
  relativePath: z.string(),
  hasResult: z.boolean(),
  eventFileCount: z.number(),
  checkpointCount: z.number(),
  hasHparams: z.boolean(),
  metrics: jsonObjectSchema,
};

const logRunPayloadSchema = z.object({
  ...logRunFields,
  experiment: z.string().nullish(),
});

const logRunOutputSchema = z.object({
  ...logRunFields,
  experiment: z.string(),
});

export const logRunSchema = logRunPayloadSchema
  .transform((run) => ({
    ...run,
    experiment: run.experiment ?? experimentFromRelativePath(run.relativePath),
  }))
  .pipe(logRunOutputSchema);

export const logRunTagsSchema = z.object({
  runId: z.string(),
  scalarTags: z.array(z.string()),
  histogramTags: z.array(z.string()),
  imageTags: z.array(z.string()),
  textTags: z.array(z.string()),
});

export const logScalarPointSchema = z.object({
  step: z.number(),
  wallTime: z.number(),
  value: z.number(),
});

export const logScalarSeriesSchema = z.object({
  runId: z.string(),
  tag: z.string(),
  points: z.array(logScalarPointSchema),
  sourcePointCount: z.number().optional().nullable(),
  truncated: z.boolean().optional().nullable(),
});

export const logImageSummarySchema = z.object({
  runId: z.string(),
  tag: z.string(),
  step: z.number(),
  wallTime: z.number(),
  mimeType: z.string(),
  dataUrl: z.string(),
});

export const logTextSummarySchema = z.object({
  runId: z.string(),
  tag: z.string(),
  step: z.number(),
  wallTime: z.number(),
  text: z.string(),
});

export const logExperimentSchema = z.object({
  experiment: z.string(),
  runCount: z.number(),
  relativePath: z.string(),
});

const paginationSchema = z.object({
  total: z.number().optional(),
  limit: z.number().optional(),
  offset: z.number().optional(),
  hasMore: z.boolean().optional(),
});

const logRunsSchema = paginationSchema.extend({ runs: z.array(logRunSchema) });

const logExperimentsSchema = paginationSchema.extend({
  experiments: z.array(logExperimentSchema),
});

export const logCheckpointSchema = z.object({
  id: z.string(),
  runId: z.string(),
  filename: z.string(),
  relativePath: z.string(),
  epoch: z.number().nullable(),
  step: z.number().nullable(),
  sizeBytes: z.number(),
  modifiedAt: z.string(),
});

export const logRunArtifactSchema = z.object({
  id: z.string(),
  kind: z.string(),
  label: z.string(),
  relativePath: z.string(),
  sizeBytes: z.number(),
  modifiedAt: z.string(),
});

const logTagsSchema = z.object({ runs: z.array(logRunTagsSchema) });
const logScalarsSchema = z.object({ series: z.array(logScalarSeriesSchema) });
const logMediaSchema = z.object({
  images: z.array(logImageSummarySchema),
  texts: z.array(logTextSummarySchema),
});
export const logCheckpointsSchema = z.object({
  checkpoints: z.array(logCheckpointSchema),
});
export const logRunArtifactsSchema = z.object({
  runId: z.string(),
  params: jsonObjectSchema,
  metrics: jsonObjectSchema,
  artifacts: z.array(logRunArtifactSchema),
  checkpoints: z.array(logCheckpointSchema),
});

export type LogRun = z.infer<typeof logRunSchema>;
export type LogExperiment = z.infer<typeof logExperimentSchema>;
export type LogRunTags = z.infer<typeof logRunTagsSchema>;
export type LogScalarPoint = z.infer<typeof logScalarPointSchema>;
export type LogScalarSeries = z.infer<typeof logScalarSeriesSchema>;
export type LogImageSummary = z.infer<typeof logImageSummarySchema>;
export type LogTextSummary = z.infer<typeof logTextSummarySchema>;
export type LogMedia = z.infer<typeof logMediaSchema>;
export type LogCheckpoint = z.infer<typeof logCheckpointSchema>;
export type LogRunArtifact = z.infer<typeof logRunArtifactSchema>;
export type LogRunArtifacts = z.infer<typeof logRunArtifactsSchema>;

const DEFAULT_LOG_PAGE_LIMIT = 500;
export const DEFAULT_LOG_SCALAR_MAX_POINTS = 500;
export const LOG_SCALAR_TAG_REQUEST_LIMIT = 50;
export const LOG_SCALAR_SAMPLING = "tail";

type PaginatedPage = {
  total?: number;
  limit?: number;
  offset?: number;
  hasMore?: boolean;
};

type PaginatedParams = Record<
  string,
  string | number | boolean | readonly string[] | undefined
>;

type FetchPaginatedOptions<TPage extends PaginatedPage, TItem> = {
  endpoint: string;
  schema: z.ZodType<TPage, z.ZodTypeDef, unknown>;
  params?: PaginatedParams;
  pagination?: { limit: number; offset?: number };
  includeAllPages?: boolean;
  getItems(page: TPage): TItem[];
  getTotal?(page: TPage): number | undefined;
};

type FetchPaginatedResult<TPage, TItem> = {
  firstPage: TPage;
  items: TItem[];
  total: number;
  limit: number;
  offset: number;
};

function paginatedPath(
  endpoint: string,
  params?: PaginatedParams,
  pagination?: { limit: number; offset: number },
) {
  const searchParams = new URLSearchParams();
  Object.entries(params ?? {}).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      value.forEach((entry) => searchParams.append(key, entry));
    } else if (value !== undefined) {
      searchParams.set(key, String(value));
    }
  });
  if (pagination) {
    searchParams.set("limit", String(pagination.limit));
    searchParams.set("offset", String(pagination.offset));
  }
  const query = searchParams.toString();
  return query ? `${endpoint}?${query}` : endpoint;
}

async function fetchPaginated<TPage extends PaginatedPage, TItem>({
  endpoint,
  schema,
  params,
  pagination,
  includeAllPages = false,
  getItems,
  getTotal,
}: FetchPaginatedOptions<TPage, TItem>): Promise<
  FetchPaginatedResult<TPage, TItem>
> {
  const firstPage = await requestJson(
    paginatedPath(
      endpoint,
      params,
      pagination
        ? { limit: pagination.limit, offset: pagination.offset ?? 0 }
        : undefined,
    ),
    schema,
  );
  const items = [...getItems(firstPage)];
  let limit =
    firstPage.limit && firstPage.limit > 0
      ? firstPage.limit
      : items.length || DEFAULT_LOG_PAGE_LIMIT;
  let offset = firstPage.offset ?? 0;
  let hasMore = firstPage.hasMore ?? false;

  while (includeAllPages && hasMore) {
    const nextOffset = offset + limit;
    const page = await requestJson(
      paginatedPath(endpoint, params, { limit, offset: nextOffset }),
      schema,
    );
    items.push(...getItems(page));
    limit = page.limit && page.limit > 0 ? page.limit : limit;
    offset = page.offset ?? nextOffset;
    hasMore = page.hasMore ?? false;
  }

  return {
    firstPage,
    items,
    total: getTotal?.(firstPage) ?? firstPage.total ?? items.length,
    limit,
    offset: firstPage.offset ?? 0,
  };
}

export type FetchLogRunsFilters = {
  experiment?: readonly string[];
  model?: readonly string[];
  preset?: readonly string[];
  dataset?: readonly string[];
  hasEventFiles?: boolean;
};

export type FetchLogRunsInput = {
  filters?: FetchLogRunsFilters;
  pagination?: { limit: number; offset?: number };
  includeAllPages?: boolean;
};

export async function fetchLogRuns(input: FetchLogRunsInput = {}) {
  const filters = input.filters;
  const page = await fetchPaginated({
    endpoint: "/logs/runs",
    schema: logRunsSchema,
    params: filters
      ? {
          experiment: filters.experiment,
          model: filters.model,
          preset: filters.preset,
          dataset: filters.dataset,
          hasEventFiles: filters.hasEventFiles,
        }
      : undefined,
    pagination: input.pagination,
    includeAllPages: input.includeAllPages,
    getItems: (logPage) => logPage.runs,
    getTotal: (logPage) => logPage.total,
  });

  return {
    ...page.firstPage,
    runs: page.items,
    total: page.total,
    limit: page.limit,
    offset: page.offset,
    hasMore: input.includeAllPages ? false : page.firstPage.hasMore ?? false,
  };
}

export async function fetchLogExperiments() {
  const page = await fetchPaginated({
    endpoint: "/logs/experiments",
    schema: logExperimentsSchema,
    includeAllPages: true,
    getItems: (logPage) => logPage.experiments,
    getTotal: (logPage) => logPage.total,
  });

  return {
    ...page.firstPage,
    experiments: page.items,
    total: page.total,
    limit: page.limit,
    offset: page.offset,
    hasMore: false,
  };
}

export function fetchLogTags(input: { runIds: string[] }) {
  return requestJson("/logs/tags", logTagsSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function fetchLogScalars(input: {
  runIds: string[];
  tags: string[];
  maxPoints?: number;
  sampling?: typeof LOG_SCALAR_SAMPLING;
}) {
  const request = {
    maxPoints: DEFAULT_LOG_SCALAR_MAX_POINTS,
    sampling: LOG_SCALAR_SAMPLING,
    ...input,
  };
  const tagChunks = Array.from(
    { length: Math.ceil(input.tags.length / LOG_SCALAR_TAG_REQUEST_LIMIT) },
    (_, index) =>
      input.tags.slice(
        index * LOG_SCALAR_TAG_REQUEST_LIMIT,
        (index + 1) * LOG_SCALAR_TAG_REQUEST_LIMIT,
      ),
  );

  if (tagChunks.length <= 1) {
    return requestJson("/logs/scalars", logScalarsSchema, {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  return Promise.all(
    tagChunks.map((tags) =>
      requestJson("/logs/scalars", logScalarsSchema, {
        method: "POST",
        body: JSON.stringify({
          ...request,
          tags,
        }),
      }),
    ),
  ).then((pages) => ({
    series: pages.flatMap((page) => page.series),
  }));
}

export function fetchLogMedia(input: {
  runIds: string[];
  imageTags: string[];
  textTags: string[];
}) {
  return requestJson("/logs/media", logMediaSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function fetchLogCheckpoints(input: { runIds: string[] }) {
  return requestJson("/logs/checkpoints", logCheckpointsSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function fetchLogRunArtifacts(runId: string) {
  return requestJson(
    `/logs/runs/${encodeURIComponent(runId)}/artifacts`,
    logRunArtifactsSchema,
  );
}
