import { z } from "zod";

import { requestJson } from "@/lib/api/client";
import { mapWithConcurrency } from "@/lib/api/concurrency";
import { type ModelIdentity } from "@/lib/api/models";
import {
  imageDataUrlSchema,
  imageMimeTypeSchema,
  jsonObjectSchema,
} from "@/lib/api/schemas";

type ApiRequestOptions = {
  signal?: AbortSignal;
};

function experimentFromRelativePath(relativePath: string) {
  return relativePath.split("/").find(Boolean) ?? "unknown";
}

const logRunFields = {
  id: z.string(),
  group: z.string().nullable(),
  modelType: z.string(),
  model: z.string(),
  preset: z.string(),
  experimentTask: z.string().nullable().optional(),
  dataset: z.string(),
  runName: z.string(),
  timestamp: z.string().nullable(),
  version: z.string(),
  relativePath: z.string(),
  hasResult: z.boolean(),
  eventFileCount: z.number(),
  checkpointCount: z.number(),
  hasHparams: z.boolean(),
  hasLayerMonitorData: z.boolean().nullable().optional(),
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

const responseMetadataSchema = z.object({
  eventBytes: z.number().optional().nullable(),
  skippedEventFiles: z.number().optional().nullable(),
  sourceItemCount: z.number().optional().nullable(),
  returnedItemCount: z.number().optional().nullable(),
  truncated: z.boolean().optional().nullable(),
  truncationReason: z.string().optional().nullable(),
});

export const logRunSchema = logRunPayloadSchema
  .transform((run) => ({
    ...run,
    experiment: run.experiment ?? experimentFromRelativePath(run.relativePath),
  }))
  .pipe(logRunOutputSchema);

export const logRunTagsSchema = z.object({
  runId: z.string(),
  hasLayerMonitorData: z.boolean().nullable().optional(),
  ...responseMetadataSchema.shape,
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
  mimeType: imageMimeTypeSchema,
  dataUrl: imageDataUrlSchema,
  ...responseMetadataSchema.shape,
});

export const logTextSummarySchema = z.object({
  runId: z.string(),
  tag: z.string(),
  step: z.number(),
  wallTime: z.number(),
  text: z.string(),
  ...responseMetadataSchema.shape,
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
  ...responseMetadataSchema.shape,
  images: z.array(logImageSummarySchema),
  texts: z.array(logTextSummarySchema),
});
export const logCheckpointsSchema = z.object({
  ...responseMetadataSchema.shape,
  checkpoints: z.array(logCheckpointSchema),
});
export const logRunArtifactsSchema = z.object({
  runId: z.string(),
  params: jsonObjectSchema,
  metrics: jsonObjectSchema,
  ...responseMetadataSchema.shape,
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
export const LOG_TAG_RUN_REQUEST_LIMIT = 20;
export const LOG_SCALAR_RUN_REQUEST_LIMIT = 2;
export const LOG_SCALAR_TAG_REQUEST_LIMIT = 50;
export const LOG_MEDIA_TAG_REQUEST_LIMIT = 20;
export const LOG_TAG_REQUEST_CONCURRENCY = 1;
export const LOG_TENSORBOARD_REQUEST_CONCURRENCY = 2;
export const LOG_SCALAR_REQUEST_CONCURRENCY = 1;
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
  signal?: AbortSignal;
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

const LOG_SCALAR_GLOBAL_REQUEST_CONCURRENCY = 1;
let activeLogScalarRequestCount = 0;
const pendingLogScalarRequests: Array<() => void> = [];

function queueLogScalarRequest<TResponse>(request: () => Promise<TResponse>) {
  // TensorBoard scalar reads are disk-heavy; keep one backend scalar request
  // active so large visible chart sets do not pile up blocking work.
  return new Promise<TResponse>((resolve, reject) => {
    const run = () => {
      activeLogScalarRequestCount += 1;
      Promise.resolve()
        .then(request)
        .then(resolve, reject)
        .finally(() => {
          activeLogScalarRequestCount = Math.max(
            0,
            activeLogScalarRequestCount - 1,
          );
          const next = pendingLogScalarRequests.shift();
          next?.();
        });
    };

    if (activeLogScalarRequestCount < LOG_SCALAR_GLOBAL_REQUEST_CONCURRENCY) {
      run();
      return;
    }

    pendingLogScalarRequests.push(run);
  });
}

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
  signal,
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
    { signal },
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
      { signal },
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
  models?: readonly ModelIdentity[];
  preset?: readonly string[];
  dataset?: readonly string[];
  hasEventFiles?: boolean;
};

export type FetchLogRunsInput = {
  filters?: FetchLogRunsFilters;
  pagination?: { limit: number; offset?: number };
  includeAllPages?: boolean;
};

export async function fetchLogRuns(
  input: FetchLogRunsInput = {},
  options: ApiRequestOptions = {},
) {
  const filters = input.filters;
  const page = await fetchPaginated({
    endpoint: "/logs/runs",
    schema: logRunsSchema,
    params: filters
      ? {
          experiment: filters.experiment,
          modelType: filters.models?.map((model) => model.modelType),
          model: filters.models?.map((model) => model.model),
          preset: filters.preset,
          dataset: filters.dataset,
          hasEventFiles: filters.hasEventFiles,
        }
      : undefined,
    pagination: input.pagination,
    includeAllPages: input.includeAllPages,
    signal: options.signal,
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

export async function fetchLogExperiments(options: ApiRequestOptions = {}) {
  const page = await fetchPaginated({
    endpoint: "/logs/experiments",
    schema: logExperimentsSchema,
    includeAllPages: true,
    signal: options.signal,
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

export async function fetchLogTags(
  input: { runIds: string[] },
  options: ApiRequestOptions = {},
) {
  const runIdChunks = chunkList(input.runIds, LOG_TAG_RUN_REQUEST_LIMIT);
  if (runIdChunks.length <= 1) {
    return requestJson("/logs/tags", logTagsSchema, {
      method: "POST",
      signal: options.signal,
      body: JSON.stringify(input),
    });
  }

  const pages = await mapWithConcurrency(
    runIdChunks,
    LOG_TAG_REQUEST_CONCURRENCY,
    (runIds) =>
      requestJson("/logs/tags", logTagsSchema, {
        method: "POST",
        signal: options.signal,
        body: JSON.stringify({ runIds }),
      }),
  );
  return {
    runs: pages.flatMap((page) => page.runs),
  };
}

export function fetchLogScalars(input: {
  runIds: string[];
  tags: string[];
  maxPoints?: number;
  sampling?: typeof LOG_SCALAR_SAMPLING;
}, options: ApiRequestOptions = {}) {
  const request = {
    maxPoints: DEFAULT_LOG_SCALAR_MAX_POINTS,
    sampling: LOG_SCALAR_SAMPLING,
    ...input,
  };
  const runIdChunks = chunkListForRequestDimension(
    input.runIds,
    LOG_SCALAR_RUN_REQUEST_LIMIT,
  );
  const tagChunks = chunkListForRequestDimension(
    input.tags,
    LOG_SCALAR_TAG_REQUEST_LIMIT,
  );
  const requests = runIdChunks.flatMap((runIds) =>
    tagChunks.map((tags) => ({ runIds, tags })),
  );

  if (requests.length <= 1) {
    return queueLogScalarRequest(() =>
      requestJson("/logs/scalars", logScalarsSchema, {
        method: "POST",
        signal: options.signal,
        body: JSON.stringify(request),
      }),
    );
  }

  return mapWithConcurrency(
    requests,
    LOG_SCALAR_REQUEST_CONCURRENCY,
    ({ runIds, tags }) =>
      queueLogScalarRequest(() =>
        requestJson("/logs/scalars", logScalarsSchema, {
          method: "POST",
          signal: options.signal,
          body: JSON.stringify({
            ...request,
            runIds,
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
}, options: ApiRequestOptions = {}) {
  const imageChunks = chunkList(input.imageTags, LOG_MEDIA_TAG_REQUEST_LIMIT);
  const textChunks = chunkList(input.textTags, LOG_MEDIA_TAG_REQUEST_LIMIT);
  if (imageChunks.length <= 1 && textChunks.length <= 1) {
    return requestJson("/logs/media", logMediaSchema, {
      method: "POST",
      signal: options.signal,
      body: JSON.stringify(input),
    });
  }

  const requests = [
    ...imageChunks.map((imageTags) => ({
      runIds: input.runIds,
      imageTags,
      textTags: [] as string[],
    })),
    ...textChunks.map((textTags) => ({
      runIds: input.runIds,
      imageTags: [] as string[],
      textTags,
    })),
  ];
  if (requests.length === 0) {
    requests.push({ runIds: input.runIds, imageTags: [], textTags: [] });
  }

  return mapWithConcurrency(
    requests,
    LOG_TENSORBOARD_REQUEST_CONCURRENCY,
    (request) =>
      requestJson("/logs/media", logMediaSchema, {
        method: "POST",
        signal: options.signal,
        body: JSON.stringify(request),
      }),
  ).then((pages) => ({
    eventBytes: pages.reduce((total, page) => total + (page.eventBytes ?? 0), 0) || null,
    skippedEventFiles:
      pages.reduce((total, page) => total + (page.skippedEventFiles ?? 0), 0) ||
      null,
    sourceItemCount: pages.reduce(
      (total, page) => total + (page.sourceItemCount ?? 0),
      0,
    ),
    returnedItemCount: pages.reduce(
      (total, page) => total + (page.returnedItemCount ?? 0),
      0,
    ),
    truncated: pages.some((page) => Boolean(page.truncated)),
    truncationReason:
      pages.find((page) => page.truncationReason)?.truncationReason ?? null,
    images: pages.flatMap((page) => page.images),
    texts: pages.flatMap((page) => page.texts),
  }));
}

function chunkList<TItem>(items: TItem[], chunkSize: number) {
  return Array.from(
    { length: Math.ceil(items.length / chunkSize) },
    (_, index) => items.slice(index * chunkSize, (index + 1) * chunkSize),
  );
}

function chunkListForRequestDimension<TItem>(items: TItem[], chunkSize: number) {
  const chunks = chunkList(items, chunkSize);
  return chunks.length > 0 ? chunks : [items];
}

export function fetchLogCheckpoints(
  input: { runIds: string[] },
  options: ApiRequestOptions = {},
) {
  return requestJson("/logs/checkpoints", logCheckpointsSchema, {
    method: "POST",
    signal: options.signal,
    body: JSON.stringify(input),
  });
}

export function fetchLogRunArtifacts(
  runId: string,
  options: ApiRequestOptions = {},
) {
  return requestJson(
    `/logs/runs/${encodeURIComponent(runId)}/artifacts`,
    logRunArtifactsSchema,
    { signal: options.signal },
  );
}
