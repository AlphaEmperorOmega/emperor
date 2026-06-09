import { z } from "zod";

import { requestJson } from "@/lib/api/client";

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
  metrics: z.record(z.unknown()),
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

const logTagsSchema = z.object({ runs: z.array(logRunTagsSchema) });
const logScalarsSchema = z.object({ series: z.array(logScalarSeriesSchema) });

export type LogRun = z.infer<typeof logRunSchema>;
export type LogExperiment = z.infer<typeof logExperimentSchema>;
export type LogRunTags = z.infer<typeof logRunTagsSchema>;
export type LogScalarPoint = z.infer<typeof logScalarPointSchema>;
export type LogScalarSeries = z.infer<typeof logScalarSeriesSchema>;

const DEFAULT_LOG_PAGE_LIMIT = 500;

type PaginatedPage = {
  total?: number;
  limit?: number;
  offset?: number;
  hasMore?: boolean;
};

type PaginatedParams = Record<string, string | number | boolean | undefined>;

type FetchPaginatedOptions<TPage extends PaginatedPage, TItem> = {
  endpoint: string;
  schema: z.ZodType<TPage, z.ZodTypeDef, unknown>;
  params?: PaginatedParams;
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
    if (value !== undefined) {
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
  getItems,
  getTotal,
}: FetchPaginatedOptions<TPage, TItem>): Promise<
  FetchPaginatedResult<TPage, TItem>
> {
  const firstPage = await requestJson(paginatedPath(endpoint, params), schema);
  const items = [...getItems(firstPage)];
  let limit =
    firstPage.limit && firstPage.limit > 0
      ? firstPage.limit
      : items.length || DEFAULT_LOG_PAGE_LIMIT;
  let offset = firstPage.offset ?? 0;
  let hasMore = firstPage.hasMore ?? false;

  while (hasMore) {
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

export async function fetchLogRuns() {
  const page = await fetchPaginated({
    endpoint: "/logs/runs",
    schema: logRunsSchema,
    getItems: (logPage) => logPage.runs,
    getTotal: (logPage) => logPage.total,
  });

  return {
    ...page.firstPage,
    runs: page.items,
    total: page.total,
    limit: page.limit,
    offset: page.offset,
    hasMore: false,
  };
}

export async function fetchLogExperiments() {
  const page = await fetchPaginated({
    endpoint: "/logs/experiments",
    schema: logExperimentsSchema,
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

export function fetchLogScalars(input: { runIds: string[]; tags: string[] }) {
  return requestJson("/logs/scalars", logScalarsSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}
