import { z } from "zod";
import { getSessionAuthToken } from "@/lib/auth-token";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_VIEWER_API_URL ?? "http://127.0.0.1:9999";

export const configValueSchema = z.union([z.string(), z.number(), z.boolean(), z.null()]);

export const presetSchema = z.object({
  name: z.string(),
  label: z.string(),
  description: z.string(),
});

export const datasetSchema = z.object({
  name: z.string(),
  label: z.string(),
  inputDim: z.number(),
  outputDim: z.number(),
});

export const monitorOptionSchema = z.object({
  name: z.string(),
  label: z.string(),
  description: z.string(),
  kinds: z.array(z.string()),
  defaultEnabled: z.boolean(),
});

export const configFieldSchema = z.object({
  key: z.string(),
  configKey: z.string(),
  flag: z.string(),
  label: z.string(),
  section: z.string(),
  type: z.string(),
  default: configValueSchema,
  nullable: z.boolean(),
  choices: z.array(configValueSchema),
  locked: z.boolean().optional(),
  lockedValue: configValueSchema.optional(),
  lockedReason: z.string().optional(),
});

export const searchAxisSchema = z.object({
  key: z.string(),
  configKey: z.string(),
  searchKey: z.string(),
  label: z.string(),
  section: z.string(),
  type: z.string(),
  values: z.array(configValueSchema),
  locked: z.boolean().optional(),
  lockedValue: configValueSchema.optional(),
  lockedReason: z.string().optional(),
});

export const graphConfigSchema = z.object({
  typeName: z.string(),
  fields: z.array(
    z.object({
      key: z.string(),
      value: z.unknown(),
    }),
  ),
});

export const graphNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  typeName: z.string(),
  path: z.string(),
  graphRole: z.enum(["architecture", "internal", "runtime"]),
  parameterCount: z.number(),
  details: z.record(z.unknown()),
  config: graphConfigSchema.nullable(),
});

export const graphEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
});

export const inspectResponseSchema = z.object({
  model: z.string(),
  preset: z.string(),
  parameterCount: z.number(),
  nodes: z.array(graphNodeSchema),
  edges: z.array(graphEdgeSchema),
});

export const trainingRunChangeSchema = z.object({
  key: z.string(),
  label: z.string(),
  value: configValueSchema,
  source: z.enum(["override", "search"]),
});

export const trainingRunSchema = z.object({
  id: z.string(),
  index: z.number(),
  status: z.enum([
    "Pending",
    "Running",
    "Completed",
    "Failed",
    "Cancelled",
    "Skipped",
  ]),
  preset: z.string(),
  snapshotId: z.string().nullable().optional(),
  snapshotName: z.string().nullable().optional(),
  dataset: z.string(),
  changes: z.array(trainingRunChangeSchema),
  overrides: z.record(configValueSchema),
  command: z.string(),
  totalEpochs: z.number(),
  currentEpoch: z.number(),
  metrics: z.record(z.unknown()),
  logDir: z.string().nullable(),
  error: z.string().nullable(),
  errorTraceback: z.string().nullable().optional(),
});

export const trainingRunPlanSummarySchema = z.object({
  totalRuns: z.number(),
  completedRuns: z.number(),
  runningRuns: z.number(),
  pendingRuns: z.number(),
  failedRuns: z.number(),
  cancelledRuns: z.number(),
  skippedRuns: z.number(),
  totalEpochs: z.number(),
  completedEpochs: z.number(),
  remainingEpochs: z.number(),
});

export const trainingRunPlanSchema = z.object({
  model: z.string(),
  preset: z.string(),
  presets: z.array(z.string()),
  datasets: z.array(z.string()),
  overrides: z.record(z.unknown()),
  search: z
    .object({
      mode: z.enum(["grid", "random"]),
      values: z.record(z.array(configValueSchema)),
      randomSamples: z.number().nullable().optional(),
    })
    .nullable(),
  logFolder: z.string(),
  isRandomSearch: z.boolean(),
  runs: z.array(trainingRunSchema),
  summary: trainingRunPlanSummarySchema,
});

export const trainingJobSchema = z.object({
  id: z.string(),
  status: z.string(),
  model: z.string(),
  preset: z.string(),
  presets: z.array(z.string()).optional(),
  datasets: z.array(z.string()),
  overrides: z.record(z.unknown()),
  search: z
    .object({
      mode: z.enum(["grid", "random"]),
      values: z.record(z.array(configValueSchema)),
      randomSamples: z.number().nullable().optional(),
    })
    .nullable()
    .optional(),
  plannedRunCount: z.number().optional(),
  runPlan: trainingRunPlanSchema.nullable().optional(),
  monitors: z.array(z.string()),
  logFolder: z.string(),
  createdAt: z.string(),
  updatedAt: z.string(),
  exitCode: z.number().nullable(),
  pid: z.number(),
  currentPreset: z.string().nullable().optional(),
  currentDataset: z.string().nullable(),
  epoch: z.number().nullable(),
  step: z.number().nullable(),
  metrics: z.record(z.unknown()),
  logDir: z.string().nullable(),
  events: z.array(z.record(z.unknown())),
  logTail: z.array(z.string()),
  resultLinks: z.array(
    z.object({
      preset: z.string().nullable().optional(),
      dataset: z.string().nullable().optional(),
      logDir: z.string().nullable().optional(),
    }),
  ),
});

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

export const logExperimentDeleteSchema = z.object({
  experiment: z.string(),
  deletedRunIds: z.array(z.string()),
  deletedRunCount: z.number(),
  deletedRelativePath: z.string(),
});

export const logRunDeleteCandidateSchema = z.object({
  id: z.string(),
  experiment: z.string(),
  model: z.string(),
  preset: z.string(),
  dataset: z.string(),
  runName: z.string(),
  version: z.string(),
  relativePath: z.string(),
});

export const logRunDeleteAffectedValuesSchema = z.object({
  experiments: z.array(z.string()),
  datasets: z.array(z.string()),
  models: z.array(z.string()),
  presets: z.array(z.string()),
  runIds: z.array(z.string()),
});

export const logRunDeleteCountsSchema = z.object({
  runs: z.number(),
  experiments: z.number(),
  datasets: z.number(),
  models: z.number(),
  presets: z.number(),
});

export const logRunDeleteBlockerSchema = z.object({
  id: z.string(),
  logFolder: z.string(),
  status: z.string(),
});

export const logRunDeletePlanSchema = z.object({
  candidateCount: z.number(),
  counts: logRunDeleteCountsSchema,
  affected: logRunDeleteAffectedValuesSchema,
  candidates: z.array(logRunDeleteCandidateSchema),
  blockedByActiveJobs: z.array(logRunDeleteBlockerSchema),
  canDelete: z.boolean(),
});

export const logRunDeleteSchema = logRunDeletePlanSchema.extend({
  deletedRunIds: z.array(z.string()),
  deletedRunCount: z.number(),
  deletedRelativePaths: z.array(z.string()),
});

export const logExperimentSchema = z.object({
  experiment: z.string(),
  runCount: z.number(),
  relativePath: z.string(),
});

const errorBodySchema = z.object({ detail: z.unknown() }).partial();

const healthSchema = z.object({ status: z.string() });
const dataSourceCapabilityPlaceholderSchema = z.object({}).strict();
const capabilitiesSchema = z.object({
  authMode: z.enum(["none", "bearer"]),
  trainingEnabled: z.boolean(),
  logDeletionEnabled: z.boolean(),
  historicalLogsEnabled: z.boolean(),
  liveMonitorDataEnabled: z.boolean(),
  historicalMonitorDataEnabled: z.boolean(),
  uploadsEnabled: z.boolean().default(false),
  maxUploadSize: z.number().int().nonnegative().nullable().default(null),
  dataSourcesEnabled: z.boolean().default(false),
  dataSources: z.array(dataSourceCapabilityPlaceholderSchema).default([]),
});
const modelsSchema = z.object({ models: z.array(z.string()) });
const presetsSchema = z.object({ model: z.string(), presets: z.array(presetSchema) });
const datasetsSchema = z.object({ model: z.string(), datasets: z.array(datasetSchema) });
const monitorsSchema = z.object({
  model: z.string(),
  monitors: z.array(monitorOptionSchema),
});
const configSchema = z.object({ model: z.string(), fields: z.array(configFieldSchema) });
const searchSpaceSchema = z.object({
  model: z.string(),
  preset: z.string().nullable().optional(),
  axes: z.array(searchAxisSchema),
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

export type ConfigValue = z.infer<typeof configValueSchema>;
export type Preset = z.infer<typeof presetSchema>;
export type Dataset = z.infer<typeof datasetSchema>;
export type MonitorOption = z.infer<typeof monitorOptionSchema>;
export type ConfigField = z.infer<typeof configFieldSchema>;
export type SearchAxis = z.infer<typeof searchAxisSchema>;
export type SearchSpace = z.infer<typeof searchSpaceSchema>;
export type Capabilities = z.infer<typeof capabilitiesSchema>;
export type GraphConfig = z.infer<typeof graphConfigSchema>;
export type GraphNode = z.infer<typeof graphNodeSchema>;
export type GraphEdge = z.infer<typeof graphEdgeSchema>;
export type InspectResponse = z.infer<typeof inspectResponseSchema>;
export type TrainingRunChange = z.infer<typeof trainingRunChangeSchema>;
export type TrainingRun = z.infer<typeof trainingRunSchema>;
export type TrainingRunPlanSummary = z.infer<typeof trainingRunPlanSummarySchema>;
export type TrainingRunPlan = z.infer<typeof trainingRunPlanSchema>;
export type TrainingJob = z.infer<typeof trainingJobSchema>;
export type MonitorData = z.infer<typeof monitorDataSchema>;
export type LogRun = z.infer<typeof logRunSchema>;
export type LogExperiment = z.infer<typeof logExperimentSchema>;
export type LogRunTags = z.infer<typeof logRunTagsSchema>;
export type LogScalarPoint = z.infer<typeof logScalarPointSchema>;
export type LogScalarSeries = z.infer<typeof logScalarSeriesSchema>;
export type LogExperimentDeleteResponse = z.infer<typeof logExperimentDeleteSchema>;
export type LogRunDeleteCandidate = z.infer<typeof logRunDeleteCandidateSchema>;
export type LogRunDeletePlan = z.infer<typeof logRunDeletePlanSchema>;
export type LogRunDeleteResponse = z.infer<typeof logRunDeleteSchema>;

export type LogRunDeleteFilters = {
  experiments: string[];
  datasets: string[];
  models: string[];
  presets: string[];
  runIds: string[];
};

export type TrainingSearchCreateInput = {
  mode: "grid" | "random";
  values: Record<string, ConfigValue[]>;
  randomSamples?: number;
};

export type TrainingSearchSubmitInput = {
  mode: "grid" | "random";
  values: Record<string, ConfigValue[]>;
  randomSamples?: number | null;
};

export type TrainingRunSubmitChangeInput = {
  key: string;
  label: string;
  value: ConfigValue;
  source: "override" | "search";
};

export type TrainingRunSubmitInput = {
  id: string;
  index: number;
  status: TrainingRun["status"];
  preset: string;
  snapshotId?: string | null;
  snapshotName?: string | null;
  dataset: string;
  changes: TrainingRunSubmitChangeInput[];
  overrides: Record<string, ConfigValue>;
  command: string;
  totalEpochs: number;
  currentEpoch: number;
  metrics: Record<string, unknown>;
  logDir: string | null;
  error: string | null;
  errorTraceback?: string | null;
};

export type TrainingRunPlanSubmitSummaryInput = {
  totalRuns: number;
  completedRuns: number;
  runningRuns: number;
  pendingRuns: number;
  failedRuns: number;
  cancelledRuns: number;
  skippedRuns: number;
  totalEpochs: number;
  completedEpochs: number;
  remainingEpochs: number;
};

export type TrainingRunPlanSubmitInput = {
  model: string;
  preset: string;
  presets: string[];
  datasets: string[];
  overrides: Record<string, unknown>;
  search: TrainingSearchSubmitInput | null;
  logFolder: string;
  isRandomSearch: boolean;
  runs: TrainingRunSubmitInput[];
  summary: TrainingRunPlanSubmitSummaryInput;
};

export type TrainingJobCreateInput = {
  model: string;
  preset: string;
  presets?: string[];
  datasets: string[];
  overrides: Record<string, string>;
  logFolder: string;
  monitors: string[];
  search?: TrainingSearchCreateInput;
  runPlan?: TrainingRunPlanSubmitInput;
};

export type TrainingRunPlanCreateInput = {
  model: string;
  preset: string;
  presets?: string[];
  datasets: string[];
  overrides: Record<string, string>;
  logFolder?: string;
  search?: TrainingSearchCreateInput;
};

type ApiErrorInit = {
  status: number;
  method: string;
  path: string;
  detail: string;
};

export type UnauthorizedApiError = Error & {
  status: 401;
  method: string;
  path: string;
  detail: string;
};

class ApiError extends Error {
  readonly status: number;
  readonly method: string;
  readonly path: string;
  readonly detail: string;

  constructor({ status, method, path, detail }: ApiErrorInit) {
    const messageDetail = detail || "Request failed";
    super(`${method} ${path} from ${API_BASE_URL} failed with ${status}: ${messageDetail}`);
    this.name = "ApiError";
    this.status = status;
    this.method = method;
    this.path = path;
    this.detail = messageDetail;
  }
}

export function isUnauthorizedApiError(error: unknown): error is UnauthorizedApiError {
  return error instanceof ApiError && error.status === 401;
}

function requestMethod(init?: RequestInit) {
  return String(init?.method ?? "GET").toUpperCase();
}

function formatIssuePath(path: Array<string | number>) {
  return path.length > 0 ? path.map(String).join(".") : "<root>";
}

function formatZodIssues(issues: z.ZodIssue[]) {
  const visibleIssues = issues
    .slice(0, 5)
    .map((issue) => `${formatIssuePath(issue.path)}: ${issue.message}`);
  if (issues.length > visibleIssues.length) {
    visibleIssues.push(`${issues.length - visibleIssues.length} more issue(s)`);
  }
  return visibleIssues.join("; ");
}

function detailText(detail: unknown) {
  if (typeof detail === "string") {
    return detail;
  }
  if (detail === null || detail === undefined) {
    return "";
  }
  try {
    return JSON.stringify(detail);
  } catch {
    return String(detail);
  }
}

function requestHeaders(initHeaders?: HeadersInit) {
  const headers = new Headers({ "content-type": "application/json" });
  if (initHeaders) {
    new Headers(initHeaders).forEach((value, key) => {
      headers.set(key, value);
    });
  }
  const token = getSessionAuthToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  return headers;
}

async function requestJson<TSchema extends z.ZodTypeAny>(
  path: string,
  schema: TSchema,
  init?: RequestInit,
): Promise<z.output<TSchema>> {
  const method = requestMethod(init);
  const request = {
    ...init,
    headers: requestHeaders(init?.headers),
  };
  const response = await fetch(`${API_BASE_URL}${path}`, request);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = errorBodySchema.safeParse(await response.json());
      if (payload.success && payload.data.detail) {
        detail = detailText(payload.data.detail);
      }
    } catch {
      // Response was not JSON; keep status text.
    }
    throw new ApiError({
      status: response.status,
      method,
      path,
      detail,
    });
  }
  const payload = await response.json();
  const parsed = schema.safeParse(payload);
  if (!parsed.success) {
    throw new Error(
      `Invalid API response for ${method} ${path} from ${API_BASE_URL}: ${formatZodIssues(
        parsed.error.issues,
      )}`,
    );
  }
  return parsed.data;
}

export function fetchHealth() {
  return requestJson("/health", healthSchema);
}

export function fetchCapabilities() {
  return requestJson("/capabilities", capabilitiesSchema);
}

export function fetchModels() {
  return requestJson("/models", modelsSchema);
}

export function fetchPresets(model: string) {
  return requestJson(`/models/${model}/presets`, presetsSchema);
}

export function fetchDatasets(model: string) {
  return requestJson(`/models/${model}/datasets`, datasetsSchema);
}

export function fetchMonitors(model: string) {
  return requestJson(`/models/${model}/monitors`, monitorsSchema);
}

export function fetchConfigSchema(model: string, preset?: string) {
  const query = preset ? `?preset=${encodeURIComponent(preset)}` : "";
  return requestJson(`/models/${model}/config-schema${query}`, configSchema);
}

export function fetchSearchSpace(model: string, preset?: string) {
  const query = preset ? `?preset=${encodeURIComponent(preset)}` : "";
  return requestJson(`/models/${model}/search-space${query}`, searchSpaceSchema);
}

export function inspectModel(input: {
  model: string;
  preset: string;
  overrides: Record<string, string>;
  dataset?: string;
}) {
  return requestJson("/inspect", inspectResponseSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function createTrainingJob(input: TrainingJobCreateInput) {
  return requestJson("/training/jobs", trainingJobSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function fetchTrainingRunPlan(input: TrainingRunPlanCreateInput) {
  return requestJson("/training/run-plan", trainingRunPlanSchema, {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function fetchTrainingJob(id: string) {
  return requestJson(`/training/jobs/${id}`, trainingJobSchema);
}

export function cancelTrainingJob(id: string) {
  return requestJson(`/training/jobs/${id}/cancel`, trainingJobSchema, {
    method: "POST",
  });
}

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

export function deleteLogExperiment(experiment: string) {
  return requestJson(
    `/logs/experiments/${encodeURIComponent(experiment)}`,
    logExperimentDeleteSchema,
    {
      method: "DELETE",
    },
  );
}

export function createLogRunDeletePlan(filters: LogRunDeleteFilters) {
  return requestJson("/logs/runs/delete-plan", logRunDeletePlanSchema, {
    method: "POST",
    body: JSON.stringify(filters),
  });
}

export function deleteLogRuns(filters: LogRunDeleteFilters) {
  return requestJson("/logs/runs/delete", logRunDeleteSchema, {
    method: "POST",
    body: JSON.stringify(filters),
  });
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
