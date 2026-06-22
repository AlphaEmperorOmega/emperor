import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cancelTrainingJob,
  createConfigSnapshot,
  createLogRunDeletePlan,
  createTrainingJob,
  configOverridesSchema,
  deleteConfigSnapshot,
  deleteLogExperiment,
  deleteLogRuns,
  fetchCapabilities,
  fetchConfigSchema,
  fetchConfigSnapshotLibrary,
  fetchConfigSnapshots,
  fetchDatasets,
  fetchHealth,
  fetchLogExperiments,
  fetchLogCheckpoints,
  fetchLogParameterStatus,
  fetchLogRunArtifacts,
  fetchLogRunMonitorData,
  fetchLogRuns,
  fetchLogMedia,
  fetchLogScalars,
  fetchLogTags,
  fetchModels,
  fetchMonitorData,
  fetchMonitorParameterStatus,
  fetchMonitors,
  fetchPresets,
  fetchSearchSpace,
  fetchTrainingJob,
  fetchTrainingJobEvents,
  fetchTrainingRunPlan,
  getViewerApiBaseUrl,
  importLogArchive,
  inspectOperationGraph,
  inspectModel,
  isUnauthorizedApiError,
  jsonObjectSchema,
  jsonValueSchema,
  logImageSummarySchema,
  logArchiveImportSchema,
  monitorDataSchema,
  normalizeViewerApiBaseUrl,
  operationGraphResponseSchema,
  renameConfigSnapshot,
  resetViewerApiBaseUrl,
  setViewerApiBaseUrl,
  trainingJobSchema,
  trainingJobEventsSchema,
  trainingProgressEventSchema,
  trainingRunPlanSchema,
  updateConfigSnapshot,
  VIEWER_API_BASE_URL_STORAGE_KEY,
} from "@/lib/api";
import { mapWithConcurrency } from "@/lib/api/concurrency";
import { setSessionAuthToken } from "@/lib/auth-token";

// Characterization tests: assert the CURRENT behavior of the API client
// (URL/verb/body construction and requestJson error handling) so later
// refactors can be verified green-to-green. Not aspirational.

const BASE = "http://127.0.0.1:9999";
const linearIdentity = { modelType: "linears", model: "linear" } as const;

const capabilitiesResponse = {
  authMode: "none",
  trainingEnabled: true,
  trainingCancellationCapability: "strict-cgroup",
  logDeletionEnabled: true,
  configSnapshotsEnabled: true,
  historicalLogsEnabled: true,
  liveMonitorDataEnabled: true,
  historicalMonitorDataEnabled: true,
  uploadsEnabled: false,
  maxUploadSize: null,
  dataSourcesEnabled: false,
  dataSources: [],
};

type FakeResponseInit = {
  ok?: boolean;
  status?: number;
  statusText?: string;
  json?: () => Promise<unknown>;
};

function fakeResponse(init: FakeResponseInit) {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: init.statusText ?? "OK",
    json: init.json ?? (() => Promise.resolve({})),
  } as unknown as Response;
}

type FetchFn = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

function stubFetch(response: Response) {
  const fetchMock = vi.fn<FetchFn>(() => Promise.resolve(response));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}

async function flushAsyncWork() {
  for (let index = 0; index < 5; index += 1) {
    await Promise.resolve();
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  resetViewerApiBaseUrl();
  window.localStorage.clear();
  window.sessionStorage.clear();
});

describe("shared API value schemas", () => {
  it("accepts nested JSON values and rejects non-JSON values", () => {
    expect(
      jsonValueSchema.parse({
        nested: [1, "two", true, null, { score: 0.75 }],
      }),
    ).toEqual({
      nested: [1, "two", true, null, { score: 0.75 }],
    });
    expect(jsonObjectSchema.parse({ trainLoss: 0.42 })).toEqual({
      trainLoss: 0.42,
    });
    expect(() => jsonValueSchema.parse(Symbol("not-json"))).toThrow();
    expect(() => jsonObjectSchema.parse({ missing: undefined })).toThrow();
  });

  it("accepts only primitive config override values", () => {
    expect(
      configOverridesSchema.parse({
        hidden_dim: 128,
        learning_rate: 0.01,
        use_bias: true,
        activation: "RELU",
        optional_layer: null,
      }),
    ).toEqual({
      hidden_dim: 128,
      learning_rate: 0.01,
      use_bias: true,
      activation: "RELU",
      optional_layer: null,
    });
    expect(() =>
      configOverridesSchema.parse({ scheduler: { name: "cosine" } }),
    ).toThrow();
  });

  it("types known training progress events while allowing future event payloads", () => {
    expect(
      trainingProgressEventSchema.parse({
        type: "validation",
        status: "running",
        jobId: "job-1",
        dataset: "Mnist",
        preset: "baseline",
        runId: "run-1",
        runIndex: 1,
        epoch: 0,
        step: 7,
        metrics: { "validation/accuracy": 0.75 },
      }),
    ).toMatchObject({
      type: "validation",
      metrics: { "validation/accuracy": 0.75 },
    });
    expect(
      trainingProgressEventSchema.parse({
        type: "cluster_initialized",
        node: "main.cluster",
        count: 2,
        capacity: [2, 2, 2],
        coordinates: [
          [0, 0, 0],
          [0, 0, 1],
        ],
      }),
    ).toMatchObject({ type: "cluster_initialized", capacity: [2, 2, 2] });
    expect(
      trainingProgressEventSchema.parse({
        type: "future_event",
        customField: { still: "allowed" },
      }),
    ).toEqual({
      type: "future_event",
      customField: { still: "allowed" },
    });
    expect(() =>
      trainingProgressEventSchema.parse({ status: "running" }),
    ).toThrow();
    expect(() =>
      trainingProgressEventSchema.parse({
        type: "validation",
        status: "completed",
      }),
    ).toThrow();
    expect(() =>
      trainingProgressEventSchema.parse({
        type: "cluster_initialized",
        node: "main.cluster",
        count: 2,
        coordinates: [[0, 0, 0]],
      }),
    ).toThrow();
    expect(() =>
      trainingProgressEventSchema.parse({
        type: "neurons_added",
        node: "main.cluster",
        coordinates: [[0, 0, 0]],
        coordinateCount: 1,
        count: 3,
      }),
    ).toThrow();
  });
});

describe("API request scheduling", () => {
  it("preserves output order with bounded concurrency", async () => {
    const result = await mapWithConcurrency([3, 1, 2], 2, async (value) => {
      await new Promise((resolve) => setTimeout(resolve, value));
      return value * 10;
    });

    expect(result).toEqual([30, 10, 20]);
  });

  it("never exceeds the configured concurrency", async () => {
    let active = 0;
    let maxActive = 0;

    await mapWithConcurrency([1, 2, 3, 4, 5], 2, async (value) => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      await new Promise((resolve) => setTimeout(resolve, 1));
      active -= 1;
      return value;
    });

    expect(maxActive).toBe(2);
  });

  it("stops launching new work and propagates request failures", async () => {
    const error = new Error("request failed");
    const calls: number[] = [];

    await expect(
      mapWithConcurrency([1, 2, 3, 4], 2, async (value) => {
        calls.push(value);
        if (value === 1) {
          throw error;
        }
        return new Promise<number>(() => {});
      }),
    ).rejects.toBe(error);
    expect(calls).toEqual([1, 2]);
  });
});

async function validateSuccessfulFixture<T>(
  payload: unknown,
  callApi: () => Promise<T>,
) {
  const fetchMock = stubFetch(
    fakeResponse({ json: () => Promise.resolve(payload) }),
  );

  const result = await callApi();

  expect(fetchMock).toHaveBeenCalledTimes(1);
  return result;
}

const successfulPresetResponse = {
  modelType: "linears",
  model: "linear",
  presets: [
    {
      name: "baseline",
      label: "Baseline",
      description: "Small linear baseline",
    },
    {
      name: "wide",
      label: "Wide",
      description: "Wider hidden layer preset",
    },
  ],
};

const successfulDatasetResponse = {
  modelType: "linears",
  model: "linear",
  datasets: [
    {
      name: "Mnist",
      label: "MNIST",
      inputDim: 784,
      outputDim: 10,
    },
  ],
};

const successfulMonitorResponse = {
  modelType: "linears",
  model: "linear",
  monitors: [
    {
      name: "weights",
      label: "Weights",
      description: "Layer weight distributions",
      kinds: ["scalar", "histogram"],
      defaultEnabled: true,
    },
    {
      name: "activations",
      label: "Activations",
      description: "Activation image summaries",
      kinds: ["image"],
      defaultEnabled: false,
    },
  ],
};

const successfulConfigSchemaResponse = {
  modelType: "linears",
  model: "linear",
  fields: [
    {
      key: "learning_rate",
      configKey: "LEARNING_RATE",
      flag: "--learning-rate",
      label: "Learning rate",
      section: "Optimisation",
      type: "float",
      default: 0.01,
      nullable: false,
      choices: [0.001, 0.01],
      locked: false,
      lockedValue: null,
      lockedReason: "",
    },
    {
      key: "checkpoint",
      configKey: "CHECKPOINT",
      flag: "--checkpoint",
      label: "Checkpoint",
      section: "Runtime",
      type: "str",
      default: null,
      nullable: true,
      choices: ["none", null],
      locked: true,
      lockedValue: "none",
      lockedReason: "Hosted fixture keeps checkpoint loading disabled",
    },
  ],
};

const successfulSearchSpaceResponse = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  axes: [
    {
      key: "hidden_dim",
      configKey: "HIDDEN_DIM",
      searchKey: "SEARCH_SPACE_HIDDEN_DIM",
      label: "Hidden dimension",
      section: "Layer Stack Options",
      type: "int",
      values: [64, 128],
      locked: false,
      lockedValue: null,
      lockedReason: "",
    },
    {
      key: "activation",
      configKey: "ACTIVATION",
      searchKey: "SEARCH_SPACE_ACTIVATION",
      label: "Activation",
      section: "Layer Stack Options",
      type: "str",
      values: ["relu", "gelu"],
      locked: true,
      lockedValue: "relu",
      lockedReason: "Preset fixes the activation",
    },
  ],
};

const successfulInspectResponse = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  parameterCount: 7850,
  parameterSizeBytes: 31400,
  nodes: [
    {
      id: "input",
      label: "Input",
      typeName: "InputNode",
      path: "root.input",
      graphRole: "architecture",
      parameterCount: 0,
      parameterSizeBytes: 0,
      details: {
        shape: [784],
      },
      config: null,
    },
    {
      id: "classifier",
      label: "Classifier",
      typeName: "Linear",
      path: "root.classifier",
      graphRole: "runtime",
      parameterCount: 7850,
      parameterSizeBytes: 31400,
      details: {
        weightShape: [10, 784],
        biasShape: [10],
      },
      config: {
        typeName: "LinearConfig",
        fields: [
          { key: "input_dim", value: 784 },
          { key: "output_dim", value: 10 },
        ],
      },
    },
  ],
  edges: [{ id: "input-classifier", source: "input", target: "classifier" }],
};

const successfulOperationGraphResponse = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  source: "torch-export",
  status: "ok",
  nodes: [
    {
      id: "op_0000",
      label: "input x",
      opKind: "placeholder",
      target: "x",
      modulePath: null,
      groupId: "__inputs__",
      details: {
        inputKind: "user_input",
        shape: [1, 1, 28, 28],
        dtype: "float32",
      },
    },
    {
      id: "op_0001",
      label: "linear",
      opKind: "call_function",
      target: "aten.linear.default",
      modulePath: "classifier",
      groupId: "classifier",
      details: {
        shape: [1, 10],
        dtype: "float32",
      },
    },
  ],
  edges: [{ id: "op_0000-op_0001", source: "op_0000", target: "op_0001" }],
  warnings: [],
};

const unsupportedOperationGraphResponse = {
  ...successfulOperationGraphResponse,
  status: "unsupported",
  nodes: [],
  edges: [],
  warnings: ["torch.export.export failed: unsupported fixture"],
};

const successfulTrainingRunFixture = {
  id: "run-0001",
  index: 0,
  status: "Running",
  preset: "baseline",
  snapshotId: "snapshot-1",
  snapshotName: "Warm start",
  dataset: "Mnist",
  changes: [
    {
      key: "learning_rate",
      label: "Learning rate",
      value: 0.01,
      source: "override",
    },
    {
      key: "hidden_dim",
      label: "Hidden dimension",
      value: 128,
      source: "search",
    },
  ],
  overrides: {
    learning_rate: 0.01,
    use_bias: true,
  },
  command:
    "python experiment.py linear --preset baseline --dataset Mnist --learning-rate 0.01",
  totalEpochs: 10,
  currentEpoch: 3,
  metrics: {
    trainLoss: 0.42,
    validation: {
      accuracy: 0.91,
    },
  },
  logDir: "runs/viewer-training/job-123/linear/baseline/Mnist/run-0001",
  error: null,
  errorTraceback: null,
};

const successfulTrainingRunPlanFixture = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  presets: ["baseline", "wide"],
  datasets: ["Mnist"],
  overrides: {
    learning_rate: 0.01,
    use_bias: true,
  },
  search: {
    mode: "grid",
    values: {
      hidden_dim: [64, 128],
      use_bias: [true, false],
      activation: ["relu"],
    },
    randomSamples: null,
  },
  logFolder: "viewer-training/job-123",
  isRandomSearch: false,
  runs: [successfulTrainingRunFixture],
  summary: {
    totalRuns: 1,
    completedRuns: 0,
    runningRuns: 1,
    pendingRuns: 0,
    failedRuns: 0,
    cancelledRuns: 0,
    skippedRuns: 0,
    totalEpochs: 10,
    completedEpochs: 3,
    remainingEpochs: 7,
  },
};

const successfulTrainingJobFixture = {
  id: "job-123",
  status: "running",
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  presets: ["baseline", "wide"],
  datasets: ["Mnist"],
  overrides: {
    learning_rate: "0.01",
    use_bias: "true",
  },
  search: {
    mode: "grid",
    values: {
      hidden_dim: [64, 128],
      use_bias: [true, false],
    },
    randomSamples: null,
  },
  plannedRunCount: 1,
  runPlan: successfulTrainingRunPlanFixture,
  monitors: ["weights", "activations"],
  logFolder: "viewer-training/job-123",
  createdAt: "2026-06-01T00:00:00Z",
  updatedAt: "2026-06-01T00:03:00Z",
  exitCode: null,
  pid: 12345,
  currentPreset: "baseline",
  currentDataset: "Mnist",
  epoch: 3,
  step: 30,
  metrics: {
    trainLoss: 0.42,
    validation: {
      accuracy: 0.91,
    },
  },
  logDir: "runs/viewer-training/job-123/linear/baseline/Mnist/run-0001",
  events: [
    {
      type: "epoch",
      epoch: 3,
      message: "trainLoss=0.42",
    },
  ],
  eventCount: 1,
  eventCounts: {
    epoch: 1,
  },
  eventsTruncated: false,
  clusterGrowth: [
    {
      node: "root.cluster",
      count: 2,
      capacityTotal: 8,
      additionCount: 1,
      additions: [{ coord: [1, 0, 0], step: 30, epoch: 3 }],
    },
  ],
  logTail: ["epoch 3 trainLoss=0.42"],
  resultLinks: [
    {
      preset: "baseline",
      dataset: "Mnist",
      logDir: "runs/viewer-training/job-123/linear/baseline/Mnist/run-0001",
    },
  ],
};

const successfulMonitorDataFixture = {
  jobId: "job-123",
  nodePath: "root.classifier",
  preset: "baseline",
  dataset: "Mnist",
  logDir: "runs/viewer-training/job-123/linear/baseline/Mnist/run-0001",
  scalarSeries: [
    {
      tag: "train/loss",
      label: "Train loss",
      points: [
        { step: 1, wallTime: 1770000000, value: 0.72 },
        { step: 2, wallTime: 1770000030, value: 0.55 },
      ],
    },
  ],
  histograms: [
    {
      tag: "weights/classifier",
      step: 2,
      wallTime: 1770000030,
      buckets: [
        { left: -1, right: 0, count: 12 },
        { left: 0, right: 1, count: 18 },
      ],
    },
  ],
  images: [
    {
      tag: "activations/classifier",
      step: 2,
      wallTime: 1770000030,
      mimeType: "image/png",
      dataUrl: "data:image/png;base64,AAAA",
    },
  ],
};

const successfulHistoricalMonitorDataFixture = {
  ...successfulMonitorDataFixture,
  jobId: "run-1",
};

const successfulParameterStatusFixture = {
  sourceId: "job-123",
  preset: "baseline",
  dataset: "Mnist",
  logDir: "runs/viewer-training/job-123/linear/baseline/Mnist/run-0001",
  nodes: [
    {
      nodePath: "root.classifier",
      weights: {
        status: "updated",
        metric: "root.classifier/weights/relative_delta_norm",
        lastStep: 2,
        observedPoints: 1,
      },
      bias: {
        status: "missing",
        metric: null,
        lastStep: null,
        observedPoints: 0,
      },
    },
  ],
};

const successfulLogParameterStatusFixture = {
  runs: [
    {
      ...successfulParameterStatusFixture,
      sourceId: "run-1",
    },
  ],
};

const successfulLogRunFixture = {
  id: "run-1",
  group: "viewer-training/job-123",
  experiment: "viewer-training",
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  dataset: "Mnist",
  runName: "run-0001",
  timestamp: "2026-06-01 01:02:03",
  version: "version_0",
  relativePath: "viewer-training/linear/baseline/Mnist/run-0001/version_0",
  hasResult: true,
  eventFileCount: 2,
  checkpointCount: 1,
  hasHparams: true,
  metrics: {
    trainLoss: 0.42,
    validationAccuracy: 0.91,
  },
};

const successfulLogRunsResponse = {
  total: 1,
  limit: 500,
  offset: 0,
  hasMore: false,
  runs: [successfulLogRunFixture],
};

const successfulLogExperimentsResponse = {
  total: 1,
  limit: 500,
  offset: 0,
  hasMore: false,
  experiments: [
    {
      experiment: "viewer-training",
      runCount: 1,
      relativePath: "viewer-training",
    },
  ],
};

const successfulLogDeletePlanFixture = {
  candidateCount: 1,
  counts: {
    runs: 1,
    experiments: 1,
    datasets: 1,
    models: 1,
    presets: 1,
  },
  affected: {
    experiments: ["viewer-training"],
    datasets: ["Mnist"],
    models: [linearIdentity],
    presets: ["baseline"],
    runIds: ["run-1"],
  },
  candidates: [
    {
      id: "run-1",
      experiment: "viewer-training",
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      dataset: "Mnist",
      runName: "run-0001",
      version: "version_0",
      relativePath: "viewer-training/linear/baseline/Mnist/run-0001/version_0",
    },
  ],
  blockedByActiveJobs: [],
  canDelete: true,
};

const successfulLogDeleteResponse = {
  ...successfulLogDeletePlanFixture,
  deletedRunIds: ["run-1"],
  deletedRunCount: 1,
  deletedRelativePaths: [
    "viewer-training/linear/baseline/Mnist/run-0001/version_0",
  ],
};

const logDeleteFilters = {
  experiments: ["viewer-training"],
  datasets: ["Mnist"],
  models: [linearIdentity],
  presets: ["baseline"],
  runIds: ["run-1"],
};

const successfulLogTagsResponse = {
  runs: [
    {
      runId: "run-1",
      scalarTags: ["train/loss", "validation/accuracy"],
      histogramTags: ["weights/classifier"],
      imageTags: ["activations/classifier"],
      textTags: ["validation/examples/predictions/text_summary"],
    },
  ],
};

const successfulLogScalarsResponse = {
  series: [
    {
      runId: "run-1",
      tag: "train/loss",
      points: [
        { step: 1, wallTime: 1770000000, value: 0.72 },
        { step: 2, wallTime: 1770000030, value: 0.55 },
      ],
    },
  ],
};

const successfulLogArchiveImportResponse = {
  extractedFileCount: 2,
  skippedFileCount: 1,
  destinationRoot: "/workspace/logs",
};

describe("successful API fixtures", () => {
  it("accepts a health response fixture", async () => {
    const result = await validateSuccessfulFixture({ status: "ok" }, fetchHealth);

    expect(result.status).toBe("ok");
  });

  it("accepts a capabilities response fixture", async () => {
    const result = await validateSuccessfulFixture(
      capabilitiesResponse,
      fetchCapabilities,
    );

    expect(result).toEqual(capabilitiesResponse);
  });

  it("accepts a log archive import response fixture", () => {
    expect(
      logArchiveImportSchema.parse(successfulLogArchiveImportResponse),
    ).toEqual(successfulLogArchiveImportResponse);
  });

  it("accepts a models response fixture", async () => {
    const result = await validateSuccessfulFixture(
      {
        models: [
          linearIdentity,
          { modelType: "transformer_encoder", model: "bert_linear" },
        ],
      },
      fetchModels,
    );

    expect(result.models).toEqual([
      linearIdentity,
      { modelType: "transformer_encoder", model: "bert_linear" },
    ]);
  });

  it("accepts a presets response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulPresetResponse, () =>
      fetchPresets(linearIdentity),
    );

    expect(result.presets.map((preset) => preset.name)).toEqual([
      "baseline",
      "wide",
    ]);
  });

  it("accepts a datasets response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulDatasetResponse, () =>
      fetchDatasets(linearIdentity),
    );

    expect(result.datasets[0]).toMatchObject({
      name: "Mnist",
      inputDim: 784,
      outputDim: 10,
    });
  });

  it("accepts a monitors response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulMonitorResponse, () =>
      fetchMonitors(linearIdentity),
    );

    expect(result.monitors[0].kinds).toEqual(["scalar", "histogram"]);
  });

  it("accepts a config schema response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulConfigSchemaResponse, () =>
      fetchConfigSchema(linearIdentity, "baseline"),
    );

    expect(result.fields.map((field) => field.key)).toEqual([
      "learning_rate",
      "checkpoint",
    ]);
  });

  it("accepts a search-space response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulSearchSpaceResponse, () =>
      fetchSearchSpace(linearIdentity, "baseline"),
    );

    expect(result.axes[0].values).toEqual([64, 128]);
  });

  it("accepts an inspect response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulInspectResponse, () =>
      inspectModel({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        overrides: {},
        dataset: "Mnist",
      }),
    );

    expect(result.nodes[1].config?.fields).toHaveLength(2);
  });

  it("accepts operation graph success and unsupported fixtures", async () => {
    const result = await validateSuccessfulFixture(
      successfulOperationGraphResponse,
      () =>
        inspectOperationGraph({
          model: "linear",
          modelType: "linears",
          preset: "baseline",
          overrides: {},
          dataset: "Mnist",
        }),
    );

    expect(result.status).toBe("ok");
    expect(result.nodes[1].target).toBe("aten.linear.default");
    expect(operationGraphResponseSchema.parse(unsupportedOperationGraphResponse))
      .toMatchObject({
        status: "unsupported",
        warnings: ["torch.export.export failed: unsupported fixture"],
      });
  });

  it("rejects extra operation graph contract fields", () => {
    const node = successfulOperationGraphResponse.nodes[0];
    const edge = successfulOperationGraphResponse.edges[0];

    expect(() =>
      operationGraphResponseSchema.parse({
        ...successfulOperationGraphResponse,
        extra: true,
      }),
    ).toThrow();
    expect(() =>
      operationGraphResponseSchema.parse({
        ...successfulOperationGraphResponse,
        nodes: [{ ...node, extra: true }],
      }),
    ).toThrow();
    expect(() =>
      operationGraphResponseSchema.parse({
        ...successfulOperationGraphResponse,
        edges: [{ ...edge, extra: true }],
      }),
    ).toThrow();
  });

  it("accepts a training job creation response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulTrainingJobFixture, () =>
      createTrainingJob({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        presets: ["baseline", "wide"],
        datasets: ["Mnist"],
        overrides: {
          learning_rate: "0.01",
        },
        logFolder: "viewer-training/job-123",
        monitors: ["weights", "activations"],
      }),
    );

    expect(result.runPlan?.runs[0].changes).toHaveLength(2);
  });

  it("accepts a training run plan response fixture", async () => {
    const result = await validateSuccessfulFixture(
      successfulTrainingRunPlanFixture,
      () =>
        fetchTrainingRunPlan({
          modelType: "linears",
          model: "linear",
          preset: "baseline",
          presets: ["baseline", "wide"],
          datasets: ["Mnist"],
          overrides: {
            learning_rate: "0.01",
          },
          logFolder: "viewer-training/job-123",
        }),
    );

    expect(result.summary.remainingEpochs).toBe(7);
  });

  it("rejects nested training run-plan response override objects", () => {
    expect(() =>
      trainingRunPlanSchema.parse({
        ...successfulTrainingRunPlanFixture,
        overrides: {
          scheduler: {
            name: "cosine",
          },
        },
      }),
    ).toThrow();
  });

  it("accepts a training job fetch response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulTrainingJobFixture, () =>
      fetchTrainingJob("job-123"),
    );

    expect(result.metrics).toMatchObject({ trainLoss: 0.42 });
    expect(result.eventCount).toBe(1);
    expect(result.clusterGrowth[0]?.additionCount).toBe(1);
  });

  it("accepts a paginated training job event history fixture", async () => {
    const result = await validateSuccessfulFixture(
      {
        jobId: "job-123",
        offset: 10,
        limit: 2,
        totalCount: 13,
        nextOffset: 12,
        events: [
          { type: "step", status: "running", step: 10 },
          { type: "validation", status: "running", step: 11, metrics: {} },
        ],
      },
      () => fetchTrainingJobEvents("job-123", { offset: 10, limit: 2 }),
    );

    expect(trainingJobEventsSchema.parse(result).nextOffset).toBe(12);
  });

  it("accepts completed dataset events in training job and event history payloads", async () => {
    const completedDatasetEvent = {
      type: "dataset_completed",
      status: "completed",
      dataset: "Mnist",
      preset: "baseline",
      runId: "run-1",
      epoch: 2,
      metrics: { "train/loss": 0.1 },
      logDir: "logs/collaborator",
    };

    const jobResult = await validateSuccessfulFixture(
      {
        ...successfulTrainingJobFixture,
        events: [completedDatasetEvent],
        eventCounts: { dataset_completed: 1 },
      },
      () => fetchTrainingJob("job-123"),
    );

    expect(jobResult.events[0]?.status).toBe("completed");

    const eventsResult = await validateSuccessfulFixture(
      {
        jobId: "job-123",
        offset: 0,
        limit: 1,
        totalCount: 1,
        nextOffset: null,
        events: [completedDatasetEvent],
      },
      () => fetchTrainingJobEvents("job-123", { offset: 0, limit: 1 }),
    );

    expect(eventsResult.events[0]?.status).toBe("completed");
  });

  it("rejects nested training job response override objects", () => {
    expect(() =>
      trainingJobSchema.parse({
        ...successfulTrainingJobFixture,
        overrides: {
          scheduler: {
            name: "cosine",
          },
        },
      }),
    ).toThrow();
  });

  it("accepts backend training job lifecycle statuses and rejects unsupported ones", () => {
    expect(
      trainingJobSchema.parse({
        ...successfulTrainingJobFixture,
        status: "unknown",
      }).status,
    ).toBe("unknown");
    expect(() =>
      trainingJobSchema.parse({
        ...successfulTrainingJobFixture,
        status: "finished",
      }),
    ).toThrow();
  });

  it("accepts a training job cancel response fixture", async () => {
    const result = await validateSuccessfulFixture(
      {
        ...successfulTrainingJobFixture,
        status: "cancelled",
        exitCode: 0,
      },
      () => cancelTrainingJob("job-123"),
    );

    expect(result.status).toBe("cancelled");
  });

  it("accepts a live monitor-data response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulMonitorDataFixture, () =>
      fetchMonitorData({
        jobId: "job-123",
        nodePath: "root.classifier",
        preset: "baseline",
        dataset: "Mnist",
      }),
    );

    expect(result.histograms[0].buckets).toHaveLength(2);
    expect(result.images[0].mimeType).toBe("image/png");
  });

  it("rejects unsafe image URLs in monitor and log media payloads", () => {
    expect(() =>
      monitorDataSchema.parse({
        ...successfulMonitorDataFixture,
        images: [
          {
            ...successfulMonitorDataFixture.images[0],
            mimeType: "text/html",
            dataUrl: "data:text/html;base64,PHNjcmlwdD4=",
          },
        ],
      }),
    ).toThrow();
    expect(() =>
      logImageSummarySchema.parse({
        runId: "run-1",
        tag: "validation/examples",
        step: 1,
        wallTime: 1000,
        mimeType: "image/png",
        dataUrl: "http://127.0.0.1/image.png",
      }),
    ).toThrow();
    expect(
      logImageSummarySchema.parse({
        runId: "run-1",
        tag: "validation/examples",
        step: 1,
        wallTime: 1000,
        mimeType: "image/png",
        dataUrl: "",
      }).dataUrl,
    ).toBe("");
  });

  it("accepts a historical monitor-data response fixture", async () => {
    const result = await validateSuccessfulFixture(
      successfulHistoricalMonitorDataFixture,
      () =>
        fetchLogRunMonitorData({
          runId: "run-1",
          nodePath: "root.classifier",
        }),
    );

    expect(result.scalarSeries[0].points).toHaveLength(2);
  });

  it("accepts an active parameter-status response fixture", async () => {
    const result = await validateSuccessfulFixture(
      successfulParameterStatusFixture,
      () =>
        fetchMonitorParameterStatus({
          jobId: "job-123",
          preset: "baseline",
          dataset: "Mnist",
        }),
    );

    expect(result.nodes[0].weights.status).toBe("updated");
    expect(result.nodes[0].bias.status).toBe("missing");
  });

  it("accepts a historical parameter-status response fixture", async () => {
    const result = await validateSuccessfulFixture(
      successfulLogParameterStatusFixture,
      () => fetchLogParameterStatus({ runIds: ["run-1"] }),
    );

    expect(result.runs[0].sourceId).toBe("run-1");
  });

  it("accepts a log runs response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulLogRunsResponse, () =>
      fetchLogRuns(),
    );

    expect(result.hasMore).toBe(false);
    expect(result.runs[0].experiment).toBe("viewer-training");
  });

  it("accepts a log experiments response fixture", async () => {
    const result = await validateSuccessfulFixture(
      successfulLogExperimentsResponse,
      () => fetchLogExperiments(),
    );

    expect(result.hasMore).toBe(false);
    expect(result.experiments[0].runCount).toBe(1);
  });

  it("accepts a delete experiment response fixture", async () => {
    const result = await validateSuccessfulFixture(
      {
        experiment: "viewer-training",
        deletedRunIds: ["run-1"],
        deletedRunCount: 1,
        deletedRelativePath: "viewer-training",
      },
      () => deleteLogExperiment("viewer-training"),
    );

    expect(result.deletedRunIds).toEqual(["run-1"]);
  });

  it("accepts a delete plan response fixture", async () => {
    const result = await validateSuccessfulFixture(
      successfulLogDeletePlanFixture,
      () => createLogRunDeletePlan(logDeleteFilters),
    );

    expect(result.affected.runIds).toEqual(["run-1"]);
    expect(result.candidates[0].relativePath).toContain("version_0");
  });

  it("accepts a delete runs response fixture", async () => {
    const result = await validateSuccessfulFixture(
      successfulLogDeleteResponse,
      () => deleteLogRuns(logDeleteFilters),
    );

    expect(result.deletedRunCount).toBe(1);
    expect(result.counts.experiments).toBe(1);
  });

  it("accepts a log tags response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulLogTagsResponse, () =>
      fetchLogTags({ runIds: ["run-1"] }),
    );

    expect(result.runs[0].histogramTags).toEqual(["weights/classifier"]);
  });

  it("accepts a log scalars response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulLogScalarsResponse, () =>
      fetchLogScalars({
        runIds: ["run-1"],
        tags: ["train/loss"],
      }),
    );

    expect(result.series[0].points[1].value).toBe(0.55);
  });
});

describe("requestJson success", () => {
  it("returns parsed data and sends content-type json", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ status: "ok" }) }),
    );

    const result = await fetchHealth();

    expect(result).toEqual({ status: "ok" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/health`);
    const headers = new Headers((init as RequestInit).headers);
    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.has("Authorization")).toBe(false);
  });

  it("attaches no authorization header when no session token is stored", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ models: [] }) }),
    );

    await fetchModels();

    const headers = new Headers((fetchMock.mock.calls[0][1] as RequestInit).headers);
    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.has("Authorization")).toBe(false);
  });

  it("attaches a bearer authorization header when a session token is stored", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ models: [] }) }),
    );
    setSessionAuthToken("hosted-secret");

    await fetchModels();

    const headers = new Headers((fetchMock.mock.calls[0][1] as RequestInit).headers);
    expect(headers.get("Authorization")).toBe("Bearer hosted-secret");
    expect(headers.get("content-type")).toBe("application/json");
  });

  it("uses the runtime API URL at request time", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ status: "ok" }) }),
    );

    setViewerApiBaseUrl(" https://api.example.test/viewer/// ");
    await fetchHealth();
    setViewerApiBaseUrl("http://127.0.0.1:7777");
    await fetchHealth();

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "https://api.example.test/viewer/health",
      "http://127.0.0.1:7777/health",
    ]);
  });

  it("falls back to the configured default API URL when no runtime URL is set", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ status: "ok" }) }),
    );

    resetViewerApiBaseUrl();

    expect(getViewerApiBaseUrl()).toBe(BASE);
    await fetchHealth();

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/health`);
  });

  it("uses a valid stored runtime API URL before an in-memory switch", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ status: "ok" }) }),
    );
    window.localStorage.setItem(
      VIEWER_API_BASE_URL_STORAGE_KEY,
      "https://stored-api.example.test/viewer",
    );

    expect(getViewerApiBaseUrl()).toBe("https://stored-api.example.test/viewer");
    await fetchHealth();

    expect(fetchMock.mock.calls[0][0]).toBe(
      "https://stored-api.example.test/viewer/health",
    );
  });

  it("ignores invalid stored runtime API URLs", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ status: "ok" }) }),
    );
    window.localStorage.setItem(VIEWER_API_BASE_URL_STORAGE_KEY, "ftp://api.invalid");

    expect(getViewerApiBaseUrl()).toBe(BASE);
    await fetchHealth();

    expect(window.localStorage.getItem(VIEWER_API_BASE_URL_STORAGE_KEY)).toBeNull();
    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/health`);
  });

  it("prefers a runtime API URL switch when storage persistence throws", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ status: "ok" }) }),
    );
    window.localStorage.setItem(
      VIEWER_API_BASE_URL_STORAGE_KEY,
      "https://stored-api.example.test",
    );
    vi.spyOn(window.localStorage, "setItem").mockImplementation(() => {
      throw new Error("storage locked");
    });

    setViewerApiBaseUrl("https://runtime-api.example.test");

    expect(getViewerApiBaseUrl()).toBe("https://runtime-api.example.test");
    await fetchHealth();
    expect(fetchMock.mock.calls[0][0]).toBe(
      "https://runtime-api.example.test/health",
    );
  });

  it("prefers a runtime API URL reset when storage clearing throws", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ status: "ok" }) }),
    );
    window.localStorage.setItem(
      VIEWER_API_BASE_URL_STORAGE_KEY,
      "https://stored-api.example.test",
    );
    vi.spyOn(window.localStorage, "removeItem").mockImplementation(() => {
      throw new Error("storage locked");
    });

    resetViewerApiBaseUrl();

    expect(window.localStorage.getItem(VIEWER_API_BASE_URL_STORAGE_KEY)).toBe(
      "https://stored-api.example.test",
    );
    expect(getViewerApiBaseUrl()).toBe(BASE);
    await fetchHealth();
    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/health`);
  });

  it("rejects runtime API URLs with query strings or fragments", () => {
    expect(normalizeViewerApiBaseUrl("https://api.example.test?debug=true"))
      .toBeNull();
    expect(normalizeViewerApiBaseUrl("https://api.example.test/viewer#status"))
      .toBeNull();
    expect(() => setViewerApiBaseUrl("https://api.example.test?debug=true"))
      .toThrow(/without a query string or fragment/i);
  });

  it("preserves POST request init fields when attaching the bearer token", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            modelType: "linears",
            model: "linear",
            preset: "base",
            parameterCount: 0,
            parameterSizeBytes: 0,
            nodes: [],
            edges: [],
          }),
      }),
    );
    setSessionAuthToken("hosted-secret");

    const input = {
      modelType: "linears",
      model: "linear",
      preset: "base",
      overrides: {},
      dataset: "mnist",
    };
    await inspectModel(input);

    const [, init] = fetchMock.mock.calls[0];
    const request = init as RequestInit;
    const headers = new Headers(request.headers);
    expect(request.method).toBe("POST");
    expect(request.body).toBe(JSON.stringify(input));
    expect(headers.get("Authorization")).toBe("Bearer hosted-secret");
  });
});

describe("requestMultipartJson success", () => {
  it("uploads log archives as multipart form data without forcing content-type", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () => Promise.resolve(successfulLogArchiveImportResponse),
      }),
    );
    setSessionAuthToken("hosted-secret");

    const file = new File(["zip-bytes"], "logs.zip", {
      type: "application/zip",
    });
    const result = await importLogArchive(file);

    expect(result).toEqual(successfulLogArchiveImportResponse);
    const [url, init] = fetchMock.mock.calls[0];
    const request = init as RequestInit;
    const headers = new Headers(request.headers);
    expect(url).toBe(`${BASE}/logs/import`);
    expect(request.method).toBe("POST");
    expect(request.body).toBeInstanceOf(FormData);
    expect(headers.get("Authorization")).toBe("Bearer hosted-secret");
    expect(headers.has("content-type")).toBe(false);
    const archive = (request.body as FormData).get("archive");
    expect(archive).toBeInstanceOf(File);
    expect((archive as File).name).toBe("logs.zip");
  });
});

describe("requestJson error handling", () => {
  it("classifies 401 responses as unauthorized while preserving request context", async () => {
    stubFetch(
      fakeResponse({
        ok: false,
        status: 401,
        statusText: "Unauthorized",
        json: () => Promise.resolve({ detail: "Missing or invalid bearer credentials" }),
      }),
    );

    let caught: unknown;
    try {
      await fetchModels();
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(Error);
    expect((caught as Error).message).toBe(
      `GET /models from ${BASE} failed with 401: Missing or invalid bearer credentials`,
    );
    expect(caught).toMatchObject({
      status: 401,
      method: "GET",
      path: "/models",
      detail: "Missing or invalid bearer credentials",
    });
    expect(isUnauthorizedApiError(caught)).toBe(true);
  });

  it("throws with request context and JSON `detail` when the response is not ok", async () => {
    stubFetch(
      fakeResponse({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: () => Promise.resolve({ detail: "model import failed" }),
      }),
    );

    let caught: unknown;
    try {
      await fetchModels();
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(Error);
    expect((caught as Error).message).toBe(
      `GET /models from ${BASE} failed with 400: model import failed`,
    );
    expect(isUnauthorizedApiError(caught)).toBe(false);
  });

  it("uses statusText when the error body is not JSON", async () => {
    stubFetch(
      fakeResponse({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        json: () => Promise.reject(new Error("not json")),
      }),
    );

    await expect(fetchModels()).rejects.toThrow(
      `GET /models from ${BASE} failed with 500: Internal Server Error`,
    );
  });

  it("throws with request context when a successful body fails schema validation", async () => {
    stubFetch(
      fakeResponse({ json: () => Promise.resolve({ models: "not-an-array" }) }),
    );

    await expect(fetchModels()).rejects.toThrow(
      `Invalid API response for GET /models from ${BASE}: models:`,
    );
  });

  it("includes nested paths in log run schema validation errors", async () => {
    stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            total: 2,
            limit: 1,
            offset: 0,
            hasMore: false,
            runs: [
              {
	                id: "run-1",
	                group: null,
	                experiment: "linear",
	                modelType: "linears",
	                preset: "BASELINE",
                dataset: "Mnist",
                runName: "aaa_20260601_010203",
                timestamp: "2026-06-01 01:02:03",
                version: "version_0",
                relativePath: "linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
                hasResult: false,
                eventFileCount: 1,
                checkpointCount: 0,
                hasHparams: true,
                metrics: {},
              },
            ],
          }),
      }),
    );

    await expect(fetchLogRuns()).rejects.toThrow(
      `Invalid API response for GET /logs/runs from ${BASE}: runs.0.model:`,
    );
  });

  it("includes nested paths in scalar schema validation errors", async () => {
    stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            series: [
              {
                runId: "run-1",
                tag: "train/loss",
                points: [{ step: 1, wallTime: 1000, value: "bad" }],
              },
            ],
          }),
      }),
    );

    await expect(
      fetchLogScalars({ runIds: ["run-1"], tags: ["train/loss"] }),
    ).rejects.toThrow(
      `Invalid API response for POST /logs/scalars from ${BASE}: series.0.points.0.value:`,
    );
  });

  it("includes nested paths in checkpoint schema validation errors", async () => {
    stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            checkpoints: [
              {
                id: "ckpt-1",
                runId: "run-1",
                filename: "epoch=0-step=1.ckpt",
                relativePath: "run/checkpoints/epoch=0-step=1.ckpt",
                epoch: 0,
                step: "bad",
                sizeBytes: 8,
                modifiedAt: "2026-06-01T00:00:00Z",
              },
            ],
          }),
      }),
    );

    await expect(fetchLogCheckpoints({ runIds: ["run-1"] })).rejects.toThrow(
      `Invalid API response for POST /logs/checkpoints from ${BASE}: checkpoints.0.step:`,
    );
  });

  it("includes request context in delete experiment schema validation errors", async () => {
    stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            experiment: "test_model",
            deletedRunIds: [123],
            deletedRunCount: 1,
            deletedRelativePath: "test_model",
          }),
      }),
    );

    await expect(deleteLogExperiment("test_model")).rejects.toThrow(
      `Invalid API response for DELETE /logs/experiments/test_model from ${BASE}: deletedRunIds.0:`,
    );
  });

  it("includes nested paths in log experiment schema validation errors", async () => {
    stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            experiments: [
              {
                experiment: "test_model",
                runCount: "one",
                relativePath: "test_model",
              },
            ],
          }),
      }),
    );

    await expect(fetchLogExperiments()).rejects.toThrow(
      `Invalid API response for GET /logs/experiments from ${BASE}: experiments.0.runCount:`,
    );
  });

  it("throws with request context when capabilities fail schema validation", async () => {
    stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            authMode: "none",
            trainingEnabled: "yes",
            trainingCancellationCapability: "strict-cgroup",
            logDeletionEnabled: true,
            historicalLogsEnabled: true,
            liveMonitorDataEnabled: true,
            historicalMonitorDataEnabled: true,
            uploadsEnabled: false,
            maxUploadSize: null,
            dataSourcesEnabled: false,
            dataSources: [],
          }),
      }),
    );

    await expect(fetchCapabilities()).rejects.toThrow(
      `Invalid API response for GET /capabilities from ${BASE}: trainingEnabled:`,
    );
  });

  it.each([
    ["uploadsEnabled", { uploadsEnabled: "no" }, "uploadsEnabled:"],
    ["maxUploadSize", { maxUploadSize: "unlimited" }, "maxUploadSize:"],
    ["dataSourcesEnabled", { dataSourcesEnabled: "no" }, "dataSourcesEnabled:"],
    ["dataSources", { dataSources: "default-server-data" }, "dataSources:"],
  ])(
    "throws with request context when capabilities placeholder %s is malformed",
    async (_field, override, issuePath) => {
      stubFetch(
        fakeResponse({
          json: () =>
            Promise.resolve({
              ...capabilitiesResponse,
              ...override,
            }),
        }),
      );

      await expect(fetchCapabilities()).rejects.toThrow(
        `Invalid API response for GET /capabilities from ${BASE}: ${issuePath}`,
      );
    },
  );
});

describe("URL and query construction", () => {
  it("fetches capabilities from the root capabilities route", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve(capabilitiesResponse) }),
    );

    const result = await fetchCapabilities();

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/capabilities`);
    expect(result).toEqual(capabilitiesResponse);
  });

  it("defaults additive capabilities placeholders from older payloads", async () => {
    const legacyCapabilities = {
      authMode: "none",
      trainingEnabled: true,
      logDeletionEnabled: true,
      historicalLogsEnabled: true,
      liveMonitorDataEnabled: true,
      historicalMonitorDataEnabled: true,
    };
    stubFetch(fakeResponse({ json: () => Promise.resolve(legacyCapabilities) }));

    const result = await fetchCapabilities();

    expect(result).toEqual({
      ...capabilitiesResponse,
      trainingCancellationCapability: "unsupported",
    });
  });

  it("encodes the preset query for fetchConfigSchema", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({ modelType: "linears", model: "linear", fields: [] }),
      }),
    );

    await fetchConfigSchema(linearIdentity, "preset/one");

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${BASE}/models/linears/linear/config-schema?preset=preset%2Fone`,
    );
  });

  it("encodes the preset query for fetchSearchSpace", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            modelType: "linears",
            model: "linear",
            preset: "baseline",
            axes: [
              {
                key: "hidden_dim",
                configKey: "HIDDEN_DIM",
                searchKey: "SEARCH_SPACE_HIDDEN_DIM",
                label: "hidden dim",
                section: "Layer Stack Options",
                type: "int",
                values: [64, 128],
                locked: false,
                lockedValue: null,
                lockedReason: "",
              },
            ],
          }),
      }),
    );

    const result = await fetchSearchSpace(linearIdentity, "baseline/preset", [
      "baseline/preset",
      "post-norm",
    ]);

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${BASE}/models/linears/linear/search-space?preset=baseline%2Fpreset&presets=baseline%2Fpreset%2Cpost-norm`,
    );
    expect(result.axes[0]).toMatchObject({
      key: "hidden_dim",
      values: [64, 128],
    });
  });

  it("omits the query when no preset is given", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({ modelType: "linears", model: "linear", fields: [] }),
      }),
    );

    await fetchConfigSchema(linearIdentity);

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/models/linears/linear/config-schema`);
  });

  it("interpolates the model into preset/dataset/job paths", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({ modelType: "linears", model: "linear", presets: [] }),
      }),
    );

    await fetchPresets(linearIdentity);

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/models/linears/linear/presets`);
  });

  it("builds the monitor-data query with nodePath, preset, and dataset", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            jobId: "j1",
            nodePath: "root.block",
            preset: "baseline",
            dataset: "mnist",
            logDir: null,
            scalarSeries: [],
            histograms: [],
            images: [],
          }),
      }),
    );

    await fetchMonitorData({
      jobId: "j1",
      nodePath: "root.block",
      preset: "baseline",
      dataset: "mnist",
    });

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${BASE}/training/jobs/j1/monitor-data?nodePath=root.block&preset=baseline&dataset=mnist`,
    );
  });

  it("encodes active training job ids in monitor URLs", async () => {
    const monitorFetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            jobId: "job/a?b",
            nodePath: "root.block",
            dataset: null,
            logDir: null,
            scalarSeries: [],
            histograms: [],
            images: [],
          }),
      }),
    );

    await fetchMonitorData({
      jobId: "job/a?b",
      nodePath: "root.block",
    });

    expect(monitorFetchMock.mock.calls[0][0]).toBe(
      `${BASE}/training/jobs/job%2Fa%3Fb/monitor-data?nodePath=root.block`,
    );

    const statusFetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            sourceId: "job/a?b",
            logDir: null,
            nodes: [],
          }),
      }),
    );

    await fetchMonitorParameterStatus({ jobId: "job/a?b" });

    expect(statusFetchMock.mock.calls[0][0]).toBe(
      `${BASE}/training/jobs/job%2Fa%3Fb/monitor-parameter-status`,
    );
  });

  it("builds the historical log-run monitor-data query", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            jobId: "run/1",
            nodePath: "root.block",
            dataset: "mnist",
            logDir: "logs/run",
            scalarSeries: [],
            histograms: [],
            images: [],
          }),
      }),
    );

    const result = await fetchLogRunMonitorData({
      runId: "run/1",
      nodePath: "root.block",
    });

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${BASE}/logs/runs/run%2F1/monitor-data?nodePath=root.block`,
    );
    expect(result).toMatchObject({
      jobId: "run/1",
      nodePath: "root.block",
      dataset: "mnist",
    });
  });

  it("builds the active parameter-status query with preset and dataset", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            sourceId: "j1",
            preset: "baseline",
            dataset: "mnist",
            logDir: null,
            nodes: [],
          }),
      }),
    );

    await fetchMonitorParameterStatus({
      jobId: "j1",
      preset: "baseline",
      dataset: "mnist",
    });

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${BASE}/training/jobs/j1/monitor-parameter-status?preset=baseline&dataset=mnist`,
    );
  });

  it("posts historical parameter-status run ids", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () => Promise.resolve({ runs: [] }),
      }),
    );

    await fetchLogParameterStatus({ runIds: ["run/1", "run-2"] });

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/parameter-status`);
    expect(fetchMock.mock.calls[0][1]).toMatchObject({
      method: "POST",
      body: JSON.stringify({ runIds: ["run/1", "run-2"] }),
    });
  });

  it("fetches log runs with the expected schema", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            total: 2,
            limit: 1,
            offset: 0,
            hasMore: false,
            runs: [
              {
	                id: "run-1",
	                group: null,
	                experiment: "linear",
	                modelType: "linears",
	                model: "linear",
                preset: "BASELINE",
                dataset: "Mnist",
                runName: "aaa_20260601_010203",
                timestamp: "2026-06-01 01:02:03",
                version: "version_0",
                relativePath: "linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
                hasResult: false,
                eventFileCount: 1,
                checkpointCount: 0,
                hasHparams: true,
                metrics: {},
              },
            ],
          }),
      }),
    );

    const result = await fetchLogRuns();

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/runs`);
    expect(result).toMatchObject({
      total: 2,
      limit: 1,
      offset: 0,
      hasMore: false,
    });
    expect(result.runs[0]).toMatchObject({
      id: "run-1",
      experiment: "linear",
      dataset: "Mnist",
      hasResult: false,
    });
  });

  it("keeps log run pagination scoped by default", async () => {
	    const firstRun = {
	      id: "run-1",
	      group: null,
	      experiment: "linear",
	      modelType: "linears",
	      model: "linear",
      preset: "BASELINE",
      dataset: "Mnist",
      runName: "aaa_20260601_010203",
      timestamp: "2026-06-01 01:02:03",
      version: "version_0",
      relativePath: "linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
      hasResult: false,
      eventFileCount: 1,
      checkpointCount: 0,
      hasHparams: true,
      metrics: {},
    };
    const secondRun = {
      ...firstRun,
      id: "run-2",
      runName: "bbb_20260601_020304",
      relativePath: "linear/BASELINE/Mnist/bbb_20260601_020304/version_0",
    };
    const fetchMock = vi
      .fn<FetchFn>()
      .mockResolvedValueOnce(
        fakeResponse({
          json: () =>
            Promise.resolve({
              total: 2,
              limit: 1,
              offset: 0,
              hasMore: true,
              runs: [firstRun],
            }),
        }),
      )
      .mockResolvedValueOnce(
        fakeResponse({
          json: () =>
            Promise.resolve({
              total: 2,
              limit: 1,
              offset: 1,
              hasMore: false,
              runs: [secondRun],
            }),
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchLogRuns();

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      `${BASE}/logs/runs`,
    ]);
    expect(result.runs.map((run) => run.id)).toEqual(["run-1"]);
    expect(result.hasMore).toBe(true);
  });

  it("fetches all log run pages when requested", async () => {
	    const firstRun = {
	      id: "run-1",
	      group: null,
	      experiment: "linear",
	      modelType: "linears",
	      model: "linear",
      preset: "BASELINE",
      dataset: "Mnist",
      runName: "aaa_20260601_010203",
      timestamp: "2026-06-01 01:02:03",
      version: "version_0",
      relativePath: "linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
      hasResult: false,
      eventFileCount: 1,
      checkpointCount: 0,
      hasHparams: true,
      metrics: {},
    };
    const secondRun = {
      ...firstRun,
      id: "run-2",
      runName: "bbb_20260601_020304",
      relativePath: "linear/BASELINE/Mnist/bbb_20260601_020304/version_0",
    };
    const fetchMock = vi
      .fn<FetchFn>()
      .mockResolvedValueOnce(
        fakeResponse({
          json: () =>
            Promise.resolve({
              total: 2,
              limit: 1,
              offset: 0,
              hasMore: true,
              runs: [firstRun],
            }),
        }),
      )
      .mockResolvedValueOnce(
        fakeResponse({
          json: () =>
            Promise.resolve({
              total: 2,
              limit: 1,
              offset: 1,
              hasMore: false,
              runs: [secondRun],
            }),
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchLogRuns({ includeAllPages: true });

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      `${BASE}/logs/runs`,
      `${BASE}/logs/runs?limit=1&offset=1`,
    ]);
    expect(result.runs.map((run) => run.id)).toEqual(["run-1", "run-2"]);
    expect(result.hasMore).toBe(false);
  });

  it("fetches scoped log run pages with filters and pagination", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            total: 0,
            limit: 5,
            offset: 0,
            hasMore: false,
            runs: [],
          }),
      }),
    );

    await fetchLogRuns({
      filters: {
        models: [linearIdentity],
        preset: ["BASELINE"],
        dataset: ["Mnist", "Cifar10"],
        hasEventFiles: true,
      },
      pagination: { limit: 5, offset: 0 },
    });

	    expect(fetchMock.mock.calls[0][0]).toBe(
	      `${BASE}/logs/runs?modelType=linears&model=linear&preset=BASELINE&dataset=Mnist&dataset=Cifar10&hasEventFiles=true&limit=5&offset=0`,
	    );
  });

  it("fetches log experiments with run counts", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            total: 1,
            limit: 500,
            offset: 0,
            hasMore: false,
            experiments: [
              {
                experiment: "test_model",
                runCount: 2,
                relativePath: "test_model",
              },
            ],
          }),
      }),
    );

    const result = await fetchLogExperiments();

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/experiments`);
    expect(result.total).toBe(1);
    expect(result.hasMore).toBe(false);
    expect(result.experiments[0]).toEqual({
      experiment: "test_model",
      runCount: 2,
      relativePath: "test_model",
    });
  });

  it("fetches all log experiment pages with limit and offset", async () => {
    const fetchMock = vi
      .fn<FetchFn>()
      .mockResolvedValueOnce(
        fakeResponse({
          json: () =>
            Promise.resolve({
              total: 2,
              limit: 1,
              offset: 0,
              hasMore: true,
              experiments: [
                {
                  experiment: "test_model",
                  runCount: 1,
                  relativePath: "test_model",
                },
              ],
            }),
        }),
      )
      .mockResolvedValueOnce(
        fakeResponse({
          json: () =>
            Promise.resolve({
              total: 2,
              limit: 1,
              offset: 1,
              hasMore: false,
              experiments: [
                {
                  experiment: "test_model_2",
                  runCount: 1,
                  relativePath: "test_model_2",
                },
              ],
            }),
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchLogExperiments();

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      `${BASE}/logs/experiments`,
      `${BASE}/logs/experiments?limit=1&offset=1`,
    ]);
    expect(result.experiments.map((entry) => entry.experiment)).toEqual([
      "test_model",
      "test_model_2",
    ]);
    expect(result.hasMore).toBe(false);
  });

  it("derives log run experiments from legacy API payloads", async () => {
    stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            runs: [
              {
	                id: "run-1",
	                group: null,
	                modelType: "linears",
	                model: "linear",
                preset: "BASELINE",
                dataset: "Mnist",
                runName: "aaa_20260601_010203",
                timestamp: "2026-06-01 01:02:03",
                version: "version_0",
                relativePath: "linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
                hasResult: false,
                eventFileCount: 1,
                checkpointCount: 0,
                hasHparams: true,
                metrics: {},
              },
              {
	                id: "run-2",
	                group: "viewer-training/job-1",
	                modelType: "linears",
	                model: "linear",
                preset: "BASELINE",
                dataset: "Cifar10",
                runName: "bbb_20260601_020304",
                timestamp: "2026-06-01 02:03:04",
                version: "version_0",
                relativePath:
                  "viewer-training/job-1/linear/BASELINE/Cifar10/bbb_20260601_020304/version_0",
                hasResult: false,
                eventFileCount: 1,
                checkpointCount: 0,
                hasHparams: true,
                metrics: {},
              },
            ],
          }),
      }),
    );

    const result = await fetchLogRuns();

    expect(result.runs.map((run) => run.experiment)).toEqual([
      "linear",
      "viewer-training",
    ]);
  });

  it("does not fallback on 404 from the configured API base", async () => {
    const fetchMock = vi.fn<FetchFn>().mockResolvedValueOnce(
      fakeResponse({
        ok: false,
        status: 404,
        statusText: "Not Found",
        json: () => Promise.resolve({ detail: "Not Found" }),
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchLogRuns()).rejects.toThrow(
      `GET /logs/runs from ${BASE} failed with 404: Not Found`,
    );

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([`${BASE}/logs/runs`]);
  });

  it("deletes a log experiment with DELETE", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            experiment: "test_model",
            deletedRunIds: ["run-1"],
            deletedRunCount: 1,
            deletedRelativePath: "test_model",
          }),
      }),
    );

    const result = await deleteLogExperiment("test_model");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/logs/experiments/test_model`);
    expect((init as RequestInit).method).toBe("DELETE");
    expect(result.deletedRunIds).toEqual(["run-1"]);
  });

  it("encodes delete log experiment names in the path", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            experiment: "test model/one",
            deletedRunIds: ["run-1"],
            deletedRunCount: 1,
            deletedRelativePath: "test model/one",
          }),
      }),
    );

    await deleteLogExperiment("test model/one");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/logs/experiments/test%20model%2Fone`);
    expect((init as RequestInit).method).toBe("DELETE");
  });
});

describe("POST requests", () => {
  it("posts the inspect body as JSON", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            modelType: "linears",
            model: "linear",
            preset: "base",
            parameterCount: 0,
            parameterSizeBytes: 0,
            nodes: [],
            edges: [],
          }),
      }),
    );

    const input = {
      modelType: "linears",
      model: "linear",
      preset: "base",
      overrides: {},
      dataset: "mnist",
    };
    await inspectModel(input);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/inspect`);
    expect((init as RequestInit).method).toBe("POST");
    expect((init as RequestInit).body).toBe(JSON.stringify(input));
  });

  it("posts the operation graph inspect body as JSON", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () => Promise.resolve(unsupportedOperationGraphResponse),
      }),
    );

    const input = {
      modelType: "linears",
      model: "linear",
      preset: "base",
      overrides: {},
      dataset: "mnist",
    };
    await inspectOperationGraph(input);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/inspect/operation-graph`);
    expect((init as RequestInit).method).toBe("POST");
    expect((init as RequestInit).body).toBe(JSON.stringify(input));
  });

  it("posts log tag and scalar requests as JSON", async () => {
    const tagFetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            runs: [
              {
                runId: "run-1",
                scalarTags: ["train/loss"],
                histogramTags: [],
                imageTags: [],
                textTags: [],
              },
            ],
          }),
      }),
    );

    await fetchLogTags({ runIds: ["run-1"] });

    expect(tagFetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/tags`);
    expect((tagFetchMock.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect((tagFetchMock.mock.calls[0][1] as RequestInit).body).toBe(
      JSON.stringify({ runIds: ["run-1"] }),
    );

    const scalarFetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            series: [
              {
                runId: "run-1",
                tag: "train/loss",
                points: [{ step: 1, wallTime: 1000, value: 0.25 }],
              },
            ],
          }),
      }),
    );

    const scalars = await fetchLogScalars({
      runIds: ["run-1"],
      tags: ["train/loss"],
    });

    expect(scalarFetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/scalars`);
    expect((scalarFetchMock.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect((scalarFetchMock.mock.calls[0][1] as RequestInit).body).toBe(
      JSON.stringify({
        maxPoints: 500,
        sampling: "tail",
        runIds: ["run-1"],
        tags: ["train/loss"],
      }),
    );
    expect(scalars.series[0].points[0].value).toBe(0.25);
  });

  it("chunks oversized log tag requests with bounded concurrency", async () => {
    const pending: Array<{
      body: { runIds: string[] };
      resolved: boolean;
      resolve: (response: Response) => void;
    }> = [];
    let active = 0;
    let maxActive = 0;
    const tagFetchMock = vi.fn<FetchFn>((_input, init) => {
      const response = createDeferred<Response>();
      const body = JSON.parse(String((init as RequestInit).body)) as {
        runIds: string[];
      };
      active += 1;
      maxActive = Math.max(maxActive, active);
      pending.push({
        body,
        resolved: false,
        resolve: response.resolve,
      });
      return response.promise;
    });
    vi.stubGlobal("fetch", tagFetchMock);
    const resolveRequest = (index: number) => {
      const request = pending[index];
      if (!request || request.resolved) {
        return;
      }
      request.resolved = true;
      active -= 1;
      request.resolve(
        fakeResponse({
          json: () =>
            Promise.resolve({
              runs: request.body.runIds.map((runId) => ({
                runId,
                scalarTags: ["validation/accuracy"],
                histogramTags: [],
                imageTags: [],
                textTags: [],
              })),
            }),
        }),
      );
    };
    const runIds = Array.from({ length: 126 }, (_, index) => `run-${index}`);

    const tagsPromise = fetchLogTags({ runIds });

    expect(pending).toHaveLength(2);
    resolveRequest(0);
    await flushAsyncWork();
    expect(pending).toHaveLength(3);
    pending.forEach((_, index) => resolveRequest(index));
    const tags = await tagsPromise;

    expect(tagFetchMock).toHaveBeenCalledTimes(3);
    expect(maxActive).toBe(2);
    expect(pending.map((request) => request.body.runIds.length)).toEqual([
      50,
      50,
      26,
    ]);
    expect(tags.runs).toHaveLength(126);
  });

  it("chunks oversized log scalar tag requests to match backend limits", async () => {
    const pending: Array<{
      body: { tags: string[] };
      resolved: boolean;
      resolve: (response: Response) => void;
    }> = [];
    let active = 0;
    let maxActive = 0;
    const scalarFetchMock = vi.fn<FetchFn>((_input, init) => {
      const response = createDeferred<Response>();
      const body = JSON.parse(String((init as RequestInit).body)) as {
        tags: string[];
      };
      active += 1;
      maxActive = Math.max(maxActive, active);
      pending.push({
        body,
        resolved: false,
        resolve: response.resolve,
      });
      return response.promise;
    });
    vi.stubGlobal("fetch", scalarFetchMock);
    const resolveRequest = (index: number) => {
      const request = pending[index];
      if (!request || request.resolved) {
        return;
      }
      request.resolved = true;
      active -= 1;
      request.resolve(
        fakeResponse({
          json: () =>
            Promise.resolve({
              series: [],
            }),
        }),
      );
    };
    const tags = Array.from({ length: 146 }, (_, index) => `validation/tag-${index}`);

    const scalarsPromise = fetchLogScalars({
      runIds: ["run-1"],
      tags,
    });

    expect(pending).toHaveLength(2);
    resolveRequest(0);
    await flushAsyncWork();
    expect(pending).toHaveLength(3);
    pending.forEach((_, index) => resolveRequest(index));
    await scalarsPromise;

    expect(scalarFetchMock).toHaveBeenCalledTimes(3);
    expect(maxActive).toBe(2);
    expect(pending.map((request) => request.body.tags.length)).toEqual([
      50,
      50,
      46,
    ]);
    for (const [, init] of scalarFetchMock.mock.calls) {
      const body = JSON.parse(String((init as RequestInit).body)) as {
        maxPoints: number;
        sampling: string;
        runIds: string[];
        tags: string[];
      };
      expect(body.maxPoints).toBe(500);
      expect(body.sampling).toBe("tail");
      expect(body.runIds).toEqual(["run-1"]);
      expect(body.tags.length).toBeLessThanOrEqual(50);
    }
  });

  it("posts log media requests as JSON", async () => {
    const mediaFetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            images: [
              {
                runId: "run-1",
                tag: "validation/examples/predictions",
                step: 3,
                wallTime: 1000,
                mimeType: "image/png",
                dataUrl: "data:image/png;base64,AAAA",
              },
            ],
            texts: [
              {
                runId: "run-1",
                tag: "validation/examples/predictions/text_summary",
                step: 3,
                wallTime: 1000,
                text: "cat -> dog",
              },
            ],
          }),
      }),
    );

    const media = await fetchLogMedia({
      runIds: ["run-1"],
      imageTags: ["validation/examples/predictions"],
      textTags: ["validation/examples/predictions/text_summary"],
    });

    expect(mediaFetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/media`);
    expect((mediaFetchMock.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect((mediaFetchMock.mock.calls[0][1] as RequestInit).body).toBe(
      JSON.stringify({
        runIds: ["run-1"],
        imageTags: ["validation/examples/predictions"],
        textTags: ["validation/examples/predictions/text_summary"],
      }),
    );
    expect(media.images[0].dataUrl).toBe("data:image/png;base64,AAAA");
    expect(media.texts[0].text).toBe("cat -> dog");
  });

  it("chunks oversized log media tag requests to match backend limits", async () => {
    const pending: Array<{
      body: { imageTags: string[]; textTags: string[] };
      resolved: boolean;
      resolve: (response: Response) => void;
    }> = [];
    let active = 0;
    let maxActive = 0;
    const mediaFetchMock = vi.fn<FetchFn>((_input, init) => {
      const response = createDeferred<Response>();
      const body = JSON.parse(String((init as RequestInit).body)) as {
        imageTags: string[];
        textTags: string[];
      };
      active += 1;
      maxActive = Math.max(maxActive, active);
      pending.push({
        body,
        resolved: false,
        resolve: response.resolve,
      });
      return response.promise;
    });
    vi.stubGlobal("fetch", mediaFetchMock);
    const resolveRequest = (index: number) => {
      const request = pending[index];
      if (!request || request.resolved) {
        return;
      }
      request.resolved = true;
      active -= 1;
      request.resolve(
        fakeResponse({
          json: () =>
            Promise.resolve({
              images: [],
              texts: [],
            }),
        }),
      );
    };
    const imageTags = Array.from(
      { length: 43 },
      (_, index) => `validation/image-${index}`,
    );
    const textTags = Array.from(
      { length: 22 },
      (_, index) => `validation/text-${index}/text_summary`,
    );

    const mediaPromise = fetchLogMedia({
      runIds: ["run-1"],
      imageTags,
      textTags,
    });

    expect(pending).toHaveLength(2);
    for (let index = 0; index < 5; index += 1) {
      while (pending.length <= index) {
        await flushAsyncWork();
      }
      resolveRequest(index);
      await flushAsyncWork();
    }
    await mediaPromise;

    expect(mediaFetchMock).toHaveBeenCalledTimes(5);
    expect(maxActive).toBe(2);
    expect(
      pending.map((request) => [
        request.body.imageTags.length,
        request.body.textTags.length,
      ]),
    ).toEqual([
      [20, 0],
      [20, 0],
      [3, 0],
      [0, 20],
      [0, 2],
    ]);
  });

  it("fetches checkpoint metadata and run artifacts", async () => {
    const checkpointFetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            checkpoints: [
              {
                id: "ckpt-1",
                runId: "run/1",
                filename: "epoch=0-step=2.ckpt",
                relativePath: "run/checkpoints/epoch=0-step=2.ckpt",
                epoch: 0,
                step: 2,
                sizeBytes: 2048,
                modifiedAt: "2026-06-01T00:00:00Z",
              },
            ],
          }),
      }),
    );

    const checkpoints = await fetchLogCheckpoints({ runIds: ["run/1"] });

    expect(checkpointFetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/checkpoints`);
    expect((checkpointFetchMock.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect((checkpointFetchMock.mock.calls[0][1] as RequestInit).body).toBe(
      JSON.stringify({ runIds: ["run/1"] }),
    );
    expect(checkpoints.checkpoints[0].step).toBe(2);

    const artifactFetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            runId: "run/1",
            params: { batch_size: 4 },
            metrics: { "test/accuracy": 0.9 },
            artifacts: [
              {
                id: "event-1",
                kind: "event_file",
                label: "events.out.tfevents.1",
                relativePath: "run/events.out.tfevents.1",
                sizeBytes: 4096,
                modifiedAt: "2026-06-01T00:00:00Z",
              },
            ],
            checkpoints: checkpoints.checkpoints,
          }),
      }),
    );

    const artifacts = await fetchLogRunArtifacts("run/1");

    expect(artifactFetchMock.mock.calls[0][0]).toBe(
      `${BASE}/logs/runs/run%2F1/artifacts`,
    );
    expect(artifacts.params.batch_size).toBe(4);
    expect(artifacts.checkpoints[0].filename).toBe("epoch=0-step=2.ckpt");
  });

  it("posts filtered log-run delete planning requests as JSON", async () => {
    const filters = {
      experiments: ["test_model"],
      datasets: ["Mnist"],
      models: [linearIdentity],
      presets: ["BASELINE"],
      runIds: ["run-1"],
    };
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            candidateCount: 1,
            counts: {
              runs: 1,
              experiments: 1,
              datasets: 1,
              models: 1,
              presets: 1,
            },
            affected: {
              experiments: ["test_model"],
              datasets: ["Mnist"],
              models: [linearIdentity],
              presets: ["BASELINE"],
              runIds: ["run-1"],
            },
            candidates: [
              {
                id: "run-1",
                experiment: "test_model",
                modelType: "linears",
                model: "linear",
                preset: "BASELINE",
                dataset: "Mnist",
                runName: "aaa_20260601_010203",
                version: "version_0",
                relativePath:
                  "test_model/linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
              },
            ],
            blockedByActiveJobs: [],
            canDelete: true,
          }),
      }),
    );

    const result = await createLogRunDeletePlan(filters);

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/runs/delete-plan`);
    expect((fetchMock.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect((fetchMock.mock.calls[0][1] as RequestInit).body).toBe(
      JSON.stringify(filters),
    );
    expect(result.candidateCount).toBe(1);
    expect(result.candidates[0].relativePath).toContain("version_0");
  });

  it("posts filtered log-run delete requests as JSON", async () => {
    const filters = {
      experiments: ["test_model"],
      datasets: ["Mnist"],
      models: [linearIdentity],
      presets: ["BASELINE"],
      runIds: ["run-1"],
    };
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            deletedRunIds: ["run-1"],
            deletedRunCount: 1,
            deletedRelativePaths: [
              "test_model/linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
            ],
            candidateCount: 1,
            counts: {
              runs: 1,
              experiments: 1,
              datasets: 1,
              models: 1,
              presets: 1,
            },
            affected: {
              experiments: ["test_model"],
              datasets: ["Mnist"],
              models: [linearIdentity],
              presets: ["BASELINE"],
              runIds: ["run-1"],
            },
            candidates: [
              {
                id: "run-1",
                experiment: "test_model",
                modelType: "linears",
                model: "linear",
                preset: "BASELINE",
                dataset: "Mnist",
                runName: "aaa_20260601_010203",
                version: "version_0",
                relativePath:
                  "test_model/linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
              },
            ],
            blockedByActiveJobs: [],
            canDelete: true,
          }),
      }),
    );

    const result = await deleteLogRuns(filters);

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/logs/runs/delete`);
    expect((fetchMock.mock.calls[0][1] as RequestInit).method).toBe("POST");
    expect((fetchMock.mock.calls[0][1] as RequestInit).body).toBe(
      JSON.stringify(filters),
    );
    expect(result.deletedRunIds).toEqual(["run-1"]);
  });

  it("posts to the cancel endpoint without a body", async () => {
    const job = {
      id: "j1",
      status: "cancelled",
      modelType: "linears",
      model: "linear",
      preset: "base",
      datasets: [],
      overrides: {},
      monitors: [],
      logFolder: "test_model",
      createdAt: "t",
      updatedAt: "t",
      exitCode: null,
      pid: 1,
      cancellationMode: "strict-cgroup",
      currentDataset: null,
      epoch: null,
      step: null,
      metrics: {},
      logDir: null,
      events: [],
      logTail: [],
      resultLinks: [],
    };
    const fetchMock = stubFetch(fakeResponse({ json: () => Promise.resolve(job) }));

    await cancelTrainingJob("j1");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/training/jobs/j1/cancel`);
    expect((init as RequestInit).method).toBe("POST");
  });

  it("encodes training job ids in job, event, and cancel paths", async () => {
    const job = {
      id: "job/a?b",
      status: "running",
      modelType: "linears",
      model: "linear",
      preset: "base",
      datasets: [],
      overrides: {},
      monitors: [],
      logFolder: "test_model",
      createdAt: "t",
      updatedAt: "t",
      exitCode: null,
      pid: 1,
      currentDataset: null,
      epoch: null,
      step: null,
      metrics: {},
      logDir: null,
      events: [],
      logTail: [],
      resultLinks: [],
    };
    const jobFetchMock = stubFetch(fakeResponse({ json: () => Promise.resolve(job) }));

    await fetchTrainingJob("job/a?b");

    expect(jobFetchMock.mock.calls[0][0]).toBe(
      `${BASE}/training/jobs/job%2Fa%3Fb`,
    );

    const eventsFetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            jobId: "job/a?b",
            offset: 1,
            limit: 2,
            totalCount: 3,
            nextOffset: 3,
            events: [],
          }),
      }),
    );

    await fetchTrainingJobEvents("job/a?b", { offset: 1, limit: 2 });

    expect(eventsFetchMock.mock.calls[0][0]).toBe(
      `${BASE}/training/jobs/job%2Fa%3Fb/events?offset=1&limit=2`,
    );

    const cancelFetchMock = stubFetch(fakeResponse({ json: () => Promise.resolve(job) }));

    await cancelTrainingJob("job/a?b");

    expect(cancelFetchMock.mock.calls[0][0]).toBe(
      `${BASE}/training/jobs/job%2Fa%3Fb/cancel`,
    );
  });

  it("preserves backend detail on delete experiment failure", async () => {
    stubFetch(
      fakeResponse({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: () => Promise.resolve({ detail: "Unknown log experiment: missing" }),
      }),
    );

    await expect(deleteLogExperiment("missing")).rejects.toThrow(
      `DELETE /logs/experiments/missing from ${BASE} failed with 400: Unknown log experiment: missing`,
    );
  });

  it("preserves backend detail on filtered run delete failure", async () => {
    stubFetch(
      fakeResponse({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: () =>
          Promise.resolve({
            detail: "A training job is still writing to this log folder.",
          }),
      }),
    );

    await expect(
      deleteLogRuns({
        experiments: ["test_model"],
        datasets: ["Mnist"],
        models: [linearIdentity],
        presets: ["BASELINE"],
        runIds: ["run-1"],
      }),
    ).rejects.toThrow(
      `POST /logs/runs/delete from ${BASE} failed with 400: A training job is still writing to this log folder.`,
    );
  });

  it("preserves backend detail on log archive import failure", async () => {
    stubFetch(
      fakeResponse({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: () =>
          Promise.resolve({
            detail: "Unsafe archive path contains traversal: ../escaped.txt",
          }),
      }),
    );

    await expect(
      importLogArchive(new File(["zip-bytes"], "logs.zip")),
    ).rejects.toThrow(
      `POST /logs/import from ${BASE} failed with 400: Unsafe archive path contains traversal: ../escaped.txt`,
    );
  });

  it("preserves backend detail on historical monitor-data failure", async () => {
    stubFetch(
      fakeResponse({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: () => Promise.resolve({ detail: "Unknown log run id: missing" }),
      }),
    );

    await expect(
      fetchLogRunMonitorData({ runId: "missing", nodePath: "root.block" }),
    ).rejects.toThrow(
      `GET /logs/runs/missing/monitor-data?nodePath=root.block from ${BASE} failed with 400: Unknown log run id: missing`,
    );
  });

  it("fetches a training job by id with GET", async () => {
    const job = {
	      id: "j2",
	      status: "running",
	      modelType: "linears",
	      model: "linear",
      preset: "base",
      datasets: [],
      overrides: {},
      monitors: [],
      logFolder: "test_model",
      createdAt: "t",
      updatedAt: "t",
      exitCode: null,
      pid: 2,
      currentDataset: null,
      epoch: null,
      step: null,
      metrics: {},
      logDir: null,
      events: [],
      logTail: [],
      resultLinks: [],
    };
    const fetchMock = stubFetch(fakeResponse({ json: () => Promise.resolve(job) }));

    await fetchTrainingJob("j2");

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/training/jobs/j2`);
  });

  it("fetches paginated training job events with GET", async () => {
    const payload = {
      jobId: "j2",
      offset: 50,
      limit: 25,
      totalCount: 120,
      nextOffset: 75,
      events: [],
    };
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve(payload) }),
    );

    await fetchTrainingJobEvents("j2", { offset: 50, limit: 25 });

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${BASE}/training/jobs/j2/events?offset=50&limit=25`,
    );
  });
});

describe("createTrainingJob", () => {
  it("posts the job request body as JSON", async () => {
    const job = {
      id: "j3",
      status: "queued",
      modelType: "linears",
      model: "linear",
      preset: "base",
      datasets: ["mnist"],
      overrides: {},
      monitors: [],
      logFolder: "test_model",
      createdAt: "t",
      updatedAt: "t",
      exitCode: null,
      pid: 3,
      currentDataset: null,
      epoch: null,
      step: null,
      metrics: {},
      logDir: null,
      events: [],
      logTail: [],
      resultLinks: [],
    };
    const fetchMock = stubFetch(fakeResponse({ json: () => Promise.resolve(job) }));

    const input = {
      modelType: "linears",
      model: "linear",
      preset: "base",
      presets: ["base", "gating"],
      datasets: ["mnist"],
      overrides: {},
      logFolder: "test_model",
      monitors: [],
    };
    await createTrainingJob(input);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/training/jobs`);
    expect((init as RequestInit).method).toBe("POST");
    expect((init as RequestInit).body).toBe(JSON.stringify(input));
  });
});

describe("fetchTrainingRunPlan", () => {
  it("posts the run plan request body as JSON", async () => {
    const plan = {
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      presets: ["baseline"],
      datasets: ["Mnist"],
      overrides: {},
      search: null,
      logFolder: "test_model",
      isRandomSearch: false,
      runs: [
        {
          id: "run-0001",
          index: 1,
          status: "Pending",
          preset: "baseline",
          dataset: "Mnist",
          changes: [],
          overrides: {},
          command:
            "source experiment.sh --model-type linears --model linear --preset baseline --datasets Mnist",
          totalEpochs: 30,
          currentEpoch: 0,
          metrics: {},
          logDir: null,
          error: null,
        },
      ],
      summary: {
        totalRuns: 1,
        completedRuns: 0,
        runningRuns: 0,
        pendingRuns: 1,
        failedRuns: 0,
        cancelledRuns: 0,
        skippedRuns: 0,
        totalEpochs: 30,
        completedEpochs: 0,
        remainingEpochs: 30,
      },
    };
    const fetchMock = stubFetch(fakeResponse({ json: () => Promise.resolve(plan) }));
    const input = {
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      presets: ["baseline"],
      datasets: ["Mnist"],
      overrides: {},
      logFolder: "test_model",
    };

    const result = await fetchTrainingRunPlan(input);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/training/run-plan`);
    expect((init as RequestInit).method).toBe("POST");
    expect((init as RequestInit).body).toBe(JSON.stringify(input));
    expect(result.summary.remainingEpochs).toBe(30);
  });
});

describe("config snapshots", () => {
  const snapshot = {
    id: "snap-1",
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    name: "tuned lr",
    overrides: { learning_rate: "0.01" },
    createdAt: "t",
    updatedAt: "t",
  };

  it("fetches snapshots for a model with an encoded query", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            modelType: "linears",
            model: "linear",
            snapshots: [snapshot],
          }),
      }),
    );

    const result = await fetchConfigSnapshots(linearIdentity);

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${BASE}/config-snapshots?modelType=linears&model=linear`,
    );
    expect(result.snapshots[0].id).toBe("snap-1");
  });

  it("fetches the global snapshot library", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () => Promise.resolve({ snapshots: [snapshot] }),
      }),
    );

    const result = await fetchConfigSnapshotLibrary();

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/config-snapshots/library`);
    expect(result.snapshots[0].id).toBe("snap-1");
  });

  it("posts the create request body as JSON", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve(snapshot) }),
    );
    const input = {
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      name: "tuned lr",
      overrides: { learning_rate: "0.01" },
    };

    await createConfigSnapshot(input);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/config-snapshots`);
    expect((init as RequestInit).method).toBe("POST");
    expect((init as RequestInit).body).toBe(JSON.stringify(input));
  });

  it("patches the rename request with the snapshot id and name", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve(snapshot) }),
    );

    await renameConfigSnapshot("snap-1", "new name");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/config-snapshots/snap-1`);
    expect((init as RequestInit).method).toBe("PATCH");
    expect((init as RequestInit).body).toBe(JSON.stringify({ name: "new name" }));
  });

  it("patches snapshot updates with optional name and overrides", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve(snapshot) }),
    );
    const input = {
      name: "new name",
      overrides: { learning_rate: "0.02" },
    };

    await updateConfigSnapshot("snap-1", input);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/config-snapshots/snap-1`);
    expect((init as RequestInit).method).toBe("PATCH");
    expect((init as RequestInit).body).toBe(JSON.stringify(input));
  });

  it("deletes a snapshot by id and returns the remaining list", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({ modelType: "linears", model: "linear", snapshots: [] }),
      }),
    );

    const result = await deleteConfigSnapshot("snap-1");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/config-snapshots/snap-1`);
    expect((init as RequestInit).method).toBe("DELETE");
    expect(result.snapshots).toEqual([]);
  });

  it("rejects malformed snapshot payloads with request context", async () => {
    stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            modelType: "linears",
            model: "linear",
            snapshots: [{ ...snapshot, id: 7 }],
          }),
      }),
    );

    await expect(fetchConfigSnapshots(linearIdentity)).rejects.toThrow(
      `Invalid API response for GET /config-snapshots?modelType=linears&model=linear from ${BASE}: snapshots.0.id:`,
    );
  });
});
