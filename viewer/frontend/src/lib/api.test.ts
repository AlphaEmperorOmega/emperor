import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cancelTrainingJob,
  createConfigSnapshot,
  createLogRunDeletePlan,
  createTrainingJob,
  deleteConfigSnapshot,
  deleteLogExperiment,
  deleteLogRuns,
  fetchCapabilities,
  fetchConfigSchema,
  fetchConfigSnapshots,
  fetchDatasets,
  fetchHealth,
  fetchLogExperiments,
  fetchLogRunMonitorData,
  fetchLogRuns,
  fetchLogScalars,
  fetchLogTags,
  fetchModels,
  fetchMonitorData,
  fetchMonitors,
  fetchPresets,
  fetchSearchSpace,
  fetchTrainingJob,
  fetchTrainingRunPlan,
  inspectModel,
  isUnauthorizedApiError,
  renameConfigSnapshot,
} from "@/lib/api";
import { setSessionAuthToken } from "@/lib/auth-token";

// Characterization tests: assert the CURRENT behavior of the API client
// (URL/verb/body construction and requestJson error handling) so later
// refactors can be verified green-to-green. Not aspirational.

const BASE = "http://127.0.0.1:9999";

const capabilitiesResponse = {
  authMode: "none",
  trainingEnabled: true,
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

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  window.sessionStorage.clear();
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
  model: "linear",
  preset: "baseline",
  parameterCount: 7850,
  nodes: [
    {
      id: "input",
      label: "Input",
      typeName: "InputNode",
      path: "root.input",
      graphRole: "architecture",
      parameterCount: 0,
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
  model: "linear",
  preset: "baseline",
  presets: ["baseline", "wide"],
  datasets: ["Mnist"],
  overrides: {
    learning_rate: 0.01,
    scheduler: {
      name: "cosine",
    },
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

const successfulLogRunFixture = {
  id: "run-1",
  group: "viewer-training/job-123",
  experiment: "viewer-training",
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
    models: ["linear"],
    presets: ["baseline"],
    runIds: ["run-1"],
  },
  candidates: [
    {
      id: "run-1",
      experiment: "viewer-training",
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
  models: ["linear"],
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

  it("accepts a models response fixture", async () => {
    const result = await validateSuccessfulFixture(
      { models: ["linear", "bert_linear"] },
      fetchModels,
    );

    expect(result.models).toEqual(["linear", "bert_linear"]);
  });

  it("accepts a presets response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulPresetResponse, () =>
      fetchPresets("linear"),
    );

    expect(result.presets.map((preset) => preset.name)).toEqual([
      "baseline",
      "wide",
    ]);
  });

  it("accepts a datasets response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulDatasetResponse, () =>
      fetchDatasets("linear"),
    );

    expect(result.datasets[0]).toMatchObject({
      name: "Mnist",
      inputDim: 784,
      outputDim: 10,
    });
  });

  it("accepts a monitors response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulMonitorResponse, () =>
      fetchMonitors("linear"),
    );

    expect(result.monitors[0].kinds).toEqual(["scalar", "histogram"]);
  });

  it("accepts a config schema response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulConfigSchemaResponse, () =>
      fetchConfigSchema("linear", "baseline"),
    );

    expect(result.fields.map((field) => field.key)).toEqual([
      "learning_rate",
      "checkpoint",
    ]);
  });

  it("accepts a search-space response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulSearchSpaceResponse, () =>
      fetchSearchSpace("linear", "baseline"),
    );

    expect(result.axes[0].values).toEqual([64, 128]);
  });

  it("accepts an inspect response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulInspectResponse, () =>
      inspectModel({
        model: "linear",
        preset: "baseline",
        overrides: {},
        dataset: "Mnist",
      }),
    );

    expect(result.nodes[1].config?.fields).toHaveLength(2);
  });

  it("accepts a training job creation response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulTrainingJobFixture, () =>
      createTrainingJob({
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

  it("accepts a training job fetch response fixture", async () => {
    const result = await validateSuccessfulFixture(successfulTrainingJobFixture, () =>
      fetchTrainingJob("job-123"),
    );

    expect(result.metrics).toMatchObject({ trainLoss: 0.42 });
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

  it("preserves POST request init fields when attaching the bearer token", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
            model: "linear",
            preset: "base",
            parameterCount: 0,
            nodes: [],
            edges: [],
          }),
      }),
    );
    setSessionAuthToken("hosted-secret");

    const input = { model: "linear", preset: "base", overrides: {}, dataset: "mnist" };
    await inspectModel(input);

    const [, init] = fetchMock.mock.calls[0];
    const request = init as RequestInit;
    const headers = new Headers(request.headers);
    expect(request.method).toBe("POST");
    expect(request.body).toBe(JSON.stringify(input));
    expect(headers.get("Authorization")).toBe("Bearer hosted-secret");
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

    expect(result).toEqual(capabilitiesResponse);
  });

  it("encodes the preset query for fetchConfigSchema", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ model: "m", fields: [] }) }),
    );

    await fetchConfigSchema("linear", "preset/one");

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${BASE}/models/linear/config-schema?preset=preset%2Fone`,
    );
  });

  it("encodes the preset query for fetchSearchSpace", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () =>
          Promise.resolve({
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

    const result = await fetchSearchSpace("linear", "baseline/preset");

    expect(fetchMock.mock.calls[0][0]).toBe(
      `${BASE}/models/linear/search-space?preset=baseline%2Fpreset`,
    );
    expect(result.axes[0]).toMatchObject({
      key: "hidden_dim",
      values: [64, 128],
    });
  });

  it("omits the query when no preset is given", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ model: "m", fields: [] }) }),
    );

    await fetchConfigSchema("linear");

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/models/linear/config-schema`);
  });

  it("interpolates the model into preset/dataset/job paths", async () => {
    const fetchMock = stubFetch(
      fakeResponse({
        json: () => Promise.resolve({ model: "linear", presets: [] }),
      }),
    );

    await fetchPresets("linear");

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/models/linear/presets`);
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

  it("fetches all log run pages with limit and offset", async () => {
    const firstRun = {
      id: "run-1",
      group: null,
      experiment: "linear",
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
      `${BASE}/logs/runs?limit=1&offset=1`,
    ]);
    expect(result.runs.map((run) => run.id)).toEqual(["run-1", "run-2"]);
    expect(result.hasMore).toBe(false);
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
            model: "linear",
            preset: "base",
            parameterCount: 0,
            nodes: [],
            edges: [],
          }),
      }),
    );

    const input = { model: "linear", preset: "base", overrides: {}, dataset: "mnist" };
    await inspectModel(input);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/inspect`);
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
      JSON.stringify({ runIds: ["run-1"], tags: ["train/loss"] }),
    );
    expect(scalars.series[0].points[0].value).toBe(0.25);
  });

  it("posts filtered log-run delete planning requests as JSON", async () => {
    const filters = {
      experiments: ["test_model"],
      datasets: ["Mnist"],
      models: ["linear"],
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
              models: ["linear"],
              presets: ["BASELINE"],
              runIds: ["run-1"],
            },
            candidates: [
              {
                id: "run-1",
                experiment: "test_model",
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
      models: ["linear"],
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
              models: ["linear"],
              presets: ["BASELINE"],
              runIds: ["run-1"],
            },
            candidates: [
              {
                id: "run-1",
                experiment: "test_model",
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
    const fetchMock = stubFetch(fakeResponse({ json: () => Promise.resolve(job) }));

    await cancelTrainingJob("j1");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/training/jobs/j1/cancel`);
    expect((init as RequestInit).method).toBe("POST");
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
        models: ["linear"],
        presets: ["BASELINE"],
        runIds: ["run-1"],
      }),
    ).rejects.toThrow(
      `POST /logs/runs/delete from ${BASE} failed with 400: A training job is still writing to this log folder.`,
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
});

describe("createTrainingJob", () => {
  it("posts the job request body as JSON", async () => {
    const job = {
      id: "j3",
      status: "queued",
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
          command: "source experiment.sh linear --preset baseline --datasets Mnist",
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
        json: () => Promise.resolve({ model: "linear", snapshots: [snapshot] }),
      }),
    );

    const result = await fetchConfigSnapshots("linear");

    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/config-snapshots?model=linear`);
    expect(result.snapshots[0].id).toBe("snap-1");
  });

  it("posts the create request body as JSON", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve(snapshot) }),
    );
    const input = {
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

  it("deletes a snapshot by id and returns the remaining list", async () => {
    const fetchMock = stubFetch(
      fakeResponse({ json: () => Promise.resolve({ model: "linear", snapshots: [] }) }),
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
          Promise.resolve({ model: "linear", snapshots: [{ ...snapshot, id: 7 }] }),
      }),
    );

    await expect(fetchConfigSnapshots("linear")).rejects.toThrow(
      `Invalid API response for GET /config-snapshots?model=linear from ${BASE}: snapshots.0.id:`,
    );
  });
});
