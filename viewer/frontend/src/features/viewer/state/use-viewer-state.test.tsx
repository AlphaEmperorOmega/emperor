import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  fetchHealth: vi.fn(),
  fetchCapabilities: vi.fn(),
  fetchModels: vi.fn(),
  fetchPresets: vi.fn(),
  fetchDatasets: vi.fn(),
  fetchMonitors: vi.fn(),
  fetchConfigSchema: vi.fn(),
  fetchSearchSpace: vi.fn(),
  fetchConfigSnapshots: vi.fn(),
  fetchConfigSnapshotLibrary: vi.fn(),
  createConfigSnapshot: vi.fn(),
  renameConfigSnapshot: vi.fn(),
  updateConfigSnapshot: vi.fn(),
  deleteConfigSnapshot: vi.fn(),
  fetchLogRuns: vi.fn(),
  fetchLogExperiments: vi.fn(),
  fetchLogTags: vi.fn(),
  inspectModel: vi.fn(),
  getViewerApiBaseUrl: vi.fn(),
  normalizeViewerApiBaseUrl: vi.fn(),
  setViewerApiBaseUrl: vi.fn(),
  resetViewerApiBaseUrl: vi.fn(),
  fetchTrainingRunPlan: vi.fn(),
  createTrainingJob: vi.fn(),
  fetchTrainingJob: vi.fn(),
  cancelTrainingJob: vi.fn(),
  fetchMonitorParameterStatus: vi.fn(),
  fetchLogParameterStatus: vi.fn(),
}));
vi.mock("@/lib/api", () => mocks);

import { useViewerState } from "@/features/viewer/state/use-viewer-state";
import { ConnectedTrainingWorkspace } from "@/features/viewer/components/connected-training-panel";
import { ViewerProviders } from "@/features/viewer/providers/viewer-providers";
import { TargetPresetPanel } from "@/features/viewer/components/screen/target-preset-panel";
import {
  clearPersistedTargetSelection,
} from "@/features/viewer/state/target/target-selection-storage";
import {
  type GraphNode,
  type InspectResponse,
  type LogParameterStatusResponse,
  type LogRun,
  type ModelIdentity,
  type TrainingJob,
} from "@/lib/api";

const DEFAULT_VIEWER_API_BASE_URL = "http://127.0.0.1:9999";

function renderViewerState(options: Parameters<typeof useViewerState>[0] = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const viewerOptions = { activeWorkspace: "logs" as const, ...options };

  return renderHook(() => useViewerState(viewerOptions), {
    wrapper: ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children),
  });
}

function renderTargetPresetPanel(onOpenFullConfig = vi.fn()) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return render(
    <QueryClientProvider client={client}>
      <ViewerProviders>
        <TargetPresetPanel onOpenFullConfig={onOpenFullConfig} />
      </ViewerProviders>
    </QueryClientProvider>,
  );
}

function renderTrainingPanel() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return render(
    <QueryClientProvider client={client}>
      <ViewerProviders>
        <ConnectedTrainingWorkspace onOpenFullConfig={vi.fn()} />
      </ViewerProviders>
    </QueryClientProvider>,
  );
}

function renderTrainingPanelWithExperiments() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return render(
    <QueryClientProvider client={client}>
      <ViewerProviders>
        <TargetPresetPanel onOpenFullConfig={vi.fn()} />
        <ConnectedTrainingWorkspace onOpenFullConfig={vi.fn()} />
      </ViewerProviders>
    </QueryClientProvider>,
  );
}

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
};

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve;
  });
  return { promise, resolve };
}

function logRun(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  return {
    id: overrides.id,
    group: overrides.group ?? overrides.experiment ?? "exp_linear",
    experiment: overrides.experiment ?? "exp_linear",
    modelType: overrides.modelType ?? "linears",
    model: overrides.model ?? "linear",
    preset: overrides.preset ?? "Fast",
    dataset: overrides.dataset ?? "FashionMnist",
    runName: overrides.runName ?? `${overrides.id}_20260601_010203`,
    timestamp: overrides.timestamp ?? "2026-06-01 01:02:03",
    version: overrides.version ?? "version_0",
    relativePath:
      overrides.relativePath ??
      "exp_linear/linear/Fast/FashionMnist/run/version_0",
    hasResult: overrides.hasResult ?? false,
    eventFileCount: overrides.eventFileCount ?? 1,
    checkpointCount: overrides.checkpointCount ?? 0,
    hasHparams: overrides.hasHparams ?? true,
    metrics: overrides.metrics ?? {},
  };
}

function graphNode(
  id: string,
  path: string,
  typeName: string,
  overrides: Partial<GraphNode> = {},
): GraphNode {
  return {
    id,
    label: overrides.label ?? id,
    typeName,
    path,
    graphRole: overrides.graphRole ?? "architecture",
    parameterCount: overrides.parameterCount ?? 0,
    parameterSizeBytes: overrides.parameterSizeBytes ?? (overrides.parameterCount ?? 0) * 4,
    details: overrides.details ?? {},
    config: overrides.config ?? null,
  };
}

function monitorGraph(): InspectResponse {
  const root = graphNode("root", "main_model", "LayerStack");
  const wrapper = graphNode("layer-0", "main_model.0", "Layer");
  const linear = graphNode("linear-0", "main_model.0.model", "LinearLayer");

  return {
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    parameterCount: 0,
    parameterSizeBytes: 0,
    nodes: [root, wrapper, linear],
    edges: [
      { id: "root-layer-0", source: root.id, target: wrapper.id },
      { id: "layer-0-linear-0", source: wrapper.id, target: linear.id },
    ],
  };
}

function experimentMonitorGraph(
  request: {
    modelType: string;
    model: string;
    preset: string;
  },
  suffix: string,
): InspectResponse {
  const root = graphNode(`${suffix}-root`, "main_model", "LayerStack");
  const wrapper = graphNode(`${suffix}-layer`, "main_model.0", "Layer");
  const linear = graphNode(
    `${suffix}-linear`,
    "main_model.0.model",
    "LinearLayer",
    {
      details: { weightShape: [10, 10], biasShape: [10] },
      parameterCount: 110,
    },
  );

  return {
    modelType: request.modelType,
    model: request.model,
    preset: request.preset,
    parameterCount: 110,
    parameterSizeBytes: 440,
    nodes: [root, wrapper, linear],
    edges: [
      { id: `${suffix}-root-layer`, source: root.id, target: wrapper.id },
      { id: `${suffix}-layer-linear`, source: wrapper.id, target: linear.id },
    ],
  };
}

function parameterLogTags(runIds: string[]) {
  return {
    runs: runIds.map((runId) => ({
      runId,
      scalarTags: [
        "main_model.0.model/weights/mean",
        "main_model.0.model/bias/mean",
      ],
      histogramTags: [],
      imageTags: [],
      textTags: [],
    })),
  };
}

function performanceLogTags(runIds: string[]) {
  return {
    runs: runIds.map((runId) => ({
      runId,
      scalarTags: [
        "epoch",
        "train/loss",
        "test/accuracy",
        "parameters/global_norm",
        "gradients/global_norm",
        "train/confusion_matrix/class_0/class_1",
      ],
      histogramTags: [],
      imageTags: ["validation/examples/predictions"],
      textTags: [],
    })),
  };
}

function parameterStatus(runIds: string[]) {
  return {
    runs: runIds.map((runId) => ({
      sourceId: runId,
      preset: "baseline",
      dataset: "Mnist",
      logDir: `logs/${runId}`,
      nodes: [
        {
          nodePath: "main_model.0.model",
          weights: {
            status: "updated" as const,
            metric: "main_model.0.model/weights/relative_delta_norm",
            lastStep: 12,
            observedPoints: 2,
          },
          bias: {
            status: "updated" as const,
            metric: "main_model.0.model/bias/delta_norm",
            lastStep: 12,
            observedPoints: 2,
          },
        },
      ],
    })),
  };
}

function linearPreviewGraph(request: {
  modelType: string;
  model: string;
  preset: string;
}): InspectResponse {
  const root = graphNode("__root__", "model", "Model");
  const linear = graphNode(
    "main_model.layers.0.model",
    "main_model.layers.0.model",
    "LinearLayer",
  );

  return {
    modelType: request.modelType,
    model: request.model,
    preset: request.preset,
    parameterCount: 0,
    parameterSizeBytes: 0,
    nodes: [root, linear],
    edges: [
      {
        id: "__root__-main_model.layers.0.model",
        source: root.id,
        target: linear.id,
      },
    ],
  };
}

function expertsPreviewGraph(request: {
  modelType: string;
  model: string;
  preset: string;
}): InspectResponse {
  const root = graphNode("__root__", "model", "Model");
  const experts = graphNode(
    "main_model.layers.0.model",
    "main_model.layers.0.model",
    "MixtureOfExperts",
  );
  const expertsModel = graphNode(
    "main_model.layers.0.model.experts",
    "main_model.layers.0.model.experts",
    "MixtureOfExpertsModel",
  );

  return {
    modelType: request.modelType,
    model: request.model,
    preset: request.preset,
    parameterCount: 0,
    parameterSizeBytes: 0,
    nodes: [root, experts, expertsModel],
    edges: [
      {
        id: "__root__-main_model.layers.0.model",
        source: root.id,
        target: experts.id,
      },
      {
        id: "main_model.layers.0.model-main_model.layers.0.model.experts",
        source: experts.id,
        target: expertsModel.id,
      },
    ],
  };
}

function trainingJob(overrides: Partial<TrainingJob> = {}): TrainingJob {
  return {
    id: overrides.id ?? "job-1",
    status: overrides.status ?? "running",
    modelType: overrides.modelType ?? "linears",
    model: overrides.model ?? "linear",
    preset: overrides.preset ?? "baseline",
    presets: overrides.presets ?? ["baseline"],
    datasets: overrides.datasets ?? ["Mnist"],
    overrides: overrides.overrides ?? {},
    search: overrides.search ?? null,
    plannedRunCount: overrides.plannedRunCount ?? 1,
    runPlan: overrides.runPlan ?? null,
    monitors: overrides.monitors ?? ["linear"],
    logFolder: overrides.logFolder ?? "runs",
    createdAt: overrides.createdAt ?? "2026-06-01T00:00:00.000Z",
    updatedAt: overrides.updatedAt ?? "2026-06-01T00:00:00.000Z",
    exitCode: overrides.exitCode ?? null,
    pid: overrides.pid ?? 1,
    currentPreset: overrides.currentPreset ?? "baseline",
    currentDataset: overrides.currentDataset ?? "Mnist",
    epoch: overrides.epoch ?? 1,
    step: overrides.step ?? 10,
    metrics: overrides.metrics ?? {},
    logDir: overrides.logDir ?? "runs/job-1",
    events: overrides.events ?? [],
    eventCount: overrides.eventCount ?? 0,
    eventCounts: overrides.eventCounts ?? {},
    eventsTruncated: overrides.eventsTruncated ?? false,
    clusterGrowth: overrides.clusterGrowth ?? [],
    logTail: overrides.logTail ?? [],
    resultLinks: overrides.resultLinks ?? [],
  };
}

function activeMonitorJob(job: TrainingJob) {
  return {
    id: job.id,
    status: job.status,
    monitors: job.monitors,
    preset: job.preset,
    presets: job.presets,
    datasets: job.datasets,
    logFolder: job.logFolder,
    currentPreset: job.currentPreset,
    currentDataset: job.currentDataset,
  };
}

function modelIdentityKey(identity: ModelIdentity) {
  return `${identity.modelType}/${identity.model}`;
}

function mockPublicModelCatalog() {
  const presetsByModel = new Map([
    [
      "linears/linear",
      [
        { name: "baseline", label: "Baseline", description: "" },
        { name: "fast", label: "Fast", description: "" },
      ],
    ],
    [
      "linears/linear_adaptive",
      [{ name: "adaptive", label: "Adaptive", description: "" }],
    ],
    [
      "experts/experts_linear",
      [{ name: "expert-baseline", label: "Expert baseline", description: "" }],
    ],
    [
      "transformer_encoder/bert_linear",
      [{ name: "bert-baseline", label: "BERT baseline", description: "" }],
    ],
  ]);
  const datasetsByModel = new Map([
    [
      "linears/linear",
      [
        { name: "Mnist", label: "MNIST", inputDim: 784, outputDim: 10 },
        {
          name: "FashionMnist",
          label: "Fashion MNIST",
          inputDim: 784,
          outputDim: 10,
        },
      ],
    ],
    [
      "linears/linear_adaptive",
      [{ name: "Mnist", label: "MNIST", inputDim: 784, outputDim: 10 }],
    ],
    [
      "experts/experts_linear",
      [
        {
          name: "ExpertToy",
          label: "Expert Toy",
          inputDim: 64,
          outputDim: 4,
        },
      ],
    ],
    [
      "transformer_encoder/bert_linear",
      [{ name: "ToyText", label: "Toy Text", inputDim: 128, outputDim: 2 }],
    ],
  ]);

  mocks.fetchModels.mockResolvedValue({
    models: [
      { modelType: "linears", model: "linear" },
      { modelType: "linears", model: "linear_adaptive" },
      { modelType: "experts", model: "experts_linear" },
      { modelType: "transformer_encoder", model: "bert_linear" },
    ],
  });
  mocks.fetchPresets.mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({
      ...identity,
      presets: presetsByModel.get(modelIdentityKey(identity)) ?? [],
    }),
  );
  mocks.fetchDatasets.mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({
      ...identity,
      datasets: datasetsByModel.get(modelIdentityKey(identity)) ?? [],
    }),
  );
  mocks.fetchMonitors.mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({ ...identity, monitors: [] }),
  );
  mocks.fetchConfigSchema.mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({ ...identity, fields: [] }),
  );
  mocks.fetchSearchSpace.mockImplementation((identity: ModelIdentity, preset: string) =>
    Promise.resolve({ ...identity, preset, axes: [] }),
  );
  mocks.inspectModel.mockImplementation(
    (request: { modelType: string; model: string; preset: string }) =>
      Promise.resolve({
        modelType: request.modelType,
        model: request.model,
        preset: request.preset,
        parameterCount: 0,
        parameterSizeBytes: 0,
        nodes: [],
        edges: [],
      }),
  );
}

beforeEach(() => {
  let viewerApiBaseUrl = DEFAULT_VIEWER_API_BASE_URL;
  mocks.normalizeViewerApiBaseUrl.mockReset().mockImplementation((url: string) => {
    const trimmedUrl = url.trim();
    if (!trimmedUrl) {
      return null;
    }
    try {
      const parsedUrl = new URL(trimmedUrl);
      if (parsedUrl.protocol !== "http:" && parsedUrl.protocol !== "https:") {
        return null;
      }
      if (parsedUrl.search || parsedUrl.hash) {
        return null;
      }
    } catch {
      return null;
    }
    return trimmedUrl.replace(/\/+$/, "");
  });
  mocks.getViewerApiBaseUrl.mockReset().mockImplementation(() => viewerApiBaseUrl);
  mocks.setViewerApiBaseUrl.mockReset().mockImplementation((url: string) => {
    const normalizedUrl = mocks.normalizeViewerApiBaseUrl(url);
    if (!normalizedUrl) {
      throw new Error("Invalid API base URL");
    }
    viewerApiBaseUrl = normalizedUrl;
    return viewerApiBaseUrl;
  });
  mocks.resetViewerApiBaseUrl.mockReset().mockImplementation(() => {
    viewerApiBaseUrl = DEFAULT_VIEWER_API_BASE_URL;
    return viewerApiBaseUrl;
  });
  clearPersistedTargetSelection();
  mocks.fetchHealth.mockReset().mockResolvedValue({ status: "ok" });
  mocks.fetchCapabilities.mockReset().mockResolvedValue({
    authMode: "none",
    trainingEnabled: true,
    trainingCancellationCapability: "unsupported",
    logDeletionEnabled: true,
    configSnapshotsEnabled: true,
    historicalLogsEnabled: true,
    liveMonitorDataEnabled: true,
    historicalMonitorDataEnabled: true,
    uploadsEnabled: false,
    maxUploadSize: null,
    dataSourcesEnabled: false,
    dataSources: [],
  });
  mocks.fetchModels.mockReset().mockResolvedValue({
    models: [
      { modelType: "linears", model: "linear" },
      { modelType: "transformer_encoder", model: "bert_linear" },
    ],
  });
  mocks.fetchPresets.mockReset().mockImplementation((identity: ModelIdentity) =>
    Promise.resolve(
      identity.model === "bert_linear"
        ? {
            ...identity,
            presets: [
              { name: "bert-baseline", label: "BERT baseline", description: "" },
            ],
          }
        : {
            modelType: "linears",
            model: "linear",
            presets: [
              { name: "baseline", label: "Baseline", description: "" },
              { name: "fast", label: "Fast", description: "" },
            ],
          },
    ),
  );
  mocks.fetchDatasets.mockReset().mockImplementation((identity: ModelIdentity) =>
    Promise.resolve(
      identity.model === "bert_linear"
        ? {
            ...identity,
            datasets: [
              { name: "ToyText", label: "Toy Text", inputDim: 128, outputDim: 2 },
            ],
          }
        : {
            modelType: "linears",
            model: "linear",
            datasets: [
              { name: "Mnist", label: "MNIST", inputDim: 784, outputDim: 10 },
              {
                name: "FashionMnist",
                label: "Fashion MNIST",
                inputDim: 784,
                outputDim: 10,
              },
            ],
          },
    ),
  );
  mocks.fetchMonitors.mockReset().mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({
      ...identity,
      monitors: [],
    }),
  );
  mocks.fetchConfigSchema.mockReset().mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({
      ...identity,
      fields: [],
    }),
  );
  mocks.fetchSearchSpace.mockReset().mockImplementation((identity: ModelIdentity, preset: string) =>
    Promise.resolve({
      ...identity,
      preset,
      axes: [],
    }),
  );
  mocks.fetchConfigSnapshots.mockReset().mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({
      ...identity,
      snapshots: [],
    }),
  );
  mocks.fetchConfigSnapshotLibrary.mockReset().mockResolvedValue({ snapshots: [] });
  mocks.createConfigSnapshot.mockReset().mockImplementation((input) =>
    Promise.resolve({
      id: "snapshot-1",
      ...input,
      createdAt: "2026-06-01T00:00:00.000Z",
      updatedAt: "2026-06-01T00:00:00.000Z",
    }),
  );
  mocks.renameConfigSnapshot.mockReset().mockImplementation((id: string, name: string) =>
    Promise.resolve({
      id,
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      name,
      overrides: {},
      createdAt: "2026-06-01T00:00:00.000Z",
      updatedAt: "2026-06-01T00:00:00.000Z",
    }),
  );
  mocks.updateConfigSnapshot.mockReset().mockImplementation(
    (
      id: string,
      input: {
        name?: string;
        overrides?: Record<string, string>;
      },
    ) =>
      Promise.resolve({
        id,
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        name: input.name ?? "snapshot",
        overrides: input.overrides ?? {},
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      }),
  );
  mocks.deleteConfigSnapshot.mockReset().mockResolvedValue({
    modelType: "linears",
    model: "linear",
    snapshots: [],
  });
  mocks.fetchLogRuns.mockReset().mockResolvedValue({ runs: [] });
  mocks.fetchLogExperiments.mockReset().mockResolvedValue({ experiments: [] });
  // Default: every requested run carries per-layer monitor data, so its
  // experiment qualifies for the Target Experiments tab.
  mocks.fetchLogTags
    .mockReset()
    .mockImplementation((input: { runIds: string[] }) =>
      Promise.resolve({
        runs: input.runIds.map((runId) => ({
          runId,
          scalarTags: ["main_model.0.model/weights/mean"],
          histogramTags: [],
          imageTags: [],
          textTags: [],
        })),
      }),
    );
  mocks.inspectModel.mockReset().mockResolvedValue({
    model: "linear",
    preset: "baseline",
    parameterCount: 0,
    parameterSizeBytes: 0,
    nodes: [],
    edges: [],
  });
  mocks.fetchTrainingRunPlan.mockReset().mockImplementation((request) =>
    Promise.resolve({
      model: request.model,
      preset: request.preset,
      presets: request.presets,
      datasets: request.datasets,
      overrides: request.overrides,
      search: null,
      logFolder: request.logFolder ?? "",
      isRandomSearch: false,
      runs: [],
      summary: {
        totalRuns: 0,
        completedRuns: 0,
        runningRuns: 0,
        pendingRuns: 0,
        failedRuns: 0,
        cancelledRuns: 0,
        skippedRuns: 0,
        totalEpochs: 0,
        completedEpochs: 0,
        remainingEpochs: 0,
      },
    }),
  );
  mocks.createTrainingJob.mockReset();
  mocks.fetchTrainingJob.mockReset();
  mocks.cancelTrainingJob.mockReset();
  mocks.fetchMonitorParameterStatus.mockReset().mockResolvedValue({
    sourceId: "job-1",
    preset: null,
    dataset: null,
    logDir: null,
    nodes: [],
  });
  mocks.fetchLogParameterStatus.mockReset().mockResolvedValue({ runs: [] });
});

describe("useViewerState", () => {
  it("keeps target and history context identities stable on unrelated rerenders", async () => {
    const { result, rerender } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedTrainingDatasets).toEqual(["Mnist"]);
      expect(result.current.target.trainingSearchAxesLoading).toBe(false);
      expect(result.current.history.experimentsLoading).toBe(false);
    });
    const target = result.current.target;
    const history = result.current.history;

    rerender();

    expect(result.current.target).toBe(target);
    expect(result.current.history).toBe(history);
  });

  it("keeps target and history context identities stable when only the active job changes", async () => {
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.history.experimentsLoading).toBe(false);
    });
    const target = result.current.target;
    const history = result.current.history;

    act(() => {
      result.current.activeJob.onJobChange(trainingJob({ step: 20 }));
    });

    expect(result.current.activeJob.activeTrainingJob?.step).toBe(20);
    expect(result.current.target).toBe(target);
    expect(result.current.history).toBe(history);
  });

  it("uses enabled local defaults while loading capabilities", () => {
    mocks.fetchCapabilities.mockRejectedValueOnce(new Error("capabilities unavailable"));

    const { result } = renderViewerState();

    expect(result.current.target.capabilities).toMatchObject({
      trainingEnabled: true,
      logDeletionEnabled: true,
      uploadsEnabled: true,
    });
  });

  it("surfaces fetched hosted capability flags", async () => {
    mocks.fetchCapabilities.mockResolvedValueOnce({
      authMode: "bearer",
      trainingEnabled: false,
      trainingCancellationCapability: "unsupported",
      logDeletionEnabled: false,
      configSnapshotsEnabled: false,
      historicalLogsEnabled: true,
      liveMonitorDataEnabled: true,
      historicalMonitorDataEnabled: true,
      uploadsEnabled: false,
      maxUploadSize: null,
      dataSourcesEnabled: false,
      dataSources: [],
    });

    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.capabilities).toMatchObject({
        authMode: "bearer",
        trainingEnabled: false,
        logDeletionEnabled: false,
      });
    });
  });

  it("defers historical tag reads on the model workspace until experiment mode is active", async () => {
    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "linear-history",
          preset: "Fast",
          dataset: "FashionMnist",
          timestamp: "2026-06-02 01:02:03",
        }),
      ],
    });
    const { result } = renderViewerState({ activeWorkspace: "model" });

    await waitFor(() => {
      expect(mocks.fetchLogRuns).toHaveBeenCalled();
      expect(result.current.target.selectedModel).toBe("linear");
    });
    expect(mocks.fetchLogTags).not.toHaveBeenCalled();

    act(() => {
      result.current.target.activateTargetExperimentMode();
    });

    await waitFor(() => {
      expect(mocks.fetchLogTags).toHaveBeenCalledWith(
        {
          runIds: ["linear-history"],
        },
        expect.any(Object),
      );
    });
  });

  it("settles the auto-selected training preset without an update loop", async () => {
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    });
  });

  it("auto-selects the first public model type and calls APIs with split identity", async () => {
    mockPublicModelCatalog();

    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModelType).toBe("linears");
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedDatasets).toEqual(["Mnist"]);
    });
    await waitFor(() => {
      expect(mocks.fetchConfigSchema).toHaveBeenCalledWith(
        { modelType: "linears", model: "linear" },
        "baseline",
      );
    });
    expect(mocks.fetchPresets).toHaveBeenCalledWith({
      modelType: "linears",
      model: "linear",
    });
    expect(mocks.fetchDatasets).toHaveBeenCalledWith({
      modelType: "linears",
      model: "linear",
    });
    expect(mocks.fetchMonitors).toHaveBeenCalledWith({
      modelType: "linears",
      model: "linear",
    });
    expect(mocks.fetchSearchSpace).toHaveBeenCalledWith(
      { modelType: "linears", model: "linear" },
      "baseline",
      ["baseline"],
    );
    expect(mocks.inspectModel.mock.calls.map(([request]) => request))
      .toContainEqual({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: {},
      });
  });

  it("selects the first model in a new type through the model reset cascade", async () => {
    mockPublicModelCatalog();
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
    });

    act(() => {
      result.current.target.updateOverride("hidden_size", "128");
    });

    await waitFor(() => {
      expect(result.current.target.overrides).toEqual({ hidden_size: "128" });
    });

    act(() => {
      result.current.target.selectModelType("experts");
    });

    await waitFor(() => {
      expect(result.current.target.selectedModelType).toBe("experts");
      expect(result.current.target.selectedModel).toBe("experts_linear");
      expect(result.current.target.selectedPreset).toBe("expert-baseline");
      expect(result.current.target.selectedDatasets).toEqual(["ExpertToy"]);
      expect(result.current.target.overrides).toEqual({});
    });
    expect(mocks.fetchPresets).toHaveBeenCalledWith({
      modelType: "experts",
      model: "experts_linear",
    });
    expect(mocks.fetchDatasets).toHaveBeenCalledWith({
      modelType: "experts",
      model: "experts_linear",
    });
    expect(mocks.fetchMonitors).toHaveBeenCalledWith({
      modelType: "experts",
      model: "experts_linear",
    });
    expect(mocks.inspectModel.mock.calls.map(([request]) => request))
      .toContainEqual({
        modelType: "experts",
        model: "experts_linear",
        preset: "expert-baseline",
        dataset: "ExpertToy",
        overrides: {},
      });
  });

  it("clears the experts graph and settles on the linear graph when changing model", async () => {
    mockPublicModelCatalog();
    mocks.inspectModel.mockImplementation(
      (request: { modelType: string; model: string; preset: string }) =>
        Promise.resolve(
          request.modelType === "experts" && request.model === "experts_linear"
            ? expertsPreviewGraph(request)
            : linearPreviewGraph(request),
        ),
    );
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.graph.graph).toMatchObject({
        modelType: "linears",
        model: "linear",
      });
    });

    act(() => {
      result.current.target.selectModelType("experts");
    });

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("experts_linear");
      expect(result.current.graph.graph).toMatchObject({
        modelType: "experts",
        model: "experts_linear",
      });
    });
    expect(result.current.graph.graph?.nodes.map((node) => node.typeName))
      .toEqual(expect.arrayContaining(["MixtureOfExperts"]));

    act(() => {
      result.current.target.selectModel("linear", "linears");
    });

    expect(result.current.graph.graph).toBeUndefined();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.graph.graph).toMatchObject({
        modelType: "linears",
        model: "linear",
      });
    });
    const latestNodeTypes =
      result.current.graph.graph?.nodes.map((node) => node.typeName) ?? [];
    expect(latestNodeTypes).toContain("LinearLayer");
    expect(
      latestNodeTypes.some((typeName) => typeName.startsWith("MixtureOfExperts")),
    ).toBe(false);
  });

  it("clears and re-requests the current preview when switching API base URLs", async () => {
    type PreviewRequest = {
      modelType: string;
      model: string;
      preset: string;
    };
    const previewResponses: Array<
      Deferred<InspectResponse> & { request: PreviewRequest }
    > = [];
    mocks.inspectModel.mockReset().mockImplementation((request: PreviewRequest) => {
      const response = deferred<InspectResponse>();
      previewResponses.push({ ...response, request });
      return response.promise;
    });
    const previewGraph = (request: PreviewRequest, label: string): InspectResponse => ({
      modelType: request.modelType,
      model: request.model,
      preset: request.preset,
      parameterCount: 0,
      parameterSizeBytes: 0,
      nodes: [graphNode(label, "model", "PreviewSource", { label })],
      edges: [],
    });
    const { result } = renderViewerState({ activeWorkspace: "model" });

    await waitFor(() => expect(previewResponses).toHaveLength(1));

    act(() => {
      result.current.apiConnection.setApiBaseUrl("https://api-alt.example.test");
    });

    expect(result.current.graph.graph).toBeUndefined();
    await waitFor(() => expect(previewResponses.length).toBeGreaterThanOrEqual(2));
    expect(previewResponses[1]?.request).toEqual(previewResponses[0]?.request);

    const firstPreviewResponse = previewResponses[0];
    if (!firstPreviewResponse) {
      throw new Error("Expected an initial preview response");
    }
    await act(async () => {
      firstPreviewResponse.resolve(
        previewGraph(firstPreviewResponse.request, "old backend"),
      );
      await firstPreviewResponse.promise;
    });
    expect(result.current.graph.graph).toBeUndefined();

    const latestPreviewResponse = previewResponses.at(-1);
    if (!latestPreviewResponse) {
      throw new Error("Expected a re-requested preview response");
    }
    await act(async () => {
      latestPreviewResponse.resolve(
        previewGraph(latestPreviewResponse.request, "new backend"),
      );
      await latestPreviewResponse.promise;
    });

    await waitFor(() => {
      expect(result.current.apiConnection.apiBaseUrl).toBe(
        "https://api-alt.example.test",
      );
      expect(result.current.graph.graph?.nodes[0]?.label).toBe("new backend");
    });
  });

  it("clears historical run selection when switching model type", async () => {
    mockPublicModelCatalog();
    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "linears-history",
          modelType: "linears",
          model: "linear",
          preset: "Fast",
          dataset: "FashionMnist",
        }),
      ],
    });
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.history.visibleHistoricalRuns.map((run) => run.id))
        .toEqual(["linears-history"]);
    });

    act(() => {
      result.current.history.selectLogRun("linears-history");
    });

    await waitFor(() => {
      expect(result.current.history.selectedLogRunId).toBe("linears-history");
      expect(result.current.target.selectedTargetMode).toBe("experiment");
      expect(result.current.target.selectedExperimentRunId).toBe("linears-history");
      expect(result.current.target.selectedPreset).toBe("fast");
    });

    act(() => {
      result.current.target.selectModelType("experts");
    });

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("experts_linear");
      expect(result.current.history.selectedLogRunId).toBeNull();
      expect(result.current.history.selectedHistoricalExperimentFilter).toBe("");
      expect(result.current.history.selectedHistoricalDatasetFilter).toBe("");
      expect(result.current.history.selectedHistoricalPreset).toBe("");
      expect(result.current.target.selectedTargetMode).toBe("preset");
      expect(result.current.target.selectedExperimentRunId).toBe("");
    });
  });

  it("settles model changes on the new model defaults", async () => {
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedDatasets).toEqual(["Mnist"]);
    });

    act(() => {
      result.current.target.selectModel("bert_linear", "transformer_encoder");
    });

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("bert_linear");
      expect(result.current.target.selectedPreset).toBe("bert-baseline");
      expect(result.current.target.selectedDatasets).toEqual(["ToyText"]);
      expect(result.current.target.selectedTrainingModel).toBe("linear");
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
      expect(result.current.target.selectedTrainingDatasets).toEqual(["Mnist"]);
    });
    expect(mocks.inspectModel.mock.calls.map(([request]) => request)).toContainEqual({
      modelType: "transformer_encoder",
      model: "bert_linear",
      preset: "bert-baseline",
      dataset: "ToyText",
      overrides: {},
    });
  });

  it("resets graph selection and expansion when selecting another model", async () => {
    mocks.inspectModel.mockResolvedValue(monitorGraph());
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.graph.graph?.nodes.map((node) => node.id)).toContain(
        "linear-0",
      );
    });

    act(() => {
      result.current.graph.revealGraphNode("linear-0");
    });

    await waitFor(() => {
      expect(result.current.graph.selectedNodeId).toBe("linear-0");
      expect(result.current.graph.expandedGraphNodeIds.size).toBeGreaterThan(0);
    });

    act(() => {
      result.current.target.selectModel("bert_linear", "transformer_encoder");
    });

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("bert_linear");
      expect(result.current.graph.selectedNodeId).toBeNull();
      expect(result.current.graph.expandedGraphNodeIds.size).toBe(0);
    });
  });

  it("does not auto-select a run, then syncs target config once when one is picked", async () => {
    mocks.fetchModels.mockResolvedValueOnce({
      models: [
        { modelType: "transformer_encoder", model: "bert_linear" },
        { modelType: "linears", model: "linear" },
      ],
    });
    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "linear-history",
          preset: "Fast",
          dataset: "FashionMnist",
          timestamp: "2026-06-02 01:02:03",
        }),
      ],
    });
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("bert_linear");
      expect(result.current.target.selectedPreset).toBe("bert-baseline");
      expect(result.current.target.selectedDatasets).toEqual(["ToyText"]);
    });

    act(() => {
      result.current.target.selectModel("linear", "linears");
    });

    // The run becomes visible but nothing is auto-selected.
    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(
        result.current.history.visibleHistoricalRuns.map((run) => run.id),
      ).toEqual(["linear-history"]);
    });
    expect(result.current.history.selectedLogRunId).toBeNull();

    mocks.inspectModel.mockClear();
    act(() => {
      result.current.history.selectLogRun("linear-history");
    });

    await waitFor(() => {
      expect(result.current.history.selectedLogRunId).toBe("linear-history");
      expect(result.current.target.selectedTargetMode).toBe("experiment");
      expect(result.current.target.selectedExperimentRunId).toBe("linear-history");
      expect(result.current.target.selectedExperimentPreset).toBe("Fast");
      expect(result.current.target.selectedExperimentDataset).toBe("FashionMnist");
      expect(result.current.target.selectedPreset).toBe("fast");
      expect(result.current.target.selectedDatasets).toEqual(["FashionMnist"]);
      expect(result.current.target.selectedTrainingPresets).toEqual(["bert-baseline"]);
      expect(result.current.target.selectedTrainingDatasets).toEqual(["ToyText"]);
      expect(result.current.target.overrides).toEqual({});
    });

    const finalHistoricalRequests = mocks.inspectModel.mock.calls.filter(
      ([request]) =>
        request.modelType === "linears" &&
        request.model === "linear" &&
        request.preset === "Fast" &&
        request.dataset === "FashionMnist" &&
        request.logRunId === "linear-history",
    );
    expect(finalHistoricalRequests).toHaveLength(1);

    const requestCount = mocks.inspectModel.mock.calls.length;
    act(() => {
      result.current.history.selectLogRun("linear-history");
    });
    expect(result.current.history.selectedLogRunId).toBe("linear-history");
    expect(result.current.target.selectedTargetMode).toBe("experiment");
    expect(mocks.inspectModel.mock.calls.length).toBe(requestCount);
  });

  it("selects the newest matching run from the experiment dataset preset cascade", async () => {
    mocks.inspectModel.mockResolvedValue(monitorGraph());
    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "baseline-old",
          experiment: "exp_linear",
          preset: "baseline",
          dataset: "Mnist",
          timestamp: "2026-06-01 01:02:03",
        }),
        logRun({
          id: "baseline-new",
          experiment: "exp_linear",
          preset: "baseline",
          dataset: "Mnist",
          timestamp: "2026-06-03 01:02:03",
        }),
        logRun({
          id: "fast-history",
          experiment: "exp_linear",
          preset: "fast",
          dataset: "Mnist",
          timestamp: "2026-06-02 01:02:03",
        }),
        logRun({
          id: "fashion-history",
          experiment: "exp_linear",
          preset: "fast",
          dataset: "FashionMnist",
          timestamp: "2026-06-04 01:02:03",
        }),
      ],
    });
    mocks.fetchLogParameterStatus.mockImplementation(
      (input: { runIds: string[] }) =>
        Promise.resolve({
          runs: input.runIds.map((runId) => ({
            sourceId: runId,
            preset: "baseline",
            dataset: "Mnist",
            logDir: `logs/${runId}`,
            nodes: [
              {
                nodePath: "main_model.0.model",
                weights: {
                  status: runId === "baseline-new" ? "updated" : "unchanged",
                  metric: "main_model.0.model/weights/relative_delta_norm",
                  lastStep: 12,
                  observedPoints: 2,
                },
                bias: {
                  status: "unchanged",
                  metric: "main_model.0.model/bias/delta_norm",
                  lastStep: 12,
                  observedPoints: 1,
                },
              },
            ],
          })),
        }),
    );
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.history.visibleHistoricalRuns.map((run) => run.id))
        .toEqual([
          "fashion-history",
          "baseline-new",
          "fast-history",
          "baseline-old",
        ]);
      expect(result.current.history.historicalExperimentOptions).toEqual([
        { value: "exp_linear", label: "exp_linear", count: 4 },
      ]);
    });

    act(() => {
      result.current.history.setSelectedHistoricalExperimentFilter("exp_linear");
    });

    await waitFor(() => {
      expect(result.current.history.visibleHistoricalRuns.map((run) => run.id))
        .toEqual([
          "fashion-history",
          "baseline-new",
          "fast-history",
          "baseline-old",
        ]);
      expect(result.current.history.selectedHistoricalDatasetFilter).toBe("");
      expect(result.current.history.selectedHistoricalPreset).toBe("");
      expect(result.current.history.selectedLogRunId).toBeNull();
      expect(result.current.history.historicalDatasetOptions).toEqual([
        { value: "FashionMnist", label: "FashionMnist", count: 1 },
        { value: "Mnist", label: "Mnist", count: 3 },
      ]);
    });

    act(() => {
      result.current.history.setSelectedHistoricalDatasetFilter("Mnist");
    });

    await waitFor(() => {
      expect(result.current.history.visibleHistoricalRuns.map((run) => run.id))
        .toEqual(["baseline-new", "fast-history", "baseline-old"]);
      expect(result.current.history.selectedHistoricalPreset).toBe("");
      expect(result.current.history.selectedLogRunId).toBeNull();
      expect(result.current.history.historicalPresetOptions).toEqual([
        { value: "baseline", label: "baseline", count: 2 },
        { value: "fast", label: "fast", count: 1 },
      ]);
    });

    mocks.inspectModel.mockClear();
    mocks.fetchLogTags.mockClear();
    mocks.fetchLogParameterStatus.mockClear();
    act(() => {
      result.current.history.setSelectedHistoricalPreset("baseline");
    });

    await waitFor(() => {
      expect(result.current.history.selectedLogRunId).toBe("baseline-new");
      expect(result.current.target.selectedTargetMode).toBe("experiment");
      expect(result.current.target.selectedExperimentRunId).toBe("baseline-new");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
      expect(result.current.target.selectedDatasets).toEqual(["Mnist"]);
      expect(result.current.history.historicalMonitorRuns.map((run) => run.id))
        .toEqual(["baseline-new", "baseline-old"]);
    });
    await waitFor(() => {
      expect(mocks.fetchLogTags).toHaveBeenCalledWith({
        runIds: ["baseline-new", "baseline-old"],
      }, expect.any(Object));
      expect(mocks.fetchLogParameterStatus).toHaveBeenCalledWith({
        runIds: ["baseline-new"],
      }, expect.any(Object));
      expect(mocks.fetchLogParameterStatus).toHaveBeenCalledWith({
        runIds: ["baseline-old"],
      }, expect.any(Object));
    });
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalledWith({
      runIds: ["fast-history"],
    }, expect.any(Object));
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalledWith({
      runIds: ["fashion-history"],
    }, expect.any(Object));
    expect(result.current.graphMonitor.graphMonitorSource?.kind)
      .toBe("historical-run-group");
    expect(
      result.current.graphMonitor.graphMonitorSource?.kind ===
        "historical-run-group"
        ? result.current.graphMonitor.graphMonitorSource.runs.map((run) => run.id)
        : [],
    ).toEqual(["baseline-new", "baseline-old"]);
    await waitFor(() => {
      expect(result.current.graph.nodes.map((node) => node.id)).toContain(
        "layer-0",
      );
    });
    await waitFor(() => {
      const layerNode = result.current.graph.nodes.find(
        (node) => node.id === "layer-0",
      );
      expect(layerNode?.data.parameterActivity).toMatchObject({
        targetPath: "main_model.0.model",
        weights: {
          status: "mixed",
          source: "historical",
          totalRuns: 2,
        },
      });
    });
    expect(result.current.history.visibleHistoricalRuns.map((run) => run.id))
      .toEqual(["baseline-new", "baseline-old"]);
    expect(mocks.inspectModel.mock.calls.map(([request]) => request))
      .toContainEqual({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: {},
        logRunId: "baseline-new",
      });

    mocks.fetchLogTags.mockClear();
    mocks.fetchLogParameterStatus.mockClear();
    act(() => {
      result.current.history.setSelectedHistoricalDatasetFilter("FashionMnist");
    });

    await waitFor(() => {
      expect(result.current.history.selectedHistoricalPreset).toBe("");
      expect(result.current.history.selectedLogRunId).toBeNull();
      expect(result.current.target.selectedExperimentRunId).toBe("");
      expect(result.current.history.visibleHistoricalRuns.map((run) => run.id))
        .toEqual(["fashion-history"]);
    });
    expect(result.current.history.historicalMonitorRuns).toEqual([]);
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalled();

    act(() => {
      result.current.history.setSelectedHistoricalPreset("fast");
    });

    await waitFor(() => {
      expect(result.current.history.selectedLogRunId).toBe("fashion-history");
      expect(result.current.target.selectedExperimentRunId).toBe("fashion-history");
      expect(result.current.target.selectedDatasets).toEqual(["FashionMnist"]);
      expect(result.current.history.historicalMonitorRuns.map((run) => run.id))
        .toEqual(["fashion-history"]);
      expect(mocks.fetchLogTags).toHaveBeenCalledWith({
        runIds: ["fashion-history"],
      }, expect.any(Object));
      expect(mocks.fetchLogParameterStatus).toHaveBeenCalledWith({
        runIds: ["fashion-history"],
      }, expect.any(Object));
    });
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalledWith({
      runIds: ["baseline-new"],
    }, expect.any(Object));
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalledWith({
      runIds: ["baseline-old"],
    }, expect.any(Object));
  });

  it("adds parameter activity after selecting a non-current experiment preset through the cascade", async () => {
    mocks.inspectModel.mockImplementation(
      (request: { modelType: string; model: string; preset: string }) =>
        Promise.resolve({
          ...monitorGraph(),
          modelType: request.modelType,
          model: request.model,
          preset: request.preset,
        }),
    );
    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "baseline-run",
          experiment: "exp_linear",
          preset: "baseline",
          dataset: "Mnist",
          timestamp: "2026-06-01 01:02:03",
        }),
        logRun({
          id: "fast-run",
          experiment: "exp_linear",
          preset: "fast",
          dataset: "Mnist",
          timestamp: "2026-06-02 01:02:03",
        }),
      ],
    });
    mocks.fetchLogParameterStatus.mockImplementation(
      (input: { runIds: string[] }) =>
        Promise.resolve({
          runs: input.runIds.map((runId) => ({
            sourceId: runId,
            preset: "fast",
            dataset: "Mnist",
            logDir: `logs/${runId}`,
            nodes: [
              {
                nodePath: "main_model.0.model",
                weights: {
                  status: "updated",
                  metric: "main_model.0.model/weights/relative_delta_norm",
                  lastStep: 12,
                  observedPoints: 2,
                },
                bias: {
                  status: "unchanged",
                  metric: "main_model.0.model/bias/delta_norm",
                  lastStep: 12,
                  observedPoints: 1,
                },
              },
            ],
          })),
        }),
    );
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.history.historicalExperimentOptions).toEqual([
        { value: "exp_linear", label: "exp_linear", count: 2 },
      ]);
    });

    act(() => {
      result.current.history.setSelectedHistoricalExperimentFilter("exp_linear");
    });
    await waitFor(() => {
      expect(result.current.history.historicalDatasetOptions).toEqual([
        { value: "Mnist", label: "Mnist", count: 2 },
      ]);
    });

    act(() => {
      result.current.history.setSelectedHistoricalDatasetFilter("Mnist");
    });
    await waitFor(() => {
      expect(result.current.history.historicalPresetOptions).toEqual([
        { value: "fast", label: "fast", count: 1 },
        { value: "baseline", label: "baseline", count: 1 },
      ]);
    });

    act(() => {
      result.current.history.setSelectedHistoricalPreset("fast");
    });

    await waitFor(() => {
      expect(result.current.history.selectedLogRunId).toBe("fast-run");
      expect(result.current.target.selectedTargetMode).toBe("experiment");
      expect(result.current.target.selectedExperimentRunId).toBe("fast-run");
      expect(result.current.target.selectedPreset).toBe("fast");
      expect(result.current.history.historicalMonitorRuns.map((run) => run.id))
        .toEqual(["fast-run"]);
    });
    await waitFor(() => {
      expect(mocks.fetchLogParameterStatus).toHaveBeenCalledWith(
        { runIds: ["fast-run"] },
        expect.any(Object),
      );
    });
    await waitFor(() => {
      const layerNode = result.current.graph.nodes.find(
        (node) => node.id === "layer-0",
      );
      expect(layerNode?.data.parameterActivity).toMatchObject({
        targetPath: "main_model.0.model",
        weights: {
          status: "updated",
          source: "historical",
          totalRuns: 1,
        },
      });
    });
  });

  it("keeps an experiment switch pending and shows loading parameter activity until status resolves", async () => {
    const statusB = deferred<ReturnType<typeof parameterStatus>>();

    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "run-a",
          experiment: "exp_a",
          preset: "baseline",
          dataset: "Mnist",
          timestamp: "2026-06-01 01:02:03",
        }),
        logRun({
          id: "run-b",
          experiment: "exp_b",
          preset: "baseline",
          dataset: "Mnist",
          timestamp: "2026-06-02 01:02:03",
        }),
      ],
    });
    mocks.fetchLogParameterStatus.mockImplementation(
      (input: { runIds: string[] }) => {
        if (input.runIds.length === 1 && input.runIds[0] === "run-b") {
          return statusB.promise;
        }
        return Promise.resolve(parameterStatus(input.runIds));
      },
    );
    mocks.inspectModel.mockImplementation(
      (request: {
        modelType: string;
        model: string;
        preset: string;
        logRunId?: string;
      }) =>
        Promise.resolve(
          experimentMonitorGraph(
            request,
            request.logRunId === "run-b" ? "b" : "a",
          ),
        ),
    );

    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.history.historicalExperimentOptions).toEqual([
        { value: "exp_b", label: "exp_b", count: 1 },
        { value: "exp_a", label: "exp_a", count: 1 },
      ]);
    });

    act(() => {
      result.current.history.setSelectedHistoricalExperimentFilter("exp_a");
    });
    await waitFor(() => {
      expect(result.current.history.historicalDatasetOptions).toEqual([
        { value: "Mnist", label: "Mnist", count: 1 },
      ]);
    });
    act(() => {
      result.current.history.setSelectedHistoricalDatasetFilter("Mnist");
    });
    await waitFor(() => {
      expect(result.current.history.historicalPresetOptions).toEqual([
        { value: "baseline", label: "baseline", count: 1 },
      ]);
    });
    act(() => {
      result.current.history.setSelectedHistoricalPreset("baseline");
    });

    await waitFor(() => {
      const layerNode = result.current.graph.nodes.find(
        (node) => node.id === "a-layer",
      );
      expect(layerNode?.data.parameterActivity).toMatchObject({
        weights: { status: "updated" },
        bias: { status: "updated" },
      });
    });

    mocks.inspectModel.mockClear();
    act(() => {
      result.current.history.setSelectedHistoricalExperimentFilter("exp_b");
    });

    await waitFor(() => {
      expect(result.current.target.selectedTargetMode).toBe("experiment");
      expect(result.current.target.selectedExperimentRunId).toBe("");
      expect(result.current.history.historicalDatasetOptions).toEqual([
        { value: "Mnist", label: "Mnist", count: 1 },
      ]);
    });
    expect(mocks.inspectModel).not.toHaveBeenCalled();
    expect(result.current.graph.graph).toBeUndefined();

    act(() => {
      result.current.history.setSelectedHistoricalDatasetFilter("Mnist");
    });
    await waitFor(() => {
      expect(result.current.history.historicalPresetOptions).toEqual([
        { value: "baseline", label: "baseline", count: 1 },
      ]);
    });
    act(() => {
      result.current.history.setSelectedHistoricalPreset("baseline");
    });

    await waitFor(() => {
      expect(result.current.target.selectedTargetMode).toBe("experiment");
      expect(result.current.target.selectedExperimentRunId).toBe("run-b");
      expect(result.current.graph.nodes.map((node) => node.id)).toContain(
        "b-layer",
      );
    });
    expect(result.current.graph.isParameterStatusLoading).toBe(true);
    expect(
      result.current.graph.nodes.find((node) => node.id === "b-layer")
        ?.data.parameterActivity,
    ).toMatchObject({
      weights: { status: "loading" },
      bias: { status: "loading" },
    });

    act(() => {
      statusB.resolve(parameterStatus(["run-b"]));
    });

    await waitFor(() => {
      expect(result.current.graph.isParameterStatusLoading).toBe(false);
      expect(
        result.current.graph.nodes.find((node) => node.id === "b-layer")
          ?.data.parameterActivity,
      ).toMatchObject({
        weights: { status: "updated" },
        bias: { status: "updated" },
      });
    });
  });

  it("omits experiments that only have classifier or global metric tags", async () => {
    const statusForRun = (
      runId: string,
      weightsStatus: "updated" | "unchanged",
      biasStatus: "updated" | "unchanged",
    ): LogParameterStatusResponse => ({
      runs: [
        {
          sourceId: runId,
          preset: "baseline",
          dataset: "Mnist",
          logDir: `logs/${runId}`,
          nodes: [
            {
              nodePath: "main_model.0.model",
              weights: {
                status: weightsStatus,
                metric: "main_model.0.model/weights/test_delta_norm",
                lastStep: 12,
                observedPoints: 2,
              },
              bias: {
                status: biasStatus,
                metric: "main_model.0.model/bias/test_delta_norm",
                lastStep: 12,
                observedPoints: 2,
              },
            },
          ],
        },
      ],
    });

    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "test-linear-run",
          experiment: "test_linear",
          preset: "baseline",
          dataset: "Mnist",
          timestamp: "2026-06-01 01:02:03",
        }),
        logRun({
          id: "kaggle-linear-run",
          experiment: "kaggle_linear_all",
          preset: "KAGGLE_LINEAR",
          dataset: "KaggleDigits",
          timestamp: "2026-06-02 01:02:03",
        }),
      ],
    });
    mocks.fetchLogTags.mockImplementation((input: { runIds: string[] }) =>
      Promise.resolve({
        runs: [
          ...parameterLogTags(
            input.runIds.filter((runId) => runId !== "kaggle-linear-run"),
          ).runs,
          ...performanceLogTags(
            input.runIds.filter((runId) => runId === "kaggle-linear-run"),
          ).runs,
        ],
      }),
    );
    mocks.fetchLogParameterStatus.mockImplementation(
      (input: { runIds: string[] }) =>
        Promise.resolve(
          statusForRun(input.runIds[0] ?? "test-linear-run", "updated", "unchanged"),
        ),
    );
    mocks.inspectModel.mockImplementation(
      (request: {
        modelType: string;
        model: string;
        preset: string;
        logRunId?: string;
      }) =>
        Promise.resolve(
          experimentMonitorGraph(
            request,
            "test",
          ),
        ),
    );

    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
    });

    act(() => {
      result.current.target.activateTargetExperimentMode();
    });

    await waitFor(() => {
      expect(result.current.history.historicalExperimentOptions).toEqual([
        { value: "test_linear", label: "test_linear", count: 1 },
      ]);
    });
    expect(result.current.history.historicalExperimentOptions).not.toContainEqual(
      { value: "kaggle_linear_all", label: "kaggle_linear_all", count: 1 },
    );

    act(() => {
      result.current.history.setSelectedHistoricalExperimentFilter("test_linear");
    });
    await waitFor(() => {
      expect(result.current.history.historicalDatasetOptions).toEqual([
        { value: "Mnist", label: "Mnist", count: 1 },
      ]);
    });
    act(() => {
      result.current.history.setSelectedHistoricalDatasetFilter("Mnist");
    });
    await waitFor(() => {
      expect(result.current.history.historicalPresetOptions).toEqual([
        { value: "baseline", label: "baseline", count: 1 },
      ]);
    });
    act(() => {
      result.current.history.setSelectedHistoricalPreset("baseline");
    });

    await waitFor(() => {
      expect(result.current.target.selectedExperimentRunId).toBe("test-linear-run");
      expect(result.current.graph.nodes.map((node) => node.id)).toContain(
        "test-layer",
      );
    });
    await waitFor(() => {
      const layerNode = result.current.graph.nodes.find(
        (node) => node.id === "test-layer",
      );
      expect(layerNode?.data.parameterActivity).toMatchObject({
        weights: {
          status: "updated",
          metric: "main_model.0.model/weights/test_delta_norm",
        },
        bias: {
          status: "unchanged",
          metric: "main_model.0.model/bias/test_delta_norm",
        },
      });
    });

    mocks.inspectModel.mockClear();
    mocks.fetchLogParameterStatus.mockClear();
    act(() => {
      result.current.history.setSelectedHistoricalExperimentFilter(
        "kaggle_linear_all",
      );
    });

    await waitFor(() => {
      expect(result.current.target.selectedTargetMode).toBe("experiment");
      expect(result.current.target.selectedExperimentRunId).toBe("");
      expect(result.current.history.selectedHistoricalExperimentFilter).toBe("");
      expect(result.current.history.historicalDatasetOptions).toEqual([]);
      expect(result.current.history.visibleHistoricalRuns.map((run) => run.id))
        .toEqual(["test-linear-run"]);
    });
    expect(mocks.inspectModel).not.toHaveBeenCalled();
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalled();
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalledWith(
      { runIds: ["kaggle-linear-run"] },
      expect.any(Object),
    );
  });

  it("switches from an experiment target back to a preset target with empty overrides", async () => {
    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "linear-history",
          preset: "Fast",
          dataset: "FashionMnist",
          timestamp: "2026-06-02 01:02:03",
        }),
      ],
    });
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.history.visibleHistoricalRuns.map((run) => run.id))
        .toEqual(["linear-history"]);
    });

    act(() => {
      result.current.history.selectLogRun("linear-history");
    });
    await waitFor(() => {
      expect(result.current.target.selectedTargetMode).toBe("experiment");
      expect(result.current.history.selectedLogRunId).toBe("linear-history");
    });

    mocks.inspectModel.mockClear();
    act(() => {
      result.current.target.selectTargetPreset("baseline");
    });

    await waitFor(() => {
      expect(result.current.target.selectedTargetMode).toBe("preset");
      expect(result.current.target.selectedExperimentRunId).toBe("");
      expect(result.current.history.selectedLogRunId).toBeNull();
      expect(result.current.target.selectedSnapshotId).toBe("");
      expect(result.current.target.overrides).toEqual({});
    });
	    expect(mocks.inspectModel.mock.calls.at(-1)?.[0]).toEqual({
	      modelType: "linears",
	      model: "linear",
      preset: "baseline",
      dataset: "FashionMnist",
      overrides: {},
    });
  });

  it("switches from an experiment target to a snapshot target with saved overrides", async () => {
    mocks.fetchConfigSnapshots.mockResolvedValue({
      modelType: "linears",
      model: "linear",
      snapshots: [
        {
          id: "snapshot-wide",
          modelType: "linears",
          model: "linear",
          preset: "baseline",
          name: "Wide",
          overrides: { hidden_size: "256" },
          createdAt: "2026-06-01T00:00:00.000Z",
          updatedAt: "2026-06-01T00:00:00.000Z",
        },
      ],
    });
    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "linear-history",
          preset: "Fast",
          dataset: "FashionMnist",
          timestamp: "2026-06-02 01:02:03",
        }),
      ],
    });
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.allConfigSnapshotCount).toBe(1);
      expect(result.current.history.visibleHistoricalRuns.map((run) => run.id))
        .toEqual(["linear-history"]);
    });

    act(() => {
      result.current.history.selectLogRun("linear-history");
    });
    await waitFor(() => {
      expect(result.current.target.selectedTargetMode).toBe("experiment");
      expect(result.current.history.selectedLogRunId).toBe("linear-history");
    });

    mocks.inspectModel.mockClear();
    act(() => {
      expect(result.current.target.selectTargetSnapshot("snapshot-wide")).toBe(true);
    });

    await waitFor(() => {
      expect(result.current.target.selectedTargetMode).toBe("snapshot");
      expect(result.current.target.selectedSnapshotId).toBe("snapshot-wide");
      expect(result.current.target.selectedExperimentRunId).toBe("");
      expect(result.current.history.selectedLogRunId).toBeNull();
      expect(result.current.target.overrides).toEqual({ hidden_size: "256" });
    });
	    expect(mocks.inspectModel.mock.calls.at(-1)?.[0]).toEqual({
	      modelType: "linears",
	      model: "linear",
      preset: "baseline",
      dataset: "FashionMnist",
      overrides: { hidden_size: "256" },
    });
  });

  it("updates the selected config snapshot without detaching snapshot mode", async () => {
    mocks.fetchConfigSnapshots.mockResolvedValue({
      modelType: "linears",
      model: "linear",
      snapshots: [
        {
          id: "snapshot-wide",
          modelType: "linears",
          model: "linear",
          preset: "baseline",
          name: "Wide",
          overrides: { hidden_size: "256" },
          createdAt: "2026-06-01T00:00:00.000Z",
          updatedAt: "2026-06-01T00:00:00.000Z",
        },
      ],
    });
    mocks.fetchConfigSchema.mockResolvedValue({
      modelType: "linears",
      model: "linear",
      fields: [
        {
          key: "hidden_size",
          configKey: "HIDDEN_SIZE",
          flag: "--hidden-size",
          label: "Hidden size",
          section: "Model",
          type: "int",
          default: 64,
          nullable: false,
          choices: [],
        },
      ],
    });
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.allConfigSnapshotCount).toBe(1);
    });

    act(() => {
      expect(result.current.target.selectTargetSnapshot("snapshot-wide")).toBe(true);
    });

    await waitFor(() => {
      expect(result.current.target.selectedTargetMode).toBe("snapshot");
      expect(result.current.target.selectedSnapshotId).toBe("snapshot-wide");
      expect(result.current.target.selectedConfigSnapshot?.name).toBe("Wide");
    });

    act(() => {
      result.current.target.updateSnapshotEditorDraftOverride("hidden_size", "512");
    });

    expect(result.current.target.selectedTargetMode).toBe("snapshot");
    expect(result.current.target.selectedSnapshotId).toBe("snapshot-wide");
    expect(result.current.target.overrides).toEqual({ hidden_size: "512" });

    act(() => {
      const resultValue =
        result.current.target.updateSelectedConfigSnapshot("Wide edited");
      expect(resultValue.ok).toBe(true);
    });

    await waitFor(() => {
      expect(mocks.updateConfigSnapshot).toHaveBeenCalledWith("snapshot-wide", {
        name: "Wide edited",
        overrides: { hidden_size: "512" },
      });
    });
    expect(mocks.createConfigSnapshot).not.toHaveBeenCalled();
    expect(result.current.target.selectedTargetMode).toBe("snapshot");
    expect(result.current.target.selectedSnapshotId).toBe("snapshot-wide");
  });

  it("requests parameter status for the active linear training job", async () => {
    mocks.inspectModel.mockResolvedValue(monitorGraph());
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
    });

    act(() => {
      result.current.activeJob.onJobChange(trainingJob());
    });

    await waitFor(() => {
      expect(mocks.fetchMonitorParameterStatus).toHaveBeenCalledWith({
        jobId: "job-1",
        preset: "baseline",
        dataset: "Mnist",
      }, expect.any(Object));
    });
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalled();
  });

  it("opens graph monitor charts from the active training job source", async () => {
    mocks.inspectModel.mockResolvedValue(monitorGraph());
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.graph.graph?.nodes.map((node) => node.id)).toContain(
        "linear-0",
      );
    });
    const linearNode = result.current.graph.graph?.nodes.find(
      (node) => node.id === "linear-0",
    );
    expect(linearNode).toBeDefined();
    const job = trainingJob();

    act(() => {
      result.current.activeJob.onJobChange(job);
    });
    act(() => {
      result.current.graphMonitor.openGraphNodeMonitor(linearNode as GraphNode);
    });

    await waitFor(() => {
      expect(result.current.graphMonitor.graphMonitorNode?.id).toBe("linear-0");
      expect(result.current.graphMonitor.graphMonitorSource).toEqual({
        kind: "active-job",
        job: activeMonitorJob(job),
      });
    });
  });

  it("keeps graph monitor identity stable across active job progress updates", async () => {
    mocks.inspectModel.mockResolvedValue(monitorGraph());
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.graph.graph?.nodes.map((node) => node.id)).toContain(
        "linear-0",
      );
    });
    const linearNode = result.current.graph.graph?.nodes.find(
      (node) => node.id === "linear-0",
    );
    expect(linearNode).toBeDefined();

    act(() => {
      result.current.activeJob.onJobChange(
        trainingJob({ epoch: 1, step: 10, metrics: { loss: 1 } }),
      );
    });
    act(() => {
      result.current.graphMonitor.openGraphNodeMonitor(linearNode as GraphNode);
    });

    await waitFor(() => {
      expect(result.current.graphMonitor.graphMonitorSource).toEqual({
        kind: "active-job",
        job: activeMonitorJob(trainingJob({ epoch: 1, step: 10 })),
      });
    });
    const activeJob = result.current.activeJob;
    const graphMonitor = result.current.graphMonitor;
    const graphMonitorSource = result.current.graphMonitor.graphMonitorSource;

    act(() => {
      result.current.activeJob.onJobChange(
        trainingJob({ epoch: 2, step: 20, metrics: { loss: 0.5 } }),
      );
    });

    expect(result.current.activeJob).not.toBe(activeJob);
    expect(result.current.activeJob.activeTrainingJob?.step).toBe(20);
    expect(result.current.graphMonitor).toBe(graphMonitor);
    expect(result.current.graphMonitor.graphMonitorSource).toBe(graphMonitorSource);
  });

  it("requests historical parameter status for the selected monitor run group", async () => {
    mocks.inspectModel.mockResolvedValue(monitorGraph());
    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "run-new",
          preset: "baseline",
          dataset: "Mnist",
          timestamp: "2026-06-02 01:02:03",
        }),
        logRun({
          id: "run-old",
          preset: "baseline",
          dataset: "Mnist",
          timestamp: "2026-06-01 01:02:03",
        }),
        logRun({
          id: "run-fast",
          preset: "fast",
          dataset: "Mnist",
          timestamp: "2026-06-01 12:02:03",
        }),
      ],
    });
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.history.visibleHistoricalRuns.map((run) => run.id))
        .toEqual(["run-new", "run-fast", "run-old"]);
    });

    act(() => {
      result.current.history.selectLogRun("run-new");
    });

    await waitFor(() => {
      expect(result.current.history.historicalMonitorRuns.map((run) => run.id))
        .toEqual(["run-new", "run-old"]);
      expect(mocks.fetchLogParameterStatus).toHaveBeenCalledWith({
        runIds: ["run-new"],
      }, expect.any(Object));
      expect(mocks.fetchLogParameterStatus).toHaveBeenCalledWith({
        runIds: ["run-old"],
      }, expect.any(Object));
    });
    expect(result.current.graphMonitor.graphMonitorSource?.kind)
      .toBe("historical-run-group");
    expect(
      result.current.graphMonitor.graphMonitorSource?.kind ===
        "historical-run-group"
        ? result.current.graphMonitor.graphMonitorSource.runs.map((run) => run.id)
        : [],
    ).toEqual(["run-new", "run-old"]);
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalledWith({
      runIds: ["run-fast"],
    }, expect.any(Object));
    expect(mocks.fetchMonitorParameterStatus).not.toHaveBeenCalled();
  });

  it("keeps the final experiment graph and monitor controls after stale switch responses resolve", async () => {
    const oldInspect = deferred<InspectResponse>();
    const finalInspect = deferred<InspectResponse>();
    const oldTags = deferred<ReturnType<typeof parameterLogTags>>();
    const finalTags = deferred<ReturnType<typeof parameterLogTags>>();
    const oldStatus = deferred<ReturnType<typeof parameterStatus>>();
    const finalStatus = deferred<ReturnType<typeof parameterStatus>>();

    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "run-old",
          experiment: "exp_old",
          preset: "baseline",
          dataset: "Mnist",
          timestamp: "2026-06-01 01:02:03",
        }),
        logRun({
          id: "run-final",
          experiment: "exp_final",
          preset: "baseline",
          dataset: "Mnist",
          timestamp: "2026-06-02 01:02:03",
        }),
      ],
    });
    mocks.fetchLogTags.mockImplementation((input: { runIds: string[] }) => {
      if (input.runIds.length === 1 && input.runIds[0] === "run-old") {
        return oldTags.promise;
      }
      if (input.runIds.length === 1 && input.runIds[0] === "run-final") {
        return finalTags.promise;
      }
      return Promise.resolve(parameterLogTags(input.runIds));
    });
    mocks.fetchLogParameterStatus.mockImplementation(
      (input: { runIds: string[] }) => {
        if (input.runIds.length === 1 && input.runIds[0] === "run-old") {
          return oldStatus.promise;
        }
        if (input.runIds.length === 1 && input.runIds[0] === "run-final") {
          return finalStatus.promise;
        }
        return Promise.resolve(parameterStatus(input.runIds));
      },
    );
    mocks.inspectModel.mockImplementation(
      (request: {
        modelType: string;
        model: string;
        preset: string;
        logRunId?: string;
      }) => {
        if (request.logRunId === "run-old") {
          return oldInspect.promise;
        }
        if (request.logRunId === "run-final") {
          return finalInspect.promise;
        }
        return Promise.resolve(experimentMonitorGraph(request, "initial"));
      },
    );

    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.history.visibleHistoricalRuns.map((run) => run.id))
        .toEqual(["run-final", "run-old"]);
    });

    act(() => {
      result.current.history.selectLogRun("run-old");
    });
    await waitFor(() => {
      expect(result.current.target.selectedExperimentRunId).toBe("run-old");
      expect(mocks.inspectModel).toHaveBeenCalledWith(
        expect.objectContaining({ logRunId: "run-old" }),
      );
    });

    act(() => {
      result.current.history.selectLogRun("run-final");
    });
    await waitFor(() => {
      expect(result.current.target.selectedExperimentRunId).toBe("run-final");
      expect(mocks.inspectModel).toHaveBeenCalledWith(
        expect.objectContaining({ logRunId: "run-final" }),
      );
    });

    act(() => {
      result.current.target.updateOverride("hidden_size", "128", {
        preserveTargetSelection: true,
      });
    });
    await waitFor(() => {
      expect(mocks.inspectModel).toHaveBeenCalledWith(
        expect.objectContaining({
          logRunId: "run-final",
          overrides: { hidden_size: "128" },
        }),
      );
    });

    act(() => {
      finalInspect.resolve(
        experimentMonitorGraph(
          { modelType: "linears", model: "linear", preset: "baseline" },
          "final",
        ),
      );
      finalTags.resolve(parameterLogTags(["run-final"]));
      finalStatus.resolve(parameterStatus(["run-final"]));
      oldInspect.resolve(
        experimentMonitorGraph(
          { modelType: "linears", model: "linear", preset: "baseline" },
          "old",
        ),
      );
      oldTags.resolve(parameterLogTags(["run-old"]));
      oldStatus.resolve(parameterStatus(["run-old"]));
    });

    await waitFor(() => {
      expect(result.current.graph.nodes.map((node) => node.id)).toContain(
        "final-layer",
      );
    });
    expect(result.current.graph.nodes.map((node) => node.id)).not.toContain(
      "old-layer",
    );
    expect(
      result.current.graphMonitor.graphMonitorSource?.kind ===
        "historical-run-group"
        ? result.current.graphMonitor.graphMonitorSource.runs.map((run) => run.id)
        : [],
    ).toEqual(["run-final"]);

    await waitFor(() => {
      const layerNode = result.current.graph.nodes.find(
        (node) => node.id === "final-layer",
      );
      expect(layerNode?.data.canOpenMonitor).toBe(true);
      expect(layerNode?.data.parameterActivity).toMatchObject({
        targetPath: "main_model.0.model",
        weights: {
          status: "updated",
          source: "historical",
          totalRuns: 1,
        },
        bias: {
          status: "updated",
          source: "historical",
          totalRuns: 1,
        },
      });
    });
  });

  it("uses footer-style icon labels in the sidebar target selectors", async () => {
    mockPublicModelCatalog();
    renderTargetPresetPanel();

    await screen.findByRole("combobox", { name: /^model type$/i });

    for (const label of ["Model Type", "Model Name", "Configuration Source"]) {
      const heading = screen.getByText(label).closest("div");
      expect(heading).toHaveClass(
        "flex",
        "items-center",
        "gap-2",
        "uppercase",
        "tracking-[0.09em]",
      );
      expect(heading?.querySelector("svg")).toBeInTheDocument();
    }
  });

  it("switches the sidebar model dropdown without an update loop", async () => {
    mockPublicModelCatalog();
    renderTargetPresetPanel();
    const user = userEvent.setup();

    const modelControl = await screen.findByRole("combobox", { name: /^model$/i });
    await waitFor(() => expect(modelControl).toHaveTextContent("linear"));

    await user.click(modelControl);
    const listbox = await screen.findByRole("listbox", { name: /^model options$/i });
    await user.click(within(listbox).getByRole("option", { name: "linear_adaptive" }));

    await waitFor(() => {
      expect(modelControl).toHaveTextContent("linear_adaptive");
      expect(screen.getByRole("combobox", { name: /^preset$/i }))
        .toHaveTextContent("adaptive");
    });
  });

  it("filters the sidebar model dropdown by model type", async () => {
    mockPublicModelCatalog();
    renderTargetPresetPanel();
    const user = userEvent.setup();

    const modelTypeControl = await screen.findByRole("combobox", {
      name: /^model type$/i,
    });
    const modelControl = await screen.findByRole("combobox", { name: /^model$/i });

    await waitFor(() => {
      expect(modelTypeControl).toHaveTextContent("Linears");
      expect(modelControl).toHaveTextContent("linear");
      expect(modelControl).not.toHaveTextContent("linears/linear");
    });

    await user.click(modelControl);
    const modelListbox = await screen.findByRole("listbox", {
      name: /^model options$/i,
    });
    expect(
      within(modelListbox).getByRole("option", { name: "linear" }),
    ).toBeInTheDocument();
    expect(
      within(modelListbox).getByRole("option", { name: "linear_adaptive" }),
    ).toBeInTheDocument();
    expect(
      within(modelListbox).queryByRole("option", { name: "linears/linear" }),
    ).not.toBeInTheDocument();
    expect(
      within(modelListbox).queryByRole("option", {
        name: "experts_linear",
      }),
    ).not.toBeInTheDocument();
    await user.keyboard("{Escape}");

    await user.click(modelTypeControl);
    const typeListbox = await screen.findByRole("listbox", {
      name: /^model type options$/i,
    });
    await user.click(within(typeListbox).getByRole("option", { name: "Experts" }));

    await waitFor(() => {
      expect(modelTypeControl).toHaveTextContent("Experts");
      expect(modelControl).toHaveTextContent("experts_linear");
      expect(modelControl).not.toHaveTextContent("experts/experts_linear");
      expect(screen.getByRole("combobox", { name: /^preset$/i }))
        .toHaveTextContent("expert-baseline");
    });
  });

  it("renders preset snapshot creation under the active preset selector", async () => {
    const onOpenFullConfig = vi.fn();
    renderTargetPresetPanel(onOpenFullConfig);
    const user = userEvent.setup();

    const createButton = await screen.findByRole("button", {
      name: "Create Snapshot",
    });
    await waitFor(() => expect(createButton).toBeEnabled());

    expect(screen.queryByRole("button", { name: "Edit" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Duplicate" }))
      .not.toBeInTheDocument();

    await user.click(createButton);

    expect(onOpenFullConfig).toHaveBeenCalledWith("snapshotDraft");
  });

  it("renders edit and duplicate actions under the active snapshot selector", async () => {
    mocks.fetchConfigSnapshots.mockResolvedValue({
      modelType: "linears",
      model: "linear",
      snapshots: [
        {
          id: "snapshot-wide",
          modelType: "linears",
          model: "linear",
          preset: "baseline",
          name: "Wide",
          overrides: { hidden_size: "256" },
          createdAt: "2026-06-01T00:00:00.000Z",
          updatedAt: "2026-06-01T00:00:00.000Z",
        },
      ],
    });
    const onOpenFullConfig = vi.fn();
    renderTargetPresetPanel(onOpenFullConfig);
    const user = userEvent.setup();

    const snapshotsButton = await screen.findByRole("radio", {
      name: /snapshots/i,
    });
    await waitFor(() => expect(snapshotsButton).toBeEnabled());
    await user.click(snapshotsButton);

    expect(await screen.findByRole("combobox", { name: /^snapshot$/i }))
      .toHaveTextContent("Wide");
    expect(screen.queryByRole("button", { name: "Create Snapshot" }))
      .not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Duplicate" }));
    await user.click(screen.getByRole("button", { name: "Edit" }));

    expect(onOpenFullConfig).toHaveBeenCalledWith("snapshotDraft");
    expect(onOpenFullConfig).toHaveBeenCalledWith("snapshotEdit");
  });

  it("does not render snapshot actions in experiment target mode", async () => {
    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "linear-history",
          preset: "baseline",
          dataset: "Mnist",
          timestamp: "2026-06-02 01:02:03",
        }),
      ],
    });
    renderTargetPresetPanel();
    const user = userEvent.setup();

    const experimentsButton = await screen.findByRole("radio", {
      name: /experiments/i,
    });
    await waitFor(() => expect(experimentsButton).toBeEnabled());
    await user.click(experimentsButton);

    expect(await screen.findByRole("combobox", { name: "Experiment" }))
      .toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Dataset" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Preset" })).toBeDisabled();
    expect(screen.queryByRole("combobox", { name: /^experiment run$/i }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create Snapshot" }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Duplicate" }))
      .not.toBeInTheDocument();
  });

  it("switches the training model dropdown without an update loop", async () => {
    mockPublicModelCatalog();
    renderTrainingPanel();
    const user = userEvent.setup();

    const panel = await screen.findByRole("region", {
      name: "Training workspace",
    });
    const modelControl = await within(panel).findByRole("combobox", {
      name: /^training model$/i,
    });
    await waitFor(() => expect(modelControl).toHaveTextContent("linear"));

    await user.click(modelControl);
    const listbox = await within(panel).findByRole("listbox", {
      name: /^training model options$/i,
    });
    await user.click(within(listbox).getByRole("option", { name: "linear_adaptive" }));

    await waitFor(() => {
      expect(modelControl).toHaveTextContent("linear_adaptive");
      expect(
        within(panel).getByRole("combobox", {
          name: /^presets\s+1\s*\/\s*1 selected$/i,
        }),
      ).toHaveTextContent("adaptive");
    });
  });

  it("filters the training model dropdown by model type", async () => {
    mockPublicModelCatalog();
    renderTrainingPanel();
    const user = userEvent.setup();

    const panel = await screen.findByRole("region", {
      name: "Training workspace",
    });
    const modelTypeControl = await within(panel).findByRole("combobox", {
      name: /^training model type$/i,
    });
    const modelControl = await within(panel).findByRole("combobox", {
      name: /^training model$/i,
    });

    await waitFor(() => {
      expect(modelTypeControl).toHaveTextContent("Linears");
      expect(modelControl).toHaveTextContent("linear");
      expect(modelControl).not.toHaveTextContent("linears/linear");
    });

    await user.click(modelControl);
    const modelListbox = await within(panel).findByRole("listbox", {
      name: /^training model options$/i,
    });
    expect(
      within(modelListbox).getByRole("option", { name: "linear" }),
    ).toBeInTheDocument();
    expect(
      within(modelListbox).getByRole("option", { name: "linear_adaptive" }),
    ).toBeInTheDocument();
    expect(
      within(modelListbox).queryByRole("option", {
        name: "experts_linear",
      }),
    ).not.toBeInTheDocument();
    await user.keyboard("{Escape}");

    await user.click(modelTypeControl);
    const typeListbox = await within(panel).findByRole("listbox", {
      name: /^training model type options$/i,
    });
    await user.click(within(typeListbox).getByRole("option", { name: "Experts" }));

    await waitFor(() => {
      expect(modelTypeControl).toHaveTextContent("Experts");
      expect(modelControl).toHaveTextContent("experts_linear");
      expect(modelControl).not.toHaveTextContent("experts/experts_linear");
      expect(
        within(panel).getByRole("combobox", {
          name: /^presets\s+1\s*\/\s*1 selected$/i,
        }),
      ).toHaveTextContent("expert-baseline");
    });
  });

  it("blocks training while a historical experiment run is selected", async () => {
    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "historical-run",
          experiment: "exp_locked",
          preset: "baseline",
          dataset: "Mnist",
          timestamp: "2026-06-02 01:02:03",
        }),
      ],
    });
    mocks.fetchLogExperiments.mockResolvedValueOnce({
      experiments: [
        {
          experiment: "scratch",
          runCount: 0,
          relativePath: "scratch",
        },
      ],
    });
    renderTrainingPanelWithExperiments();
    const user = userEvent.setup();

    await user.click(screen.getByRole("radio", { name: /new folder/i }));
    await user.type(screen.getByLabelText(/^new log folder$/i), "scratch_run");

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^start training$/i }))
        .toBeEnabled();
    });

    await user.click(await screen.findByRole("radio", { name: "Experiments" }));
    const experimentControl = await screen.findByRole("combobox", {
      name: "Experiment",
    });
    await user.click(experimentControl);
    await user.click(
      within(
        await screen.findByRole("listbox", { name: /^experiment options$/i }),
      ).getByRole("option", {
        name: "exp_locked",
      }),
    );
    const datasetControl = screen.getByRole("combobox", { name: "Dataset" });
    await user.click(datasetControl);
    await user.click(
      within(
        await screen.findByRole("listbox", { name: /^dataset options$/i }),
      ).getByRole("option", {
        name: "Mnist",
      }),
    );
    const presetControl = screen.getByRole("combobox", { name: "Preset" });
    await user.click(presetControl);
    await user.click(
      within(
        await screen.findByRole("listbox", { name: /^preset options$/i }),
      ).getByRole("option", {
        name: "baseline",
      }),
    );

    await waitFor(() => {
      expect(
        screen.getAllByText(
          "Cannot perform training while experiment exp_locked is selected.",
        ).length,
      ).toBeGreaterThan(0);
      expect(screen.getByRole("button", { name: /^start training$/i }))
        .toBeDisabled();
    });
    await user.click(screen.getByRole("button", { name: /^start training$/i }));
    expect(mocks.createTrainingJob).not.toHaveBeenCalled();
  });
});
