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
  getWorkbenchApiBaseUrl: vi.fn(),
  normalizeWorkbenchApiBaseUrl: vi.fn(),
  setWorkbenchApiBaseUrl: vi.fn(),
  resetWorkbenchApiBaseUrl: vi.fn(),
  fetchTrainingRunPlan: vi.fn(),
  createTrainingJob: vi.fn(),
  fetchTrainingJob: vi.fn(),
  cancelTrainingJob: vi.fn(),
  fetchMonitorParameterStatus: vi.fn(),
  fetchLogParameterStatus: vi.fn(),
}));
vi.mock("@/lib/api", () => mocks);

import { useWorkbenchState } from "@/features/workbench/state/use-workbench-state";
import { ConnectedTrainingWorkspace } from "@/features/workbench/components/connected-training-panel";
import {
  useModelTargetConfig,
  useTrainingTargetConfig,
  WorkbenchProviders,
} from "@/features/workbench/providers/workbench-providers";
import { TargetPresetPanel } from "@/features/workbench/components/screen/target-preset-panel";
import {
  clearPersistedTargetSelection,
} from "@/features/workbench/state/target/target-selection-storage";
import {
  type GraphNode,
  type InspectResponse,
  type LogParameterStatusResponse,
  type LogRun,
  type ModelIdentity,
  type TrainingJob,
} from "@/lib/api";

const DEFAULT_WORKBENCH_API_BASE_URL = "http://127.0.0.1:9999";

function renderWorkbenchState(options: Parameters<typeof useWorkbenchState>[0] = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const workbenchOptions = { activeWorkspace: "logs" as const, ...options };

  return renderHook(() => useWorkbenchState(workbenchOptions), {
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
      <WorkbenchProviders>
        <TargetPresetPanel onOpenFullConfig={onOpenFullConfig} />
      </WorkbenchProviders>
    </QueryClientProvider>,
  );
}

function renderTrainingPanel() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return render(
    <QueryClientProvider client={client}>
      <WorkbenchProviders activeWorkspace="training">
        <ConnectedTrainingWorkspace onOpenFullConfig={vi.fn()} />
      </WorkbenchProviders>
    </QueryClientProvider>,
  );
}

function renderTrainingPanelWithExperiments() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return render(
    <QueryClientProvider client={client}>
      <WorkbenchProviders activeWorkspace="training">
        <TargetPresetPanel onOpenFullConfig={vi.fn()} />
        <ConnectedTrainingWorkspace onOpenFullConfig={vi.fn()} />
      </WorkbenchProviders>
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
    hasLayerMonitorData: overrides.hasLayerMonitorData,
    metrics: overrides.metrics ?? {},
  };
}

function historicalOption(
  value: string,
  count: number,
  monitorEligibility: "checking" | "eligible" | "ineligible" = "checking",
) {
  const description =
    monitorEligibility === "eligible"
      ? "monitor data"
      : monitorEligibility === "ineligible"
        ? "no monitor data"
        : "monitor checking";
  return {
    value,
    label: value,
    count,
    monitorEligibility,
    description,
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
      "experts/linear",
      [{ name: "expert-baseline", label: "Expert baseline", description: "" }],
    ],
    [
      "bert/linear",
      [{ name: "bert-baseline", label: "BERT baseline", description: "" }],
    ],
    [
      "bert/linear_adaptive",
      [{ name: "bert-baseline", label: "BERT baseline", description: "" }],
    ],
    [
      "bert/expert_linear",
      [{ name: "bert-baseline", label: "BERT baseline", description: "" }],
    ],
    [
      "bert/expert_linear_adaptive",
      [{ name: "bert-baseline", label: "BERT baseline", description: "" }],
    ],
    [
      "vit/linear",
      [{ name: "baseline", label: "ViT baseline", description: "" }],
    ],
    [
      "vit/linear_adaptive",
      [{ name: "baseline", label: "ViT baseline", description: "" }],
    ],
    [
      "vit/expert_linear",
      [{ name: "baseline", label: "ViT baseline", description: "" }],
    ],
    [
      "vit/expert_linear_adaptive",
      [{ name: "baseline", label: "ViT baseline", description: "" }],
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
      "experts/linear",
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
      "bert/linear",
      [{ name: "ToyText", label: "Toy Text", inputDim: 128, outputDim: 2 }],
    ],
    [
      "bert/linear_adaptive",
      [{ name: "ToyText", label: "Toy Text", inputDim: 128, outputDim: 2 }],
    ],
    [
      "bert/expert_linear",
      [{ name: "ToyText", label: "Toy Text", inputDim: 128, outputDim: 2 }],
    ],
    [
      "bert/expert_linear_adaptive",
      [{ name: "ToyText", label: "Toy Text", inputDim: 128, outputDim: 2 }],
    ],
    [
      "vit/linear",
      [{ name: "Mnist", label: "MNIST", inputDim: 784, outputDim: 10 }],
    ],
    [
      "vit/linear_adaptive",
      [{ name: "Mnist", label: "MNIST", inputDim: 784, outputDim: 10 }],
    ],
    [
      "vit/expert_linear",
      [{ name: "Mnist", label: "MNIST", inputDim: 784, outputDim: 10 }],
    ],
    [
      "vit/expert_linear_adaptive",
      [{ name: "Mnist", label: "MNIST", inputDim: 784, outputDim: 10 }],
    ],
  ]);

  mocks.fetchModels.mockResolvedValue({
    models: [
      { modelType: "linears", model: "linear" },
      { modelType: "linears", model: "linear_adaptive" },
      { modelType: "experts", model: "linear" },
      { modelType: "bert", model: "linear" },
      { modelType: "bert", model: "linear_adaptive" },
      { modelType: "bert", model: "expert_linear" },
      { modelType: "bert", model: "expert_linear_adaptive" },
      { modelType: "vit", model: "linear" },
      { modelType: "vit", model: "linear_adaptive" },
      { modelType: "vit", model: "expert_linear" },
      { modelType: "vit", model: "expert_linear_adaptive" },
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
      defaultExperimentTask: "image-classification",
      datasetGroups: [
        {
          experimentTask: "image-classification",
          label: "Image Classification",
          datasets: datasetsByModel.get(modelIdentityKey(identity)) ?? [],
        },
      ],
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
  let workbenchApiBaseUrl = DEFAULT_WORKBENCH_API_BASE_URL;
  mocks.normalizeWorkbenchApiBaseUrl.mockReset().mockImplementation((url: string) => {
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
  mocks.getWorkbenchApiBaseUrl.mockReset().mockImplementation(() => workbenchApiBaseUrl);
  mocks.setWorkbenchApiBaseUrl.mockReset().mockImplementation((url: string) => {
    const normalizedUrl = mocks.normalizeWorkbenchApiBaseUrl(url);
    if (!normalizedUrl) {
      throw new Error("Invalid API base URL");
    }
    workbenchApiBaseUrl = normalizedUrl;
    return workbenchApiBaseUrl;
  });
  mocks.resetWorkbenchApiBaseUrl.mockReset().mockImplementation(() => {
    workbenchApiBaseUrl = DEFAULT_WORKBENCH_API_BASE_URL;
    return workbenchApiBaseUrl;
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
      { modelType: "bert", model: "linear" },
      { modelType: "bert", model: "linear_adaptive" },
      { modelType: "bert", model: "expert_linear" },
      { modelType: "bert", model: "expert_linear_adaptive" },
      { modelType: "vit", model: "linear" },
      { modelType: "vit", model: "linear_adaptive" },
      { modelType: "vit", model: "expert_linear" },
      { modelType: "vit", model: "expert_linear_adaptive" },
    ],
  });
  mocks.fetchPresets.mockReset().mockImplementation((identity: ModelIdentity) =>
    Promise.resolve(
      identity.modelType === "bert"
        ? {
            ...identity,
            presets: [
              { name: "bert-baseline", label: "BERT baseline", description: "" },
            ],
          }
        : identity.modelType === "vit"
          ? {
              ...identity,
              presets: [
                { name: "baseline", label: "ViT baseline", description: "" },
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
      identity.modelType === "bert"
        ? {
            ...identity,
            defaultExperimentTask: "bert-pretraining",
            datasetGroups: [
              {
                experimentTask: "bert-pretraining",
                label: "Bert Pretraining",
                datasets: [
                  { name: "ToyText", label: "Toy Text", inputDim: 128, outputDim: 2 },
                ],
              },
            ],
          }
        : identity.modelType === "vit"
          ? {
              ...identity,
              defaultExperimentTask: "image-classification",
              datasetGroups: [
                {
                  experimentTask: "image-classification",
                  label: "Image Classification",
                  datasets: [
                    { name: "Mnist", label: "MNIST", inputDim: 784, outputDim: 10 },
                  ],
                },
              ],
            }
        : {
            modelType: "linears",
            model: "linear",
            defaultExperimentTask: "image-classification",
            datasetGroups: [
              {
                experimentTask: "image-classification",
                label: "Image Classification",
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

describe("useWorkbenchState", () => {
  it("keeps target and history context identities stable on unrelated rerenders", async () => {
    const { result, rerender } = renderWorkbenchState();

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
    const { result } = renderWorkbenchState();

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

  it("does not rerender Training consumers for a Model-only override", async () => {
    let trainingRenderCount = 0;

    function ModelTargetProbe() {
      const { selectedModel, overrides, updateOverride } =
        useModelTargetConfig();
      return (
        <button
          type="button"
          onClick={() => updateOverride("hidden_size", "128")}
        >
          {selectedModel}:{overrides.hidden_size ?? "default"}
        </button>
      );
    }

    function TrainingTargetProbe() {
      const {
        selectedTrainingModel,
        selectedTrainingPrimaryPreset,
        trainingSchemaLoading,
        trainingSearchAxesLoading,
      } = useTrainingTargetConfig();
      trainingRenderCount += 1;
      return (
        <output
          data-testid="training-target-probe"
          data-ready={
            !trainingSchemaLoading && !trainingSearchAxesLoading
              ? "true"
              : "false"
          }
        >
          {selectedTrainingModel}:{selectedTrainingPrimaryPreset}
        </output>
      );
    }

    const client = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    render(
      <QueryClientProvider client={client}>
        <WorkbenchProviders activeWorkspace="training">
          <ModelTargetProbe />
          <TrainingTargetProbe />
        </WorkbenchProviders>
      </QueryClientProvider>,
    );
    const user = userEvent.setup();

    await screen.findByRole("button", { name: "linear:default" });
    await waitFor(() => {
      expect(screen.getByTestId("training-target-probe")).toHaveTextContent(
        "linear:baseline",
      );
      expect(screen.getByTestId("training-target-probe")).toHaveAttribute(
        "data-ready",
        "true",
      );
    });
    const settledTrainingRenderCount = trainingRenderCount;

    await user.click(screen.getByRole("button", { name: "linear:default" }));

    expect(
      await screen.findByRole("button", { name: "linear:128" }),
    ).toBeInTheDocument();
    expect(trainingRenderCount).toBe(settledTrainingRenderCount);
  });

  it("does not rerender Model consumers for a Training-only override", async () => {
    let modelRenderCount = 0;

    function ModelTargetProbe() {
      const { selectedModel, overrides } = useModelTargetConfig();
      modelRenderCount += 1;
      return (
        <output data-testid="model-target-probe">
          {selectedModel}:{overrides.hidden_size ?? "default"}
        </output>
      );
    }

    function TrainingTargetProbe() {
      const {
        selectedTrainingModel,
        trainingOverrides,
        updateTrainingOverride,
        trainingSchemaLoading,
        trainingSearchAxesLoading,
      } = useTrainingTargetConfig();
      return (
        <button
          type="button"
          data-ready={
            !trainingSchemaLoading && !trainingSearchAxesLoading
              ? "true"
              : "false"
          }
          onClick={() => updateTrainingOverride("epochs", "12")}
        >
          {selectedTrainingModel}:{trainingOverrides.epochs ?? "default"}
        </button>
      );
    }

    const client = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    render(
      <QueryClientProvider client={client}>
        <WorkbenchProviders activeWorkspace="training">
          <ModelTargetProbe />
          <TrainingTargetProbe />
        </WorkbenchProviders>
      </QueryClientProvider>,
    );
    const user = userEvent.setup();

    await waitFor(() => {
      expect(screen.getByTestId("model-target-probe")).toHaveTextContent(
        "linear:default",
      );
      expect(
        screen.getByRole("button", { name: "linear:default" }),
      ).toHaveAttribute("data-ready", "true");
    });
    const settledModelRenderCount = modelRenderCount;

    await user.click(screen.getByRole("button", { name: "linear:default" }));

    expect(
      await screen.findByRole("button", { name: "linear:12" }),
    ).toBeInTheDocument();
    expect(modelRenderCount).toBe(settledModelRenderCount);
  });

  it("uses enabled local defaults while loading capabilities", () => {
    mocks.fetchCapabilities.mockRejectedValueOnce(new Error("capabilities unavailable"));

    const { result } = renderWorkbenchState();

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

    const { result } = renderWorkbenchState();

    await waitFor(() => {
      expect(result.current.target.capabilities).toMatchObject({
        authMode: "bearer",
        trainingEnabled: false,
        logDeletionEnabled: false,
      });
    });
  });

  it("defers historical tag reads until an experiment run group is selected", async () => {
    mocks.fetchLogRuns.mockImplementation(
      (input?: { filters?: { models?: ModelIdentity[] } }) => {
        const runs = input?.filters?.models?.some(
          (model) => model.modelType === "linears" && model.model === "linear",
        )
          ? [
              logRun({
                id: "linear-history",
                experiment: "exp_linear",
                preset: "Fast",
                dataset: "FashionMnist",
                timestamp: "2026-06-02 01:02:03",
              }),
              logRun({
                id: "other-history",
                experiment: "exp_linear",
                preset: "baseline",
                dataset: "Mnist",
                timestamp: "2026-06-03 01:02:03",
              }),
            ]
          : [];
        return Promise.resolve({ runs });
      },
    );
    const { result } = renderWorkbenchState({ activeWorkspace: "model" });

    await waitFor(() => {
      expect(mocks.fetchLogRuns).toHaveBeenCalledWith(
        expect.objectContaining({
          filters: {
            models: [{ modelType: "linears", model: "linear" }],
          },
          includeAllPages: true,
        }),
        expect.any(Object),
      );
      expect(result.current.target.selectedModel).toBe("linear");
    });
    expect(mocks.fetchLogTags).not.toHaveBeenCalled();

    act(() => {
      result.current.target.activateTargetExperimentMode();
    });
    expect(mocks.fetchLogTags).not.toHaveBeenCalled();

    act(() => {
      result.current.history.setSelectedHistoricalExperimentFilter("exp_linear");
    });
    act(() => {
      result.current.history.setSelectedHistoricalDatasetFilter("FashionMnist");
    });
    act(() => {
      result.current.history.setSelectedHistoricalPreset("Fast");
    });
    await waitFor(() => {
      expect(mocks.fetchLogTags).toHaveBeenCalledWith(
        {
          runIds: ["linear-history"],
        },
        expect.any(Object),
      );
    });
    expect(mocks.fetchLogTags).not.toHaveBeenCalledWith(
      {
        runIds: ["linear-history", "other-history"],
      },
      expect.any(Object),
    );
  });

  it("settles the auto-selected training preset without an update loop", async () => {
    const { result } = renderWorkbenchState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    });
  });

  it("auto-selects the first public model type and calls APIs with split identity", async () => {
    mockPublicModelCatalog();

    const { result } = renderWorkbenchState();

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
        expect.objectContaining({ signal: expect.anything() }),
      );
    });
    expect(mocks.fetchPresets).toHaveBeenCalledWith(
      { modelType: "linears", model: "linear" },
      expect.objectContaining({ signal: expect.anything() }),
    );
    expect(mocks.fetchDatasets).toHaveBeenCalledWith(
      { modelType: "linears", model: "linear" },
      expect.objectContaining({ signal: expect.anything() }),
    );
    expect(mocks.fetchMonitors).toHaveBeenCalledWith(
      { modelType: "linears", model: "linear" },
      expect.objectContaining({ signal: expect.anything() }),
    );
    expect(mocks.fetchSearchSpace).toHaveBeenCalledWith(
      { modelType: "linears", model: "linear" },
      "baseline",
      ["baseline"],
      expect.objectContaining({ signal: expect.anything() }),
    );
    expect(mocks.inspectModel.mock.calls.map(([request]) => request))
      .toContainEqual({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        experimentTask: "image-classification",
        dataset: "Mnist",
        overrides: {},
      });
  });

  it("aborts obsolete model metadata when the selected model changes", async () => {
    mockPublicModelCatalog();
    const signals = new Map<string, AbortSignal | undefined>();
    mocks.fetchPresets.mockImplementation(
      (
        identity: ModelIdentity,
        options: { signal?: AbortSignal } = {},
      ) =>
        new Promise((resolve, reject) => {
          const key = modelIdentityKey(identity);
          signals.set(key, options.signal);
          options.signal?.addEventListener(
            "abort",
            () => reject(new DOMException("Aborted", "AbortError")),
            { once: true },
          );
          if (key === "experts/linear") {
            resolve({
              ...identity,
              presets: [
                {
                  name: "expert-baseline",
                  label: "Expert baseline",
                  description: "",
                },
              ],
            });
          }
        }),
    );
    const { result } = renderWorkbenchState({ activeWorkspace: "model" });

    await waitFor(() => expect(signals.has("linears/linear")).toBe(true));

    act(() => result.current.target.selectModelType("experts"));

    await waitFor(() => {
      expect(signals.get("linears/linear")?.aborted).toBe(true);
      expect(signals.has("experts/linear")).toBe(true);
      expect(result.current.target.selectedModelType).toBe("experts");
    });
    expect(result.current.target.isPresetsError).toBe(false);
  });

  it("selects the first model in a new type through the model reset cascade", async () => {
    mockPublicModelCatalog();
    const { result } = renderWorkbenchState();

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
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("expert-baseline");
      expect(result.current.target.selectedDatasets).toEqual(["ExpertToy"]);
      expect(result.current.target.overrides).toEqual({});
    });
    expect(mocks.fetchPresets).toHaveBeenCalledWith(
      { modelType: "experts", model: "linear" },
      expect.objectContaining({ signal: expect.anything() }),
    );
    expect(mocks.fetchDatasets).toHaveBeenCalledWith(
      { modelType: "experts", model: "linear" },
      expect.objectContaining({ signal: expect.anything() }),
    );
    expect(mocks.fetchMonitors).toHaveBeenCalledWith(
      { modelType: "experts", model: "linear" },
      expect.objectContaining({ signal: expect.anything() }),
    );
    expect(mocks.inspectModel.mock.calls.map(([request]) => request))
      .toContainEqual({
        modelType: "experts",
        model: "linear",
        preset: "expert-baseline",
        experimentTask: "image-classification",
        dataset: "ExpertToy",
        overrides: {},
      });
  });

  it("clears the experts graph and settles on the linear graph when changing model", async () => {
    mockPublicModelCatalog();
    mocks.inspectModel.mockImplementation(
      (request: { modelType: string; model: string; preset: string }) =>
        Promise.resolve(
          request.modelType === "experts" && request.model === "linear"
            ? expertsPreviewGraph(request)
            : linearPreviewGraph(request),
        ),
    );
    const { result } = renderWorkbenchState();

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
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.graph.graph).toMatchObject({
        modelType: "experts",
        model: "linear",
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
    const { result } = renderWorkbenchState({ activeWorkspace: "model" });

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
    const { result } = renderWorkbenchState();

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
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.history.selectedLogRunId).toBeNull();
      expect(result.current.history.selectedHistoricalExperimentFilter).toBe("");
      expect(result.current.history.selectedHistoricalDatasetFilter).toBe("");
      expect(result.current.history.selectedHistoricalPreset).toBe("");
      expect(result.current.target.selectedTargetMode).toBe("preset");
      expect(result.current.target.selectedExperimentRunId).toBe("");
    });
  });

  it("settles model changes on the new model defaults", async () => {
    const { result } = renderWorkbenchState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedDatasets).toEqual(["Mnist"]);
    });

    act(() => {
      result.current.target.selectModel("linear", "bert");
    });

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("bert-baseline");
      expect(result.current.target.selectedDatasets).toEqual(["ToyText"]);
      expect(result.current.target.selectedTrainingModel).toBe("linear");
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
      expect(result.current.target.selectedTrainingDatasets).toEqual(["Mnist"]);
    });
    expect(mocks.inspectModel.mock.calls.map(([request]) => request)).toContainEqual({
      modelType: "bert",
      model: "linear",
      preset: "bert-baseline",
      experimentTask: "bert-pretraining",
      dataset: "ToyText",
      overrides: {},
    });
  });

  it("resets graph selection and expansion when selecting another model", async () => {
    mocks.inspectModel.mockResolvedValue(monitorGraph());
    const { result } = renderWorkbenchState();

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
      result.current.target.selectModel("linear", "bert");
    });

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.graph.selectedNodeId).toBeNull();
      expect(result.current.graph.expandedGraphNodeIds.size).toBe(0);
    });
  });

  it("does not auto-select a run, then syncs target config once when one is picked", async () => {
    mocks.fetchModels.mockResolvedValueOnce({
      models: [
        { modelType: "bert", model: "linear" },
        { modelType: "linears", model: "linear" },
      ],
    });
    mocks.fetchLogRuns.mockImplementation(
      (input?: { filters?: { models?: ModelIdentity[] } }) =>
        Promise.resolve({
          runs: input?.filters?.models?.some(
            (model) => model.modelType === "linears" && model.model === "linear",
          )
            ? [
                logRun({
                  id: "linear-history",
                  preset: "Fast",
                  dataset: "FashionMnist",
                  timestamp: "2026-06-02 01:02:03",
                }),
              ]
            : [],
        }),
    );
    const { result } = renderWorkbenchState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
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
    const { result } = renderWorkbenchState();

    await waitFor(() => {
      expect(result.current.history.visibleHistoricalRuns.map((run) => run.id))
        .toEqual([
          "fashion-history",
          "baseline-new",
          "fast-history",
          "baseline-old",
        ]);
      expect(result.current.history.historicalExperimentOptions).toEqual([
        historicalOption("exp_linear", 4, "checking"),
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
        historicalOption("FashionMnist", 1, "checking"),
        historicalOption("Mnist", 3, "checking"),
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
        historicalOption("baseline", 2, "checking"),
        historicalOption("fast", 1, "checking"),
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
        runIds: ["baseline-new", "baseline-old"],
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
    const { result } = renderWorkbenchState();

    await waitFor(() => {
      expect(result.current.history.historicalExperimentOptions).toEqual([
        historicalOption("exp_linear", 2, "checking"),
      ]);
    });

    act(() => {
      result.current.history.setSelectedHistoricalExperimentFilter("exp_linear");
    });
    await waitFor(() => {
      expect(result.current.history.historicalDatasetOptions).toEqual([
        historicalOption("Mnist", 2, "checking"),
      ]);
    });

    act(() => {
      result.current.history.setSelectedHistoricalDatasetFilter("Mnist");
    });
    await waitFor(() => {
      expect(result.current.history.historicalPresetOptions).toEqual([
        historicalOption("fast", 1, "checking"),
        historicalOption("baseline", 1, "checking"),
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

    const { result } = renderWorkbenchState();

    await waitFor(() => {
      expect(result.current.history.historicalExperimentOptions).toEqual([
        historicalOption("exp_b", 1, "checking"),
        historicalOption("exp_a", 1, "checking"),
      ]);
    });

    act(() => {
      result.current.history.setSelectedHistoricalExperimentFilter("exp_a");
    });
    await waitFor(() => {
      expect(result.current.history.historicalDatasetOptions).toEqual([
        historicalOption("Mnist", 1, "checking"),
      ]);
    });
    act(() => {
      result.current.history.setSelectedHistoricalDatasetFilter("Mnist");
    });
    await waitFor(() => {
      expect(result.current.history.historicalPresetOptions).toEqual([
        historicalOption("baseline", 1, "checking"),
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
        historicalOption("Mnist", 1, "checking"),
      ]);
    });
    expect(mocks.inspectModel).not.toHaveBeenCalled();
    expect(result.current.graph.graph).toBeUndefined();

    act(() => {
      result.current.history.setSelectedHistoricalDatasetFilter("Mnist");
    });
    await waitFor(() => {
      expect(result.current.history.historicalPresetOptions).toEqual([
        historicalOption("baseline", 1, "checking"),
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

  it("keeps performance-only experiments selectable without graph activity", async () => {
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

    const { result } = renderWorkbenchState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
    });

    act(() => {
      result.current.target.activateTargetExperimentMode();
    });

    await waitFor(() => {
      expect(result.current.history.historicalExperimentOptions).toEqual([
        historicalOption("kaggle_linear_all", 1, "checking"),
        historicalOption("test_linear", 1, "checking"),
      ]);
    });

    act(() => {
      result.current.history.setSelectedHistoricalExperimentFilter("test_linear");
    });
    await waitFor(() => {
      expect(result.current.history.historicalDatasetOptions).toEqual([
        historicalOption("Mnist", 1, "checking"),
      ]);
    });
    act(() => {
      result.current.history.setSelectedHistoricalDatasetFilter("Mnist");
    });
    await waitFor(() => {
      expect(result.current.history.historicalPresetOptions).toEqual([
        historicalOption("baseline", 1, "checking"),
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
      expect(result.current.history.selectedHistoricalExperimentFilter).toBe(
        "kaggle_linear_all",
      );
      expect(result.current.history.historicalDatasetOptions).toEqual([
        historicalOption("KaggleDigits", 1, "checking"),
      ]);
      expect(result.current.history.visibleHistoricalRuns.map((run) => run.id))
        .toEqual(["kaggle-linear-run"]);
    });
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalled();
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalledWith(
      { runIds: ["kaggle-linear-run"] },
      expect.any(Object),
    );

    act(() => {
      result.current.history.setSelectedHistoricalDatasetFilter("KaggleDigits");
    });
    await waitFor(() => {
      expect(result.current.history.historicalPresetOptions).toEqual([
        historicalOption("KAGGLE_LINEAR", 1, "checking"),
      ]);
    });

    act(() => {
      result.current.history.setSelectedHistoricalPreset("KAGGLE_LINEAR");
    });

    await waitFor(() => {
      expect(result.current.history.selectedLogRunId).toBe("kaggle-linear-run");
      expect(result.current.target.selectedExperimentRunId).toBe(
        "kaggle-linear-run",
      );
      expect(result.current.history.selectedLogRunMonitorEligibility).toBe(
        "ineligible",
      );
      expect(result.current.history.historicalMonitorRuns).toEqual([]);
      expect(result.current.graph.nodes.map((node) => node.id)).toContain(
        "test-layer",
      );
    });
    expect(mocks.inspectModel).toHaveBeenCalledWith(
      expect.objectContaining({
        modelType: "linears",
        model: "linear",
        preset: "KAGGLE_LINEAR",
        dataset: "KaggleDigits",
        overrides: {},
        logRunId: "kaggle-linear-run",
      }),
      expect.objectContaining({ signal: expect.anything() }),
    );
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalled();
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalledWith(
      { runIds: ["kaggle-linear-run"] },
      expect.any(Object),
    );
  });

  it("does not block experiment options or preview selection while tags load", async () => {
    const delayedTags = deferred<ReturnType<typeof parameterLogTags>>();

    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "slow-tags-run",
          experiment: "slow_exp",
          preset: "baseline",
          dataset: "Mnist",
          timestamp: "2026-06-01 01:02:03",
          hasLayerMonitorData: null,
        }),
      ],
    });
    mocks.fetchLogTags.mockImplementation((input: { runIds: string[] }) => {
      if (input.runIds.includes("slow-tags-run")) {
        return delayedTags.promise;
      }
      return Promise.resolve(parameterLogTags(input.runIds));
    });
    mocks.inspectModel.mockImplementation(
      (request: {
        modelType: string;
        model: string;
        preset: string;
        logRunId?: string;
      }) => Promise.resolve(experimentMonitorGraph(request, "slow")),
    );

    const { result } = renderWorkbenchState();

    await waitFor(() => {
      expect(result.current.history.historicalExperimentOptions).toEqual([
        historicalOption("slow_exp", 1, "checking"),
      ]);
    });
    expect(mocks.fetchLogTags).not.toHaveBeenCalled();

    act(() => {
      result.current.target.activateTargetExperimentMode();
    });
    expect(mocks.fetchLogTags).not.toHaveBeenCalled();
    act(() => {
      result.current.history.setSelectedHistoricalExperimentFilter("slow_exp");
    });
    await waitFor(() => {
      expect(result.current.history.historicalDatasetOptions).toEqual([
        historicalOption("Mnist", 1, "checking"),
      ]);
    });
    act(() => {
      result.current.history.setSelectedHistoricalDatasetFilter("Mnist");
    });
    await waitFor(() => {
      expect(result.current.history.historicalPresetOptions).toEqual([
        historicalOption("baseline", 1, "checking"),
      ]);
    });

    mocks.inspectModel.mockClear();
    mocks.fetchLogParameterStatus.mockClear();
    act(() => {
      result.current.history.setSelectedHistoricalPreset("baseline");
    });
    await waitFor(() => {
      expect(mocks.fetchLogTags).toHaveBeenCalledWith(
        { runIds: ["slow-tags-run"] },
        expect.any(Object),
      );
    });

    await waitFor(() => {
      expect(result.current.history.selectedLogRunId).toBe("slow-tags-run");
      expect(result.current.target.selectedExperimentRunId).toBe("slow-tags-run");
      expect(result.current.history.selectedLogRunMonitorEligibility).toBe(
        "checking",
      );
      expect(result.current.history.historicalMonitorRuns).toEqual([]);
      expect(result.current.graph.nodes.map((node) => node.id)).toContain(
        "slow-layer",
      );
    });
    expect(mocks.inspectModel).toHaveBeenCalledWith(
      expect.objectContaining({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: {},
        logRunId: "slow-tags-run",
      }),
      expect.objectContaining({ signal: expect.anything() }),
    );
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalled();

    act(() => {
      delayedTags.resolve(parameterLogTags(["slow-tags-run"]));
    });

    await waitFor(() => {
      expect(result.current.history.selectedLogRunMonitorEligibility).toBe(
        "eligible",
      );
      expect(mocks.fetchLogParameterStatus).toHaveBeenCalledWith(
        { runIds: ["slow-tags-run"] },
        expect.any(Object),
      );
    });
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
    const { result } = renderWorkbenchState();

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
      expect(result.current.target.selectedExperimentDataset).toBe("FashionMnist");
      expect(result.current.target.selectedDatasets).toEqual(["FashionMnist"]);
    });
    await waitFor(() => {
      expect(mocks.inspectModel).toHaveBeenCalledWith(
        expect.objectContaining({
          modelType: "linears",
          model: "linear",
          preset: "Fast",
          dataset: "FashionMnist",
          overrides: {},
          logRunId: "linear-history",
        }),
        expect.objectContaining({ signal: expect.anything() }),
      );
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
      experimentTask: "image-classification",
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
    const { result } = renderWorkbenchState();

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
      expect(result.current.target.selectedExperimentDataset).toBe("FashionMnist");
      expect(result.current.target.selectedDatasets).toEqual(["FashionMnist"]);
    });
    await waitFor(() => {
      expect(mocks.inspectModel).toHaveBeenCalledWith(
        expect.objectContaining({
          modelType: "linears",
          model: "linear",
          preset: "Fast",
          dataset: "FashionMnist",
          overrides: {},
          logRunId: "linear-history",
        }),
        expect.objectContaining({ signal: expect.anything() }),
      );
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
      experimentTask: "image-classification",
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
          sectionPath: ["Model"],
          type: "int",
          default: 64,
          nullable: false,
          choices: [],
        },
      ],
    });
    const { result } = renderWorkbenchState();

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
    const { result } = renderWorkbenchState();

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
    const { result } = renderWorkbenchState();

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
    const { result } = renderWorkbenchState();

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
    const { result } = renderWorkbenchState();

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
        runIds: ["run-new", "run-old"],
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
      (
        request: {
          modelType: string;
          model: string;
          preset: string;
          logRunId?: string;
        },
        options: { signal?: AbortSignal } = {},
      ) => {
        const abortable = (response: Promise<InspectResponse>) =>
          new Promise<InspectResponse>((resolve, reject) => {
            response.then(resolve, reject);
            options.signal?.addEventListener(
              "abort",
              () => reject(new DOMException("Aborted", "AbortError")),
              { once: true },
            );
          });
        if (request.logRunId === "run-old") {
          return abortable(oldInspect.promise);
        }
        if (request.logRunId === "run-final") {
          return abortable(finalInspect.promise);
        }
        return Promise.resolve(experimentMonitorGraph(request, "initial"));
      },
    );

    const { result } = renderWorkbenchState();

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
        expect.objectContaining({ signal: expect.anything() }),
      );
    });

    act(() => {
      result.current.history.selectLogRun("run-final");
    });
    await waitFor(() => {
      expect(result.current.target.selectedExperimentRunId).toBe("run-final");
      expect(mocks.inspectModel).toHaveBeenCalledWith(
        expect.objectContaining({ logRunId: "run-final" }),
        expect.objectContaining({ signal: expect.anything() }),
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
        expect.objectContaining({ signal: expect.anything() }),
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
      within(modelListbox)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["linear", "linear_adaptive"]);
    await user.keyboard("{Escape}");

    await user.click(modelTypeControl);
    const typeListbox = await screen.findByRole("listbox", {
      name: /^model type options$/i,
    });
    expect(within(typeListbox).getByRole("option", { name: "Vit" }))
      .toBeInTheDocument();
    await user.click(within(typeListbox).getByRole("option", { name: "Experts" }));

    await waitFor(() => {
      expect(modelTypeControl).toHaveTextContent("Experts");
      expect(modelControl).toHaveTextContent("linear");
      expect(modelControl).not.toHaveTextContent("experts/linear");
      expect(screen.getByRole("combobox", { name: /^preset$/i }))
        .toHaveTextContent("expert-baseline");
    });

    await user.click(modelControl);
    const expertModelListbox = await screen.findByRole("listbox", {
      name: /^model options$/i,
    });
    expect(
      within(expertModelListbox)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["linear"]);
    expect(
      within(expertModelListbox).queryByRole("option", {
        name: "linear_adaptive",
      }),
    ).not.toBeInTheDocument();
    await user.keyboard("{Escape}");

    await user.click(modelTypeControl);
    const vitTypeListbox = await screen.findByRole("listbox", {
      name: /^model type options$/i,
    });
    await user.click(within(vitTypeListbox).getByRole("option", { name: "Vit" }));

    await waitFor(() => {
      expect(modelTypeControl).toHaveTextContent("Vit");
      expect(modelControl).toHaveTextContent("linear");
      expect(screen.getByRole("combobox", { name: /^preset$/i }))
        .toHaveTextContent("baseline");
    });

    await user.click(modelControl);
    const vitModelListbox = await screen.findByRole("listbox", {
      name: /^model options$/i,
    });
    expect(
      within(vitModelListbox)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual([
      "linear",
      "linear_adaptive",
      "expert_linear",
      "expert_linear_adaptive",
    ]);
    await user.keyboard("{Escape}");
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
      within(modelListbox)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["linear", "linear_adaptive"]);
    await user.keyboard("{Escape}");

    await user.click(modelTypeControl);
    const typeListbox = await within(panel).findByRole("listbox", {
      name: /^training model type options$/i,
    });
    expect(within(typeListbox).getByRole("option", { name: "Vit" }))
      .toBeInTheDocument();
    await user.click(within(typeListbox).getByRole("option", { name: "Experts" }));

    await waitFor(() => {
      expect(modelTypeControl).toHaveTextContent("Experts");
      expect(modelControl).toHaveTextContent("linear");
      expect(modelControl).not.toHaveTextContent("experts/linear");
      expect(
        within(panel).getByRole("combobox", {
          name: /^presets\s+1\s*\/\s*1 selected$/i,
        }),
      ).toHaveTextContent("expert-baseline");
    });

    await user.click(modelControl);
    const expertModelListbox = await within(panel).findByRole("listbox", {
      name: /^training model options$/i,
    });
    expect(
      within(expertModelListbox)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["linear"]);
    expect(
      within(expertModelListbox).queryByRole("option", {
        name: "linear_adaptive",
      }),
    ).not.toBeInTheDocument();
    await user.keyboard("{Escape}");

    await user.click(modelTypeControl);
    const vitTypeListbox = await within(panel).findByRole("listbox", {
      name: /^training model type options$/i,
    });
    await user.click(within(vitTypeListbox).getByRole("option", { name: "Vit" }));

    await waitFor(() => {
      expect(modelTypeControl).toHaveTextContent("Vit");
      expect(modelControl).toHaveTextContent("linear");
      expect(
        within(panel).getByRole("combobox", {
          name: /^presets\s+1\s*\/\s*1 selected$/i,
        }),
      ).toHaveTextContent("baseline");
    });

    await user.click(modelControl);
    const vitModelListbox = await within(panel).findByRole("listbox", {
      name: /^training model options$/i,
    });
    expect(
      within(vitModelListbox)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual([
      "linear",
      "linear_adaptive",
      "expert_linear",
      "expert_linear_adaptive",
    ]);
    await user.keyboard("{Escape}");
  });

  it("allows training while a historical experiment run is selected", async () => {
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
    mocks.createTrainingJob.mockResolvedValueOnce(
      trainingJob({
        id: "job-after-selected-experiment",
        status: "queued",
        logFolder: "scratch_run",
      }),
    );
    mocks.fetchTrainingJob.mockResolvedValueOnce(
      trainingJob({
        id: "job-after-selected-experiment",
        status: "queued",
        logFolder: "scratch_run",
      }),
    );
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
        screen.queryByText(
          "Cannot perform training while experiment exp_locked is selected.",
        ),
      ).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: /^start training$/i }))
        .toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: /^start training$/i }));
    await waitFor(() => {
      expect(mocks.createTrainingJob).toHaveBeenCalledWith(
        expect.objectContaining({
          modelType: "linears",
          model: "linear",
          preset: "baseline",
          presets: ["baseline"],
          datasets: ["Mnist"],
          logFolder: "scratch_run",
        }),
        expect.anything(),
      );
    });
  });
});
