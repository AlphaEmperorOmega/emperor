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
  deleteConfigSnapshot: vi.fn(),
  fetchLogRuns: vi.fn(),
  fetchLogExperiments: vi.fn(),
  fetchLogTags: vi.fn(),
  inspectModel: vi.fn(),
  fetchTrainingRunPlan: vi.fn(),
  createTrainingJob: vi.fn(),
  fetchTrainingJob: vi.fn(),
  cancelTrainingJob: vi.fn(),
  fetchMonitorParameterStatus: vi.fn(),
  fetchLogParameterStatus: vi.fn(),
}));
vi.mock("@/lib/api", () => mocks);

import { useViewerState } from "@/features/viewer/state/use-viewer-state";
import { ConnectedTrainingPanel } from "@/features/viewer/components/connected-training-panel";
import { ViewerProviders } from "@/features/viewer/providers/viewer-providers";
import { TargetPresetPanel } from "@/features/viewer/components/screen/target-preset-panel";
import {
  type GraphNode,
  type InspectResponse,
  type LogRun,
  type TrainingJob,
} from "@/lib/api";

function renderViewerState() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return renderHook(() => useViewerState(), {
    wrapper: ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children),
  });
}

function renderTargetPresetPanel() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return render(
    <QueryClientProvider client={client}>
      <ViewerProviders>
        <TargetPresetPanel />
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
        <ConnectedTrainingPanel onOpenFullConfig={vi.fn()} />
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
        <TargetPresetPanel />
        <ConnectedTrainingPanel onOpenFullConfig={vi.fn()} />
      </ViewerProviders>
    </QueryClientProvider>,
  );
}

function logRun(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  return {
    id: overrides.id,
    group: overrides.group ?? overrides.experiment ?? "exp_linear",
    experiment: overrides.experiment ?? "exp_linear",
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

function trainingJob(overrides: Partial<TrainingJob> = {}): TrainingJob {
  return {
    id: overrides.id ?? "job-1",
    status: overrides.status ?? "running",
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
      "linears/linear",
      "linears/linear_adaptive",
      "experts/experts_linear",
      "transformer_encoder/bert_linear",
    ],
  });
  mocks.fetchPresets.mockImplementation((model: string) =>
    Promise.resolve({ model, presets: presetsByModel.get(model) ?? [] }),
  );
  mocks.fetchDatasets.mockImplementation((model: string) =>
    Promise.resolve({ model, datasets: datasetsByModel.get(model) ?? [] }),
  );
  mocks.fetchMonitors.mockImplementation((model: string) =>
    Promise.resolve({ model, monitors: [] }),
  );
  mocks.fetchConfigSchema.mockImplementation((model: string) =>
    Promise.resolve({ model, fields: [] }),
  );
  mocks.fetchSearchSpace.mockImplementation((model: string, preset: string) =>
    Promise.resolve({ model, preset, axes: [] }),
  );
  mocks.inspectModel.mockImplementation(
    (request: { model: string; preset: string }) =>
      Promise.resolve({
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
  mocks.fetchHealth.mockReset().mockResolvedValue({ status: "ok" });
  mocks.fetchCapabilities.mockReset().mockResolvedValue({
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
  });
  mocks.fetchModels.mockReset().mockResolvedValue({ models: ["linear", "bert_linear"] });
  mocks.fetchPresets.mockReset().mockImplementation((model: string) =>
    Promise.resolve(
      model === "bert_linear"
        ? {
            model,
            presets: [
              { name: "bert-baseline", label: "BERT baseline", description: "" },
            ],
          }
        : {
            model: "linear",
            presets: [
              { name: "baseline", label: "Baseline", description: "" },
              { name: "fast", label: "Fast", description: "" },
            ],
          },
    ),
  );
  mocks.fetchDatasets.mockReset().mockImplementation((model: string) =>
    Promise.resolve(
      model === "bert_linear"
        ? {
            model,
            datasets: [
              { name: "ToyText", label: "Toy Text", inputDim: 128, outputDim: 2 },
            ],
          }
        : {
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
  mocks.fetchMonitors.mockReset().mockImplementation((model: string) =>
    Promise.resolve({
      model,
      monitors: [],
    }),
  );
  mocks.fetchConfigSchema.mockReset().mockImplementation((model: string) =>
    Promise.resolve({
      model,
      fields: [],
    }),
  );
  mocks.fetchSearchSpace.mockReset().mockImplementation((model: string, preset: string) =>
    Promise.resolve({
      model,
      preset,
      axes: [],
    }),
  );
  mocks.fetchConfigSnapshots.mockReset().mockImplementation((model: string) =>
    Promise.resolve({
      model,
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
      model: "linear",
      preset: "baseline",
      name,
      overrides: {},
      createdAt: "2026-06-01T00:00:00.000Z",
      updatedAt: "2026-06-01T00:00:00.000Z",
    }),
  );
  mocks.deleteConfigSnapshot.mockReset().mockResolvedValue({
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
  it("uses enabled local defaults while loading capabilities", () => {
    mocks.fetchCapabilities.mockRejectedValueOnce(new Error("capabilities unavailable"));

    const { result } = renderViewerState();

    expect(result.current.target.capabilities).toMatchObject({
      trainingEnabled: true,
      logDeletionEnabled: true,
    });
  });

  it("surfaces fetched hosted capability flags", async () => {
    mocks.fetchCapabilities.mockResolvedValueOnce({
      authMode: "bearer",
      trainingEnabled: false,
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

  it("settles the auto-selected training preset without an update loop", async () => {
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    });
  });

  it("auto-selects the first public model type and keeps API calls on full IDs", async () => {
    mockPublicModelCatalog();

    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModelType).toBe("linears");
      expect(result.current.target.selectedModel).toBe("linears/linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedDatasets).toEqual(["Mnist"]);
    });
    await waitFor(() => {
      expect(mocks.fetchConfigSchema).toHaveBeenCalledWith(
        "linears/linear",
        "baseline",
      );
    });
    expect(mocks.fetchPresets).toHaveBeenCalledWith("linears/linear");
    expect(mocks.fetchDatasets).toHaveBeenCalledWith("linears/linear");
    expect(mocks.fetchMonitors).toHaveBeenCalledWith("linears/linear");
    expect(mocks.fetchSearchSpace).toHaveBeenCalledWith(
      "linears/linear",
      "baseline",
    );
    expect(mocks.inspectModel.mock.calls.map(([request]) => request))
      .toContainEqual({
        model: "linears/linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: {},
      });
  });

  it("selects the first model in a new type through the model reset cascade", async () => {
    mockPublicModelCatalog();
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linears/linear");
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
      expect(result.current.target.selectedModel).toBe("experts/experts_linear");
      expect(result.current.target.selectedPreset).toBe("expert-baseline");
      expect(result.current.target.selectedDatasets).toEqual(["ExpertToy"]);
      expect(result.current.target.overrides).toEqual({});
    });
    expect(mocks.fetchPresets).toHaveBeenCalledWith("experts/experts_linear");
    expect(mocks.fetchDatasets).toHaveBeenCalledWith("experts/experts_linear");
    expect(mocks.fetchMonitors).toHaveBeenCalledWith("experts/experts_linear");
    expect(mocks.inspectModel.mock.calls.map(([request]) => request))
      .toContainEqual({
        model: "experts/experts_linear",
        preset: "expert-baseline",
        dataset: "ExpertToy",
        overrides: {},
      });
  });

  it("clears historical run selection when switching model type", async () => {
    mockPublicModelCatalog();
    mocks.fetchLogRuns.mockResolvedValueOnce({
      runs: [
        logRun({
          id: "linears-history",
          model: "linears/linear",
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
      expect(result.current.target.selectedModel).toBe("experts/experts_linear");
      expect(result.current.history.selectedLogRunId).toBeNull();
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
      result.current.target.selectModel("bert_linear");
    });

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("bert_linear");
      expect(result.current.target.selectedPreset).toBe("bert-baseline");
      expect(result.current.target.selectedTrainingPresets).toEqual(["bert-baseline"]);
      expect(result.current.target.selectedDatasets).toEqual(["ToyText"]);
    });
    expect(mocks.inspectModel.mock.calls.map(([request]) => request)).toContainEqual({
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
      result.current.target.selectModel("bert_linear");
    });

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("bert_linear");
      expect(result.current.graph.selectedNodeId).toBeNull();
      expect(result.current.graph.expandedGraphNodeIds.size).toBe(0);
    });
  });

  it("does not auto-select a run, then syncs target config once when one is picked", async () => {
    mocks.fetchModels.mockResolvedValueOnce({ models: ["bert_linear", "linear"] });
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
      result.current.target.selectModel("linear");
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
      expect(result.current.target.selectedPreset).toBe("fast");
      expect(result.current.target.selectedTrainingPresets).toEqual(["fast"]);
      expect(result.current.target.selectedDatasets).toEqual(["FashionMnist"]);
      expect(result.current.target.overrides).toEqual({});
    });

    const finalHistoricalRequests = mocks.inspectModel.mock.calls.filter(
      ([request]) =>
        request.model === "linear" &&
        request.preset === "fast" &&
        request.dataset === "FashionMnist",
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
      model: "linear",
      preset: "baseline",
      dataset: "FashionMnist",
      overrides: {},
    });
  });

  it("switches from an experiment target to a snapshot target with saved overrides", async () => {
    mocks.fetchConfigSnapshots.mockResolvedValue({
      model: "linear",
      snapshots: [
        {
          id: "snapshot-wide",
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
      model: "linear",
      preset: "baseline",
      dataset: "FashionMnist",
      overrides: { hidden_size: "256" },
    });
  });

  it("requests parameter status for the active linear training job", async () => {
    mocks.inspectModel.mockResolvedValue(monitorGraph());
    const { result } = renderViewerState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
    });

    act(() => {
      result.current.training.onJobChange(trainingJob());
    });

    await waitFor(() => {
      expect(mocks.fetchMonitorParameterStatus).toHaveBeenCalledWith({
        jobId: "job-1",
        preset: "baseline",
        dataset: "Mnist",
      });
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
      result.current.training.onJobChange(job);
    });
    act(() => {
      result.current.training.openGraphNodeMonitor(linearNode as GraphNode);
    });

    await waitFor(() => {
      expect(result.current.training.graphMonitorNode?.id).toBe("linear-0");
      expect(result.current.training.graphMonitorSource).toEqual({
        kind: "active-job",
        job,
      });
    });
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
        runIds: ["run-new", "run-old"],
      });
    });
    expect(mocks.fetchLogParameterStatus).not.toHaveBeenCalledWith({
      runIds: ["run-new", "run-fast", "run-old"],
    });
    expect(mocks.fetchMonitorParameterStatus).not.toHaveBeenCalled();
  });

  it("switches the sidebar model dropdown without an update loop", async () => {
    renderTargetPresetPanel();
    const user = userEvent.setup();

    const modelControl = await screen.findByRole("combobox", { name: /^model$/i });
    await waitFor(() => expect(modelControl).toHaveTextContent("linear"));

    await user.click(modelControl);
    const listbox = await screen.findByRole("listbox", { name: /^model options$/i });
    await user.click(within(listbox).getByRole("option", { name: "bert_linear" }));

    await waitFor(() => {
      expect(modelControl).toHaveTextContent("bert_linear");
      expect(screen.getByRole("combobox", { name: /^preset$/i }))
        .toHaveTextContent("bert-baseline");
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

  it("switches the training model dropdown without an update loop", async () => {
    renderTrainingPanel();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^training/i }));
    const details = document.getElementById("training-panel-details");
    expect(details).toBeInstanceOf(HTMLElement);
    const panel = details as HTMLElement;
    const modelControl = await within(panel).findByRole("combobox", {
      name: /^training model$/i,
    });
    await waitFor(() => expect(modelControl).toHaveTextContent("linear"));

    await user.click(modelControl);
    const listbox = await within(panel).findByRole("listbox", {
      name: /^training model options$/i,
    });
    await user.click(within(listbox).getByRole("option", { name: "bert_linear" }));

    await waitFor(() => {
      expect(modelControl).toHaveTextContent("bert_linear");
      expect(
        within(panel).getByRole("combobox", {
          name: /^presets\s+1\s*\/\s*1 selected$/i,
        }),
      ).toHaveTextContent("bert-baseline");
    });
  });

  it("filters the training model dropdown by model type", async () => {
    mockPublicModelCatalog();
    renderTrainingPanel();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^training/i }));
    const details = document.getElementById("training-panel-details");
    expect(details).toBeInstanceOf(HTMLElement);
    const panel = details as HTMLElement;
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

    await user.click(await screen.findByRole("button", { name: /^training/i }));
    await user.click(screen.getByRole("tab", { name: /new folder/i }));
    await user.type(screen.getByLabelText(/^new log folder$/i), "scratch_run");

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^start training$/i }))
        .toBeEnabled();
    });

    await user.click(await screen.findByRole("tab", { name: "Experiments" }));
    const experimentRunControl = await screen.findByRole("combobox", {
      name: /^experiment run$/i,
    });
    await user.click(experimentRunControl);
    await user.click(
      within(
        await screen.findByRole("listbox", { name: /^experiment run options$/i }),
      ).getByRole("option", {
        name: /exp_locked · baseline · Mnist · 2026-06-02 01:02:03/i,
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
