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
  fetchLogRuns: vi.fn(),
  fetchLogExperiments: vi.fn(),
  fetchLogTags: vi.fn(),
  inspectModel: vi.fn(),
  fetchTrainingRunPlan: vi.fn(),
  createTrainingJob: vi.fn(),
  fetchTrainingJob: vi.fn(),
  cancelTrainingJob: vi.fn(),
}));
vi.mock("@/lib/api", () => mocks);

import { useViewerState } from "@/components/features/viewer/use-viewer-state";
import { ConnectedTrainingPanel } from "@/components/features/viewer/connected-training-panel";
import { ViewerProviders } from "@/components/features/viewer/providers/viewer-providers";
import { TargetPresetPanel } from "@/components/features/viewer/screen/target-preset-panel";
import { type LogRun } from "@/lib/api";

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
  mocks.fetchLogRuns.mockReset().mockResolvedValue({ runs: [] });
  mocks.fetchLogExperiments.mockReset().mockResolvedValue({ experiments: [] });
  mocks.fetchLogTags.mockReset().mockResolvedValue({ runs: [] });
  mocks.inspectModel.mockReset().mockResolvedValue({
    model: "linear",
    preset: "baseline",
    parameterCount: 0,
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

  it("syncs a newly selected historical run once when switching models", async () => {
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

    mocks.inspectModel.mockClear();
    act(() => {
      result.current.target.selectModel("linear");
    });

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.history.selectedLogRunId).toBe("linear-history");
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
});
