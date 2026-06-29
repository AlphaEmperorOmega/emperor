import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  createLogRunDeletePlan: vi.fn(),
  deleteLogExperiment: vi.fn(),
  deleteLogRuns: vi.fn(),
  fetchLogCheckpoints: vi.fn(),
  fetchLogExperiments: vi.fn(),
  fetchLogMedia: vi.fn(),
  fetchLogRunArtifacts: vi.fn(),
  fetchLogRuns: vi.fn(),
  fetchLogScalars: vi.fn(),
  fetchLogTags: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  DEFAULT_LOG_SCALAR_MAX_POINTS: 500,
  LOG_SCALAR_SAMPLING: "auto",
  createLogRunDeletePlan: mocks.createLogRunDeletePlan,
  deleteLogExperiment: mocks.deleteLogExperiment,
  deleteLogRuns: mocks.deleteLogRuns,
  fetchLogCheckpoints: mocks.fetchLogCheckpoints,
  fetchLogExperiments: mocks.fetchLogExperiments,
  fetchLogMedia: mocks.fetchLogMedia,
  fetchLogRunArtifacts: mocks.fetchLogRunArtifacts,
  fetchLogRuns: mocks.fetchLogRuns,
  fetchLogScalars: mocks.fetchLogScalars,
  fetchLogTags: mocks.fetchLogTags,
}));

import {
  useLogsWorkspaceState,
  type LogsTargetScope,
} from "@/features/viewer/state/logs/use-logs-workspace-state";
import { type LogRun } from "@/lib/api";

function logRun(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  const experiment = overrides.experiment ?? "exp_a";
  const model = overrides.model ?? "linear";
  const preset = overrides.preset ?? "baseline";
  const dataset = overrides.dataset ?? "Cifar10";
  return {
    id: overrides.id,
    group: overrides.group ?? experiment,
    experiment,
    modelType: overrides.modelType ?? "linears",
    model,
    preset,
    dataset,
    runName: overrides.runName ?? overrides.id,
    timestamp: overrides.timestamp ?? null,
    version: overrides.version ?? "version_0",
    relativePath:
      overrides.relativePath ??
      `${experiment}/${model}/${preset}/${dataset}/${overrides.id}/version_0`,
    hasResult: overrides.hasResult ?? true,
    eventFileCount: overrides.eventFileCount ?? 1,
    checkpointCount: overrides.checkpointCount ?? 0,
    hasHparams: overrides.hasHparams ?? true,
    metrics: overrides.metrics ?? {},
  };
}

function values(selection: Set<string>) {
  return Array.from(selection);
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve;
  });
  return { promise, resolve };
}

const targetScope: LogsTargetScope = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  datasets: ["Cifar10"],
};

function renderLogsWorkspaceState() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return renderHook(
    () =>
      useLogsWorkspaceState({
        enabled: true,
        targetScope,
      }),
    {
      wrapper: ({ children }: { children: ReactNode }) =>
        createElement(QueryClientProvider, { client }, children),
    },
  );
}

describe("useLogsWorkspaceState", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    const runs = [
      logRun({ id: "a-cifar" }),
      logRun({
        id: "a-mnist",
        dataset: "Mnist",
        model: "wide_linear",
        preset: "wide",
      }),
      logRun({ id: "b-cifar", experiment: "exp_b" }),
    ];

    mocks.fetchLogExperiments.mockResolvedValue({
      experiments: [
        { experiment: "exp_a", runCount: 2, relativePath: "exp_a" },
        { experiment: "exp_b", runCount: 1, relativePath: "exp_b" },
      ],
    });
    mocks.fetchLogRuns.mockImplementation(
      ({ filters }: { filters?: { experiment?: string[] } } = {}) => {
        const selectedExperiments = filters?.experiment;
        return Promise.resolve({
          runs: selectedExperiments
            ? runs.filter((run) => selectedExperiments.includes(run.experiment))
            : runs,
        });
      },
    );
    mocks.fetchLogTags.mockResolvedValue({ runs: [] });
  });

  it("selects only the first dataset, model, and preset when an experiment is selected", async () => {
    const { result } = renderLogsWorkspaceState();

    await waitFor(() => {
      expect(result.current.experimentOptions.map((option) => option.value))
        .toEqual(["exp_a", "exp_b"]);
    });

    act(() => {
      result.current.toggleExperiment("exp_a");
    });

    await waitFor(() => {
      expect(values(result.current.selectedDatasets)).toEqual(["Cifar10"]);
      expect(values(result.current.selectedModels)).toEqual(["linears/linear"]);
      expect(values(result.current.selectedPresets)).toEqual(["baseline"]);
    });
    expect(result.current.visibleRuns.map((run) => run.id)).toEqual(["a-cifar"]);
  });

  it("waits for fresh experiment runs before selecting first dataset, model, and preset", async () => {
    const customRuns = [
      logRun({ id: "a-cifar" }),
      logRun({
        id: "a-mnist",
        dataset: "Mnist",
        model: "wide_linear",
        preset: "wide",
      }),
    ];
    const experimentRuns = deferred<{ runs: LogRun[] }>();
    mocks.fetchLogRuns.mockImplementation(
      ({ filters }: { filters?: { experiment?: string[] } } = {}) => {
        if (filters?.experiment?.includes("exp_a")) {
          return experimentRuns.promise;
        }
        return Promise.resolve({ runs: [] });
      },
    );
    const { result } = renderLogsWorkspaceState();

    await waitFor(() => {
      expect(result.current.experimentOptions.map((option) => option.value))
        .toEqual(["exp_a", "exp_b"]);
    });

    act(() => {
      result.current.toggleExperiment("exp_a");
    });
    await waitFor(() => {
      expect(mocks.fetchLogRuns).toHaveBeenCalledWith(
        expect.objectContaining({
          filters: expect.objectContaining({ experiment: ["exp_a"] }),
        }),
        expect.any(Object),
      );
    });

    act(() => {
      experimentRuns.resolve({ runs: customRuns });
    });

    await waitFor(() => {
      expect(values(result.current.selectedDatasets)).toEqual(["Cifar10"]);
      expect(values(result.current.selectedModels)).toEqual(["linears/linear"]);
      expect(values(result.current.selectedPresets)).toEqual(["baseline"]);
    });
    expect(result.current.visibleRuns.map((run) => run.id)).toEqual(["a-cifar"]);
    await waitFor(() => {
      expect(mocks.fetchLogTags).toHaveBeenCalledWith(
        { runIds: ["a-cifar"] },
        expect.any(Object),
      );
    });
    expect(mocks.fetchLogTags).not.toHaveBeenCalledWith(
      { runIds: ["a-cifar", "a-mnist"] },
      expect.any(Object),
    );
  });

  it("uses complete custom run metadata while scalar tags load by visible-run window", async () => {
    const runs = Array.from({ length: 105 }, (_, index) => {
      const lateRun = index >= 100;
      return logRun({
        id: `large-${String(index + 1).padStart(3, "0")}`,
        experiment: "large_exp",
        dataset: lateRun ? "ZebraSet" : "Mnist",
        model: lateRun ? "wide_linear" : "linear",
        preset: lateRun ? "BASELINE" : "AAA_CONTROL",
      });
    });
    mocks.fetchLogExperiments.mockResolvedValue({
      experiments: [
        { experiment: "large_exp", runCount: runs.length, relativePath: "large_exp" },
      ],
    });
    mocks.fetchLogRuns.mockImplementation(
      ({ filters }: { filters?: { experiment?: string[] } } = {}) =>
        Promise.resolve({
          runs: filters?.experiment
            ? runs.filter((run) => filters.experiment?.includes(run.experiment))
            : runs,
          total: runs.length,
          limit: 100,
          offset: 0,
          hasMore: false,
        }),
    );
    mocks.fetchLogTags.mockImplementation(({ runIds }: { runIds: string[] }) =>
      Promise.resolve({
        runs: runIds.map((runId) => ({
          runId,
          scalarTags: ["train/loss"],
          histogramTags: [],
          imageTags: [],
          textTags: [],
        })),
      }),
    );

    const { result } = renderLogsWorkspaceState();

    await waitFor(() => {
      expect(result.current.experimentOptions.map((option) => option.value))
        .toEqual(["large_exp"]);
    });

    act(() => {
      result.current.toggleExperiment("large_exp");
    });

    await waitFor(() => {
      expect(result.current.datasetOptions.map((option) => option.value))
        .toEqual(["Mnist", "ZebraSet"]);
      expect(result.current.modelOptions.map((option) => option.value))
        .toEqual(["linears/linear", "linears/wide_linear"]);
      expect(result.current.presetOptions.map((option) => option.value))
        .toEqual(["AAA_CONTROL", "BASELINE"]);
    });
    expect(mocks.fetchLogRuns).toHaveBeenCalledWith(
      expect.objectContaining({
        filters: { experiment: ["large_exp"] },
        includeAllPages: true,
        pagination: { limit: 100, offset: 0 },
      }),
      expect.any(Object),
    );

    act(() => {
      result.current.selectAllDatasets();
      result.current.selectAllModels();
      result.current.selectAllPresets();
    });

    await waitFor(() => {
      expect(result.current.visibleRunIds).toHaveLength(105);
    });
    await waitFor(() => {
      expect(mocks.fetchLogTags).toHaveBeenCalledWith(
        { runIds: runs.slice(0, 100).map((run) => run.id) },
        expect.any(Object),
      );
    });
    expect(result.current.loadedScalarTagRunCount).toBe(100);
    expect(result.current.totalScalarTagRunCount).toBe(105);
    expect(result.current.canLoadMoreScalarTags).toBe(true);

    const datasetOptionsBefore = result.current.datasetOptions.map(
      (option) => option.value,
    );
    const modelOptionsBefore = result.current.modelOptions.map(
      (option) => option.value,
    );
    const presetOptionsBefore = result.current.presetOptions.map(
      (option) => option.value,
    );

    act(() => {
      result.current.loadMoreScalarTags();
    });

    await waitFor(() => {
      expect(mocks.fetchLogTags).toHaveBeenCalledWith(
        { runIds: runs.slice(100).map((run) => run.id) },
        expect.any(Object),
      );
    });
    await waitFor(() => {
      expect(result.current.loadedScalarTagRunCount).toBe(105);
    });
    expect(result.current.canLoadMoreScalarTags).toBe(false);
    expect(result.current.datasetOptions.map((option) => option.value))
      .toEqual(datasetOptionsBefore);
    expect(result.current.modelOptions.map((option) => option.value))
      .toEqual(modelOptionsBefore);
    expect(result.current.presetOptions.map((option) => option.value))
      .toEqual(presetOptionsBefore);
  });
});
