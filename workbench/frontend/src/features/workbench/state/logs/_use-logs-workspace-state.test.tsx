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
  useLogsWorkspaceImplementation,
  type LogsTargetScope,
} from "@/features/workbench/state/logs/_use-logs-workspace-state";
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
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, reject, resolve };
}

const targetScope: LogsTargetScope = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  datasets: ["Cifar10"],
};

function renderLogsWorkspaceState(
  initialProps: {
    enabled: boolean;
    targetScope: LogsTargetScope;
  } = { enabled: true, targetScope },
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return renderHook(
    (props) =>
      useLogsWorkspaceImplementation({
        enabled: props.enabled,
        targetScope: props.targetScope,
      }),
    {
      initialProps,
      wrapper: ({ children }: { children: ReactNode }) =>
        createElement(QueryClientProvider, { client }, children),
    },
  );
}

describe("Logs workspace Implementation", () => {
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
      ({
        filters,
        pagination = { limit: 100, offset: 0 },
      }: {
        filters?: { experiment?: string[] };
        pagination?: { limit: number; offset?: number };
      } = {}) => {
        const filteredRuns = filters?.experiment
          ? runs.filter((run) => filters.experiment?.includes(run.experiment))
          : runs;
        const offset = pagination.offset ?? 0;
        return Promise.resolve({
          runs: filteredRuns.slice(offset, offset + pagination.limit),
          total: filteredRuns.length,
          limit: pagination.limit,
          offset,
          hasMore: offset + pagination.limit < filteredRuns.length,
          facets: {
            experiments: [
              {
                experiment: "large_exp",
                runCount: 105,
                datasets: [
                  { value: "Mnist", count: 100 },
                  { value: "ZebraSet", count: 5 },
                ],
                models: [
                  { modelType: "linears", model: "linear", count: 100 },
                  { modelType: "linears", model: "wide_linear", count: 5 },
                ],
                presets: [
                  { value: "AAA_CONTROL", count: 100 },
                  { value: "BASELINE", count: 5 },
                ],
              },
            ],
          },
        });
      },
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
      expect(result.current.visibleRunIds).toHaveLength(100);
    });
    expect(result.current.loadedRunCount).toBe(100);
    expect(result.current.totalRunCount).toBe(105);
    expect(result.current.canLoadMoreRuns).toBe(true);
    expect(result.current.canLoadMoreScalarTags).toBe(false);

    act(() => {
      result.current.loadMoreRuns();
    });

    await waitFor(() => {
      expect(mocks.fetchLogRuns).toHaveBeenCalledWith(
        expect.objectContaining({
          filters: { experiment: ["large_exp"] },
          pagination: { limit: 100, offset: 100 },
        }),
        expect.any(Object),
      );
      expect(result.current.visibleRunIds).toHaveLength(105);
    });
    expect(result.current.canLoadMoreRuns).toBe(false);
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

  it("keeps an incomplete current target cold until a custom experiment is selected", async () => {
    const { result } = renderLogsWorkspaceState({
      enabled: true,
      targetScope: { ...targetScope, datasets: [] },
    });

    await waitFor(() => {
      expect(result.current.experimentOptions.map((option) => option.value))
        .toEqual(["exp_a", "exp_b"]);
    });
    expect(mocks.fetchLogRuns).not.toHaveBeenCalled();
    expect(mocks.fetchLogTags).not.toHaveBeenCalled();

    act(() => {
      result.current.toggleExperiment("exp_a");
    });

    await waitFor(() => {
      expect(mocks.fetchLogRuns).toHaveBeenCalledWith(
        expect.objectContaining({ filters: { experiment: ["exp_a"] } }),
        expect.any(Object),
      );
    });
  });

  it("retains custom selections across target changes and returns explicitly", async () => {
    const rendered = renderLogsWorkspaceState();

    await waitFor(() => {
      expect(rendered.result.current.experimentOptions).toHaveLength(2);
    });
    act(() => {
      rendered.result.current.toggleExperiment("exp_a");
    });
    await waitFor(() => {
      expect(rendered.result.current.scopeMode).toBe("custom");
      expect(values(rendered.result.current.selectedExperiments)).toEqual(["exp_a"]);
    });

    rendered.rerender({
      enabled: true,
      targetScope: {
        modelType: "bert",
        model: "linear",
        preset: "pre-norm",
        datasets: ["PennTreebank"],
      },
    });

    expect(rendered.result.current.scopeMode).toBe("custom");
    expect(values(rendered.result.current.selectedExperiments)).toEqual(["exp_a"]);

    act(() => {
      rendered.result.current.useCurrentTargetScope();
    });

    await waitFor(() => {
      expect(rendered.result.current.scopeMode).toBe("target");
      expect(values(rendered.result.current.selectedExperiments)).toEqual([]);
      expect(mocks.fetchLogRuns).toHaveBeenCalledWith(
        expect.objectContaining({
          filters: expect.objectContaining({
            models: [{ modelType: "bert", model: "linear" }],
            preset: ["pre-norm"],
            dataset: ["PennTreebank"],
          }),
        }),
        expect.any(Object),
      );
    });
  });

  it("enters custom scope for a Training-created experiment and all-runs browsing", async () => {
    const { result } = renderLogsWorkspaceState();

    await waitFor(() => {
      expect(result.current.experimentOptions).toHaveLength(2);
    });
    act(() => {
      result.current.includeStartedExperiment("fresh_run");
    });

    await waitFor(() => {
      expect(result.current.scopeMode).toBe("custom");
      expect(values(result.current.selectedExperiments)).toEqual(["fresh_run"]);
      expect(mocks.fetchLogRuns).toHaveBeenCalledWith(
        expect.objectContaining({ filters: { experiment: ["fresh_run"] } }),
        expect.any(Object),
      );
    });

    act(() => {
      result.current.useCurrentTargetScope();
      result.current.showAllRuns();
    });

    await waitFor(() => {
      expect(result.current.scopeMode).toBe("custom");
      expect(values(result.current.selectedExperiments)).toEqual([
        "exp_a",
        "exp_b",
        "fresh_run",
      ]);
    });
  });

  it("clears connection-scoped Logs selections through one private command", async () => {
    const { result } = renderLogsWorkspaceState();

    await waitFor(() => expect(result.current.experimentOptions).toHaveLength(2));
    act(() => {
      result.current.includeStartedExperiment("fresh_run");
      result.current.setSelectedDetailRunId("a-cifar");
      result.current.deletion.actions.openExperiment({
        value: "fresh_run",
        label: "fresh_run",
        count: 1,
      });
    });
    await waitFor(() => {
      expect(result.current.scopeMode).toBe("custom");
      expect(values(result.current.selectedExperiments)).toEqual(["fresh_run"]);
    });

    act(() => result.current.clearForConnectionChange());

    expect(result.current.scopeMode).toBe("target");
    expect(values(result.current.selectedExperiments)).toEqual([]);
    expect(result.current.selectedDetailRunId).toBeNull();
    expect(result.current.deletion.operation).toBeNull();
    act(() => result.current.showAllRuns());
    expect(values(result.current.selectedExperiments)).toEqual(["exp_a", "exp_b"]);
  });

  it("owns all, none, and toggle transitions for every caller-facing filter", async () => {
    mocks.fetchLogTags.mockImplementation(({ runIds }: { runIds: string[] }) =>
      Promise.resolve({
        runs: runIds.map((runId) => ({
          runId,
          scalarTags: ["train/loss_epoch", "custom/tag"],
          histogramTags: [],
          imageTags: [],
          textTags: [],
        })),
      }),
    );
    const { result } = renderLogsWorkspaceState();
    const sortedValues = (selection: Set<string>) =>
      Array.from(selection).sort((left, right) => left.localeCompare(right));

    await waitFor(() => expect(result.current.experimentOptions).toHaveLength(2));
    act(() => result.current.toggleExperiment("exp_a"));
    await waitFor(() => {
      expect(result.current.tagOptions.map((option) => option.value)).toEqual([
        "train/loss_epoch",
        "custom/tag",
      ]);
    });

    act(() => result.current.selectAllTags());
    expect(sortedValues(result.current.selectedTags)).toEqual([
      "custom/tag",
      "train/loss_epoch",
    ]);
    act(() => result.current.selectNoTags());
    expect(result.current.selectedTags.size).toBe(0);
    act(() => result.current.toggleTag("custom/tag"));
    expect(sortedValues(result.current.selectedTags)).toEqual(["custom/tag"]);

    act(() => result.current.selectAllDatasets());
    expect(sortedValues(result.current.selectedDatasets)).toEqual([
      "Cifar10",
      "Mnist",
    ]);
    act(() => result.current.selectNoDatasets());
    expect(result.current.selectedDatasets.size).toBe(0);
    act(() => result.current.toggleDataset("Mnist"));
    expect(sortedValues(result.current.selectedDatasets)).toEqual(["Mnist"]);

    act(() => result.current.selectAllModels());
    expect(sortedValues(result.current.selectedModels)).toEqual([
      "linears/linear",
      "linears/wide_linear",
    ]);
    act(() => result.current.selectNoModels());
    expect(result.current.selectedModels.size).toBe(0);
    act(() => result.current.toggleModel("linears/wide_linear"));
    expect(sortedValues(result.current.selectedModels)).toEqual([
      "linears/wide_linear",
    ]);

    act(() => result.current.selectAllPresets());
    expect(sortedValues(result.current.selectedPresets)).toEqual([
      "baseline",
      "wide",
    ]);
    act(() => result.current.selectNoPresets());
    expect(result.current.selectedPresets.size).toBe(0);
    act(() => result.current.togglePreset("wide"));
    expect(sortedValues(result.current.selectedPresets)).toEqual(["wide"]);

    act(() => result.current.selectAllExperiments());
    expect(sortedValues(result.current.selectedExperiments)).toEqual([
      "exp_a",
      "exp_b",
    ]);
    act(() => result.current.selectNoExperiments());
    expect(result.current.selectedExperiments.size).toBe(0);
    act(() => result.current.toggleExperiment("exp_b"));
    expect(sortedValues(result.current.selectedExperiments)).toEqual(["exp_b"]);
  });

  it("prunes a Training-created experiment after its last loaded run is deleted", async () => {
    const freshRun = logRun({
      id: "fresh-run",
      experiment: "fresh_run",
      preset: "baseline",
    });
    let deleted = false;
    mocks.fetchLogExperiments.mockImplementation(() =>
      Promise.resolve({
        experiments: deleted
          ? []
          : [{ experiment: "fresh_run", runCount: 1, relativePath: "fresh_run" }],
      }),
    );
    mocks.fetchLogRuns.mockImplementation(() =>
      Promise.resolve({ runs: deleted ? [] : [freshRun] }),
    );
    const plan = {
      candidateCount: 1,
      counts: { runs: 1, experiments: 1, datasets: 1, models: 1, presets: 1 },
      affected: {
        experiments: ["fresh_run"],
        datasets: ["Cifar10"],
        models: [{ modelType: "linears", model: "linear" }],
        presets: ["baseline"],
      },
      candidates: [{ id: freshRun.id, relativePath: freshRun.relativePath }],
      blockedByActiveJobs: [],
      canDelete: true,
      truncated: false,
    };
    mocks.createLogRunDeletePlan.mockResolvedValue(plan);
    mocks.deleteLogRuns.mockImplementation(() => {
      deleted = true;
      return Promise.resolve({
        ...plan,
        deletedRunIds: [freshRun.id],
        deletedRunCount: 1,
        deletedRelativePaths: [freshRun.relativePath],
      });
    });
    const { result } = renderLogsWorkspaceState();

    act(() => result.current.includeStartedExperiment("fresh_run"));
    await waitFor(() => {
      expect(result.current.visibleRunIds).toEqual([freshRun.id]);
      expect(values(result.current.selectedExperiments)).toEqual(["fresh_run"]);
    });

    act(() => {
      result.current.deletion.actions.openPreset({
        value: "baseline",
        label: "baseline",
        count: 1,
      });
    });
    await waitFor(() => {
      expect(result.current.deletion.operation?.phase).toBe("ready");
    });
    await act(async () => result.current.deletion.actions.confirm());

    await waitFor(() => {
      expect(result.current.experimentOptions).toEqual([]);
      expect(values(result.current.selectedExperiments)).toEqual([]);
      expect(result.current.visibleRunIds).toEqual([]);
      expect(result.current.selectedRun).toBeUndefined();
    });
  });

  it("never reselects a deleted detail Run while its list refresh is pending", async () => {
    const freshRun = logRun({
      id: "fresh-run",
      experiment: "fresh_run",
      preset: "baseline",
    });
    let deleted = false;
    const runsRefresh = deferred<{ runs: LogRun[] }>();
    mocks.fetchLogExperiments.mockResolvedValue({
      experiments: [
        { experiment: "fresh_run", runCount: 1, relativePath: "fresh_run" },
      ],
    });
    mocks.fetchLogRuns.mockImplementation(() =>
      deleted ? runsRefresh.promise : Promise.resolve({ runs: [freshRun] }),
    );
    const plan = {
      candidateCount: 1,
      counts: { runs: 1, experiments: 1, datasets: 1, models: 1, presets: 1 },
      affected: {
        experiments: ["fresh_run"],
        datasets: ["Cifar10"],
        models: [{ modelType: "linears", model: "linear" }],
        presets: ["baseline"],
      },
      candidates: [{ id: freshRun.id, relativePath: freshRun.relativePath }],
      blockedByActiveJobs: [],
      canDelete: true,
      truncated: false,
    };
    mocks.createLogRunDeletePlan.mockResolvedValue(plan);
    mocks.deleteLogRuns.mockImplementation(() => {
      deleted = true;
      return Promise.resolve({
        ...plan,
        deletedRunIds: [freshRun.id],
        deletedRunCount: 1,
        deletedRelativePaths: [freshRun.relativePath],
      });
    });
    const { result } = renderLogsWorkspaceState();

    act(() => result.current.includeStartedExperiment("fresh_run"));
    await waitFor(() => {
      expect(result.current.selectedRun?.id).toBe(freshRun.id);
    });
    act(() => {
      result.current.deletion.actions.openPreset({
        value: "baseline",
        label: "baseline",
        count: 1,
      });
    });
    await waitFor(() => {
      expect(result.current.deletion.operation?.phase).toBe("ready");
    });
    const runRequestCountBeforeDeletion = mocks.fetchLogRuns.mock.calls.length;

    await act(async () => result.current.deletion.actions.confirm());
    await waitFor(() => {
      expect(mocks.fetchLogRuns.mock.calls.length).toBeGreaterThan(
        runRequestCountBeforeDeletion,
      );
      expect(result.current.visibleRunIds).toEqual([]);
      expect(result.current.selectedRun).toBeUndefined();
    });

    act(() => runsRefresh.resolve({ runs: [] }));
    await waitFor(() => {
      expect(result.current.visibleRunIds).toEqual([]);
      expect(result.current.selectedRun).toBeUndefined();
    });
  });

  it("reconciles a last-Run deletion only after a hidden experiment list refetch succeeds", async () => {
    const freshRun = logRun({
      id: "fresh-run",
      experiment: "fresh_run",
      preset: "baseline",
    });
    let deleted = false;
    mocks.fetchLogExperiments.mockImplementation(() =>
      Promise.resolve({
        experiments: deleted
          ? []
          : [{ experiment: "fresh_run", runCount: 1, relativePath: "fresh_run" }],
      }),
    );
    mocks.fetchLogRuns.mockImplementation(() =>
      Promise.resolve({ runs: deleted ? [] : [freshRun] }),
    );
    const plan = {
      candidateCount: 1,
      counts: { runs: 1, experiments: 1, datasets: 1, models: 1, presets: 1 },
      affected: {
        experiments: ["fresh_run"],
        datasets: ["Cifar10"],
        models: [{ modelType: "linears", model: "linear" }],
        presets: ["baseline"],
      },
      candidates: [{ id: freshRun.id, relativePath: freshRun.relativePath }],
      blockedByActiveJobs: [],
      canDelete: true,
      truncated: false,
    };
    const response = {
      ...plan,
      deletedRunIds: [freshRun.id],
      deletedRunCount: 1,
      deletedRelativePaths: [freshRun.relativePath],
    };
    const deletion = deferred<typeof response>();
    mocks.createLogRunDeletePlan.mockResolvedValue(plan);
    mocks.deleteLogRuns.mockImplementation(() => deletion.promise);
    const rendered = renderLogsWorkspaceState();

    act(() => rendered.result.current.includeStartedExperiment("fresh_run"));
    await waitFor(() => {
      expect(rendered.result.current.visibleRunIds).toEqual([freshRun.id]);
      expect(values(rendered.result.current.selectedExperiments)).toEqual([
        "fresh_run",
      ]);
    });

    act(() => {
      rendered.result.current.deletion.actions.openPreset({
        value: "baseline",
        label: "baseline",
        count: 1,
      });
    });
    await waitFor(() => {
      expect(rendered.result.current.deletion.operation?.phase).toBe("ready");
    });
    let confirmation: Promise<void> | undefined;
    act(() => {
      confirmation = rendered.result.current.deletion.actions.confirm();
    });
    await waitFor(() => {
      expect(rendered.result.current.deletion.operation?.phase).toBe("mutating");
    });

    rendered.rerender({ enabled: false, targetScope });
    await waitFor(() => {
      expect(rendered.result.current.deletion.operation).toBeNull();
    });
    act(() => {
      deleted = true;
      deletion.resolve(response);
    });
    await act(async () => {
      await confirmation;
    });
    expect(values(rendered.result.current.selectedExperiments)).toEqual([
      "fresh_run",
    ]);

    rendered.rerender({ enabled: true, targetScope });
    await waitFor(() => {
      expect(rendered.result.current.experimentOptions).toEqual([]);
      expect(values(rendered.result.current.selectedExperiments)).toEqual([]);
      expect(rendered.result.current.visibleRunIds).toEqual([]);
    });
  });

  it("retains last-Run reconciliation through a failed experiment list refetch", async () => {
    const freshRun = logRun({
      id: "fresh-run",
      experiment: "fresh_run",
      preset: "baseline",
    });
    let deleted = false;
    let failNextExperimentRefresh = true;
    mocks.fetchLogExperiments.mockImplementation(() => {
      if (!deleted) {
        return Promise.resolve({
          experiments: [
            { experiment: "fresh_run", runCount: 1, relativePath: "fresh_run" },
          ],
        });
      }
      if (failNextExperimentRefresh) {
        failNextExperimentRefresh = false;
        return Promise.reject(new Error("experiment refresh unavailable"));
      }
      return Promise.resolve({ experiments: [] });
    });
    mocks.fetchLogRuns.mockImplementation(() =>
      Promise.resolve({ runs: deleted ? [] : [freshRun] }),
    );
    const plan = {
      candidateCount: 1,
      counts: { runs: 1, experiments: 1, datasets: 1, models: 1, presets: 1 },
      affected: {
        experiments: ["fresh_run"],
        datasets: ["Cifar10"],
        models: [{ modelType: "linears", model: "linear" }],
        presets: ["baseline"],
      },
      candidates: [{ id: freshRun.id, relativePath: freshRun.relativePath }],
      blockedByActiveJobs: [],
      canDelete: true,
      truncated: false,
    };
    mocks.createLogRunDeletePlan.mockResolvedValue(plan);
    mocks.deleteLogRuns.mockImplementation(() => {
      deleted = true;
      return Promise.resolve({
        ...plan,
        deletedRunIds: [freshRun.id],
        deletedRunCount: 1,
        deletedRelativePaths: [freshRun.relativePath],
      });
    });
    const { result } = renderLogsWorkspaceState();

    act(() => result.current.includeStartedExperiment("fresh_run"));
    await waitFor(() => {
      expect(result.current.visibleRunIds).toEqual([freshRun.id]);
      expect(values(result.current.selectedExperiments)).toEqual(["fresh_run"]);
    });
    act(() => {
      result.current.deletion.actions.openPreset({
        value: "baseline",
        label: "baseline",
        count: 1,
      });
    });
    await waitFor(() => {
      expect(result.current.deletion.operation?.phase).toBe("ready");
    });
    await act(async () => result.current.deletion.actions.confirm());

    await waitFor(() => {
      expect(result.current.experimentsQuery.error).toEqual(
        new Error("experiment refresh unavailable"),
      );
      expect(values(result.current.selectedExperiments)).toEqual(["fresh_run"]);
      expect(result.current.visibleRunIds).toEqual([]);
    });

    await act(async () => result.current.refreshLogLists());
    await waitFor(() => {
      expect(result.current.experimentsQuery.error).toBeNull();
      expect(result.current.experimentOptions).toEqual([]);
      expect(values(result.current.selectedExperiments)).toEqual([]);
    });
  });
});
