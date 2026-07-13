import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  createLogPresetDeletePlan: vi.fn(),
  createLogRunDeletePlan: vi.fn(),
  createMutationRequestOptions: vi.fn(),
  deleteLogExperiment: vi.fn(),
  deleteLogPreset: vi.fn(),
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
  createLogPresetDeletePlan: mocks.createLogPresetDeletePlan,
  createLogRunDeletePlan: mocks.createLogRunDeletePlan,
  createMutationRequestOptions: mocks.createMutationRequestOptions,
  deleteLogExperiment: mocks.deleteLogExperiment,
  deleteLogPreset: mocks.deleteLogPreset,
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
} from "@/features/workbench/state/logs/_use-logs-workspace-state";
import { type LogRun, type LogRunTags } from "@/lib/api";

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

function values(selection: readonly string[]) {
  return [...selection];
}

function logRunTags(runId: string, scalarTags: string[]): LogRunTags {
  return {
    runId,
    scalarTags,
    histogramTags: [],
    imageTags: [],
    textTags: [],
  };
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
      useLogsWorkspaceState({
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

describe("Logs workspace state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.fetchLogRunArtifacts.mockImplementation((runId: string) =>
      Promise.resolve({
        runId,
        params: {},
        metrics: {},
        artifacts: [],
        checkpoints: [],
      }),
    );
    mocks.createMutationRequestOptions.mockReturnValue({
      idempotencyKey: "logs-workspace-command",
    });
    mocks.createLogPresetDeletePlan.mockResolvedValue({
      candidateCount: 0,
      counts: { runs: 0, experiments: 0, datasets: 0, models: 0, presets: 0 },
      affected: { experiments: [], datasets: [], models: [], presets: [], runIds: [] },
      candidates: [],
      blockedByActiveJobs: [],
      canDelete: true,
    });
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
      expect(result.current.browser.filters.experiments.options.map((option) => option.value))
        .toEqual(["exp_a", "exp_b"]);
    });

    act(() => {
      result.current.browser.actions.toggleFilter("experiments", "exp_a");
    });

    await waitFor(() => {
      expect(result.current.browser.filters.datasets.selectedValues).toEqual(["Cifar10"]);
      expect(result.current.browser.filters.models.selectedValues).toEqual(["linears/linear"]);
      expect(result.current.browser.filters.presets.selectedValues).toEqual(["baseline"]);
    });
    expect(result.current.charts.visibleRuns.map((run) => run.id)).toEqual(["a-cifar"]);
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
      expect(result.current.browser.filters.experiments.options.map((option) => option.value))
        .toEqual(["exp_a", "exp_b"]);
    });

    act(() => {
      result.current.browser.actions.toggleFilter("experiments", "exp_a");
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
      expect(result.current.browser.filters.datasets.selectedValues).toEqual(["Cifar10"]);
      expect(result.current.browser.filters.models.selectedValues).toEqual(["linears/linear"]);
      expect(result.current.browser.filters.presets.selectedValues).toEqual(["baseline"]);
    });
    expect(result.current.charts.visibleRuns.map((run) => run.id)).toEqual(["a-cifar"]);
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

  it("keeps the current tag selection until a newly selected experiment has fresh tags", async () => {
    const combinedTags = deferred<{ runs: LogRunTags[] }>();
    mocks.fetchLogTags.mockImplementation(({ runIds }: { runIds: string[] }) => {
      if (runIds.includes("b-cifar")) {
        return combinedTags.promise;
      }
      return Promise.resolve({
        runs: runIds.map((runId) => logRunTags(runId, ["legacy/weight"])),
      });
    });
    const { result } = renderLogsWorkspaceState();

    await waitFor(() =>
      expect(result.current.browser.filters.experiments.options).toHaveLength(2),
    );
    act(() => {
      result.current.browser.actions.toggleFilter("experiments", "exp_a");
    });
    await waitFor(() =>
      expect(result.current.browser.filters.tags.options.map(({ value }) => value))
        .toEqual(["legacy/weight"]),
    );
    await waitFor(() =>
      expect(result.current.charts.visibleRunIds).toEqual(["a-cifar"]),
    );
    act(() => {
      result.current.browser.actions.toggleFilter("tags", "legacy/weight");
      result.current.browser.actions.toggleFilter("experiments", "exp_b");
    });

    await waitFor(() =>
      expect(mocks.fetchLogTags).toHaveBeenCalledWith(
        { runIds: ["a-cifar", "b-cifar"] },
        expect.any(Object),
      ),
    );
    expect(result.current.browser.filters.tags.selectedValues).toEqual([
      "legacy/weight",
    ]);

    act(() => {
      combinedTags.resolve({
        runs: [
          logRunTags("a-cifar", ["legacy/weight"]),
          logRunTags("b-cifar", ["validation/accuracy_epoch"]),
        ],
      });
    });

    await waitFor(() =>
      expect(result.current.browser.filters.tags.selectedValues).toEqual([
        "validation/accuracy_epoch",
      ]),
    );
  });

  it("preserves a selected tag shared by a newly selected experiment", async () => {
    mocks.fetchLogTags.mockImplementation(({ runIds }: { runIds: string[] }) =>
      Promise.resolve({
        runs: runIds.map((runId) =>
          logRunTags(
            runId,
            runId === "b-cifar"
              ? ["shared/metric", "custom/new"]
              : ["shared/metric"],
          ),
        ),
      }),
    );
    const { result } = renderLogsWorkspaceState();

    await waitFor(() =>
      expect(result.current.browser.filters.experiments.options).toHaveLength(2),
    );
    act(() => {
      result.current.browser.actions.toggleFilter("experiments", "exp_a");
    });
    await waitFor(() =>
      expect(result.current.browser.filters.tags.options.map(({ value }) => value))
        .toEqual(["shared/metric"]),
    );
    act(() => {
      result.current.browser.actions.toggleFilter("tags", "shared/metric");
    });
    await waitFor(() =>
      expect(result.current.browser.filters.tags.selectedValues).toEqual([
        "shared/metric",
      ]),
    );
    act(() => {
      result.current.browser.actions.toggleFilter("experiments", "exp_b");
    });

    await waitFor(() =>
      expect(result.current.browser.filters.tags.options.map(({ value }) => value))
        .toEqual(["custom/new", "shared/metric"]),
    );
    expect(result.current.browser.filters.tags.selectedValues).toEqual([
      "shared/metric",
    ]);
  });

  it("falls back to fresh non-standard tags when the previous selection is stale", async () => {
    mocks.fetchLogTags.mockImplementation(({ runIds }: { runIds: string[] }) =>
      Promise.resolve({
        runs: runIds.map((runId) =>
          logRunTags(
            runId,
            runId === "b-cifar"
              ? ["custom/first", "custom/second"]
              : ["legacy/weight"],
          ),
        ),
      }),
    );
    const { result } = renderLogsWorkspaceState();

    await waitFor(() =>
      expect(result.current.browser.filters.experiments.options).toHaveLength(2),
    );
    act(() => {
      result.current.browser.actions.toggleFilter("experiments", "exp_a");
    });
    await waitFor(() =>
      expect(result.current.browser.filters.tags.options.map(({ value }) => value))
        .toEqual(["legacy/weight"]),
    );
    act(() => {
      result.current.browser.actions.toggleFilter("tags", "legacy/weight");
    });
    await waitFor(() =>
      expect(result.current.browser.filters.tags.selectedValues).toEqual([
        "legacy/weight",
      ]),
    );
    act(() => {
      result.current.browser.actions.toggleFilter("experiments", "exp_b");
    });

    await waitFor(() =>
      expect(result.current.browser.filters.tags.selectedValues).toEqual([
        "custom/first",
        "custom/second",
      ]),
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
      expect(result.current.browser.filters.experiments.options.map((option) => option.value))
        .toEqual(["large_exp"]);
    });

    act(() => {
      result.current.browser.actions.toggleFilter("experiments", "large_exp");
    });

    await waitFor(() => {
      expect(result.current.browser.filters.datasets.options.map((option) => option.value))
        .toEqual(["Mnist", "ZebraSet"]);
      expect(result.current.browser.filters.models.options.map((option) => option.value))
        .toEqual(["linears/linear", "linears/wide_linear"]);
      expect(result.current.browser.filters.presets.options.map((option) => option.value))
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
      result.current.browser.actions.selectAll("datasets");
      result.current.browser.actions.selectAll("models");
      result.current.browser.actions.selectAll("presets");
    });

    await waitFor(() => {
      expect(result.current.charts.visibleRunIds).toHaveLength(100);
    });
    expect(result.current.browser.pagination.runs.loaded).toBe(100);
    expect(result.current.browser.pagination.runs.total).toBe(105);
    expect(result.current.browser.pagination.runs.canLoadMore).toBe(true);
    expect(result.current.browser.pagination.scalarTags.canLoadMore).toBe(false);

    act(() => {
      result.current.browser.actions.loadMoreRuns();
    });

    await waitFor(() => {
      expect(mocks.fetchLogRuns).toHaveBeenCalledWith(
        expect.objectContaining({
          filters: { experiment: ["large_exp"] },
          pagination: { limit: 100, offset: 100 },
        }),
        expect.any(Object),
      );
      expect(result.current.charts.visibleRunIds).toHaveLength(105);
    });
    expect(result.current.browser.pagination.runs.canLoadMore).toBe(false);
    await waitFor(() => {
      expect(mocks.fetchLogTags).toHaveBeenCalledWith(
        { runIds: runs.slice(0, 100).map((run) => run.id) },
        expect.any(Object),
      );
    });
    expect(result.current.browser.pagination.scalarTags.loadedRuns).toBe(100);
    expect(result.current.browser.pagination.scalarTags.totalRuns).toBe(105);
    expect(result.current.browser.pagination.scalarTags.canLoadMore).toBe(true);

    const datasetOptionsBefore = result.current.browser.filters.datasets.options.map(
      (option) => option.value,
    );
    const modelOptionsBefore = result.current.browser.filters.models.options.map(
      (option) => option.value,
    );
    const presetOptionsBefore = result.current.browser.filters.presets.options.map(
      (option) => option.value,
    );

    act(() => {
      result.current.browser.actions.loadMoreScalarTags();
    });

    await waitFor(() => {
      expect(mocks.fetchLogTags).toHaveBeenCalledWith(
        { runIds: runs.slice(100).map((run) => run.id) },
        expect.any(Object),
      );
    });
    await waitFor(() => {
      expect(result.current.browser.pagination.scalarTags.loadedRuns).toBe(105);
    });
    expect(result.current.browser.pagination.scalarTags.canLoadMore).toBe(false);
    expect(result.current.browser.filters.datasets.options.map((option) => option.value))
      .toEqual(datasetOptionsBefore);
    expect(result.current.browser.filters.models.options.map((option) => option.value))
      .toEqual(modelOptionsBefore);
    expect(result.current.browser.filters.presets.options.map((option) => option.value))
      .toEqual(presetOptionsBefore);
  });

  it("keeps an incomplete current target cold until a custom experiment is selected", async () => {
    const { result } = renderLogsWorkspaceState({
      enabled: true,
      targetScope: { ...targetScope, datasets: [] },
    });

    await waitFor(() => {
      expect(result.current.browser.filters.experiments.options.map((option) => option.value))
        .toEqual(["exp_a", "exp_b"]);
    });
    expect(mocks.fetchLogRuns).not.toHaveBeenCalled();
    expect(mocks.fetchLogTags).not.toHaveBeenCalled();

    act(() => {
      result.current.browser.actions.toggleFilter("experiments", "exp_a");
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
      expect(rendered.result.current.browser.filters.experiments.options).toHaveLength(2);
    });
    act(() => {
      rendered.result.current.browser.actions.toggleFilter("experiments", "exp_a");
    });
    await waitFor(() => {
      expect(rendered.result.current.browser.scope.mode).toBe("custom");
      expect(values(rendered.result.current.browser.filters.experiments.selectedValues)).toEqual(["exp_a"]);
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

    expect(rendered.result.current.browser.scope.mode).toBe("custom");
    expect(values(rendered.result.current.browser.filters.experiments.selectedValues)).toEqual(["exp_a"]);

    act(() => {
      rendered.result.current.browser.scope.useCurrentTarget();
    });

    await waitFor(() => {
      expect(rendered.result.current.browser.scope.mode).toBe("target");
      expect(values(rendered.result.current.browser.filters.experiments.selectedValues)).toEqual([]);
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
      expect(result.current.browser.filters.experiments.options).toHaveLength(2);
    });
    act(() => {
      result.current.commands.includeStartedExperiment("fresh_run");
    });

    await waitFor(() => {
      expect(result.current.browser.scope.mode).toBe("custom");
      expect(result.current.browser.filters.experiments.selectedValues).toEqual([]);
      expect(mocks.fetchLogRuns).toHaveBeenCalledWith(
        expect.objectContaining({ filters: { experiment: ["fresh_run"] } }),
        expect.any(Object),
      );
    });

    act(() => {
      result.current.browser.scope.useCurrentTarget();
      result.current.browser.scope.showAllRuns();
    });

    await waitFor(() => {
      expect(result.current.browser.scope.mode).toBe("custom");
      expect(values(result.current.browser.filters.experiments.selectedValues)).toEqual([
        "exp_a",
        "exp_b",
      ]);
    });
  });

  it("clears connection-scoped Logs selections through one private command", async () => {
    const { result } = renderLogsWorkspaceState();

    await waitFor(() => expect(result.current.browser.filters.experiments.options).toHaveLength(2));
    act(() => {
      result.current.commands.includeStartedExperiment("fresh_run");
      result.current.charts.commands.openRunDetail("a-cifar");
      result.current.deletion.actions.openExperiment({
        value: "fresh_run",
        label: "fresh_run",
        count: 1,
      });
    });
    await waitFor(() => {
      expect(result.current.browser.scope.mode).toBe("custom");
      expect(result.current.browser.filters.experiments.selectedValues).toEqual([]);
      expect(mocks.fetchLogRuns).toHaveBeenCalledWith(
        expect.objectContaining({ filters: { experiment: ["fresh_run"] } }),
        expect.any(Object),
      );
    });

    act(() => result.current.commands.clearForConnectionChange());

    expect(result.current.browser.scope.mode).toBe("target");
    expect(values(result.current.browser.filters.experiments.selectedValues)).toEqual([]);
    expect(result.current.detail.run).toBeUndefined();
    expect(result.current.deletion.operation).toBeNull();
    act(() => result.current.browser.scope.showAllRuns());
    expect(values(result.current.browser.filters.experiments.selectedValues)).toEqual(["exp_a", "exp_b"]);
  });

  it("owns selected Run Artifact loading behind the read-only detail projection", async () => {
    const artifacts = {
      runId: "a-cifar",
      params: { learning_rate: 0.001 },
      metrics: { validation_accuracy: 0.95 },
      artifacts: [],
      checkpoints: [],
    };
    mocks.fetchLogRunArtifacts.mockResolvedValue(artifacts);
    const { result } = renderLogsWorkspaceState();

    await waitFor(() => {
      expect(result.current.browser.filters.experiments.options).toHaveLength(2);
    });
    act(() => {
      result.current.browser.actions.toggleFilter("experiments", "exp_a");
    });
    await waitFor(() =>
      expect(result.current.charts.visibleRunIds).toContain("a-cifar"),
    );
    act(() => result.current.charts.commands.openRunDetail("a-cifar"));

    await waitFor(() => {
      expect(mocks.fetchLogRunArtifacts).toHaveBeenCalledWith(
        "a-cifar",
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
      expect(result.current.detail).toEqual({
        run: expect.objectContaining({ id: "a-cifar" }),
        artifacts,
        status: { isLoading: false, error: null },
      });
    });
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
    const sortedValues = (selection: readonly string[]) =>
      [...selection].sort((left, right) => left.localeCompare(right));

    await waitFor(() => expect(result.current.browser.filters.experiments.options).toHaveLength(2));
    act(() => result.current.browser.actions.toggleFilter("experiments", "exp_a"));
    await waitFor(() => {
      expect(result.current.browser.filters.tags.options.map((option) => option.value)).toEqual([
        "train/loss_epoch",
        "custom/tag",
      ]);
    });

    act(() => result.current.browser.actions.selectAll("tags"));
    expect(sortedValues(result.current.browser.filters.tags.selectedValues)).toEqual([
      "custom/tag",
      "train/loss_epoch",
    ]);
    act(() => result.current.browser.actions.selectNone("tags"));
    expect(result.current.browser.filters.tags.selectedValues).toHaveLength(0);
    act(() => result.current.browser.actions.toggleFilter("tags", "custom/tag"));
    expect(sortedValues(result.current.browser.filters.tags.selectedValues)).toEqual(["custom/tag"]);

    act(() => result.current.browser.actions.selectAll("datasets"));
    expect(sortedValues(result.current.browser.filters.datasets.selectedValues)).toEqual([
      "Cifar10",
      "Mnist",
    ]);
    act(() => result.current.browser.actions.selectNone("datasets"));
    expect(result.current.browser.filters.datasets.selectedValues).toHaveLength(0);
    act(() => result.current.browser.actions.toggleFilter("datasets", "Mnist"));
    expect(sortedValues(result.current.browser.filters.datasets.selectedValues)).toEqual(["Mnist"]);

    act(() => result.current.browser.actions.selectAll("models"));
    expect(sortedValues(result.current.browser.filters.models.selectedValues)).toEqual([
      "linears/linear",
      "linears/wide_linear",
    ]);
    act(() => result.current.browser.actions.selectNone("models"));
    expect(result.current.browser.filters.models.selectedValues).toHaveLength(0);
    act(() => result.current.browser.actions.toggleFilter("models", "linears/wide_linear"));
    expect(sortedValues(result.current.browser.filters.models.selectedValues)).toEqual([
      "linears/wide_linear",
    ]);

    act(() => result.current.browser.actions.selectAll("presets"));
    expect(sortedValues(result.current.browser.filters.presets.selectedValues)).toEqual([
      "baseline",
      "wide",
    ]);
    act(() => result.current.browser.actions.selectNone("presets"));
    expect(result.current.browser.filters.presets.selectedValues).toHaveLength(0);
    act(() => result.current.browser.actions.toggleFilter("presets", "wide"));
    expect(sortedValues(result.current.browser.filters.presets.selectedValues)).toEqual(["wide"]);

    act(() => result.current.browser.actions.selectAll("experiments"));
    expect(sortedValues(result.current.browser.filters.experiments.selectedValues)).toEqual([
      "exp_a",
      "exp_b",
    ]);
    act(() => result.current.browser.actions.selectNone("experiments"));
    expect(result.current.browser.filters.experiments.selectedValues).toHaveLength(0);
    act(() => result.current.browser.actions.toggleFilter("experiments", "exp_b"));
    expect(sortedValues(result.current.browser.filters.experiments.selectedValues)).toEqual(["exp_b"]);
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
    mocks.createLogPresetDeletePlan.mockResolvedValue(plan);
    mocks.deleteLogPreset.mockImplementation(() => {
      deleted = true;
      return Promise.resolve({
        ...plan,
        deletedRunIds: [freshRun.id],
        deletedRunCount: 1,
        deletedRelativePaths: [freshRun.relativePath],
      });
    });
    const { result } = renderLogsWorkspaceState();

    act(() => result.current.commands.includeStartedExperiment("fresh_run"));
    await waitFor(() => {
      expect(result.current.charts.visibleRunIds).toEqual([freshRun.id]);
      expect(values(result.current.browser.filters.experiments.selectedValues)).toEqual(["fresh_run"]);
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
      expect(result.current.browser.filters.experiments.options).toEqual([]);
      expect(values(result.current.browser.filters.experiments.selectedValues)).toEqual([]);
      expect(result.current.charts.visibleRunIds).toEqual([]);
      expect(result.current.detail.run).toBeUndefined();
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
    mocks.createLogPresetDeletePlan.mockResolvedValue(plan);
    mocks.deleteLogPreset.mockImplementation(() => {
      deleted = true;
      return Promise.resolve({
        ...plan,
        deletedRunIds: [freshRun.id],
        deletedRunCount: 1,
        deletedRelativePaths: [freshRun.relativePath],
      });
    });
    const { result } = renderLogsWorkspaceState();

    act(() => result.current.commands.includeStartedExperiment("fresh_run"));
    await waitFor(() => {
      expect(result.current.detail.run?.id).toBe(freshRun.id);
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
      expect(result.current.charts.visibleRunIds).toEqual([]);
      expect(result.current.detail.run).toBeUndefined();
    });

    act(() => runsRefresh.resolve({ runs: [] }));
    await waitFor(() => {
      expect(result.current.charts.visibleRunIds).toEqual([]);
      expect(result.current.detail.run).toBeUndefined();
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
    mocks.createLogPresetDeletePlan.mockResolvedValue(plan);
    mocks.deleteLogPreset.mockImplementation(() => deletion.promise);
    const rendered = renderLogsWorkspaceState();

    act(() => rendered.result.current.commands.includeStartedExperiment("fresh_run"));
    await waitFor(() => {
      expect(rendered.result.current.charts.visibleRunIds).toEqual([freshRun.id]);
      expect(values(rendered.result.current.browser.filters.experiments.selectedValues)).toEqual([
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
    expect(values(rendered.result.current.browser.filters.experiments.selectedValues)).toEqual([
      "fresh_run",
    ]);

    rendered.rerender({ enabled: true, targetScope });
    await waitFor(() => {
      expect(rendered.result.current.browser.filters.experiments.options).toEqual([]);
      expect(values(rendered.result.current.browser.filters.experiments.selectedValues)).toEqual([]);
      expect(rendered.result.current.charts.visibleRunIds).toEqual([]);
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
    mocks.createLogPresetDeletePlan.mockResolvedValue(plan);
    mocks.deleteLogPreset.mockImplementation(() => {
      deleted = true;
      return Promise.resolve({
        ...plan,
        deletedRunIds: [freshRun.id],
        deletedRunCount: 1,
        deletedRelativePaths: [freshRun.relativePath],
      });
    });
    const { result } = renderLogsWorkspaceState();

    act(() => result.current.commands.includeStartedExperiment("fresh_run"));
    await waitFor(() => {
      expect(result.current.charts.visibleRunIds).toEqual([freshRun.id]);
      expect(values(result.current.browser.filters.experiments.selectedValues)).toEqual(["fresh_run"]);
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
      expect(result.current.browser.status.experimentsError).toEqual(
        new Error("experiment refresh unavailable"),
      );
      expect(values(result.current.browser.filters.experiments.selectedValues)).toEqual(["fresh_run"]);
      expect(result.current.charts.visibleRunIds).toEqual([]);
    });

    await act(async () => result.current.browser.actions.refresh());
    await waitFor(() => {
      expect(result.current.browser.status.experimentsError).toBeNull();
      expect(result.current.browser.filters.experiments.options).toEqual([]);
      expect(values(result.current.browser.filters.experiments.selectedValues)).toEqual([]);
    });
  });
});
