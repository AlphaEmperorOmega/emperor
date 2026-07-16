import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  fetchMonitorParameterStatus: vi.fn(),
  fetchLogParameterStatus: vi.fn(),
}));

vi.mock("@/lib/api/monitor-data", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api/monitor-data")>()),
  fetchMonitorParameterStatus: apiMocks.fetchMonitorParameterStatus,
  fetchLogParameterStatus: apiMocks.fetchLogParameterStatus,
}));

import type { GraphNode, InspectResponse } from "@/lib/api/inspection";
import type { LogParameterStatusResponse, ParameterStatus } from "@/lib/api/monitor-data";
import type { LogRun } from "@/lib/api/logs";
import { useExperimentMonitorParameterActivity } from "@/features/workbench/state/graph-monitor/use-experiment-monitor-parameter-activity";
import { useMonitorSourceOrchestration } from "@/features/workbench/state/graph-monitor/use-monitor-source-orchestration";
import { type ActiveMonitorJob, type MonitorChartsSource } from "@/types/monitor";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function node(id: string, overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id,
    label: overrides.label ?? id,
    typeName: overrides.typeName ?? id,
    path: overrides.path ?? id,
    graphRole: overrides.graphRole ?? "architecture",
    parameterCount: overrides.parameterCount ?? 0,
    parameterSizeBytes: overrides.parameterSizeBytes ?? 0,
    details: overrides.details ?? {},
    config: overrides.config ?? null,
  };
}

function inspectGraph(): InspectResponse {
  return {
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    parameterCount: 2,
    parameterSizeBytes: 8,
    nodes: [
      node("model", { path: "main_model", typeName: "Model" }),
      node("linear", {
        path: "main_model.linear",
        typeName: "LinearLayer",
        parameterCount: 2,
        parameterSizeBytes: 8,
        details: { weightShape: "1 x 2", biasShape: "1" },
      }),
    ],
    edges: [{ id: "model-linear", source: "model", target: "linear" }],
  };
}

function logRun(id: string): LogRun {
  return {
    id,
    group: "exp",
    experiment: "exp",
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    dataset: "Mnist",
    runName: id,
    timestamp: "2026-06-01 00:00:00",
    version: "version_0",
    relativePath: `exp/${id}/version_0`,
    hasResult: true,
    eventFileCount: 1,
    checkpointCount: 0,
    hasHparams: true,
    metrics: {},
    hasLayerMonitorData: true,
  };
}

function parameterNode(nodePath = "main_model.linear", updated = true) {
  return {
    nodePath,
    weights: {
      status: updated ? ("updated" as const) : ("unchanged" as const),
      metric: `${nodePath}/weights`,
      lastStep: 2,
      observedPoints: 2,
    },
    bias: {
      status: "unchanged" as const,
      metric: `${nodePath}/bias`,
      lastStep: 2,
      observedPoints: 2,
    },
  };
}

function activeStatus(nodePath?: string): ParameterStatus {
  return {
    sourceId: "job-1",
    preset: "fast",
    dataset: "FashionMnist",
    logDir: "logs/job-1",
    nodes: [parameterNode(nodePath)],
  };
}

function historicalStatus(
  runs: LogRun[],
  nodePath = "main_model.linear",
): LogParameterStatusResponse {
  return {
    runs: runs.map((run, index) => ({
      sourceId: run.id,
      preset: run.preset,
      dataset: run.dataset,
      logDir: `logs/${run.id}`,
      nodes: [parameterNode(nodePath, index === 0)],
    })),
  };
}

function activeSource(
  overrides: Partial<Extract<MonitorChartsSource, { kind: "active-job" }>["job"]> = {},
): MonitorChartsSource {
  return {
    kind: "active-job",
    job: {
      id: overrides.id ?? "job-1",
      status: overrides.status ?? "running",
      monitors: overrides.monitors ?? ["linear"],
      preset: overrides.preset ?? "baseline",
      presets: overrides.presets ?? ["baseline", "fast"],
      datasets: overrides.datasets ?? ["Mnist"],
      logFolder: overrides.logFolder ?? "runs",
      currentPreset: overrides.currentPreset ?? "fast",
      currentDataset: overrides.currentDataset ?? "FashionMnist",
    },
  };
}

function historicalSource(runs: LogRun[]): MonitorChartsSource {
  return runs.length === 1
    ? { kind: "historical-run", run: runs[0]! }
    : {
        kind: "historical-run-group",
        runs,
        experiment: "exp",
        dataset: "Mnist",
        preset: "baseline",
      };
}

function createWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(QueryClientProvider, { client }, children);
  };
}

describe("useExperimentMonitorParameterActivity", () => {
  beforeEach(() => {
    apiMocks.fetchMonitorParameterStatus.mockReset();
    apiMocks.fetchLogParameterStatus.mockReset();
  });

  it("loads active Training Job activity with current Run identity", async () => {
    const request = deferred<ParameterStatus>();
    apiMocks.fetchMonitorParameterStatus.mockReturnValue(request.promise);
    const source = activeSource();
    const { result } = renderHook(
      () =>
        useExperimentMonitorParameterActivity({
          graph: inspectGraph(),
          source,
          fallbackPreset: "fallback-preset",
          fallbackDataset: "fallback-dataset",
        }),
      { wrapper: createWrapper() },
    );

    await waitFor(() =>
      expect(apiMocks.fetchMonitorParameterStatus).toHaveBeenCalledWith(
        {
          jobId: "job-1",
          preset: "fast",
          dataset: "FashionMnist",
        },
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      ),
    );
    expect(result.current.isLoading).toBe(true);
    expect(
      result.current.activityByNodePath?.get("main_model.linear")?.weights.status,
    ).toBe("loading");
    expect(apiMocks.fetchLogParameterStatus).not.toHaveBeenCalled();

    act(() => request.resolve(activeStatus()));
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(
      result.current.activityByNodePath?.get("main_model.linear")?.weights.status,
    ).toBe("updated");
    expect(result.current.isPathMismatch).toBe(false);
    expect(result.current.isError).toBe(false);
  });

  it("switches query identity from one historical Run to a Run group", async () => {
    const runA = logRun("run-a");
    const runB = logRun("run-b");
    apiMocks.fetchLogParameterStatus.mockImplementation(
      ({ runIds }: { runIds: string[] }) =>
        Promise.resolve(
          historicalStatus(
            [runA, runB].filter((run) => runIds.includes(run.id)),
          ),
        ),
    );
    const rendered = renderHook(
      ({ source }: { source: MonitorChartsSource }) =>
        useExperimentMonitorParameterActivity({
          graph: inspectGraph(),
          source,
        }),
      {
        initialProps: { source: historicalSource([runA]) },
        wrapper: createWrapper(),
      },
    );

    await waitFor(() =>
      expect(apiMocks.fetchLogParameterStatus).toHaveBeenCalledWith(
        { runIds: ["run-a"] },
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      ),
    );
    await waitFor(() =>
      expect(
        rendered.result.current.activityByNodePath?.get("main_model.linear")
          ?.weights.totalRuns,
      ).toBe(1),
    );

    rendered.rerender({ source: historicalSource([runA, runB]) });
    await waitFor(() =>
      expect(apiMocks.fetchLogParameterStatus).toHaveBeenCalledWith(
        { runIds: ["run-a", "run-b"] },
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      ),
    );
    await waitFor(() =>
      expect(
        rendered.result.current.activityByNodePath?.get("main_model.linear")
          ?.weights,
      ).toMatchObject({ status: "mixed", totalRuns: 2 }),
    );
    expect(apiMocks.fetchMonitorParameterStatus).not.toHaveBeenCalled();
  });

  it("gates requests by protected access, caller readiness, and monitor support", async () => {
    apiMocks.fetchMonitorParameterStatus.mockResolvedValue(activeStatus());
    const rendered = renderHook(
      ({
        source,
        enabled,
        protectedReadsEnabled,
      }: {
        source: MonitorChartsSource;
        enabled: boolean;
        protectedReadsEnabled: boolean;
      }) =>
        useExperimentMonitorParameterActivity({
          graph: inspectGraph(),
          source,
          enabled,
          protectedReadsEnabled,
        }),
      {
        initialProps: {
          source: activeSource(),
          enabled: true,
          protectedReadsEnabled: false,
        },
        wrapper: createWrapper(),
      },
    );

    expect(apiMocks.fetchMonitorParameterStatus).not.toHaveBeenCalled();
    expect(rendered.result.current.isLoading).toBe(false);

    rendered.rerender({
      source: activeSource(),
      enabled: false,
      protectedReadsEnabled: true,
    });
    expect(apiMocks.fetchMonitorParameterStatus).not.toHaveBeenCalled();

    rendered.rerender({
      source: activeSource({ id: "job-without-linear", monitors: ["halting"] }),
      enabled: true,
      protectedReadsEnabled: true,
    });
    expect(apiMocks.fetchMonitorParameterStatus).not.toHaveBeenCalled();

    rendered.rerender({
      source: activeSource(),
      enabled: true,
      protectedReadsEnabled: true,
    });
    await waitFor(() =>
      expect(apiMocks.fetchMonitorParameterStatus).toHaveBeenCalledTimes(1),
    );
  });

  it("publishes failures without retaining loading state", async () => {
    apiMocks.fetchLogParameterStatus.mockRejectedValue(new Error("status failed"));
    const run = logRun("failed-run");
    const { result } = renderHook(
      () =>
        useExperimentMonitorParameterActivity({
          graph: inspectGraph(),
          source: historicalSource([run]),
        }),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error).toEqual(new Error("status failed"));
    expect(result.current.isLoading).toBe(false);
    expect(result.current.isPathMismatch).toBe(false);
    expect(
      result.current.activityByNodePath?.get("main_model.linear")?.weights.status,
    ).toBe("unknown");
  });

  it("detects status paths that cannot attach to the inspected graph", async () => {
    const run = logRun("mismatch-run");
    apiMocks.fetchLogParameterStatus.mockResolvedValue(
      historicalStatus([run], "unrelated.linear"),
    );
    const { result } = renderHook(
      () =>
        useExperimentMonitorParameterActivity({
          graph: inspectGraph(),
          source: historicalSource([run]),
        }),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isPathMismatch).toBe(true));
    expect(result.current.isLoading).toBe(false);
    expect(result.current.isError).toBe(false);
  });

  it("adapts the main graph from active Job priority to historical activity", async () => {
    const run = logRun("historical-run");
    apiMocks.fetchMonitorParameterStatus.mockResolvedValue(activeStatus());
    apiMocks.fetchLogParameterStatus.mockResolvedValue(historicalStatus([run]));
    const source = activeSource();
    if (source.kind !== "active-job") {
      throw new Error("Expected active source fixture");
    }
    const rendered = renderHook(
      ({ activeTrainingJob }: { activeTrainingJob?: ActiveMonitorJob }) =>
        useMonitorSourceOrchestration({
          graph: inspectGraph(),
          activeTrainingJob,
          historicalMonitorRuns: [run],
          selectedHistoricalExperiment: run.experiment,
          selectedHistoricalDataset: run.dataset,
          selectedHistoricalPreset: run.preset,
          logRunTags: [],
          filteredHistoricalRunIds: [run.id],
          targetPreset: "baseline",
          targetDatasets: ["Mnist"],
        }),
      {
        initialProps: {
          activeTrainingJob: source.job,
        } as { activeTrainingJob?: ActiveMonitorJob },
        wrapper: createWrapper(),
      },
    );

    await waitFor(() =>
      expect(apiMocks.fetchMonitorParameterStatus).toHaveBeenCalledTimes(1),
    );
    expect(apiMocks.fetchLogParameterStatus).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(
        rendered.result.current.parameterActivityByNodePath?.get(
          "main_model.linear",
        )?.weights.source,
      ).toBe("active-job"),
    );

    rendered.rerender({ activeTrainingJob: undefined });
    await waitFor(() =>
      expect(apiMocks.fetchLogParameterStatus).toHaveBeenCalledWith(
        { runIds: [run.id] },
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      ),
    );
    await waitFor(() =>
      expect(
        rendered.result.current.parameterActivityByNodePath?.get(
          "main_model.linear",
        )?.weights,
      ).toMatchObject({ source: "historical", totalRuns: 1 }),
    );
  });
});
