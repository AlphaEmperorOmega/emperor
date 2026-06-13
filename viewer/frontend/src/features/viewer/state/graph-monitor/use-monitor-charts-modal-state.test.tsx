import { createElement, type ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  fetchLogRunMonitorData: vi.fn(),
  fetchMonitorData: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    fetchLogRunMonitorData: apiMocks.fetchLogRunMonitorData,
    fetchMonitorData: apiMocks.fetchMonitorData,
  };
});

import {
  emptyMonitorComparisonCandidateGroups,
  useMonitorChartsModalState,
} from "@/features/viewer/state/graph-monitor/use-monitor-charts-modal-state";
import {
  type GraphNode,
  type LogRun,
  type MonitorData,
  type TrainingJob,
} from "@/lib/api";
import { type MonitorChartsSource } from "@/types/monitor";

function graphNode(path: string): GraphNode {
  return {
    id: path,
    label: path,
    typeName: "LinearLayer",
    path,
    graphRole: "architecture",
    parameterCount: 2,
    parameterSizeBytes: 8,
    details: {},
    config: null,
  };
}

function trainingJob(overrides: Partial<TrainingJob> = {}): TrainingJob {
  return {
    id: overrides.id ?? "job-1",
    status: overrides.status ?? "running",
    model: overrides.model ?? "linears/linear",
    preset: overrides.preset ?? "baseline",
    presets: overrides.presets ?? ["baseline", "wide"],
    datasets: overrides.datasets ?? ["Mnist", "FashionMnist"],
    overrides: overrides.overrides ?? {},
    search: overrides.search ?? null,
    plannedRunCount: overrides.plannedRunCount ?? 1,
    runPlan: overrides.runPlan ?? null,
    monitors: overrides.monitors ?? ["linear"],
    logFolder: overrides.logFolder ?? "runs",
    createdAt: overrides.createdAt ?? "2026-06-01T00:00:00.000Z",
    updatedAt: overrides.updatedAt ?? "2026-06-01T00:00:00.000Z",
    exitCode: overrides.exitCode ?? null,
    pid: overrides.pid ?? 1234,
    currentPreset: overrides.currentPreset ?? "wide",
    currentDataset: overrides.currentDataset ?? "FashionMnist",
    epoch: overrides.epoch ?? 1,
    step: overrides.step ?? 10,
    metrics: overrides.metrics ?? {},
    logDir: overrides.logDir ?? "logs/runs",
    events: overrides.events ?? [],
    eventCount: overrides.eventCount ?? 0,
    eventCounts: overrides.eventCounts ?? {},
    eventsTruncated: overrides.eventsTruncated ?? false,
    clusterGrowth: overrides.clusterGrowth ?? [],
    logTail: overrides.logTail ?? [],
    resultLinks: overrides.resultLinks ?? [],
  };
}

function logRun(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  return {
    id: overrides.id,
    group: overrides.group ?? "exp",
    experiment: overrides.experiment ?? "exp",
    model: overrides.model ?? "linears/linear",
    preset: overrides.preset ?? "baseline",
    dataset: overrides.dataset ?? "Mnist",
    runName: overrides.runName ?? overrides.id,
    timestamp: overrides.timestamp ?? "2026-06-01 00:00:00",
    version: overrides.version ?? "version_0",
    relativePath: overrides.relativePath ?? `exp/${overrides.id}/version_0`,
    hasResult: overrides.hasResult ?? true,
    eventFileCount: overrides.eventFileCount ?? 1,
    checkpointCount: overrides.checkpointCount ?? 0,
    hasHparams: overrides.hasHparams ?? true,
    metrics: overrides.metrics ?? {},
  };
}

function monitorData(nodePath: string): MonitorData {
  return {
    jobId: "job-1",
    nodePath,
    preset: "wide",
    dataset: "FashionMnist",
    logDir: "logs/runs",
    scalarSeries: [
      {
        tag: `${nodePath}/weights_norm`,
        label: "weights_norm",
        points: [{ step: 1, wallTime: 100, value: 0.5 }],
      },
    ],
    histograms: [],
    images: [],
  };
}

function renderMonitorModalState({
  node = graphNode("main_model.0"),
  source,
  comparisonCandidateGroups = emptyMonitorComparisonCandidateGroups,
}: {
  node?: GraphNode;
  source: MonitorChartsSource;
  comparisonCandidateGroups?: Parameters<typeof useMonitorChartsModalState>[0]["comparisonCandidateGroups"];
}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return renderHook(
    () =>
      useMonitorChartsModalState({
        node,
        source,
        comparisonCandidateGroups,
      }),
    {
      wrapper: ({ children }: { children: ReactNode }) =>
        createElement(QueryClientProvider, { client }, children),
    },
  );
}

beforeEach(() => {
  apiMocks.fetchMonitorData.mockReset().mockImplementation(({ nodePath }) =>
    Promise.resolve(monitorData(nodePath)),
  );
  apiMocks.fetchLogRunMonitorData.mockReset().mockImplementation(({ nodePath }) =>
    Promise.resolve(monitorData(nodePath)),
  );
});

describe("useMonitorChartsModalState", () => {
  it("defaults active-job preset, dataset, and comparison scope from the source", async () => {
    const source: MonitorChartsSource = { kind: "active-job", job: trainingJob() };
    const { result } = renderMonitorModalState({
      source,
      comparisonCandidateGroups: {
        "same-stack": [graphNode("main_model.1")],
        "all-layers": [graphNode("main_model.1"), graphNode("main_model.2")],
      },
    });

    expect(result.current.dataset).toBe("FashionMnist");
    expect(result.current.preset).toBe("wide");
    expect(result.current.comparisonScope).toBe("same-stack");
    expect(result.current.comparisonCandidates.map((node) => node.path)).toEqual([
      "main_model.1",
    ]);

    await waitFor(() =>
      expect(apiMocks.fetchMonitorData).toHaveBeenCalledWith({
        jobId: "job-1",
        nodePath: "main_model.0",
        preset: "wide",
        dataset: "FashionMnist",
      }),
    );

    act(() => {
      result.current.setComparisonPath("main_model.1");
    });

    await waitFor(() =>
      expect(apiMocks.fetchMonitorData).toHaveBeenCalledWith({
        jobId: "job-1",
        nodePath: "main_model.1",
        preset: "wide",
        dataset: "FashionMnist",
      }),
    );
  });

  it("clears a selected comparison path when switching to a scope that excludes it", async () => {
    const source: MonitorChartsSource = { kind: "active-job", job: trainingJob() };
    const { result } = renderMonitorModalState({
      source,
      comparisonCandidateGroups: {
        "same-stack": [graphNode("main_model.1")],
        "all-layers": [graphNode("main_model.2")],
      },
    });

    act(() => {
      result.current.setComparisonPath("main_model.1");
      result.current.setComparisonScope("all-layers");
    });

    await waitFor(() => expect(result.current.comparisonPath).toBe(""));
    expect(result.current.comparisonCandidates.map((node) => node.path)).toEqual([
      "main_model.2",
    ]);
  });

  it("requests monitor data for every run in a historical run group", async () => {
    const runs = [logRun({ id: "run-1" }), logRun({ id: "run-2" })];
    const source: MonitorChartsSource = {
      kind: "historical-run-group",
      runs,
      experiment: "exp",
      dataset: "Mnist",
      preset: "baseline",
    };
    const { result } = renderMonitorModalState({ source });

    expect(result.current.historicalDataset).toBe("Mnist");
    expect(result.current.historicalPreset).toBe("baseline");
    expect(result.current.sourceDatasets).toEqual(["Mnist"]);
    expect(result.current.sourcePresets).toEqual(["baseline"]);

    await waitFor(() =>
      expect(apiMocks.fetchLogRunMonitorData).toHaveBeenCalledWith({
        runId: "run-1",
        nodePath: "main_model.0",
      }),
    );
    expect(apiMocks.fetchLogRunMonitorData).toHaveBeenCalledWith({
      runId: "run-2",
      nodePath: "main_model.0",
    });
  });
});
