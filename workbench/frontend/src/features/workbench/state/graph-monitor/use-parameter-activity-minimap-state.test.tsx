import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  fetchLogParameterStatus: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    fetchLogParameterStatus: apiMocks.fetchLogParameterStatus,
  };
});

import { useParameterActivityMinimapState } from "@/features/workbench/state/graph-monitor/use-parameter-activity-minimap-state";
import {
  type GraphNode,
  type InspectResponse,
  type LogParameterStatusResponse,
  type LogRun,
} from "@/lib/api";

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

function status(runId: string): LogParameterStatusResponse {
  return {
    runs: [
      {
        sourceId: runId,
        preset: "baseline",
        dataset: "Mnist",
        logDir: "logs",
        nodes: [
          {
            nodePath: "main_model.linear",
            weights: {
              status: "updated",
              metric: "weights",
              lastStep: 2,
              observedPoints: 2,
            },
            bias: {
              status: "unchanged",
              metric: "bias",
              lastStep: 2,
              observedPoints: 2,
            },
          },
        ],
      },
    ],
  };
}

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return createElement(QueryClientProvider, { client }, children);
}

describe("useParameterActivityMinimapState", () => {
  beforeEach(() => {
    apiMocks.fetchLogParameterStatus.mockReset();
    apiMocks.fetchLogParameterStatus.mockResolvedValue(status("run-1"));
  });

  it("queries selected-run parameter status with only selectedExperimentRunId", async () => {
    const selectedRun = logRun("run-1");
    const { result } = renderHook(
      () =>
        useParameterActivityMinimapState({
          graph: inspectGraph(),
          selectedTargetMode: "experiment",
          selectedExperimentRunId: selectedRun.id,
          selectedLogRun: selectedRun,
          selectedLogRunMonitorEligibility: "eligible",
        }),
      { wrapper },
    );

    await waitFor(() =>
      expect(apiMocks.fetchLogParameterStatus).toHaveBeenCalledWith(
        { runIds: ["run-1"] },
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      ),
    );
    await waitFor(() => expect(result.current.canOpen).toBe(true));

    expect(result.current.selectedRunSource).toEqual({
      kind: "historical-run",
      run: selectedRun,
    });
    expect(result.current.parameterNodeCount).toBe(1);
    expect(result.current.activityByNodePath?.get("main_model.linear")?.bias)
      .toBeDefined();
  });

  it("hides outside experiment mode and disables while monitor eligibility is checking", () => {
    const selectedRun = logRun("run-1");
    const presetState = renderHook(
      () =>
        useParameterActivityMinimapState({
          graph: inspectGraph(),
          selectedTargetMode: "preset",
          selectedExperimentRunId: selectedRun.id,
          selectedLogRun: selectedRun,
          selectedLogRunMonitorEligibility: "eligible",
        }),
      { wrapper },
    );
    const checkingState = renderHook(
      () =>
        useParameterActivityMinimapState({
          graph: inspectGraph(),
          selectedTargetMode: "experiment",
          selectedExperimentRunId: selectedRun.id,
          selectedLogRun: selectedRun,
          selectedLogRunMonitorEligibility: "checking",
        }),
      { wrapper },
    );

    expect(presetState.result.current.shouldRenderButton).toBe(false);
    expect(checkingState.result.current.shouldRenderButton).toBe(true);
    expect(checkingState.result.current.canOpen).toBe(false);
    expect(checkingState.result.current.disabledReason).toMatch(/checking/i);
    expect(apiMocks.fetchLogParameterStatus).not.toHaveBeenCalled();
  });

  it("disables on selected-run status errors", async () => {
    apiMocks.fetchLogParameterStatus.mockRejectedValueOnce(new Error("boom"));
    const selectedRun = logRun("run-1");
    const { result } = renderHook(
      () =>
        useParameterActivityMinimapState({
          graph: inspectGraph(),
          selectedTargetMode: "experiment",
          selectedExperimentRunId: selectedRun.id,
          selectedLogRun: selectedRun,
          selectedLogRunMonitorEligibility: "eligible",
        }),
      { wrapper },
    );

    await waitFor(() => expect(result.current.disabledReason).toMatch(/could not/i));
    expect(result.current.canOpen).toBe(false);
  });
});
