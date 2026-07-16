import { createElement, type ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { LogRun } from "@/lib/api/logs";

const mocks = vi.hoisted(() => ({
  createMutationRequestOptions: vi.fn(),
  createPlan: vi.fn(),
  deleteExperiment: vi.fn(),
  deleteRuns: vi.fn(),
}));

vi.mock("@/lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api/client")>()),
  createMutationRequestOptions: mocks.createMutationRequestOptions,
}));
vi.mock("@/lib/api/deletion", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api/deletion")>()),
  createLogPresetDeletePlan: mocks.createPlan,
  deleteLogExperiment: mocks.deleteExperiment,
  deleteLogPreset: mocks.deleteRuns,
}));

import { useLogsDeletionState } from "@/features/workbench/state/logs/_logs-deletion-state";

function deferred<Value>() {
  let resolve!: (value: Value) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<Value>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function logRun(id: string): LogRun {
  return {
    id,
    group: "exp_a",
    experiment: "exp_a",
    modelType: "linears",
    model: "linear",
    preset: "BASELINE",
    dataset: "Mnist",
    runName: id,
    timestamp: null,
    version: "version_0",
    relativePath: `exp_a/linear/BASELINE/Mnist/${id}/version_0`,
    hasResult: true,
    eventFileCount: 1,
    checkpointCount: 0,
    hasHparams: true,
    metrics: {},
  };
}

const run = logRun("run-a");
const target = { experiment: "exp_a", preset: "BASELINE" };
const plan = {
  candidateCount: 1,
  counts: { runs: 1, experiments: 1, datasets: 1, models: 1, presets: 1 },
  affected: {
    experiments: ["exp_a"],
    datasets: ["Mnist"],
    models: [{ modelType: "linears", model: "linear" }],
    presets: ["BASELINE"],
  },
  candidates: [{ id: run.id, relativePath: run.relativePath }],
  blockedByActiveJobs: [],
  canDelete: true,
  truncated: false,
};
const deleteResult = {
  ...plan,
  deletedRunIds: [run.id],
  deletedRunCount: 1,
  deletedRelativePaths: [run.relativePath],
};

function renderDeletion(
  overrides: Partial<{
    active: boolean;
    enabled: boolean;
    selectedExperiments: Set<string>;
  }> = {},
) {
  const onExperimentDeleted = vi.fn();
  const onRunsDeleted = vi.fn();
  const props = {
    active: true,
    enabled: true,
    selectedExperiments: new Set(["exp_a"]),
    ...overrides,
  };
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false }, queries: { retry: false } },
  });
  const rendered = renderHook(
    (currentProps) =>
      useLogsDeletionState({
        ...currentProps,
        onExperimentDeleted,
        onRunsDeleted,
      }),
    {
      initialProps: props,
      wrapper: ({ children }: { children: ReactNode }) =>
        createElement(QueryClientProvider, { client }, children),
    },
  );
  return { ...rendered, onExperimentDeleted, onRunsDeleted };
}

beforeEach(() => {
  mocks.createMutationRequestOptions
    .mockReset()
    .mockImplementation(() => ({
      idempotencyKey: `command-${mocks.createMutationRequestOptions.mock.calls.length}`,
    }));
  mocks.createPlan.mockReset();
  mocks.deleteExperiment.mockReset();
  mocks.deleteRuns.mockReset();
});

describe("Logs deletion lifecycle", () => {
  it("plans a semantic preset target even when no matching Runs are loaded locally", async () => {
    mocks.createPlan.mockResolvedValue(plan);
    const { result } = renderDeletion();

    act(() => {
      result.current.actions.openPreset({
        value: "BASELINE",
        label: "BASELINE",
        count: 2,
      });
    });
    await waitFor(() => expect(result.current.operation?.phase).toBe("ready"));

    expect(mocks.createPlan).toHaveBeenCalledWith(
      target,
      expect.any(Object),
    );
  });

  it("enforces a disabled capability before planning or mutation", async () => {
    const { result } = renderDeletion({ enabled: false });

    act(() => {
      result.current.actions.openExperiment({
        value: "exp_a",
        label: "exp_a",
        count: 1,
      });
      result.current.actions.openPreset({
        value: "BASELINE",
        label: "BASELINE",
        count: 1,
      });
    });
    await act(async () => result.current.actions.confirm());

    expect(result.current.operation).toBeNull();
    expect(mocks.createPlan).not.toHaveBeenCalled();
    expect(mocks.deleteExperiment).not.toHaveBeenCalled();
    expect(mocks.deleteRuns).not.toHaveBeenCalled();
  });

  it("ignores a stale plan completion after cancellation", async () => {
    const pendingPlan = deferred<typeof plan>();
    mocks.createPlan.mockReturnValue(pendingPlan.promise);
    const { result } = renderDeletion();

    act(() => {
      result.current.actions.openPreset({
        value: "BASELINE",
        label: "BASELINE",
        count: 1,
      });
    });
    expect(result.current.operation?.phase).toBe("planning");
    await waitFor(() => expect(mocks.createPlan).toHaveBeenCalled());
    expect(mocks.createPlan.mock.calls[0]?.[0]).toEqual(target);

    act(() => result.current.actions.cancel());
    await act(async () => pendingPlan.resolve(plan));

    expect(result.current.operation).toBeNull();
  });

  it("retries a failed subset plan without rebuilding caller filters", async () => {
    mocks.createPlan
      .mockRejectedValueOnce(new Error("plan unavailable"))
      .mockResolvedValueOnce(plan);
    const { result } = renderDeletion();

    act(() => {
      result.current.actions.openPreset({
        value: "BASELINE",
        label: "BASELINE",
        count: 1,
      });
    });
    await waitFor(() => expect(result.current.operation?.phase).toBe("planFailed"));

    await act(async () => result.current.actions.retryPlan());

    expect(result.current.operation?.phase).toBe("ready");
    expect(mocks.createPlan.mock.calls.map(([input]) => input)).toEqual([
      target,
      target,
    ]);
  });

  it("retains a frozen plan across mutation failure and retry", async () => {
    mocks.createPlan.mockResolvedValue(plan);
    mocks.deleteRuns
      .mockRejectedValueOnce(new Error("delete unavailable"))
      .mockResolvedValueOnce(deleteResult);
    const { result, onRunsDeleted } = renderDeletion();

    act(() => {
      result.current.actions.openPreset({
        value: "BASELINE",
        label: "BASELINE",
        count: 1,
      });
    });
    await waitFor(() => expect(result.current.operation?.phase).toBe("ready"));

    let failure: unknown;
    await act(async () => {
      try {
        await result.current.actions.confirm();
      } catch (error) {
        failure = error;
      }
    });
    expect(failure).toEqual(expect.objectContaining({ message: "delete unavailable" }));
    expect(result.current.operation?.phase).toBe("mutationFailed");

    await act(async () => result.current.actions.confirm());

    expect(result.current.operation).toBeNull();
    expect(mocks.deleteRuns.mock.calls.map(([input]) => input)).toEqual([
      target,
      target,
    ]);
    const mutationKeys = mocks.deleteRuns.mock.calls.map(
      ([, mutation]) => mutation.idempotencyKey,
    );
    expect(mutationKeys[0]).toBe(mutationKeys[1]);
    expect(onRunsDeleted).toHaveBeenCalledTimes(1);
  });

  it("lets an issued mutation finish after hiding while keeping intent reset", async () => {
    const pendingDelete = deferred<typeof deleteResult>();
    mocks.createPlan.mockResolvedValue(plan);
    mocks.deleteRuns.mockReturnValue(pendingDelete.promise);
    const rendered = renderDeletion();

    act(() => {
      rendered.result.current.actions.openPreset({
        value: "BASELINE",
        label: "BASELINE",
        count: 1,
      });
    });
    await waitFor(() =>
      expect(rendered.result.current.operation?.phase).toBe("ready"),
    );
    let mutation!: Promise<void>;
    act(() => {
      mutation = rendered.result.current.actions.confirm();
    });
    expect(rendered.result.current.operation?.phase).toBe("mutating");

    rendered.rerender({
      active: false,
      enabled: true,
      selectedExperiments: new Set(["exp_a"]),
    });
    expect(rendered.result.current.operation).toBeNull();

    await act(async () => pendingDelete.resolve(deleteResult));
    await mutation;

    expect(rendered.result.current.operation).toBeNull();
    expect(rendered.onRunsDeleted).toHaveBeenCalledTimes(1);
  });

  it("quarantines an issued mutation result after the connection changes", async () => {
    const pendingDelete = deferred<typeof deleteResult>();
    mocks.createPlan.mockResolvedValue(plan);
    mocks.deleteRuns.mockReturnValue(pendingDelete.promise);
    const rendered = renderDeletion();

    act(() => {
      rendered.result.current.actions.openPreset({
        value: "BASELINE",
        label: "BASELINE",
        count: 1,
      });
    });
    await waitFor(() =>
      expect(rendered.result.current.operation?.phase).toBe("ready"),
    );
    let mutation!: Promise<void>;
    act(() => {
      mutation = rendered.result.current.actions.confirm();
    });
    expect(rendered.result.current.operation?.phase).toBe("mutating");

    act(() => rendered.result.current.clearForConnectionChange());
    expect(rendered.result.current.operation).toBeNull();

    await act(async () => pendingDelete.resolve(deleteResult));
    await mutation;
    expect(rendered.onRunsDeleted).not.toHaveBeenCalled();
    expect(rendered.onExperimentDeleted).not.toHaveBeenCalled();
    expect(rendered.result.current.operation).toBeNull();
  });
});
