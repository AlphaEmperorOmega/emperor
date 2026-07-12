import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  cancelTrainingJob: vi.fn(),
  createTrainingJob: vi.fn(),
  fetchTrainingJob: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    cancelTrainingJob: mocks.cancelTrainingJob,
    createTrainingJob: mocks.createTrainingJob,
    fetchTrainingJob: mocks.fetchTrainingJob,
  };
});

import {
  type TrainingJob,
  type TrainingJobCreateInput,
} from "@/lib/api";
import {
  LOG_ARTIFACTS_QUERY_KEY,
  LOG_CHECKPOINTS_QUERY_KEY,
  LOG_EXPERIMENTS_QUERY_KEY,
  LOG_MEDIA_QUERY_KEY,
  LOG_RUNS_QUERY_KEY,
  LOG_SCALARS_QUERY_KEY,
  LOG_TAGS_QUERY_KEY,
  trainingQueryKeys,
} from "@/lib/query-keys";
import { useTrainingJobLifecycle } from "@/features/workbench/state/training/use-training-job-lifecycle";

const baseJob: TrainingJob = {
  id: "job-1",
  status: "running",
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  presets: ["baseline"],
  experimentTask: "image-classification",
  datasets: ["Mnist"],
  overrides: {},
  runPlan: null,
  monitors: [],
  logFolder: "runs",
  createdAt: "2026-06-01T00:00:00Z",
  updatedAt: "2026-06-01T00:00:00Z",
  exitCode: null,
  pid: 123,
  currentPreset: "baseline",
  currentDataset: "Mnist",
  epoch: 0,
  step: 0,
  metrics: {},
  logDir: null,
  events: [],
  eventCount: 0,
  eventCounts: {},
  eventsTruncated: false,
  clusterGrowth: [],
  logTail: [],
  logTailTruncated: false,
  resultLinks: [],
};

const runPlanRequest: TrainingJobCreateInput = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  presets: ["baseline"],
  experimentTask: "image-classification",
  datasets: ["Mnist"],
  overrides: {},
  monitors: [],
  logFolder: "runs",
  runPlan: {
    runs: [
      {
        id: "run-1",
        preset: "baseline",
        dataset: "Mnist",
        overrides: {},
      },
    ],
  },
};

function trainingJob(overrides: Partial<TrainingJob> = {}): TrainingJob {
  return { ...baseJob, ...overrides };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, reject, resolve };
}

function queryClientWrapper(queryClient: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(QueryClientProvider, { client: queryClient }, children);
  };
}

function renderLifecycle({
  enabled = true,
  onJobStarted,
}: {
  enabled?: boolean;
  onJobStarted?: (logFolder: string) => void;
} = {}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  const rendered = renderHook(
    (options: { enabled: boolean; onJobStarted?: (logFolder: string) => void }) =>
      useTrainingJobLifecycle(options),
    {
      initialProps: { enabled, onJobStarted },
      wrapper: queryClientWrapper(queryClient),
    },
  );
  return { ...rendered, queryClient };
}

beforeEach(() => {
  mocks.cancelTrainingJob.mockReset();
  mocks.createTrainingJob.mockReset();
  mocks.fetchTrainingJob.mockReset();
});

describe("useTrainingJobLifecycle", () => {
  it("owns launch identity, polling, and one started-folder notification", async () => {
    const onJobStarted = vi.fn();
    mocks.createTrainingJob.mockResolvedValue(trainingJob());
    mocks.fetchTrainingJob.mockResolvedValue(trainingJob({ step: 4 }));
    const { result } = renderLifecycle({ onJobStarted });

    act(() => result.current.launchRunPlan(runPlanRequest));
    await waitFor(() => {
      expect(mocks.createTrainingJob.mock.calls[0]?.[0]).toEqual(
        runPlanRequest,
      );
      expect(result.current.job?.id).toBe("job-1");
      expect(mocks.fetchTrainingJob).toHaveBeenCalledWith(
        "job-1",
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
    });
    await waitFor(() => expect(result.current.job?.step).toBe(4));
    expect(onJobStarted).toHaveBeenCalledTimes(1);
    expect(onJobStarted).toHaveBeenCalledWith("runs");
  });

  it("keeps failed launch out of active identity and permits a fresh launch", async () => {
    mocks.createTrainingJob
      .mockRejectedValueOnce(new Error("launch unavailable"))
      .mockResolvedValueOnce(trainingJob());
    mocks.fetchTrainingJob.mockResolvedValue(trainingJob());
    const { result } = renderLifecycle();

    act(() => result.current.launchRunPlan(runPlanRequest));
    await waitFor(() => {
      expect(result.current.trainingError).toBe("launch unavailable");
      expect(result.current.job).toBeUndefined();
    });

    act(() => result.current.launchRunPlan(runPlanRequest));
    await waitFor(() => {
      expect(mocks.createTrainingJob).toHaveBeenCalledTimes(2);
      expect(result.current.job?.id).toBe("job-1");
      expect(result.current.trainingError).toBe("");
    });
  });

  it("keeps cancellation terminal when an obsolete running poll resolves", async () => {
    const stalePoll = deferred<TrainingJob>();
    mocks.createTrainingJob.mockResolvedValue(trainingJob());
    mocks.fetchTrainingJob.mockReturnValue(stalePoll.promise);
    mocks.cancelTrainingJob.mockResolvedValue(
      trainingJob({ status: "cancelled", exitCode: 130 }),
    );
    const { result } = renderLifecycle();

    act(() => result.current.launchRunPlan(runPlanRequest));
    await waitFor(() => {
      expect(result.current.job?.status).toBe("running");
      expect(mocks.fetchTrainingJob).toHaveBeenCalledTimes(1);
    });
    act(() => result.current.cancelTraining());
    await waitFor(() => expect(result.current.job?.status).toBe("cancelled"));

    await act(async () => {
      stalePoll.resolve(trainingJob({ status: "running", step: 99 }));
      await stalePoll.promise;
    });
    expect(result.current.job).toMatchObject({
      status: "cancelled",
      exitCode: 130,
    });
  });

  it("retains a running Job after failed cancellation and can retry", async () => {
    mocks.createTrainingJob.mockResolvedValue(trainingJob());
    mocks.fetchTrainingJob.mockReturnValue(new Promise<TrainingJob>(() => undefined));
    mocks.cancelTrainingJob
      .mockRejectedValueOnce(new Error("cancel unavailable"))
      .mockResolvedValueOnce(trainingJob({ status: "cancelled" }));
    const { result } = renderLifecycle();

    act(() => result.current.launchRunPlan(runPlanRequest));
    await waitFor(() => expect(result.current.job?.status).toBe("running"));
    act(() => result.current.cancelTraining());
    await waitFor(() => {
      expect(result.current.job?.status).toBe("running");
      expect(result.current.trainingError).toBe("cancel unavailable");
    });

    act(() => result.current.cancelTraining());
    await waitFor(() => {
      expect(mocks.cancelTrainingJob).toHaveBeenCalledTimes(2);
      expect(result.current.job?.status).toBe("cancelled");
      expect(result.current.trainingError).toBe("");
    });
  });

  it("refreshes Logs on first directory and terminal transition", async () => {
    mocks.createTrainingJob.mockResolvedValue(trainingJob({ logDir: null }));
    mocks.fetchTrainingJob.mockReturnValue(new Promise<TrainingJob>(() => undefined));
    const { result, queryClient } = renderLifecycle();
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    const removeSpy = vi.spyOn(queryClient, "removeQueries");

    act(() => result.current.launchRunPlan(runPlanRequest));
    await waitFor(() => expect(result.current.job?.id).toBe("job-1"));
    invalidateSpy.mockClear();
    removeSpy.mockClear();

    act(() => {
      queryClient.setQueryData(
        trainingQueryKeys.job("job-1"),
        trainingJob({ logDir: "logs/runs" }),
      );
    });
    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: LOG_EXPERIMENTS_QUERY_KEY,
      });
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: LOG_RUNS_QUERY_KEY,
      });
    });
    invalidateSpy.mockClear();
    removeSpy.mockClear();

    act(() => {
      queryClient.setQueryData(
        trainingQueryKeys.job("job-1"),
        trainingJob({
          status: "completed",
          exitCode: 0,
          logDir: "logs/runs",
        }),
      );
    });
    await waitFor(() => expect(result.current.job?.status).toBe("completed"));
    await waitFor(() => {
      expect(removeSpy).toHaveBeenCalledWith({ queryKey: LOG_TAGS_QUERY_KEY });
      expect(removeSpy).toHaveBeenCalledWith({
        queryKey: LOG_CHECKPOINTS_QUERY_KEY,
      });
      expect(removeSpy).toHaveBeenCalledWith({
        queryKey: LOG_ARTIFACTS_QUERY_KEY,
      });
      expect(removeSpy).toHaveBeenCalledWith({ queryKey: LOG_MEDIA_QUERY_KEY });
      expect(removeSpy).toHaveBeenCalledWith({ queryKey: LOG_SCALARS_QUERY_KEY });
    });
  });

  it("resets a terminal Job through the semantic lifecycle action", async () => {
    mocks.createTrainingJob.mockResolvedValue(
      trainingJob({ status: "completed", exitCode: 0 }),
    );
    mocks.fetchTrainingJob.mockResolvedValue(
      trainingJob({ status: "completed", exitCode: 0 }),
    );
    const { result } = renderLifecycle();

    act(() => result.current.launchRunPlan(runPlanRequest));
    await waitFor(() => expect(result.current.canResetTraining).toBe(true));
    act(() => result.current.resetTraining());
    await waitFor(() => {
      expect(result.current.job).toBeUndefined();
      expect(result.current.canResetTraining).toBe(false);
    });
  });

  it("quarantines a launch result that completes after connection reset", async () => {
    const staleLaunch = deferred<TrainingJob>();
    mocks.createTrainingJob.mockReturnValue(staleLaunch.promise);
    const { result } = renderLifecycle();

    act(() => result.current.launchRunPlan(runPlanRequest));
    await waitFor(() => expect(result.current.isStarting).toBe(true));
    act(() => result.current.clearForConnectionChange());
    await waitFor(() => expect(result.current.isStarting).toBe(false));

    await act(async () => {
      staleLaunch.resolve(trainingJob());
      await staleLaunch.promise;
    });
    expect(result.current.job).toBeUndefined();
    expect(result.current.isStarting).toBe(false);
  });

  it("keeps launch and polling inactive until protected access is enabled", async () => {
    mocks.createTrainingJob.mockResolvedValue(trainingJob());
    mocks.fetchTrainingJob.mockResolvedValue(trainingJob());
    const rendered = renderLifecycle({ enabled: false });

    act(() => rendered.result.current.launchRunPlan(runPlanRequest));
    expect(mocks.createTrainingJob).not.toHaveBeenCalled();
    expect(mocks.fetchTrainingJob).not.toHaveBeenCalled();

    rendered.rerender({ enabled: true, onJobStarted: undefined });
    act(() => rendered.result.current.launchRunPlan(runPlanRequest));
    await waitFor(() => {
      expect(mocks.createTrainingJob).toHaveBeenCalledTimes(1);
      expect(mocks.fetchTrainingJob).toHaveBeenCalledTimes(1);
    });
  });
});
