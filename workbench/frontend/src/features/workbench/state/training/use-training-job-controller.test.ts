// @vitest-environment jsdom

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  renderHook,
  waitFor,
} from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  fetchTrainingJob: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    fetchTrainingJob: mocks.fetchTrainingJob,
  };
});

import { useActiveTrainingJobProgress } from "@/features/workbench/state/training/use-training-job-controller";
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
import { type TrainingJob } from "@/lib/api";

const baseJob: TrainingJob = {
  id: "job-1",
  status: "running",
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  presets: ["baseline"],
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
  resultLinks: [],
};

function trainingJob(overrides: Partial<TrainingJob>): TrainingJob {
  return { ...baseJob, ...overrides };
}

function queryClientWrapper(queryClient: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(QueryClientProvider, { client: queryClient }, children);
  };
}

beforeEach(() => {
  mocks.fetchTrainingJob.mockReset();
});

describe("useActiveTrainingJobProgress", () => {
  it("does not poll an active job until protected access is enabled", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0 } },
    });
    const onJobChange = vi.fn();
    mocks.fetchTrainingJob.mockResolvedValue(trainingJob({}));
    const rendered = renderHook(
      ({ enabled }) =>
        useActiveTrainingJobProgress({
          activeJobId: "job-1",
          onJobChange,
          enabled,
        }),
      {
        initialProps: { enabled: false },
        wrapper: queryClientWrapper(queryClient),
      },
    );

    expect(mocks.fetchTrainingJob).not.toHaveBeenCalled();
    rendered.rerender({ enabled: true });
    await waitFor(() => expect(mocks.fetchTrainingJob).toHaveBeenCalledTimes(1));
  });

  it("passes React Query cancellation to an obsolete job poll", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0 } },
    });
    const onJobChange = vi.fn();
    mocks.fetchTrainingJob.mockReturnValue(
      new Promise<TrainingJob>(() => undefined),
    );

    const rendered = renderHook(
      ({ activeJobId }: { activeJobId: string }) =>
        useActiveTrainingJobProgress({ activeJobId, onJobChange }),
      {
        initialProps: { activeJobId: "job-1" },
        wrapper: queryClientWrapper(queryClient),
      },
    );

    await waitFor(() => {
      expect(mocks.fetchTrainingJob.mock.calls[0]?.[1]?.signal).toBeDefined();
    });
    const firstSignal = mocks.fetchTrainingJob.mock.calls[0][1].signal;

    rendered.rerender({ activeJobId: "job-2" });

    await waitFor(() => {
      expect(mocks.fetchTrainingJob).toHaveBeenCalledTimes(2);
    });
    expect(firstSignal.aborted).toBe(true);
    rendered.unmount();
  });

  it("refreshes log lists once when a running job first exposes a log directory", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0 } },
    });
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    const onJobChange = vi.fn();
    mocks.fetchTrainingJob.mockResolvedValue(trainingJob({ logDir: null }));

    renderHook(
      () =>
        useActiveTrainingJobProgress({
          activeJobId: "job-1",
          onJobChange,
        }),
      { wrapper: queryClientWrapper(queryClient) },
    );

    await waitFor(() => {
      expect(onJobChange).toHaveBeenCalledWith(
        expect.objectContaining({ logDir: null }),
      );
    });
    invalidateSpy.mockClear();
    onJobChange.mockClear();

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
    onJobChange.mockClear();

    act(() => {
      queryClient.setQueryData(
        trainingQueryKeys.job("job-1"),
        trainingJob({ logDir: "logs/runs", step: 12 }),
      );
    });

    await waitFor(() => {
      expect(onJobChange).toHaveBeenCalledWith(
        expect.objectContaining({ logDir: "logs/runs", step: 12 }),
      );
    });
    expect(invalidateSpy).not.toHaveBeenCalled();
  });

  it("keeps terminal log refresh alive without the training panel UI mounted", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0 } },
    });
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    const removeSpy = vi.spyOn(queryClient, "removeQueries");
    const onJobChange = vi.fn();
    mocks.fetchTrainingJob.mockResolvedValue(trainingJob({ logDir: null }));

    renderHook(
      () =>
        useActiveTrainingJobProgress({
          activeJobId: "job-1",
          onJobChange,
        }),
      { wrapper: queryClientWrapper(queryClient) },
    );

    await waitFor(() => {
      expect(onJobChange).toHaveBeenCalledWith(
        expect.objectContaining({ status: "running" }),
      );
    });
    invalidateSpy.mockClear();
    removeSpy.mockClear();
    onJobChange.mockClear();

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

    await waitFor(() => {
      expect(onJobChange).toHaveBeenCalledWith(
        expect.objectContaining({ status: "completed" }),
      );
    });
    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: LOG_EXPERIMENTS_QUERY_KEY,
      });
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: LOG_RUNS_QUERY_KEY,
      });
      expect(removeSpy).toHaveBeenCalledWith({
        queryKey: LOG_TAGS_QUERY_KEY,
      });
      expect(removeSpy).toHaveBeenCalledWith({
        queryKey: LOG_CHECKPOINTS_QUERY_KEY,
      });
      expect(removeSpy).toHaveBeenCalledWith({
        queryKey: LOG_ARTIFACTS_QUERY_KEY,
      });
      expect(removeSpy).toHaveBeenCalledWith({
        queryKey: LOG_MEDIA_QUERY_KEY,
      });
      expect(removeSpy).toHaveBeenCalledWith({
        queryKey: LOG_SCALARS_QUERY_KEY,
      });
    });
    invalidateSpy.mockClear();
    removeSpy.mockClear();
    onJobChange.mockClear();

    act(() => {
      queryClient.setQueryData(
        trainingQueryKeys.job("job-1"),
        trainingJob({
          status: "completed",
          exitCode: 0,
          logDir: "logs/runs",
          updatedAt: "2026-06-01T00:00:01Z",
        }),
      );
    });

    await waitFor(() => {
      expect(onJobChange).toHaveBeenCalledWith(
        expect.objectContaining({
          status: "completed",
          updatedAt: "2026-06-01T00:00:01Z",
        }),
      );
    });
    expect(invalidateSpy).not.toHaveBeenCalled();
    expect(removeSpy).not.toHaveBeenCalled();
  });
});
