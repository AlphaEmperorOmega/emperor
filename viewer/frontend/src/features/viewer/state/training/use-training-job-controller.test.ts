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

import {
  resolveTrainingLogRefresh,
  useActiveTrainingJobProgress,
} from "@/features/viewer/state/training/use-training-job-controller";
import {
  LOG_EXPERIMENTS_QUERY_KEY,
  LOG_RUNS_QUERY_KEY,
  LOG_TAGS_QUERY_KEY,
  trainingQueryKeys,
} from "@/lib/query-keys";
import { type TrainingJob } from "@/lib/api";

const baseJob: TrainingJob = {
  id: "job-1",
  status: "running",
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

describe("resolveTrainingLogRefresh", () => {
  it("refreshes lists only when a running job first exposes a log directory", () => {
    const emptySnapshot = resolveTrainingLogRefresh(
      {
        jobId: null,
        logDir: null,
        terminalStatus: null,
      },
      undefined,
    ).snapshot;
    const firstRunningPoll = resolveTrainingLogRefresh(
      emptySnapshot,
      trainingJob({ logDir: "logs/runs" }),
    );
    const repeatedRunningPoll = resolveTrainingLogRefresh(
      firstRunningPoll.snapshot,
      trainingJob({ logDir: "logs/runs", step: 12 }),
    );

    expect(firstRunningPoll.action).toBe("lists");
    expect(repeatedRunningPoll.action).toBe("none");
  });

  it("does one details refresh when the job reaches a terminal status", () => {
    const firstRunningPoll = resolveTrainingLogRefresh(
      {
        jobId: null,
        logDir: null,
        terminalStatus: null,
      },
      trainingJob({ logDir: "logs/runs" }),
    );
    const terminalPoll = resolveTrainingLogRefresh(
      firstRunningPoll.snapshot,
      trainingJob({
        status: "completed",
        exitCode: 0,
        logDir: "logs/runs",
      }),
    );
    const repeatedTerminalPoll = resolveTrainingLogRefresh(
      terminalPoll.snapshot,
      trainingJob({
        status: "completed",
        exitCode: 0,
        logDir: "logs/runs",
        updatedAt: "2026-06-01T00:00:01Z",
      }),
    );

    expect(terminalPoll.action).toBe("details");
    expect(repeatedTerminalPoll.action).toBe("none");
  });
});

describe("useActiveTrainingJobProgress", () => {
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
    });
  });
});
