import { describe, expect, it } from "vitest";
import {
  resolveTrainingLogRefresh,
} from "@/features/viewer/state/training/use-training-job-controller";
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
