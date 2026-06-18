import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TrainingFooterRunSummary } from "@/features/viewer/components/training/training-footer-run-summary";
import { type TrainingJob, type TrainingRun, type TrainingRunPlan } from "@/lib/api";

function run(overrides: Partial<TrainingRun> = {}): TrainingRun {
  const index = overrides.index ?? 1;
  const preset = overrides.preset ?? "baseline";
  const dataset = overrides.dataset ?? "Mnist";

  return {
    id: `run-${index}`,
    index,
    status: "Pending",
    preset,
    snapshotId: null,
    snapshotName: null,
    dataset,
    changes: [],
    overrides: {},
    command: `source experiment.sh --model linear --preset ${preset} --datasets ${dataset}`,
    totalEpochs: 30,
    currentEpoch: 0,
    metrics: {},
    logDir: null,
    error: null,
    errorTraceback: null,
    ...overrides,
  };
}

function plan(runs: TrainingRun[]): TrainingRunPlan {
  const statusCounts = runs.reduce(
    (counts, candidate) => ({
      ...counts,
      [candidate.status]: counts[candidate.status] + 1,
    }),
    {
      Pending: 0,
      Running: 0,
      Completed: 0,
      Failed: 0,
      Cancelled: 0,
      Skipped: 0,
    } satisfies Record<TrainingRun["status"], number>,
  );
  const totalEpochs = runs.reduce(
    (total, candidate) => total + candidate.totalEpochs,
    0,
  );
  const completedEpochs = runs.reduce(
    (total, candidate) => total + candidate.currentEpoch,
    0,
  );

  return {
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    presets: ["baseline"],
    datasets: ["Mnist"],
    overrides: {},
    search: null,
    logFolder: "summary",
    isRandomSearch: false,
    runs,
    summary: {
      totalRuns: runs.length,
      completedRuns: statusCounts.Completed,
      runningRuns: statusCounts.Running,
      pendingRuns: statusCounts.Pending,
      failedRuns: statusCounts.Failed,
      cancelledRuns: statusCounts.Cancelled,
      skippedRuns: statusCounts.Skipped,
      totalEpochs,
      completedEpochs,
      remainingEpochs: totalEpochs - completedEpochs,
    },
  };
}

function job(
  runPlan: TrainingRunPlan,
  overrides: Partial<TrainingJob> = {},
): TrainingJob {
  return {
    id: "job-1",
    status: "running",
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    presets: ["baseline"],
    datasets: ["Mnist"],
    overrides: {},
    search: null,
    plannedRunCount: runPlan.summary.totalRuns,
    runPlan,
    monitors: [],
    logFolder: "summary",
    createdAt: "2026-06-01T00:00:00Z",
    updatedAt: "2026-06-01T00:00:01Z",
    exitCode: null,
    pid: 123,
    currentPreset: "baseline",
    currentDataset: "Mnist",
    epoch: null,
    step: null,
    metrics: {},
    logDir: null,
    events: [],
    eventCount: 0,
    eventCounts: {},
    eventsTruncated: false,
    clusterGrowth: [],
    logTail: [],
    resultLinks: [],
    ...overrides,
  };
}

describe("TrainingFooterRunSummary", () => {
  it("renders no-plan, loading, and plan-error states", () => {
    const { rerender } = render(<TrainingFooterRunSummary />);

    expect(screen.getByRole("status")).toHaveAccessibleName(
      "Training run summary: no run plan",
    );
    expect(screen.getByText("No run plan")).toBeInTheDocument();

    rerender(<TrainingFooterRunSummary isLoading />);
    expect(screen.getByRole("status")).toHaveAccessibleName(
      "Training run summary: planning training runs",
    );
    expect(screen.getByText("Planning runs")).toBeInTheDocument();

    rerender(<TrainingFooterRunSummary error="Plan failed" />);
    expect(screen.getByRole("status")).toHaveAccessibleName(
      "Training run summary: plan error: Plan failed",
    );
    expect(screen.getByText("Plan error")).toBeInTheDocument();
  });

  it("shows zero completed draft counts and the next planned run", () => {
    render(
      <TrainingFooterRunSummary
        plan={plan([run({ index: 1 }), run({ index: 2, dataset: "Cifar10" })])}
      />,
    );

    expect(screen.getByText("Runs 0 / 2")).toBeInTheDocument();
    expect(screen.getByText("Epochs 0 / 60")).toBeInTheDocument();
    expect(screen.getByText("Next run #1 0 / 30 epochs")).toBeInTheDocument();
  });

  it("shows completed run and completed epoch counts", () => {
    render(
      <TrainingFooterRunSummary
        plan={plan([
          run({ index: 1, status: "Completed", currentEpoch: 20, totalEpochs: 20 }),
          run({ index: 2, status: "Completed", currentEpoch: 30 }),
        ])}
      />,
    );

    expect(screen.getByText("Runs 2 / 2")).toBeInTheDocument();
    expect(screen.getByText("Epochs 50 / 50")).toBeInTheDocument();
    expect(screen.getByText("Result run #2 30 / 30 epochs")).toBeInTheDocument();
  });

  it("uses the running run for current epoch progress", () => {
    render(
      <TrainingFooterRunSummary
        plan={plan([
          run({ index: 1, status: "Completed", currentEpoch: 30 }),
          run({ index: 2, status: "Running", currentEpoch: 4 }),
        ])}
      />,
    );

    expect(screen.getByText("Runs 1 / 2")).toBeInTheDocument();
    expect(screen.getByText("Epochs 34 / 60")).toBeInTheDocument();
    expect(screen.getByText("Active run #2 4 / 30 epochs")).toBeInTheDocument();
  });

  it("selects failed and cancelled result runs for terminal jobs", () => {
    const failedPlan = plan([
      run({ index: 1, status: "Completed", currentEpoch: 30 }),
      run({ index: 2, status: "Failed", currentEpoch: 8 }),
      run({ index: 3, status: "Pending" }),
    ]);
    const { rerender } = render(
      <TrainingFooterRunSummary
        plan={failedPlan}
        job={job(failedPlan, { status: "failed", exitCode: 1 })}
      />,
    );

    expect(screen.getByText("Result run #2 8 / 30 epochs")).toBeInTheDocument();

    const cancelledPlan = plan([
      run({ index: 1, status: "Completed", currentEpoch: 30 }),
      run({ index: 2, status: "Cancelled", currentEpoch: 4 }),
    ]);
    rerender(
      <TrainingFooterRunSummary
        plan={cancelledPlan}
        job={job(cancelledPlan, { status: "cancelled" })}
      />,
    );

    expect(screen.getByText("Result run #2 4 / 30 epochs")).toBeInTheDocument();
  });
});
