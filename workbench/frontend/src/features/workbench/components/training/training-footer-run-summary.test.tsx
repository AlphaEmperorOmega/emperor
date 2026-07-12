import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { TrainingFooterRunSummary } from "@/features/workbench/components/training/training-footer-run-summary";
import * as trainingRunDisplay from "@/features/workbench/components/training/training-run-display";
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

function expectStatusPill(label: string, value: string) {
  const labelElement = screen.getByText(label);
  const pill = labelElement.parentElement;
  if (!(pill instanceof HTMLElement)) {
    throw new Error(`Expected ${label} to render inside a status pill`);
  }
  expect(pill).toHaveClass("h-control", "rounded-control-md", "border-line");
  expect(pill.querySelector("svg")).toBeInTheDocument();
  expect(pill).toHaveTextContent(label);
  expect(pill).toHaveTextContent(value);
}

describe("TrainingFooterRunSummary", () => {
  it("selects the display run once per render", () => {
    const selectSpy = vi.spyOn(
      trainingRunDisplay,
      "selectTrainingRunForDisplay",
    );
    const runPlan = plan([run({ index: 1 }), run({ index: 2 })]);

    render(<TrainingFooterRunSummary plan={runPlan} />);

    expect(selectSpy).toHaveBeenCalledTimes(1);
    selectSpy.mockRestore();
  });

  it("renders no-plan, loading, and plan-error states", () => {
    const { rerender } = render(<TrainingFooterRunSummary />);

    expect(screen.getByRole("status")).toHaveAccessibleName(
      "Training run summary: no run plan",
    );
    expectStatusPill("runs", "no plan");

    rerender(<TrainingFooterRunSummary isLoading />);
    expect(screen.getByRole("status")).toHaveAccessibleName(
      "Training run summary: planning training runs…",
    );
    expectStatusPill("runs", "planning");

    rerender(<TrainingFooterRunSummary error="Plan failed" />);
    expect(screen.getByRole("status")).toHaveAccessibleName(
      "Training run summary: plan error: Plan failed",
    );
    expectStatusPill("plan", "error");
  });

  it("shows zero completed draft counts and the next planned run", () => {
    render(
      <TrainingFooterRunSummary
        plan={plan([run({ index: 1 }), run({ index: 2, dataset: "Cifar10" })])}
      />,
    );
    const summary = screen.getByRole("status");

    expectStatusPill("runs", "0 / 2");
    expectStatusPill("epochs", "0 / 60");
    expectStatusPill("next run", "#1 0 / 30 epochs");
    expect(summary).not.toHaveClass("rounded-[10px]", "border-line", "bg-white/[0.025]");
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

    expectStatusPill("runs", "2 / 2");
    expectStatusPill("epochs", "50 / 50");
    expectStatusPill("result run", "#2 30 / 30 epochs");
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

    expectStatusPill("runs", "1 / 2");
    expectStatusPill("epochs", "34 / 60");
    expectStatusPill("active run", "#2 4 / 30 epochs");
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

    expectStatusPill("result run", "#2 8 / 30 epochs");

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

    expectStatusPill("result run", "#2 4 / 30 epochs");
  });
});
