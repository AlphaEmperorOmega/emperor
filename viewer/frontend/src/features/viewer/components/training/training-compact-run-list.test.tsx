import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { TrainingCompactRunList } from "@/features/viewer/components/training/training-compact-run-list";
import { type TrainingRun, type TrainingRunPlan } from "@/lib/api";

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
  const totalEpochs = runs.reduce((total, candidate) => total + candidate.totalEpochs, 0);
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
    logFolder: "compact",
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

describe("TrainingCompactRunList", () => {
  it("renders draft, running, completed, failed, and cancelled states", () => {
    render(
      <TrainingCompactRunList
        plan={plan([
          run({ index: 1, status: "Pending" }),
          run({ index: 2, status: "Running", currentEpoch: 3 }),
          run({
            index: 3,
            status: "Completed",
            currentEpoch: 30,
            metrics: { accuracy: 0.98765 },
            logDir: "logs/run-3",
          }),
          run({
            index: 4,
            status: "Failed",
            error: "loss exploded",
            errorTraceback: "Traceback: loss exploded",
          }),
          run({ index: 5, status: "Cancelled" }),
        ])}
      />,
    );

    expect(screen.getByText("Training Runs")).toBeInTheDocument();
    expect(screen.getByText("5 runs")).toBeInTheDocument();
    expect(screen.getByText("1 running")).toBeInTheDocument();
    expect(screen.getByText("1 done")).toBeInTheDocument();
    expect(screen.getByText("2 stopped")).toBeInTheDocument();
    for (const status of ["Pending", "Running", "Completed", "Failed", "Cancelled"]) {
      expect(screen.getByText(status)).toBeInTheDocument();
    }
    expect(screen.getByText("accuracy=0.9877")).toBeInTheDocument();
    expect(screen.getByText("loss exploded")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Full error for run 4" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Copy log path for run 3" }),
    ).toBeInTheDocument();
  });

  it("opens command and full-error dialogs", async () => {
    const user = userEvent.setup();
    render(
      <TrainingCompactRunList
        plan={plan([
          run({
            index: 1,
            status: "Failed",
            command: "source experiment.sh --model linear --preset baseline",
            error: "training failed",
            errorTraceback: "Traceback\nRuntimeError: training failed",
          }),
        ])}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Command for run 1" }));
    const commandDialog = screen.getByRole("dialog", {
      name: "Training Command",
    });
    expect(within(commandDialog).getByLabelText("Training command"))
      .toHaveValue("source experiment.sh --model linear --preset baseline");

    await user.click(
      within(commandDialog).getByRole("button", {
        name: "Close training command",
      }),
    );
    await user.click(screen.getByRole("button", { name: "Full error for run 1" }));

    const errorDialog = screen.getByRole("dialog", { name: "Training Error" });
    expect(within(errorDialog).getByText(/RuntimeError: training failed/))
      .toBeInTheDocument();
  });

  it("shows draft remove actions only when valid", async () => {
    const user = userEvent.setup();
    const onExcludePreset = vi.fn();
    const onExcludeSnapshot = vi.fn();
    const currentPlan = plan([
      run({ index: 1, preset: "baseline" }),
      run({
        index: 2,
        preset: "baseline",
        snapshotId: "snap-wide",
        snapshotName: "Wide snapshot",
      }),
    ]);
    const { rerender } = render(<TrainingCompactRunList plan={currentPlan} />);

    expect(
      screen.queryByRole("button", {
        name: "Remove preset baseline from this run plan",
      }),
    ).not.toBeInTheDocument();

    rerender(
      <TrainingCompactRunList
        plan={currentPlan}
        canManageDraftRuns
        onExcludePreset={onExcludePreset}
        onExcludeSnapshot={onExcludeSnapshot}
      />,
    );

    await user.click(
      screen.getByRole("button", {
        name: "Remove preset baseline from this run plan",
      }),
    );
    expect(onExcludePreset).toHaveBeenCalledWith("baseline");

    await user.click(
      screen.getByRole("button", {
        name: "Remove snapshot Wide snapshot from this run plan",
      }),
    );
    expect(onExcludeSnapshot).toHaveBeenCalledWith("snap-wide");
  });

  it("renders loading, empty, and error states", () => {
    const { rerender } = render(<TrainingCompactRunList isLoading />);
    expect(screen.getByText("Planning training runs")).toBeInTheDocument();

    rerender(<TrainingCompactRunList />);
    expect(screen.getByText("No training runs planned")).toBeInTheDocument();

    rerender(<TrainingCompactRunList error="Plan failed" />);
    expect(screen.getByRole("alert")).toHaveTextContent("Plan failed");
  });

  it("shows Resample only when enabled and calls the handler", async () => {
    const user = userEvent.setup();
    const onResample = vi.fn();
    const currentPlan = plan([run()]);
    const { rerender } = render(<TrainingCompactRunList plan={currentPlan} />);

    expect(
      screen.queryByRole("button", { name: /^resample$/i }),
    ).not.toBeInTheDocument();

    rerender(
      <TrainingCompactRunList
        plan={currentPlan}
        canResample
        onResample={onResample}
      />,
    );

    await user.click(screen.getByRole("button", { name: /^resample$/i }));
    expect(onResample).toHaveBeenCalledTimes(1);

    rerender(
      <TrainingCompactRunList
        plan={currentPlan}
        canResample
        isResampling
        onResample={onResample}
      />,
    );
    expect(screen.getByRole("button", { name: /^resample$/i })).toBeDisabled();
  });
});
