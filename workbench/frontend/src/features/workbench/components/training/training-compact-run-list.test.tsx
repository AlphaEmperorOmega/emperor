import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  TrainingAllCommandsButton,
  TrainingCompactRunList,
} from "@/features/workbench/components/training/training-compact-run-list";
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

function commandBlock(commands: string[]) {
  return ["(", "  set -e", ...commands.map((command) => `  ${command}`), ")"].join(
    "\n",
  );
}

describe("TrainingCompactRunList", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

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

    expect(screen.queryByText("Training Runs")).not.toBeInTheDocument();
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

  it("opens and copies per-run command and full-error dialogs", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    const trainingCommand = "source experiment.sh --model linear --preset baseline";
    render(
      <TrainingCompactRunList
        plan={plan([
          run({
            index: 1,
            status: "Failed",
            command: trainingCommand,
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
      .toHaveValue(trainingCommand);

    await user.click(
      within(commandDialog).getByRole("button", { name: "Copy Command" }),
    );
    expect(writeText).toHaveBeenCalledWith(trainingCommand);
    expect(within(commandDialog).getByRole("status")).toHaveTextContent(
      "Command copied",
    );

    await user.click(
      within(commandDialog).getByRole("button", {
        name: "Close Training Command",
      }),
    );
    await user.click(screen.getByRole("button", { name: "Full error for run 1" }));

    const errorDialog = screen.getByRole("dialog", { name: "Training Error" });
    expect(within(errorDialog).getByText(/RuntimeError: training failed/))
      .toBeInTheDocument();
  });

  it("switches a run command between POSIX and PowerShell projections", async () => {
    const user = userEvent.setup();
    render(
      <TrainingCompactRunList
        plan={plan([
          run({
            commands: {
              posix: "mise run experiment -- --preset 'wide run'",
              powershell: "mise run experiment -- --preset 'wide run'",
            },
          }),
        ])}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Command for run 1" }));
    const dialog = screen.getByRole("dialog", { name: "Training Command" });
    await user.click(
      within(dialog).getByRole("radio", { name: "PowerShell" }),
    );

    expect(
      within(dialog).getByRole("radio", { name: "PowerShell" }),
    ).toBeChecked();
    expect(within(dialog).getByLabelText("Training command")).toHaveValue(
      "mise run experiment -- --preset 'wide run'",
    );
  });

  it("shows a visible button for all training commands", () => {
    render(<TrainingAllCommandsButton plan={plan([run()])} />);

    const commandsButton = screen.getByRole("button", {
      name: "Commands",
    });

    expect(commandsButton).toBeInTheDocument();
    expect(commandsButton).toHaveTextContent("Commands");
    expect(
      screen.queryByRole("button", {
        name: "Copy all training commands",
      }),
    ).not.toBeInTheDocument();
  });

  it("opens an all-commands dialog with runnable commands in order", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    const firstCommand =
      "source experiment.sh --model linear --preset baseline --datasets Mnist";
    const secondCommand =
      "source experiment.sh --model linear --preset wide --datasets Cifar10";

    render(
      <TrainingAllCommandsButton
        plan={plan([
          run({ index: 1, command: firstCommand }),
          run({ index: 2, command: "" }),
          run({ index: 3, command: "   " }),
          run({ index: 4, command: secondCommand }),
        ])}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Commands" }));

    expect(writeText).not.toHaveBeenCalled();
    const dialog = screen.getByRole("dialog", { name: "Training Commands" });
    const expected = commandBlock([firstCommand, secondCommand]);
    expect(within(dialog).getByLabelText("Training commands")).toHaveValue(expected);

    await user.click(
      within(dialog).getByRole("button", { name: "Copy Commands" }),
    );

    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText).toHaveBeenCalledWith(expected);
    expect(within(dialog).getByRole("status")).toHaveTextContent(
      "Commands copied",
    );
  });

  it("includes planned commands even when their rows are hidden by the compact limit", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    const hiddenCommand =
      "source experiment.sh --model linear --preset hidden --datasets HeldOut";
    const runs = Array.from({ length: 161 }, (_, index) =>
      run({
        index: index + 1,
        command:
          index === 160
            ? hiddenCommand
            : `source experiment.sh --model linear --preset visible-${index + 1}`,
      }),
    );

    const currentPlan = plan(runs);
    render(
      <>
        <TrainingAllCommandsButton plan={currentPlan} />
        <TrainingCompactRunList plan={currentPlan} />
      </>,
    );

    await user.click(screen.getByRole("button", { name: "Commands" }));

    expect(screen.getByText(/Showing 160 of 161 planned runs/)).toBeInTheDocument();
    const dialog = screen.getByRole("dialog", { name: "Training Commands" });
    expect(
      (within(dialog).getByLabelText("Training commands") as HTMLTextAreaElement).value,
    ).toContain(hiddenCommand);

    await user.click(
      within(dialog).getByRole("button", { name: "Copy Commands" }),
    );

    expect(writeText).toHaveBeenCalledWith(expect.stringContaining(hiddenCommand));
  });

  it("omits the copy-all action when there are no runnable commands", () => {
    render(
      <TrainingAllCommandsButton
        plan={plan([
          run({ index: 1, command: "" }),
          run({ index: 2, command: undefined as unknown as string }),
        ])}
      />,
    );

    expect(
      screen.queryByRole("button", { name: "Commands" }),
    ).not.toBeInTheDocument();
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
    expect(screen.getByText("Planning training runs…")).toBeInTheDocument();

    rerender(<TrainingCompactRunList />);
    expect(screen.getByText("No training runs planned")).toBeInTheDocument();

    rerender(<TrainingCompactRunList error="Plan failed" />);
    expect(screen.getByRole("alert")).toHaveTextContent("Plan failed");
  });

});
