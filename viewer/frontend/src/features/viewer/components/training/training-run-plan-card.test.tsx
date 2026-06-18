import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TrainingLogTailCard } from "@/features/viewer/components/training/training-log-tail-card";
import { TrainingRunPlanCard } from "@/features/viewer/components/training/training-run-plan-card";
import {
  DEFAULT_TRAINING_SEARCH_STATE,
  type TrainingSearchState,
} from "@/lib/training-search";
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
    command: `source experiment.sh --model-type linears --model linear --preset ${preset} --datasets ${dataset}`,
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
    logFolder: "test_model",
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
    logFolder: "test_model",
    createdAt: "2026-06-01T00:00:00Z",
    updatedAt: "2026-06-01T00:00:01Z",
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
    ...overrides,
  };
}

function renderCard(
  props: Partial<React.ComponentProps<typeof TrainingRunPlanCard>> = {},
) {
  return render(
    <TrainingRunPlanCard
      effectiveTrainingSearch={DEFAULT_TRAINING_SEARCH_STATE}
      searchModeLabel="Off"
      activeSearchAxisCount={0}
      searchConflictCount={0}
      trainingSearchValidation={{ ready: true, message: "" }}
      displayedRunCount={props.plan?.summary.totalRuns ?? 0}
      requiresLargeGridConfirmation={false}
      selectedMonitorCount={1}
      {...props}
    />,
  );
}

describe("TrainingRunPlanCard", () => {
  it("renders a draft run plan with counts and next run identity", () => {
    const command =
      "source experiment.sh --model-type linears --model linear --preset baseline --datasets Mnist";
    renderCard({
      plan: plan([
        run({ index: 1, command }),
        run({ index: 2, dataset: "Cifar10" }),
      ]),
    });

    expect(screen.getByText("Run Plan")).toBeInTheDocument();
    expect(screen.getAllByText("2 planned runs").length).toBeGreaterThan(0);
    expect(screen.getByText("2 pending")).toBeInTheDocument();
    expect(screen.getByText("60 epochs")).toBeInTheDocument();
    expect(screen.getByText("60 left")).toBeInTheDocument();
    expect(screen.getByText("Next run")).toBeInTheDocument();
    expect(screen.getAllByText("#1").length).toBeGreaterThan(0);
    expect(screen.getAllByText("baseline").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Mnist").length).toBeGreaterThan(0);
    expect(screen.queryByTitle(command)).not.toBeInTheDocument();
    expect(screen.queryByText("Preview runs")).not.toBeInTheDocument();
  });

  it("renders active job identity, epoch and step, metrics, and log directory", () => {
    const runPlan = plan([
      run({ index: 1, status: "Completed", currentEpoch: 30 }),
      run({
        index: 2,
        status: "Running",
        preset: "recurrent-gating-halting",
        dataset: "Cifar10",
        currentEpoch: 4,
        logDir: "logs/run-2",
      }),
    ]);

    renderCard({
      plan: runPlan,
      job: job(runPlan, {
        status: "running",
        epoch: 3,
        step: 42,
        metrics: {
          validation_accuracy: 0.987654,
          loss: 1.23456,
          phase: "fit",
        },
        logDir: "logs/active-job",
      }),
    });

    expect(screen.getByText("Active Run")).toBeInTheDocument();
    expect(screen.getByText("Active run")).toBeInTheDocument();
    expect(screen.getByText("#2")).toBeInTheDocument();
    expect(screen.getByText("recurrent-gating-halting")).toBeInTheDocument();
    expect(screen.getByText("Cifar10")).toBeInTheDocument();
    expect(screen.getByText("epoch 3 / step 42")).toBeInTheDocument();
    expect(screen.getByText("validation_accuracy=0.9877")).toBeInTheDocument();
    expect(screen.getByText("loss=1.235")).toBeInTheDocument();
    expect(screen.getByText("phase=fit")).toBeInTheDocument();
    expect(screen.getByTitle("logs/active-job")).toBeInTheDocument();
  });

  it("shows the first pending run as next when a running job has not marked a run active yet", () => {
    const runPlan = plan([
      run({ index: 1, preset: "baseline", dataset: "Mnist" }),
      run({ index: 2, preset: "wide", dataset: "Cifar10" }),
    ]);

    renderCard({
      plan: runPlan,
      job: job(runPlan, {
        status: "running",
        epoch: null,
        step: null,
      }),
    });

    expect(screen.getByText("Next run")).toBeInTheDocument();
    expect(screen.getByText("#1")).toBeInTheDocument();
    expect(screen.getByText("baseline")).toBeInTheDocument();
    expect(screen.getByText("Mnist")).toBeInTheDocument();
    expect(screen.queryByText("#2")).not.toBeInTheDocument();
    expect(screen.queryByText("wide")).not.toBeInTheDocument();
  });

  it("renders terminal results with failed counts and run errors", () => {
    const runPlan = plan([
      run({ index: 1, status: "Completed", currentEpoch: 30 }),
      run({
        index: 2,
        status: "Failed",
        currentEpoch: 8,
        error: "training exploded",
        errorTraceback: "Traceback: training exploded",
      }),
    ]);

    renderCard({
      plan: runPlan,
      job: job(runPlan, { status: "failed", exitCode: 1 }),
    });

    expect(screen.getByText("Results")).toBeInTheDocument();
    expect(screen.getByText("1 completed")).toBeInTheDocument();
    expect(screen.getByText("1 failed")).toBeInTheDocument();
    expect(screen.getByText("training exploded")).toHaveClass("text-danger-text");
  });

  it("renders search validation, conflict, and large-grid warnings with distinct tones", () => {
    const search: TrainingSearchState = {
      mode: "grid",
      selectedValues: {},
      randomSamples: 10,
    };

    renderCard({
      effectiveTrainingSearch: search,
      searchModeLabel: "Grid",
      activeSearchAxisCount: 0,
      searchConflictCount: 2,
      trainingSearchValidation: {
        ready: false,
        message: "Select at least one search axis.",
      },
      displayedRunCount: 110,
      requiresLargeGridConfirmation: true,
      selectedMonitorCount: 0,
    });

    expect(screen.getByText("Select at least one search axis."))
      .toHaveClass("text-danger-text");
    expect(screen.getByText("2 overrides replaced by search."))
      .toHaveClass("text-amber");
    expect(screen.getByText("110 planned runs require confirmation before start."))
      .toHaveClass("text-amber");
    expect(screen.getByText("No monitors selected.")).toHaveClass("text-ink-faint");
  });

  it("keeps the empty state readable while planning has not produced a plan", () => {
    renderCard({ selectedMonitorCount: 0 });

    expect(screen.getByText("Run Plan")).toBeInTheDocument();
    expect(screen.getByText("No plan")).toBeInTheDocument();
    expect(
      screen.getByText("No run plan yet. Select a trainable target to preview runs."),
    ).toBeInTheDocument();
    expect(screen.getByText("No monitors selected.")).toBeInTheDocument();
  });

  it("renders the planning state as busy", () => {
    renderCard({ isPlanning: true });

    const status = screen.getByText("Building run plan...").closest("div");
    if (!(status instanceof HTMLElement)) {
      throw new Error("Expected planning status container");
    }
    expect(within(status).getByText("Building run plan...")).toBeInTheDocument();
    expect(status.querySelector("svg")).toHaveClass("animate-spin");
  });
});

describe("TrainingLogTailCard", () => {
  it("wraps long unbroken log lines inside the footer card", () => {
    const longToken = "x".repeat(240);
    render(<TrainingLogTailCard logTail={[longToken]} />);

    const pre = screen.getByText(longToken);
    expect(pre.tagName).toBe("PRE");
    expect(pre).toHaveClass("overflow-x-hidden");
    expect(pre.className).toContain("[overflow-wrap:anywhere]");
    expect(screen.getByText("1 line")).toBeInTheDocument();
  });
});
