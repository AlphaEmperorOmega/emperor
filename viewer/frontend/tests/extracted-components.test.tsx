import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { FullPageError } from "@/components/layout/page-error-status";
import { FullPageLoading, FullPageStatus } from "@/components/layout/page-status";
import { FeatureListDialog } from "@/components/features/viewer/feature-list-dialog";
import { createViewerContext } from "@/components/features/viewer/providers/create-context";
import { TrainingProgressDialog } from "@/components/features/viewer/training/training-progress-dialog";
import { ViewerWorkspaceNav } from "@/components/features/viewer/viewer-workspace-nav";
import { type TrainingRunPlan } from "@/lib/api";
import { IMPLEMENTED_FEATURES } from "@/lib/feature-catalog";

describe("FullPageStatus", () => {
  it("renders status content, detail text, icon, and action", () => {
    render(
      <FullPageStatus
        title="Backend offline"
        detail="Start the API server."
        icon={<span aria-hidden>!</span>}
        action={<button type="button">Retry</button>}
      />,
    );

    expect(screen.getByRole("heading", { name: "Backend offline" })).toBeInTheDocument();
    expect(screen.getByText("Start the API server.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("renders the loading title without optional detail text", () => {
    render(<FullPageLoading />);

    expect(screen.getByRole("heading", { name: "Loading viewer" })).toBeInTheDocument();
    expect(screen.queryByText(/Start the API/i)).not.toBeInTheDocument();
  });
});

describe("FullPageError", () => {
  it("uses the fallback error copy and calls retry", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    render(<FullPageError onRetry={onRetry} />);

    expect(screen.getByText(/unexpected error/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("renders a provided error message", () => {
    render(<FullPageError message="Exploded during render" onRetry={() => {}} />);

    expect(screen.getByText("Exploded during render")).toBeInTheDocument();
  });
});

describe("ViewerWorkspaceNav", () => {
  it("marks the active workspace and emits workspace changes", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<ViewerWorkspaceNav activeWorkspace="model" onChange={onChange} />);

    expect(screen.getByRole("button", { name: "Model" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: "Logs" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );

    await user.click(screen.getByRole("button", { name: "Logs" }));

    expect(onChange).toHaveBeenCalledWith("logs");
  });
});

describe("FeatureListDialog", () => {
  it("renders the feature catalog grouped in a modal dialog", () => {
    render(<FeatureListDialog onClose={() => {}} />);

    const dialog = screen.getByRole("dialog", { name: "Implemented Features" });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(within(dialog).getByText(`${IMPLEMENTED_FEATURES.length} features`))
      .toBeInTheDocument();
    expect(within(dialog).getByText(IMPLEMENTED_FEATURES[0].title)).toBeInTheDocument();
  });

  it("calls onClose from the close button", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<FeatureListDialog onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: "Close implemented features" }));

    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("TrainingProgressDialog", () => {
  it("opens the full traceback for a failed row", async () => {
    const user = userEvent.setup();
    const plan: TrainingRunPlan = {
      model: "linear",
      preset: "baseline",
      presets: ["baseline"],
      datasets: ["Mnist"],
      overrides: {},
      search: null,
      logFolder: "errors",
      isRandomSearch: false,
      runs: [
        {
          id: "run-0001",
          index: 1,
          status: "Failed",
          preset: "baseline",
          dataset: "Mnist",
          changes: [],
          overrides: {},
          command: "source experiment.sh linear --preset baseline",
          totalEpochs: 2,
          currentEpoch: 0,
          metrics: {},
          logDir: null,
          error: "scalar conversion failed",
          errorTraceback:
            "Traceback (most recent call last):\nRuntimeError: scalar conversion failed",
        },
      ],
      summary: {
        totalRuns: 1,
        completedRuns: 0,
        runningRuns: 0,
        pendingRuns: 0,
        failedRuns: 1,
        cancelledRuns: 0,
        skippedRuns: 0,
        totalEpochs: 2,
        completedEpochs: 0,
        remainingEpochs: 0,
      },
    };

    render(
      <TrainingProgressDialog
        plan={plan}
        isLoading={false}
        error=""
        canResample={false}
        isResampling={false}
        onResample={() => {}}
        onClose={() => {}}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Full error for run 1" }));

    const dialog = screen.getByRole("dialog", { name: "Training Error" });
    expect(within(dialog).getByText(/Traceback \(most recent call last\):/))
      .toBeInTheDocument();
    expect(within(dialog).getByText(/RuntimeError: scalar conversion failed/))
      .toBeInTheDocument();
  });
});

describe("createViewerContext", () => {
  it("returns values from its generated provider", () => {
    const [Provider, useValue] = createViewerContext<{ label: string }>("TestContext");

    function Consumer() {
      return <span>{useValue().label}</span>;
    }

    render(
      <Provider value={{ label: "provided" }}>
        <Consumer />
      </Provider>,
    );

    expect(screen.getByText("provided")).toBeInTheDocument();
  });

  it("throws a clear error when rendered outside its provider", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    const [, useValue] = createViewerContext<{ label: string }>("MissingContext");

    function Consumer() {
      useValue();
      return null;
    }

    expect(() => render(<Consumer />)).toThrow(
      "MissingContext value is missing; render within its provider.",
    );
    consoleError.mockRestore();
  });
});
