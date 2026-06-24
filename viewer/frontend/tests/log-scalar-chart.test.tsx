import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import {
  LazyLogScalarChart,
  LogScalarChart,
} from "@/features/viewer/components/logs/log-scalar-chart";
import { formatRunLabel } from "@/features/viewer/state/logs/logs-selectors";
import { type LogRun, type LogScalarSeries } from "@/lib/api";

function logRun(overrides: Partial<LogRun> = {}): LogRun {
  return {
    id: "run-1",
    group: null,
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    dataset: "Cifar10",
    runName: "run-1",
    timestamp: "2026-06-01 01:02:03",
    version: "1",
    relativePath: "kaggle_linear_all/run-1",
    experiment: "kaggle_linear_all",
    hasResult: true,
    eventFileCount: 1,
    checkpointCount: 0,
    hasHparams: false,
    metrics: {},
    ...overrides,
  };
}

function scalarSeries(overrides: Partial<LogScalarSeries> = {}): LogScalarSeries {
  return {
    runId: "run-1",
    tag: "train/accuracy",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.4 },
      { step: 2, wallTime: 1780000001, value: 0.75 },
    ],
    sourcePointCount: null,
    truncated: null,
    ...overrides,
  };
}

function renderScalarChart(tag: string) {
  return render(
    <LogScalarChart
      tag={tag}
      series={[scalarSeries({ tag })]}
      runsById={new Map([["run-1", logRun()]])}
      checkpointsByRunId={new Map()}
      runOrder={["run-1"]}
      onSelectRun={vi.fn()}
    />,
  );
}

describe("LogScalarChart", () => {
  it("renders lazy charts immediately when IntersectionObserver is unavailable", async () => {
    const originalObserver = globalThis.IntersectionObserver;
    Reflect.deleteProperty(globalThis, "IntersectionObserver");

    try {
      render(
        <LazyLogScalarChart
          tag="validation/accuracy"
          series={[scalarSeries({ tag: "validation/accuracy" })]}
          runsById={new Map([["run-1", logRun()]])}
          checkpointsByRunId={new Map()}
          runOrder={["run-1"]}
          onSelectRun={vi.fn()}
        />,
      );

      expect(
        await screen.findByRole("img", {
          name: /validation\/accuracy scalar chart/i,
        }),
      ).toBeInTheDocument();
    } finally {
      if (originalObserver) {
        Object.defineProperty(globalThis, "IntersectionObserver", {
          configurable: true,
          writable: true,
          value: originalObserver,
        });
      }
    }
  });

  it("bounds the run legend without removing chart or selection behavior", async () => {
    const user = userEvent.setup();
    const runs = Array.from({ length: 18 }, (_, index) =>
      logRun({
        id: `run-${index}`,
        runName: `run-${index}`,
        timestamp: null,
      }),
    );
    const series = runs.map((run, index) =>
      scalarSeries({
        runId: run.id,
        tag: "train/loss",
        points: [
          { step: 1, wallTime: 1780000000 + index, value: index + 0.25 },
          { step: 2, wallTime: 1780000100 + index, value: index + 0.75 },
        ],
      }),
    );
    const onSelectRun = vi.fn();

    render(
      <LogScalarChart
        tag="train/loss"
        series={series}
        runsById={new Map(runs.map((run) => [run.id, run]))}
        checkpointsByRunId={new Map()}
        runOrder={runs.map((run) => run.id)}
        onSelectRun={onSelectRun}
      />,
    );

    expect(
      screen.getByRole("img", { name: /train\/loss scalar chart/i }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("echart")).toBeInTheDocument();
    const chartCard = screen
      .getByRole("img", { name: /train\/loss scalar chart/i })
      .closest("section");
    if (!(chartCard instanceof HTMLElement)) {
      throw new Error("Expected scalar chart to render inside a surface section");
    }
    expect(chartCard).toHaveClass(
      "rounded-[10px]",
      "border",
      "border-line",
      "bg-white/[0.018]",
      "p-4",
    );
    expect(chartCard).not.toHaveClass("edge", "rounded-card");

    const selectedRun = runs[12];
    const selectedRunLabel = formatRunLabel(selectedRun);
    const selectedRunButton = screen.getByText(selectedRunLabel).closest("button");
    if (!(selectedRunButton instanceof HTMLButtonElement)) {
      throw new Error("Expected formatted run label to render inside a legend button");
    }

    const legend = selectedRunButton.parentElement;
    if (!(legend instanceof HTMLElement)) {
      throw new Error("Expected legend button to render inside a legend container");
    }

    expect(legend).toHaveClass(
      "grid",
      "max-h-48",
      "min-h-0",
      "overflow-y-auto",
      "pr-1",
      "sm:grid-cols-2",
      "xl:grid-cols-3",
    );

    await user.click(selectedRunButton);

    expect(onSelectRun).toHaveBeenCalledWith(selectedRun.id);
  });

  it("opens metric info from the scalar chart card header", async () => {
    const user = userEvent.setup();
    const infoButtonLabel = "Explain metric train/accuracy";
    renderScalarChart("train/accuracy");

    const infoButton = screen.getByRole("button", { name: infoButtonLabel });
    expect(infoButton).toBeInTheDocument();
    const chartCard = screen
      .getByRole("img", { name: /train\/accuracy scalar chart/i })
      .closest("section");
    if (!(chartCard instanceof HTMLElement)) {
      throw new Error("Expected scalar chart to render inside a card section");
    }

    await user.click(infoButton);

    const dialog = await screen.findByRole("dialog", { name: "train/accuracy" });
    expect(chartCard).not.toContainElement(dialog);
    expect(within(dialog).getByText("Training accuracy")).toBeInTheDocument();
    expect(within(dialog).getByText("Visible range")).toBeInTheDocument();
    expect(within(dialog).getByText("0.4 to 0.75")).toBeInTheDocument();
    expect(within(dialog).getByText(/training examples/i)).toBeInTheDocument();
    expect(within(dialog).getByText("Why it matters")).toBeInTheDocument();
    expect(within(dialog).getByText("Interpretation")).toBeInTheDocument();

    await waitFor(() =>
      expect(
        within(dialog).getByRole("button", { name: "Close metric info" }),
      ).toHaveFocus(),
    );
    await user.keyboard("{Escape}");

    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "train/accuracy" }),
      ).not.toBeInTheDocument(),
    );
    await waitFor(() => expect(infoButton).toHaveFocus());
  });

  it.each([
    {
      tag: "train/calibration/ece",
      displayName: "Training calibration error",
      body: /model's confidence matches reality/i,
    },
    {
      tag: "train/confidence/mean",
      displayName: "Training confidence",
      body: /how sure the model says it is/i,
    },
  ])("uses a plain-language explanation for $tag", async ({ tag, displayName, body }) => {
    const user = userEvent.setup();
    renderScalarChart(tag);

    await user.click(screen.getByRole("button", { name: `Explain metric ${tag}` }));

    const dialog = await screen.findByRole("dialog", { name: tag });
    expect(within(dialog).getByText(displayName)).toBeInTheDocument();
    expect(within(dialog).getByText(body)).toBeInTheDocument();
  });
});
