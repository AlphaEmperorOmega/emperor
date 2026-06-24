import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LogScalarChart } from "@/features/viewer/components/logs/log-scalar-chart";
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
