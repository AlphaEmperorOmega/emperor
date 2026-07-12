import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/features/workbench/components/charts/echart", () => ({
  EChart: () => <div data-testid="echart" />,
}));

import {
  HistogramChart,
  MultiRunScalarChart,
  ScalarChart,
} from "@/features/workbench/components/monitor/monitor-charts";
import { type LogRun } from "@/lib/api";

function run(): LogRun {
  return {
    id: "run-1",
    group: null,
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    dataset: "Mnist",
    runName: "Run one",
    timestamp: "2026-07-12T10:00:00Z",
    version: "1",
    relativePath: "experiment/run-1",
    experiment: "experiment",
    hasResult: true,
    eventFileCount: 1,
    checkpointCount: 0,
    hasHparams: false,
    metrics: {},
  };
}

describe("monitor chart data alternatives", () => {
  it("exposes scalar steps, wall times, values, and truncation", async () => {
    const user = userEvent.setup();
    render(
      <ScalarChart
        series={{
          tag: "layer/weights",
          label: "weights",
          points: [{ step: 4, wallTime: 1780000000, value: 0.25 }],
          sourceItemCount: 3,
          returnedItemCount: 1,
          truncated: true,
          truncationReason: "Scalar payload capped.",
        }}
      />,
    );

    await user.click(screen.getByRole("button", { name: "View chart data" }));
    const dialog = screen.getByRole("dialog", { name: "weights" });
    expect(within(dialog).getByRole("columnheader", { name: "Step" }))
      .toBeInTheDocument();
    expect(within(dialog).getByRole("columnheader", { name: "Wall time" }))
      .toBeInTheDocument();
    expect(within(dialog).getByRole("columnheader", { name: "Value" }))
      .toBeInTheDocument();
    expect(within(dialog).getByText("4")).toBeInTheDocument();
    expect(within(dialog).getByText("0.25")).toBeInTheDocument();
    expect(within(dialog).getByRole("note")).toHaveTextContent(
      "Scalar payload capped.",
    );
  });

  it("exposes histogram bucket interval semantics", async () => {
    const user = userEvent.setup();
    render(
      <HistogramChart
        histogram={{
          tag: "layer/activation/histogram",
          step: 7,
          wallTime: 1780000000,
          buckets: [{ left: -1, right: 0, count: 6 }],
        }}
      />,
    );

    await user.click(screen.getByRole("button", { name: "View chart data" }));
    const table = screen.getByRole("table");
    expect(within(table).getByRole("columnheader", { name: "Inclusive lower bound" }))
      .toBeInTheDocument();
    expect(within(table).getByRole("columnheader", { name: "Exclusive upper bound" }))
      .toBeInTheDocument();
    expect(within(table).getByRole("columnheader", { name: "Count" }))
      .toBeInTheDocument();
    expect(within(table).getByText("-1")).toBeInTheDocument();
    expect(within(table).getByText("6")).toBeInTheDocument();
  });

  it("identifies the run for every multi-run scalar row", async () => {
    const user = userEvent.setup();
    render(
      <MultiRunScalarChart
        metric={{
          key: "validation/loss",
          missingRuns: [],
          entries: [
            {
              run: run(),
              series: {
                tag: "validation/loss",
                label: "loss",
                points: [{ step: 2, wallTime: 1780000001, value: 1.25 }],
              },
            },
          ],
        }}
      />,
    );

    await user.click(screen.getByRole("button", { name: "View chart data" }));
    const table = screen.getByRole("table");
    expect(within(table).getByRole("columnheader", { name: "Run" }))
      .toBeInTheDocument();
    expect(within(table).getByText(/Run one/)).toBeInTheDocument();
    expect(within(table).getByText("1.25")).toBeInTheDocument();
  });
});
