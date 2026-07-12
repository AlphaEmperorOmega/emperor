import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const chartMocks = vi.hoisted(() => ({ options: [] as unknown[] }));

vi.mock("@/features/workbench/components/charts/echart", async () => {
  const { createElement } = await import("react");
  return {
    EChart: ({ option }: { option: unknown }) => {
      chartMocks.options.push(option);
      return createElement("div", { "data-testid": "echart" });
    },
  };
});

import {
  buildTrainValidationScalarLines,
  LazyLogScalarChart,
  LazyLogTrainValidationScalarChart,
  LogScalarChart,
  LogTrainValidationScalarChart,
} from "@/features/workbench/components/logs/log-scalar-chart";
import { formatRunLabel } from "@/features/workbench/state/logs/logs-selectors";
import {
  type LogCheckpoint,
  type LogRun,
  type LogScalarSeries,
} from "@/lib/api";

beforeEach(() => {
  chartMocks.options.length = 0;
});

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

function renderTrainValidationChart({
  series,
  runs = [logRun()],
  runOrder = runs.map((run) => run.id),
  suffix = "loss_epoch",
  trainTag = `train/${suffix}`,
  validationTag = `validation/${suffix}`,
  xMode = "step",
}: {
  series: LogScalarSeries[];
  runs?: LogRun[];
  runOrder?: string[];
  suffix?: string;
  trainTag?: string;
  validationTag?: string;
  xMode?: "step" | "wallTime";
}) {
  return render(
    <LogTrainValidationScalarChart
      suffix={suffix}
      trainTag={trainTag}
      validationTag={validationTag}
      series={series}
      runsById={new Map(runs.map((run) => [run.id, run]))}
      checkpointsByRunId={new Map()}
      runOrder={runOrder}
      onSelectRun={vi.fn()}
      xMode={xMode}
    />,
  );
}

describe("Logs scalar rendering", () => {
  describe("train-validation line Adapter", () => {
  it("builds train-validation comparison lines with phase labels and stable ids", () => {
    const run = logRun();
    const lines = buildTrainValidationScalarLines({
      trainTag: "train/accuracy_epoch",
      validationTag: "validation/accuracy_epoch",
      runsById: new Map([[run.id, run]]),
      runOrder: [run.id],
      series: [
        scalarSeries({ tag: "train/accuracy_epoch" }),
        scalarSeries({ tag: "validation/accuracy_epoch" }),
      ],
    });

    expect(lines.map((line) => line.id)).toEqual([
      "run-1::train/accuracy_epoch",
      "run-1::validation/accuracy_epoch",
    ]);
    expect(lines.map((line) => line.name)).toEqual([
      `${formatRunLabel(run)} · Train`,
      `${formatRunLabel(run)} · Validation`,
    ]);
    expect(lines[0].lineStyle?.type).toBe("solid");
    expect(lines[1].lineStyle?.type).toBe("dashed");
  });

  it("keeps same-run train and validation lines separate across multiple runs", () => {
    const runs = [
      logRun({ id: "run-1", runName: "run-1" }),
      logRun({ id: "run-2", runName: "run-2", timestamp: null }),
    ];
    const lines = buildTrainValidationScalarLines({
      trainTag: "train/loss_epoch",
      validationTag: "validation/loss_epoch",
      runsById: new Map(runs.map((run) => [run.id, run])),
      runOrder: runs.map((run) => run.id),
      series: runs.flatMap((run) => [
        scalarSeries({ runId: run.id, tag: "train/loss_epoch" }),
        scalarSeries({ runId: run.id, tag: "validation/loss_epoch" }),
      ]),
    });

    expect(new Set(lines.map((line) => line.id)).size).toBe(4);
    expect(lines.map((line) => line.phase)).toEqual([
      "Train",
      "Validation",
      "Train",
      "Validation",
    ]);
    expect(lines.filter((line) => line.runId === "run-2").map((line) => line.tag))
      .toEqual(["train/loss_epoch", "validation/loss_epoch"]);
  });

  });

  describe("shared lazy contract", () => {

  it("shows train-validation loading and error states inside the lazy chart shell", async () => {
    const originalObserver = globalThis.IntersectionObserver;
    Reflect.deleteProperty(globalThis, "IntersectionObserver");

    try {
      const props = {
        suffix: "accuracy_epoch",
        trainTag: "train/accuracy_epoch",
        validationTag: "validation/accuracy_epoch",
        series: [],
        runsById: new Map([["run-1", logRun()]]),
        checkpointsByRunId: new Map(),
        runOrder: ["run-1"],
        onSelectRun: vi.fn(),
      };
      const { rerender } = render(
        <LazyLogTrainValidationScalarChart {...props} isLoading />,
      );

      expect(await screen.findByRole("status")).toHaveTextContent(
        "Loading accuracy_epoch train vs validation scalar points",
      );

      rerender(
        <LazyLogTrainValidationScalarChart
          {...props}
          hasRequested
          isError
          error={new Error("read failed")}
        />,
      );

      expect(
        await screen.findByText("accuracy_epoch train vs validation scalar read failed"),
      ).toBeInTheDocument();
      expect(screen.getByText("read failed")).toBeInTheDocument();
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

  it("waits for viewport entry before requesting scalar data", async () => {
    const originalObserver = globalThis.IntersectionObserver;
    let onIntersect: IntersectionObserverCallback | null = null;
    class IntersectionObserverMock {
      constructor(callback: IntersectionObserverCallback) {
        onIntersect = callback;
      }
      observe = vi.fn();
      disconnect = vi.fn();
      unobserve = vi.fn();
      takeRecords = vi.fn(() => []);
      root = null;
      rootMargin = "";
      thresholds = [];
    }
    Object.defineProperty(globalThis, "IntersectionObserver", {
      configurable: true,
      writable: true,
      value: IntersectionObserverMock,
    });
    const onVisible = vi.fn();

    try {
      const { rerender } = render(
        <LazyLogScalarChart
          tag="validation/loss"
          series={[]}
          runsById={new Map([["run-1", logRun()]])}
          checkpointsByRunId={new Map()}
          runOrder={["run-1"]}
          onSelectRun={vi.fn()}
          onVisible={onVisible}
        />,
      );

      expect(
        screen.getByLabelText("validation/loss scalar chart placeholder"),
      ).toBeInTheDocument();
      expect(onVisible).not.toHaveBeenCalled();

      act(() => {
        onIntersect?.(
          [{ isIntersecting: true } as IntersectionObserverEntry],
          {} as IntersectionObserver,
        );
      });

      await waitFor(() => expect(onVisible).toHaveBeenCalledWith("validation/loss"));
      expect(screen.getByRole("status")).toHaveTextContent(
        "Loading validation/loss scalar points",
      );
      expect(
        screen.queryByRole("img", {
          name: /validation\/loss scalar chart/i,
        }),
      ).not.toBeInTheDocument();

      rerender(
        <LazyLogScalarChart
          tag="validation/loss"
          series={[scalarSeries({ tag: "validation/loss" })]}
          runsById={new Map([["run-1", logRun()]])}
          checkpointsByRunId={new Map()}
          runOrder={["run-1"]}
          onSelectRun={vi.fn()}
          hasRequested
          onVisible={onVisible}
        />,
      );

      expect(
        await screen.findByRole("img", {
          name: /validation\/loss scalar chart/i,
        }),
      ).toBeInTheDocument();
    } finally {
      if (originalObserver) {
        Object.defineProperty(globalThis, "IntersectionObserver", {
          configurable: true,
          writable: true,
          value: originalObserver,
        });
      } else {
        Reflect.deleteProperty(globalThis, "IntersectionObserver");
      }
    }
  });

  it("requests visible scalar data when the observer misses the initial viewport entry", async () => {
    const originalObserver = globalThis.IntersectionObserver;
    const originalRequestAnimationFrame = globalThis.requestAnimationFrame;
    const originalCancelAnimationFrame = globalThis.cancelAnimationFrame;
    const rectSpy = vi.spyOn(HTMLElement.prototype, "getBoundingClientRect");
    class IntersectionObserverMock {
      observe = vi.fn();
      disconnect = vi.fn();
      unobserve = vi.fn();
      takeRecords = vi.fn(() => []);
      root = null;
      rootMargin = "";
      thresholds = [];
    }
    Object.defineProperty(globalThis, "IntersectionObserver", {
      configurable: true,
      writable: true,
      value: IntersectionObserverMock,
    });
    Object.defineProperty(globalThis, "requestAnimationFrame", {
      configurable: true,
      writable: true,
      value: (callback: FrameRequestCallback) => {
        callback(0);
        return 1;
      },
    });
    Object.defineProperty(globalThis, "cancelAnimationFrame", {
      configurable: true,
      writable: true,
      value: vi.fn(),
    });
    rectSpy.mockReturnValue({
      x: 0,
      y: 80,
      top: 80,
      left: 0,
      right: 800,
      bottom: 430,
      width: 800,
      height: 350,
      toJSON: () => ({}),
    } as DOMRect);
    const onVisible = vi.fn();

    try {
      render(
        <LazyLogScalarChart
          tag="validation/loss"
          series={[]}
          runsById={new Map([["run-1", logRun()]])}
          checkpointsByRunId={new Map()}
          runOrder={["run-1"]}
          onSelectRun={vi.fn()}
          onVisible={onVisible}
        />,
      );

      await waitFor(() => expect(onVisible).toHaveBeenCalledWith("validation/loss"));
      expect(screen.getByRole("status")).toHaveTextContent(
        "Loading validation/loss scalar points",
      );
    } finally {
      rectSpy.mockRestore();
      if (originalObserver) {
        Object.defineProperty(globalThis, "IntersectionObserver", {
          configurable: true,
          writable: true,
          value: originalObserver,
        });
      } else {
        Reflect.deleteProperty(globalThis, "IntersectionObserver");
      }
      if (originalRequestAnimationFrame) {
        Object.defineProperty(globalThis, "requestAnimationFrame", {
          configurable: true,
          writable: true,
          value: originalRequestAnimationFrame,
        });
      } else {
        Reflect.deleteProperty(globalThis, "requestAnimationFrame");
      }
      if (originalCancelAnimationFrame) {
        Object.defineProperty(globalThis, "cancelAnimationFrame", {
          configurable: true,
          writable: true,
          value: originalCancelAnimationFrame,
        });
      } else {
        Reflect.deleteProperty(globalThis, "cancelAnimationFrame");
      }
    }
  });

  });

  describe("shared card contract and Adapter projections", () => {

  it.each(["ordinary", "train-validation"] as const)(
    "projects checkpoint markers once through the %s Adapter",
    (kind) => {
      const checkpoint: LogCheckpoint = {
        id: "checkpoint-1",
        runId: "run-1",
        filename: "epoch=0-step=2.ckpt",
        relativePath: "kaggle_linear_all/run-1/checkpoints/epoch=0-step=2.ckpt",
        epoch: 0,
        step: 2,
        sizeBytes: 2048,
        modifiedAt: "2026-06-01T01:03:00Z",
      };
      const sharedProps = {
        runsById: new Map([["run-1", logRun()]]),
        checkpointsByRunId: new Map([["run-1", [checkpoint]]]),
        runOrder: ["run-1"],
        onSelectRun: vi.fn(),
      };

      if (kind === "ordinary") {
        render(
          <LogScalarChart
            {...sharedProps}
            tag="train/loss"
            series={[scalarSeries({ tag: "train/loss" })]}
          />,
        );
      } else {
        render(
          <LogTrainValidationScalarChart
            {...sharedProps}
            suffix="loss_epoch"
            trainTag="train/loss_epoch"
            validationTag="validation/loss_epoch"
            series={[
              scalarSeries({ tag: "train/loss_epoch" }),
              scalarSeries({ tag: "validation/loss_epoch" }),
            ]}
          />,
        );
      }

      const option = chartMocks.options.at(-1) as {
        series?: Array<{
          markLine?: { data?: Array<{ name: string; xAxis: number }> };
        }>;
      };
      expect(option.series?.[0]?.markLine?.data).toEqual([
        {
          name: `${formatRunLabel(logRun())}: epoch=0-step=2.ckpt (epoch 0, step 2)`,
          xAxis: 2,
        },
      ]);
    },
  );

  it.each(["ordinary", "train-validation"] as const)(
    "shares legend selection, hover, focus, and highlighting through the %s Adapter",
    (kind) => {
      const runs = [
        logRun({ id: "run-1", runName: "run-1" }),
        logRun({ id: "run-2", runName: "run-2", timestamp: null }),
      ];
      const onSelectRun = vi.fn();
      const onHoverRunChange = vi.fn();
      const sharedProps = {
        runsById: new Map(runs.map((run) => [run.id, run])),
        checkpointsByRunId: new Map<string, LogCheckpoint[]>(),
        runOrder: runs.map((run) => run.id),
        onSelectRun,
        onHoverRunChange,
        highlightedRunId: "run-1",
      };
      let secondLabel: string;

      if (kind === "ordinary") {
        render(
          <LogScalarChart
            {...sharedProps}
            tag="train/loss"
            series={runs.map((run) =>
              scalarSeries({ runId: run.id, tag: "train/loss" }),
            )}
          />,
        );
        secondLabel = formatRunLabel(runs[1]);
      } else {
        render(
          <LogTrainValidationScalarChart
            {...sharedProps}
            suffix="loss_epoch"
            trainTag="train/loss_epoch"
            validationTag="validation/loss_epoch"
            series={runs.map((run) =>
              scalarSeries({ runId: run.id, tag: "train/loss_epoch" }),
            )}
          />,
        );
        secondLabel = `${formatRunLabel(runs[1])} · Train`;
      }

      const secondButton = screen.getByText(secondLabel).closest("button");
      if (!(secondButton instanceof HTMLButtonElement)) {
        throw new Error("Expected the second Run legend entry to be interactive");
      }
      expect(secondButton).toHaveClass("opacity-30");

      fireEvent.pointerEnter(secondButton);
      fireEvent.pointerLeave(secondButton);
      fireEvent.focus(secondButton);
      fireEvent.blur(secondButton);
      fireEvent.click(secondButton);

      expect(onHoverRunChange.mock.calls).toEqual([
        ["run-2"],
        [null],
        ["run-2"],
        [null],
      ]);
      expect(onSelectRun).toHaveBeenCalledWith("run-2");
    },
  );

  it("shows the chronological displayed trend for decreasing loss", () => {
    render(
      <LogScalarChart
        tag="train/loss"
        series={[
          scalarSeries({
            tag: "train/loss",
            points: [
              { step: 1, wallTime: 1780000000, value: 2.1327 },
              { step: 2, wallTime: 1780000001, value: 1.4535 },
            ],
          }),
        ]}
        runsById={new Map([["run-1", logRun()]])}
        checkpointsByRunId={new Map()}
        runOrder={["run-1"]}
        onSelectRun={vi.fn()}
      />,
    );

    expect(screen.getByText("2.1327 to 1.4535")).toBeInTheDocument();
    expect(screen.queryByText("1.4535 to 2.1327")).not.toBeInTheDocument();
  });

  it("shows the chronological displayed trend for decreasing train-validation loss", async () => {
    const user = userEvent.setup();
    renderTrainValidationChart({
      series: [
        scalarSeries({
          tag: "train/loss_epoch",
          points: [
            { step: 1, wallTime: 1780000000, value: 2.1327 },
            { step: 2, wallTime: 1780000002, value: 1.4535 },
          ],
        }),
        scalarSeries({
          tag: "validation/loss_epoch",
          points: [
            { step: 1, wallTime: 1780000001, value: 2.01 },
            { step: 2, wallTime: 1780000001, value: 1.72 },
          ],
        }),
      ],
    });

    expect(screen.getByText("2.1327 to 1.4535")).toBeInTheDocument();
    expect(screen.queryByText("1.4535 to 2.1327")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Explain metric loss_epoch" }));

    const dialog = await screen.findByRole("dialog", {
      name: "validation/loss_epoch",
    });
    expect(within(dialog).getByText("Displayed trend")).toBeInTheDocument();
    expect(within(dialog).getByText("2.1327 to 1.4535")).toBeInTheDocument();
  });

  it("summarizes overlaid runs from aggregate earliest point to aggregate latest point", () => {
    const runs = [
      logRun({
        id: "run-1",
        runName: "run-1",
      }),
      logRun({
        id: "run-2",
        runName: "run-2",
      }),
    ];

    render(
      <LogScalarChart
        tag="train/loss"
        series={[
          scalarSeries({
            runId: "run-1",
            tag: "train/loss",
            points: [
              { step: 3, wallTime: 1780000003, value: 11 },
              { step: 9, wallTime: 1780000009, value: 21 },
            ],
          }),
          scalarSeries({
            runId: "run-2",
            tag: "train/loss",
            points: [
              { step: 1, wallTime: 1780000001, value: 101 },
              { step: 7, wallTime: 1780000007, value: 107 },
            ],
          }),
        ]}
        runsById={new Map(runs.map((run) => [run.id, run]))}
        checkpointsByRunId={new Map()}
        runOrder={runs.map((run) => run.id)}
        onSelectRun={vi.fn()}
      />,
    );

    expect(screen.getByText("101 to 21")).toBeInTheDocument();
    expect(screen.queryByText("11 to 107")).not.toBeInTheDocument();
  });

  it("summarizes train-validation runs from aggregate earliest displayed point to latest displayed point", () => {
    const runs = [
      logRun({ id: "run-1", runName: "run-1" }),
      logRun({ id: "run-2", runName: "run-2" }),
    ];

    renderTrainValidationChart({
      runs,
      series: [
        scalarSeries({
          runId: "run-1",
          tag: "train/loss_epoch",
          points: [
            { step: 3, wallTime: 1780000003, value: 11 },
            { step: 9, wallTime: 1780000009, value: 21 },
          ],
        }),
        scalarSeries({
          runId: "run-1",
          tag: "validation/loss_epoch",
          points: [
            { step: 4, wallTime: 1780000004, value: 12 },
            { step: 8, wallTime: 1780000008, value: 18 },
          ],
        }),
        scalarSeries({
          runId: "run-2",
          tag: "train/loss_epoch",
          points: [
            { step: 1, wallTime: 1780000001, value: 101 },
            { step: 7, wallTime: 1780000007, value: 107 },
          ],
        }),
        scalarSeries({
          runId: "run-2",
          tag: "validation/loss_epoch",
          points: [
            { step: 2, wallTime: 1780000002, value: 202 },
            { step: 10, wallTime: 1780000010, value: 210 },
          ],
        }),
      ],
    });

    expect(screen.getByText("101 to 210")).toBeInTheDocument();
    expect(screen.queryByText("11 to 210")).not.toBeInTheDocument();
  });

  it("orders train-validation displayed trend by wall time before step", () => {
    renderTrainValidationChart({
      xMode: "wallTime",
      series: [
        scalarSeries({
          tag: "validation/loss_epoch",
          points: [
            { step: 20, wallTime: 1780000100, value: 20 },
            { step: 1, wallTime: 1780000600, value: 1 },
          ],
        }),
        scalarSeries({
          tag: "train/loss_epoch",
          points: [
            { step: 10, wallTime: 1780000100, value: 10 },
            { step: 40, wallTime: 1780000500, value: 40 },
          ],
        }),
      ],
    });

    expect(screen.getByText("2 lines · step 1 to 40")).toBeInTheDocument();
    expect(screen.getByText("10 to 1")).toBeInTheDocument();
    expect(screen.queryByText("1 to 40")).not.toBeInTheDocument();
    expect(screen.queryByText("20 to 1")).not.toBeInTheDocument();
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
    expect(within(dialog).getByText("Displayed trend")).toBeInTheDocument();
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
});
