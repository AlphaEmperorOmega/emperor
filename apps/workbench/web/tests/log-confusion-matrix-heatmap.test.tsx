import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LogConfusionMatrixHeatmaps } from "@/features/workbench/components/logs/log-confusion-matrix-heatmap";
import { type ConfusionMatrixHeatmap } from "@/features/workbench/state/logs/log-diagnostics";

function heatmap(overrides: Partial<ConfusionMatrixHeatmap> = {}) {
  return {
    key: "validation:run-a",
    split: "validation",
    runId: "run-a",
    runLabel: "run-a · MNIST · baseline",
    classCount: 2,
    cells: [
      { trueClass: 0, predictedClass: 0, value: 0.8 },
      { trueClass: 0, predictedClass: 1, value: 0.2 },
      { trueClass: 1, predictedClass: 0, value: 0.25 },
      { trueClass: 1, predictedClass: 1, value: 0.75 },
    ],
    ...overrides,
  } satisfies ConfusionMatrixHeatmap;
}

function largeHeatmap() {
  const classCount = 20;
  const cells: ConfusionMatrixHeatmap["cells"] = [];
  for (let trueClass = 0; trueClass < classCount; trueClass += 1) {
    for (let predictedClass = 0; predictedClass < classCount; predictedClass += 1) {
      cells.push({
        trueClass,
        predictedClass,
        value: trueClass === predictedClass ? 0.9 : 0.02,
      });
    }
  }
  return heatmap({ classCount, cells });
}

function rect(width: number, height: number) {
  return {
    bottom: height,
    height,
    left: 0,
    right: width,
    top: 0,
    width,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  } satisfies DOMRect;
}

afterEach(() => {
  vi.restoreAllMocks();
  Reflect.deleteProperty(globalThis, "IntersectionObserver");
});

describe("LogConfusionMatrixHeatmaps", () => {
  it("renders large matrices as one canvas instead of one DOM node per cell", () => {
    const { container } = render(
      <LogConfusionMatrixHeatmaps heatmaps={[largeHeatmap()]} />,
    );

    expect(container.querySelector("canvas")).toHaveAttribute("aria-hidden", "true");
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(container.querySelectorAll("canvas")).toHaveLength(1);
    expect(container.querySelectorAll('[role="grid"]')).toHaveLength(0);
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: /view validation confusion matrix for run-a · MNIST · baseline, 20 classes data/i,
      }),
    ).toHaveAttribute("aria-expanded", "false");
    expect(container.querySelectorAll("[title]")).toHaveLength(0);
    expect(container.querySelectorAll("*").length).toBeLessThan(25);
  });

  it("exposes every class-pair value through a keyboard-operated table", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <LogConfusionMatrixHeatmaps heatmaps={[heatmap()]} />,
    );

    expect(container.querySelector("canvas")).toHaveAttribute("aria-hidden", "true");
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();

    const disclosure = screen.getByRole("button", {
      name: /view validation confusion matrix for run-a · MNIST · baseline, 2 classes data/i,
    });
    expect(disclosure).toHaveAttribute("aria-expanded", "false");

    disclosure.focus();
    await user.keyboard("{Enter}");

    expect(disclosure).toHaveAttribute("aria-expanded", "true");
    const table = screen.getByRole("table", {
      name: /validation confusion matrix for run-a · MNIST · baseline, 2 classes/i,
    });
    expect(within(table).getAllByRole("columnheader")).toHaveLength(3);
    expect(within(table).getAllByRole("rowheader")).toHaveLength(2);
    expect(within(table).getAllByRole("cell").map((cell) => cell.textContent)).toEqual([
      "0.8",
      "0.2",
      "0.25",
      "0.75",
    ]);
  });

  it("shows the true class, predicted class, and value for the hovered cell", async () => {
    const { container } = render(
      <LogConfusionMatrixHeatmaps heatmaps={[heatmap()]} />,
    );

    const canvas = container.querySelector("canvas");
    expect(canvas).toBeInstanceOf(HTMLCanvasElement);
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new Error("Expected confusion matrix canvas");
    }
    vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue(rect(98, 98));

    fireEvent(
      canvas,
      new MouseEvent("pointermove", {
        bubbles: true,
        clientX: 49,
        clientY: 83,
      }),
    );

    const tooltip = await screen.findByText("true 1, predicted 0, 0.25");
    expect(tooltip).toHaveTextContent("true 1, predicted 0, 0.25");
    expect(tooltip).toHaveAttribute("aria-hidden", "true");

    fireEvent.pointerLeave(canvas);
    await waitFor(() => {
      expect(screen.queryByText("true 1, predicted 0, 0.25")).not.toBeInTheDocument();
    });
  });

  it("defers canvas mounting until the matrix card enters the observer margin", async () => {
    let onIntersect: IntersectionObserverCallback | null = null;
    const disconnect = vi.fn();
    const observe = vi.fn();
    class IntersectionObserverMock {
      readonly root = null;
      readonly rootMargin = "";
      readonly thresholds = [];
      constructor(callback: IntersectionObserverCallback) {
        onIntersect = callback;
      }
      disconnect = disconnect;
      observe = observe;
      takeRecords = () => [];
      unobserve = vi.fn();
    }
    Object.defineProperty(globalThis, "IntersectionObserver", {
      configurable: true,
      writable: true,
      value: IntersectionObserverMock,
    });

    const { container } = render(<LogConfusionMatrixHeatmaps heatmaps={[heatmap()]} />);

    expect(container.querySelector("canvas")).not.toBeInTheDocument();
    expect(observe).toHaveBeenCalledTimes(1);

    act(() => {
      onIntersect?.(
        [{ isIntersecting: true } as IntersectionObserverEntry],
        {} as IntersectionObserver,
      );
    });

    await waitFor(() => {
      expect(container.querySelector("canvas")).toHaveAttribute("aria-hidden", "true");
    });
    expect(disconnect).toHaveBeenCalled();
  });
});
