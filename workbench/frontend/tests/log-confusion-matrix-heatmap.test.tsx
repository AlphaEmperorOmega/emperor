import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
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

    expect(
      screen.getByRole("img", {
        name: /validation confusion matrix for run-a · MNIST · baseline, 20 classes/i,
      }),
    ).toBeInstanceOf(HTMLCanvasElement);
    expect(container.querySelectorAll("canvas")).toHaveLength(1);
    expect(container.querySelectorAll('[role="grid"]')).toHaveLength(0);
    expect(container.querySelectorAll("[title]")).toHaveLength(0);
    expect(container.querySelectorAll("*").length).toBeLessThan(20);
  });

  it("shows the true class, predicted class, and value for the hovered cell", async () => {
    render(<LogConfusionMatrixHeatmaps heatmaps={[heatmap()]} />);

    const canvas = screen.getByRole("img", {
      name: /validation confusion matrix for run-a/i,
    });
    vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue(rect(98, 98));

    fireEvent(
      canvas,
      new MouseEvent("pointermove", {
        bubbles: true,
        clientX: 49,
        clientY: 83,
      }),
    );

    const tooltip = await screen.findByRole("tooltip");
    expect(tooltip).toHaveTextContent("true 1, predicted 0, 0.25");

    fireEvent.pointerLeave(canvas);
    await waitFor(() => {
      expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
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

    expect(
      await screen.findByRole("img", {
        name: /validation confusion matrix for run-a/i,
      }),
    ).toBeInTheDocument();
    expect(disconnect).toHaveBeenCalled();
  });
});
