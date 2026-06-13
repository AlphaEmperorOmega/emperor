import "@testing-library/jest-dom/vitest";
import { createElement } from "react";
import type { CSSProperties } from "react";
import { vi } from "vitest";

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

if (typeof window !== "undefined") {
  Object.defineProperty(window, "ResizeObserver", {
    configurable: true,
    writable: true,
    value: ResizeObserverMock,
  });
}

type MockEChartDataItem = {
  name?: unknown;
  nodeId?: unknown;
  path?: unknown;
  children?: MockEChartDataItem[];
};

function collectMockTreemapItems(option: unknown) {
  const items: MockEChartDataItem[] = [];
  const optionRecord =
    typeof option === "object" && option !== null
      ? (option as { series?: unknown })
      : undefined;
  const series = Array.isArray(optionRecord?.series)
    ? optionRecord.series
    : optionRecord?.series
      ? [optionRecord.series]
      : [];

  const visit = (item: unknown) => {
    if (typeof item !== "object" || item === null) {
      return;
    }
    const treemapItem = item as MockEChartDataItem;
    if (typeof treemapItem.nodeId === "string") {
      items.push(treemapItem);
    }
    if (Array.isArray(treemapItem.children)) {
      for (const child of treemapItem.children) {
        visit(child);
      }
    }
  };

  for (const entry of series) {
    if (typeof entry !== "object" || entry === null) {
      continue;
    }
    const data = (entry as { data?: unknown }).data;
    if (Array.isArray(data)) {
      for (const item of data) {
        visit(item);
      }
    }
  }
  return items;
}

vi.mock("@/features/viewer/components/charts/echart", () => ({
  EChart: ({
    option,
    onEvents,
    className,
    style,
  }: {
    option: unknown;
    onEvents?: Record<string, (params: unknown) => void>;
    className?: string;
    style?: CSSProperties;
  }) =>
    createElement(
      "div",
      { "data-testid": "echart", className, style },
      collectMockTreemapItems(option).map((item, index) =>
        createElement("button", {
          key: `${String(item.nodeId)}-${index}`,
          type: "button",
          "aria-label": `treemap node ${String(item.path ?? item.nodeId ?? item.name)}`,
          onClick: () => onEvents?.click?.({ data: item }),
        }),
      ),
    ),
}));

// jsdom has no canvas implementation; ECharts needs a 2D context to initialise.
// Component tests mock the <EChart> wrapper, so this stub only guards against a
// stray real mount throwing "getContext is not a function".
const noop = () => {};
const canvasContextStub = {
  canvas: { width: 0, height: 0 },
  measureText: () => ({ width: 0 }),
  getImageData: () => ({ data: [] }),
  createLinearGradient: () => ({ addColorStop: noop }),
  createRadialGradient: () => ({ addColorStop: noop }),
  createPattern: () => ({}),
  getLineDash: () => [],
  setLineDash: noop,
  save: noop,
  restore: noop,
  scale: noop,
  rotate: noop,
  translate: noop,
  transform: noop,
  setTransform: noop,
  resetTransform: noop,
  beginPath: noop,
  closePath: noop,
  moveTo: noop,
  lineTo: noop,
  bezierCurveTo: noop,
  quadraticCurveTo: noop,
  arc: noop,
  arcTo: noop,
  ellipse: noop,
  rect: noop,
  fill: noop,
  stroke: noop,
  clip: noop,
  fillRect: noop,
  strokeRect: noop,
  clearRect: noop,
  fillText: noop,
  strokeText: noop,
  drawImage: noop,
  putImageData: noop,
  createImageData: () => ({ data: [] }),
};

if (typeof HTMLCanvasElement !== "undefined") {
  HTMLCanvasElement.prototype.getContext = vi.fn(
    () => canvasContextStub,
  ) as unknown as typeof HTMLCanvasElement.prototype.getContext;
}
