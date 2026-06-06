import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

Object.defineProperty(window, "ResizeObserver", {
  writable: true,
  configurable: true,
  value: ResizeObserverMock,
});

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

HTMLCanvasElement.prototype.getContext = vi.fn(
  () => canvasContextStub,
) as unknown as typeof HTMLCanvasElement.prototype.getContext;
