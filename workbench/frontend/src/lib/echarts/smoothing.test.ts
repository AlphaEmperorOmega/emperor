import { describe, expect, it } from "vitest";
import { applyEmaSmoothing } from "@/lib/echarts/smoothing";

const points = [
  { step: 0, wallTime: 10, value: 0 },
  { step: 1, wallTime: 11, value: 10 },
  { step: 2, wallTime: 12, value: 0 },
  { step: 3, wallTime: 13, value: 10 },
];

describe("applyEmaSmoothing", () => {
  it("returns the original values when weight is 0", () => {
    const result = applyEmaSmoothing(points, 0);
    expect(result.map((point) => point.value)).toEqual([0, 10, 0, 10]);
  });

  it("clamps negative weight to a no-op", () => {
    const result = applyEmaSmoothing(points, -5);
    expect(result.map((point) => point.value)).toEqual([0, 10, 0, 10]);
  });

  it("keeps the first smoothed value equal to the first raw value (debias)", () => {
    const result = applyEmaSmoothing(points, 0.9);
    expect(result[0].value).toBeCloseTo(0, 10);
  });

  it("damps oscillation so smoothed values stay between the extremes", () => {
    const result = applyEmaSmoothing(points, 0.8);
    for (const point of result) {
      expect(point.value).toBeGreaterThanOrEqual(0);
      expect(point.value).toBeLessThanOrEqual(10);
    }
    // The third raw value is 0 but smoothing pulls it up toward the running mean.
    expect(result[2].value).toBeGreaterThan(0);
  });

  it("preserves non-value fields", () => {
    const result = applyEmaSmoothing(points, 0.5);
    expect(result.map((point) => point.step)).toEqual([0, 1, 2, 3]);
    expect(result.map((point) => point.wallTime)).toEqual([10, 11, 12, 13]);
  });

  it("passes non-finite values through untouched", () => {
    const noisy = [
      { step: 0, value: 1 },
      { step: 1, value: Number.NaN },
      { step: 2, value: 3 },
    ];
    const result = applyEmaSmoothing(noisy, 0.5);
    expect(Number.isNaN(result[1].value)).toBe(true);
  });

  it("does not mutate the input array", () => {
    const input = points.map((point) => ({ ...point }));
    applyEmaSmoothing(input, 0.6);
    expect(input.map((point) => point.value)).toEqual([0, 10, 0, 10]);
  });
});
