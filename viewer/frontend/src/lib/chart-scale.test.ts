import { describe, expect, it } from "vitest";
import { buildChartPath, buildLinearScale, formatChartDomain } from "@/lib/chart-scale";

describe("buildLinearScale", () => {
  it("uses the empty chart fallback domain and empty path", () => {
    const scale = buildLinearScale([], { width: 320, height: 92, padding: 10 });

    expect(scale.domain).toEqual({
      minStep: 0,
      maxStep: 1,
      minValue: 0,
      maxValue: 1,
    });
    expect(buildChartPath([], scale)).toBe("");
  });

  it("uses fallback spans for one point", () => {
    const scale = buildLinearScale([{ step: 5, value: 7 }], {
      width: 320,
      height: 92,
      padding: 10,
    });

    expect(scale.domain).toEqual({
      minStep: 5,
      maxStep: 5,
      minValue: 7,
      maxValue: 7,
    });
    expect(scale.coordinate({ step: 5, value: 7 })).toEqual({ x: 10, y: 82 });
  });

  it("maps multiple points across the drawable chart area", () => {
    const scale = buildLinearScale(
      [
        { step: 0, value: 0 },
        { step: 10, value: 100 },
      ],
      { width: 100, height: 50, padding: 10 },
    );

    expect(scale.coordinate({ step: 0, value: 0 })).toEqual({ x: 10, y: 40 });
    expect(scale.coordinate({ step: 10, value: 100 })).toEqual({ x: 90, y: 10 });
  });

  it("uses the fallback value span when min and max values are equal", () => {
    const scale = buildLinearScale(
      [
        { step: 0, value: 5 },
        { step: 10, value: 5 },
      ],
      { width: 100, height: 50, padding: 10 },
    );

    expect(scale.coordinate({ step: 0, value: 5 })).toEqual({ x: 10, y: 40 });
    expect(scale.coordinate({ step: 10, value: 5 })).toEqual({ x: 90, y: 40 });
  });

  it("uses the fallback step span when min and max steps are equal", () => {
    const scale = buildLinearScale(
      [
        { step: 2, value: 0 },
        { step: 2, value: 10 },
      ],
      { width: 100, height: 50, padding: 10 },
    );

    expect(scale.coordinate({ step: 2, value: 0 })).toEqual({ x: 10, y: 40 });
    expect(scale.coordinate({ step: 2, value: 10 })).toEqual({ x: 10, y: 10 });
  });

  it("uses an explicit domain", () => {
    const scale = buildLinearScale([{ step: 5, value: 50 }], {
      width: 100,
      height: 60,
      padding: 10,
      domain: { minStep: 0, maxStep: 10, minValue: 0, maxValue: 100 },
    });

    expect(scale.domain).toEqual({
      minStep: 0,
      maxStep: 10,
      minValue: 0,
      maxValue: 100,
    });
    expect(scale.coordinate({ step: 5, value: 50 })).toEqual({ x: 50, y: 30 });
  });

  it("supports separate x and y padding", () => {
    const scale = buildLinearScale([{ step: 10, value: 100 }], {
      width: 100,
      height: 80,
      padding: { x: 20, y: 10 },
      domain: { minStep: 0, maxStep: 10, minValue: 0, maxValue: 100 },
    });

    expect(scale.coordinate({ step: 10, value: 100 })).toEqual({ x: 80, y: 10 });
  });

  it("can preserve series-order step domains", () => {
    const scale = buildLinearScale(
      [
        { step: 10, value: 0 },
        { step: 0, value: 1 },
      ],
      { width: 100, height: 50, padding: 10, stepDomainMode: "series" },
    );

    expect(scale.domain).toMatchObject({ minStep: 10, maxStep: 0 });
    expect(scale.coordinate({ step: 10, value: 0 })).toEqual({ x: 10, y: 40 });
    expect(scale.coordinate({ step: 0, value: 1 })).toEqual({ x: 90, y: 10 });
  });
});

describe("buildChartPath", () => {
  it("builds monitor polyline point strings", () => {
    const scale = buildLinearScale(
      [
        { step: 0, value: 0 },
        { step: 1, value: 1 },
      ],
      { width: 320, height: 92, padding: 10 },
    );

    expect(
      buildChartPath(
        [
          { step: 0, value: 0 },
          { step: 1, value: 1 },
        ],
        scale,
      ),
    ).toBe("10,82 310,10");
  });

  it("builds logs chart SVG path commands with fixed precision", () => {
    const scale = buildLinearScale(
      [
        { step: 0, value: 0 },
        { step: 2, value: 4 },
      ],
      { width: 760, height: 188, padding: { x: 34, y: 22 }, pathKind: "path" },
    );

    expect(
      buildChartPath(
        [
          { step: 0, value: 0 },
          { step: 2, value: 4 },
        ],
        scale,
      ),
    ).toBe("M 34.00 166.00 L 726.00 22.00");
  });
});

describe("formatChartDomain", () => {
  it("formats min and max labels with the shared number formatter", () => {
    expect(
      formatChartDomain({
        minStep: 0,
        maxStep: 1,
        minValue: 0.0005,
        maxValue: 1234,
      }),
    ).toEqual({ minLabel: "5.00e-4", maxLabel: "1.23e+3" });
  });
});
