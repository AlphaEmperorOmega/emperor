import { describe, expect, it } from "vitest";
import type { LineSeriesOption } from "echarts";
import {
  buildScalarLineOption,
  type ScalarLine,
} from "@/lib/echarts/scalar-options";

function lineWith(points: ScalarLine["points"]): ScalarLine {
  return { id: "run-a", name: "Run A", color: "#7c6dff", points };
}

const multiPoint = lineWith([
  { step: 0, wallTime: 100, value: 1 },
  { step: 1, wallTime: 101, value: 2 },
  { step: 2, wallTime: 102, value: 3 },
]);

function axis(option: ReturnType<typeof buildScalarLineOption>, key: "xAxis" | "yAxis") {
  const value = option[key];
  return (Array.isArray(value) ? value[0] : value) as { type?: string; min?: unknown };
}

function seriesOf(option: ReturnType<typeof buildScalarLineOption>): LineSeriesOption[] {
  return (option.series ?? []) as LineSeriesOption[];
}

describe("buildScalarLineOption", () => {
  it("uses a value x-axis in step mode and a time x-axis in wallTime mode", () => {
    expect(axis(buildScalarLineOption([multiPoint], { xMode: "step" }), "xAxis").type).toBe(
      "value",
    );
    expect(
      axis(buildScalarLineOption([multiPoint], { xMode: "wallTime" }), "xAxis").type,
    ).toBe("time");
  });

  it("switches the y-axis to log scale on request", () => {
    expect(axis(buildScalarLineOption([multiPoint], { yScale: "log" }), "yAxis").type).toBe(
      "log",
    );
    expect(axis(buildScalarLineOption([multiPoint]), "yAxis").type).toBe("value");
  });

  it("renders one series per line without smoothing", () => {
    expect(seriesOf(buildScalarLineOption([multiPoint]))).toHaveLength(1);
  });

  it("renders a raw + smoothed pair per multi-point line when smoothing is on", () => {
    const series = seriesOf(buildScalarLineOption([multiPoint], { smoothing: 0.6 }));
    expect(series).toHaveLength(2);
    expect(series.every((entry) => entry.name === "Run A")).toBe(true);
  });

  it("does not duplicate single-point lines even with smoothing", () => {
    const single = lineWith([{ step: 5, wallTime: 200, value: 9 }]);
    const series = seriesOf(buildScalarLineOption([single], { smoothing: 0.6 }));
    expect(series).toHaveLength(1);
    expect(series[0].showSymbol).toBe(true);
  });

  it("scales wallTime seconds to milliseconds on the time axis", () => {
    const series = seriesOf(buildScalarLineOption([multiPoint], { xMode: "wallTime" }));
    const first = (series[0].data as number[][])[0];
    expect(first[0]).toBe(100 * 1000);
  });

  it("includes dataZoom only when requested", () => {
    expect(buildScalarLineOption([multiPoint]).dataZoom).toBeUndefined();
    expect(buildScalarLineOption([multiPoint], { dataZoom: true })).toHaveProperty("dataZoom");
    const zoom = buildScalarLineOption([multiPoint], { dataZoom: true }).dataZoom;
    expect(Array.isArray(zoom)).toBe(true);
  });

  it("applies a fixed value domain when provided", () => {
    const option = buildScalarLineOption([multiPoint], {
      domain: { minValue: -1, maxValue: 5 },
    });
    expect(axis(option, "yAxis").min).toBe(-1);
  });
});
