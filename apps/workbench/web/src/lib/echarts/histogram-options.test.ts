import { describe, expect, it } from "vitest";
import type { CustomSeriesOption } from "echarts";
import type { HistogramData } from "@/types/monitor";
import { buildHistogramBarOption } from "@/lib/echarts/histogram-options";
import { workbenchVisualTokens } from "@/lib/visual-tokens";

const histogram: HistogramData = {
  tag: "layer/weights/histogram",
  step: 7,
  wallTime: 100,
  buckets: [
    { left: -1, right: 0, count: 3 },
    { left: 0, right: 1, count: 8 },
    { left: 1, right: 2, count: 2 },
  ],
};

function axis(option: ReturnType<typeof buildHistogramBarOption>, key: "xAxis" | "yAxis") {
  const value = option[key];
  return (Array.isArray(value) ? value[0] : value) as { type?: string; max?: unknown };
}

function customSeries(option: ReturnType<typeof buildHistogramBarOption>) {
  return (option.series as CustomSeriesOption[])[0];
}

describe("buildHistogramBarOption", () => {
  it("renders one custom series with a datum per bucket", () => {
    const series = customSeries(buildHistogramBarOption(histogram));
    const data = series.data as number[][];
    expect(series.type).toBe("custom");
    expect(data).toHaveLength(3);
    expect(data[0]).toEqual([-1, 0, 3]);
    expect((series.itemStyle as { color?: string }).color).toBe(
      workbenchVisualTokens.violet,
    );
  });

  it("uses value axes for both dimensions", () => {
    const option = buildHistogramBarOption(histogram);
    expect(axis(option, "xAxis").type).toBe("value");
    expect(axis(option, "yAxis").type).toBe("value");
  });

  it("leaves the y-axis max open when no maxCount is given", () => {
    expect(axis(buildHistogramBarOption(histogram), "yAxis").max).toBeUndefined();
  });

  it("fixes the y-axis max for shared comparison scaling", () => {
    expect(axis(buildHistogramBarOption(histogram, { maxCount: 20 }), "yAxis").max).toBe(20);
  });

  it("encodes left/right on x and count on y", () => {
    const series = customSeries(buildHistogramBarOption(histogram));
    expect(series.encode).toEqual({ x: [0, 1], y: 2 });
  });
});
