import { describe, expect, it } from "vitest";
import type { LineSeriesOption } from "echarts";
import {
  buildScalarLineOption,
  SCALAR_OPTION_DOWNSAMPLE_THRESHOLD,
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

function markLineData(option: ReturnType<typeof buildScalarLineOption>) {
  const firstSeries = seriesOf(option)[0] as LineSeriesOption | undefined;
  return (firstSeries?.markLine?.data ?? []) as Array<{
    name?: string;
    xAxis?: number;
  }>;
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

  it("draws aggregated checkpoint mark lines in step mode", () => {
    const option = buildScalarLineOption([multiPoint], {
      xMode: "step",
      checkpointMarkers: [
        {
          runId: "run-a",
          runLabel: "Run A",
          filename: "epoch=0-step=2.ckpt",
          epoch: 0,
          step: 2,
        },
        {
          runId: "run-b",
          runLabel: "Run B",
          filename: "epoch=1-step=2.ckpt",
          epoch: 1,
          step: 2,
        },
        {
          runId: "run-a",
          runLabel: "Run A",
          filename: "last.ckpt",
          epoch: null,
          step: null,
        },
      ],
    });

    const data = markLineData(option);
    expect(data).toHaveLength(1);
    expect(data[0].xAxis).toBe(2);
    expect(data[0].name).toContain("Run A: epoch=0-step=2.ckpt (epoch 0, step 2)");
    expect(data[0].name).toContain("Run B: epoch=1-step=2.ckpt (epoch 1, step 2)");
    expect(data[0].name).not.toContain("last.ckpt");
  });

  it("suppresses checkpoint mark lines in wall-time mode", () => {
    const option = buildScalarLineOption([multiPoint], {
      xMode: "wallTime",
      checkpointMarkers: [
        {
          runId: "run-a",
          runLabel: "Run A",
          filename: "epoch=0-step=2.ckpt",
          epoch: 0,
          step: 2,
        },
      ],
    });

    expect(markLineData(option)).toHaveLength(0);
  });

  it("downsamples large point arrays before building series data", () => {
    const points = Array.from({ length: 100_001 }, (_, index) => ({
      step: index,
      wallTime: 1_000 + index,
      value: index / 10,
    }));
    const option = buildScalarLineOption([lineWith(points)]);
    const data = seriesOf(option)[0].data as number[][];

    expect(data).toHaveLength(SCALAR_OPTION_DOWNSAMPLE_THRESHOLD);
    expect(data.at(-1)).toEqual([100_000, 10_000]);
  });

  it("downsamples before smoothing so raw and smoothed series stay bounded", () => {
    const points = Array.from({ length: 100_001 }, (_, index) => ({
      step: index,
      wallTime: 1_000 + index,
      value: index,
    }));
    const option = buildScalarLineOption([lineWith(points)], { smoothing: 0.5 });
    const [smoothed, raw] = seriesOf(option);

    expect(smoothed.data).toHaveLength(SCALAR_OPTION_DOWNSAMPLE_THRESHOLD);
    expect(raw.data).toHaveLength(SCALAR_OPTION_DOWNSAMPLE_THRESHOLD);
    expect((raw.data as number[][]).at(-1)).toEqual([100_000, 100_000]);
  });
});
