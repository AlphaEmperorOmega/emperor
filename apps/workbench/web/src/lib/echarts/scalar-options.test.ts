import { describe, expect, it } from "vitest";
import type { LineSeriesOption } from "echarts";
import {
  buildScalarLineOption,
  SCALAR_OPTION_DOWNSAMPLE_THRESHOLD,
  type ScalarLine,
} from "@/lib/echarts/scalar-options";
import { workbenchVisualTokens } from "@/lib/visual-tokens";

function lineWith(points: ScalarLine["points"]): ScalarLine {
  return { id: "run-a", name: "Run A", color: "#7c6dff", points };
}

function namedLine(id: string, name: string, points: ScalarLine["points"]): ScalarLine {
  return { id, name, color: id === "run-a" ? "#7c6dff" : "#19c37d", points };
}

const multiPoint = lineWith([
  { step: 0, wallTime: 100, value: 1 },
  { step: 1, wallTime: 101, value: 2 },
  { step: 2, wallTime: 102, value: 3 },
]);

const secondMultiPoint = namedLine("run-b", "Run B", [
  { step: 0, wallTime: 100, value: 4 },
  { step: 1, wallTime: 101, value: 5 },
  { step: 2, wallTime: 102, value: 6 },
]);

function axis(option: ReturnType<typeof buildScalarLineOption>, key: "xAxis" | "yAxis") {
  const value = option[key];
  return (Array.isArray(value) ? value[0] : value) as { type?: string; min?: unknown };
}

function seriesOf(option: ReturnType<typeof buildScalarLineOption>): LineSeriesOption[] {
  return (option.series ?? []) as LineSeriesOption[];
}

function lineOpacity(series: LineSeriesOption) {
  return (series.lineStyle as { opacity?: number } | undefined)?.opacity;
}

function itemOpacity(series: LineSeriesOption) {
  return (series.itemStyle as { opacity?: number } | undefined)?.opacity;
}

function markLineData(option: ReturnType<typeof buildScalarLineOption>) {
  const firstSeries = seriesOf(option)[0] as LineSeriesOption | undefined;
  return (firstSeries?.markLine?.data ?? []) as Array<{
    name?: string;
    xAxis?: number;
  }>;
}

function expectTooltipFormatter(
  formatter: unknown,
): asserts formatter is (params: unknown) => string {
  expect(typeof formatter).toBe("function");
}

function axisTooltipFormatter(option: ReturnType<typeof buildScalarLineOption>) {
  const tooltip = option.tooltip;
  if (!tooltip || Array.isArray(tooltip) || typeof tooltip !== "object") {
    throw new Error("Expected object tooltip option");
  }
  const formatter = tooltip.formatter;
  expectTooltipFormatter(formatter);
  return formatter;
}

function checkpointTooltipFormatter(option: ReturnType<typeof buildScalarLineOption>) {
  const formatter = seriesOf(option)[0].markLine?.tooltip?.formatter;
  expectTooltipFormatter(formatter);
  return formatter;
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

  it("applies optional line style to rendered series", () => {
    const [series] = seriesOf(
      buildScalarLineOption([
        { ...multiPoint, lineStyle: { type: "dashed" } },
      ]),
    );

    expect((series.lineStyle as { type?: string } | undefined)?.type).toBe(
      "dashed",
    );
  });

  it("keeps existing series opacity when no line is highlighted", () => {
    const series = seriesOf(buildScalarLineOption([multiPoint, secondMultiPoint]));

    expect(series.map(lineOpacity)).toEqual([undefined, undefined]);
    expect(series.map(itemOpacity)).toEqual([undefined, undefined]);
  });

  it("dims non-highlighted lines to the default opacity", () => {
    const series = seriesOf(
      buildScalarLineOption([multiPoint, secondMultiPoint], {
        highlightedLineId: "run-a",
      }),
    );

    expect(lineOpacity(series[0])).toBeUndefined();
    expect(itemOpacity(series[0])).toBeUndefined();
    expect(lineOpacity(series[1])).toBe(0.1);
    expect(itemOpacity(series[1])).toBe(0.1);
  });

  it("renders a raw + smoothed pair per multi-point line when smoothing is on", () => {
    const series = seriesOf(buildScalarLineOption([multiPoint], { smoothing: 0.6 }));
    expect(series).toHaveLength(2);
    expect(series.every((entry) => entry.name === "Run A")).toBe(true);
  });

  it("keeps highlighted smoothed pairs at their existing opacity", () => {
    const series = seriesOf(
      buildScalarLineOption([multiPoint, secondMultiPoint], {
        highlightedLineId: "run-a",
        smoothing: 0.6,
      }),
    );

    expect(series.map((entry) => entry.name)).toEqual(["Run A", "Run A", "Run B", "Run B"]);
    expect(series.map(lineOpacity)).toEqual([undefined, 0.25, 0.1, 0.1]);
  });

  it("does not duplicate single-point lines even with smoothing", () => {
    const single = lineWith([{ step: 5, wallTime: 200, value: 9 }]);
    const series = seriesOf(buildScalarLineOption([single], { smoothing: 0.6 }));
    expect(series).toHaveLength(1);
    expect(series[0].showSymbol).toBe(true);
  });

  it("dims single-point symbols when the run is not highlighted", () => {
    const firstSingle = namedLine("run-a", "Run A", [
      { step: 5, wallTime: 200, value: 9 },
    ]);
    const secondSingle = namedLine("run-b", "Run B", [
      { step: 5, wallTime: 200, value: 7 },
    ]);
    const series = seriesOf(
      buildScalarLineOption([firstSingle, secondSingle], {
        highlightedLineId: "run-a",
      }),
    );

    expect(series[1].showSymbol).toBe(true);
    expect(itemOpacity(series[1])).toBe(0.1);
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
    expect(seriesOf(option)[0].markLine?.lineStyle).toMatchObject({
      color: workbenchVisualTokens.checkpointMarker,
    });
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

  it("escapes untrusted axis tooltip labels before returning HTML", () => {
    const formatter = axisTooltipFormatter(buildScalarLineOption([multiPoint]));
    const html = formatter([
      {
        axisValueLabel: '<b data-unsafe="1">step</b>',
        seriesName: '<img src=x onerror="alert(1)"> & run',
        marker: "<span></span>",
        value: [1, 2],
      },
    ]);

    expect(html).toContain("&lt;b data-unsafe=&quot;1&quot;&gt;step&lt;/b&gt;");
    expect(html).toContain("&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &amp; run");
    expect(html).not.toContain("<b data-unsafe");
    expect(html).not.toContain("<img");
  });

  it("escapes untrusted checkpoint tooltip labels before returning HTML", () => {
    const option = buildScalarLineOption([multiPoint], {
      checkpointMarkers: [
        {
          runId: "run-a",
          runLabel: '<img src=x onerror="alert(1)">',
          filename: 'epoch=<script>alert("x")</script>.ckpt',
          epoch: 0,
          step: 2,
        },
      ],
    });
    const formatter = checkpointTooltipFormatter(option);
    const html = formatter({ name: markLineData(option)[0].name });

    expect(html).toContain("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;");
    expect(html).toContain("epoch=&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;.ckpt");
    expect(html).not.toContain("<img");
    expect(html).not.toContain("<script>");
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

  it("enforces the total point budget across many individually smaller lines", () => {
    const points = Array.from({ length: 4_000 }, (_, index) => ({
      step: index,
      wallTime: 1_000 + index,
      value: index,
    }));
    const lines = Array.from({ length: 50 }, (_, index): ScalarLine => ({
      id: `run-${index}`,
      name: `Run ${index}`,
      color: "#7c6dff",
      points,
    }));
    const emittedSeries = seriesOf(buildScalarLineOption(lines));
    const emittedPointCount = emittedSeries.reduce(
      (total, series) => total + ((series.data as number[][] | undefined)?.length ?? 0),
      0,
    );

    expect(emittedPointCount).toBeLessThanOrEqual(100_000);
    expect(emittedSeries).toHaveLength(50);
    expect((emittedSeries[0].data as number[][]).at(-1)).toEqual([3_999, 3_999]);
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
