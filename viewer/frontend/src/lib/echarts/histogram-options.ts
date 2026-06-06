import type {
  CustomSeriesRenderItemAPI,
  CustomSeriesRenderItemParams,
  CustomSeriesRenderItemReturn,
  EChartsOption,
} from "echarts";
import type { HistogramData } from "@/types/monitor";
import { formatNumber } from "@/lib/format";

const HISTOGRAM_COLOR = "#a78bfa";

export type HistogramBarOptions = {
  /** Fixed y-axis maximum so paired comparison histograms share a scale. */
  maxCount?: number;
  color?: string;
};

// Each datum is [left, right, count]; a custom rect respects the real bucket
// width instead of forcing uniform bars like the old SVG version.
function renderHistogramBar(
  _params: CustomSeriesRenderItemParams,
  api: CustomSeriesRenderItemAPI,
): CustomSeriesRenderItemReturn {
  const left = api.value(0) as number;
  const right = api.value(1) as number;
  const count = api.value(2) as number;
  const topLeft = api.coord([left, count]);
  const bottomRight = api.coord([right, 0]);
  const width = bottomRight[0] - topLeft[0];
  return {
    type: "rect",
    shape: {
      x: topLeft[0],
      y: topLeft[1],
      width: Math.max(width - 1, 0.5),
      height: bottomRight[1] - topLeft[1],
    },
    style: api.style(),
  };
}

export function buildHistogramBarOption(
  histogram: HistogramData,
  options: HistogramBarOptions = {},
): EChartsOption {
  const { maxCount, color = HISTOGRAM_COLOR } = options;
  const data = histogram.buckets.map((bucket) => [bucket.left, bucket.right, bucket.count]);
  return {
    animation: false,
    grid: { left: 40, right: 12, top: 12, bottom: 24 },
    tooltip: {
      trigger: "item",
      formatter: (params) => {
        const value = (params as { value?: number[] }).value ?? [];
        const [left, right, count] = value;
        return `[${formatNumber(Number(left))}, ${formatNumber(
          Number(right),
        )})<br/>count ${formatNumber(Number(count))}`;
      },
    },
    xAxis: { type: "value", scale: true },
    yAxis: { type: "value", min: 0, max: maxCount },
    series: [
      {
        type: "custom",
        renderItem: renderHistogramBar,
        encode: { x: [0, 1], y: 2 },
        itemStyle: { color, opacity: 0.85 },
        data,
      },
    ],
  };
}
