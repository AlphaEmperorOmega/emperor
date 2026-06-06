"use client";

import { useEffect, useRef, type CSSProperties } from "react";
import EChartsReactCore from "echarts-for-react/lib/core";
import type { EChartsOption } from "echarts";
import { echarts } from "@/lib/echarts/register";
import { EMPEROR_THEME_NAME, registerEmperorTheme } from "@/lib/echarts/theme";
import { cn } from "@/lib/utils";

registerEmperorTheme();

export type EChartEventHandlers = Record<string, (params: unknown) => void>;

/**
 * Thin client-only wrapper around echarts-for-react's core entry. Renders an
 * ECharts canvas with the emperor theme, resizes on container changes, and -
 * when a `group` is given - links the instance so tooltips, axis pointers, and
 * dataZoom stay synced across every chart sharing that group.
 */
export function EChart({
  option,
  group,
  className,
  style,
  onEvents,
}: {
  option: EChartsOption;
  group?: string;
  className?: string;
  style?: CSSProperties;
  onEvents?: EChartEventHandlers;
}) {
  const chartRef = useRef<EChartsReactCore>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) {
      return;
    }
    const observer = new ResizeObserver(() => {
      chartRef.current?.getEchartsInstance().resize();
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={containerRef} className={cn("h-full w-full", className)}>
      <EChartsReactCore
        ref={chartRef}
        echarts={echarts}
        option={option}
        theme={EMPEROR_THEME_NAME}
        notMerge
        lazyUpdate
        style={{ height: "100%", width: "100%", ...style }}
        opts={{ renderer: "canvas" }}
        onEvents={onEvents}
        onChartReady={(instance) => {
          if (group) {
            instance.group = group;
            echarts.connect(group);
          }
        }}
      />
    </div>
  );
}
