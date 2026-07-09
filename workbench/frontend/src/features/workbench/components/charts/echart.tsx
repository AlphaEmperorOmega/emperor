"use client";

import { type CSSProperties } from "react";
import EChartsReactCore from "echarts-for-react/lib/core";
import type { EChartsOption } from "echarts";
import { echarts } from "@/lib/echarts/register";
import { EMPEROR_THEME_NAME, registerEmperorTheme } from "@/lib/echarts/theme";
import { cn } from "@/lib/utils";

registerEmperorTheme();

export type EChartEventHandlers = Record<string, (params: unknown) => void>;

/**
 * Thin client-only wrapper around echarts-for-react's core entry. Renders an
 * ECharts canvas with the emperor theme. The core component already auto-resizes
 * via its built-in size-sensor, so the container only needs `min-w-0` to be
 * allowed to shrink below the canvas's intrinsic width inside flex/grid layouts
 * (otherwise the box stays pinned to the old canvas width and never resizes).
 * When a `group` is given, the instance is linked so tooltips, axis pointers,
 * and dataZoom stay synced across every chart sharing that group.
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
  return (
    <div className={cn("h-full w-full min-w-0", className)}>
      <EChartsReactCore
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
