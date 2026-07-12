import { echarts } from "@/lib/echarts/register";
import { scalarSeriesColors } from "@/lib/charts";
import {
  workbenchFontTokens,
  workbenchVisualTokens,
} from "@/lib/visual-tokens";

export const EMPEROR_THEME_NAME = "emperor";

const MONO_FALLBACK = workbenchFontTokens.mono;

function resolveMonoFontFamily(): string {
  if (typeof document === "undefined") {
    return MONO_FALLBACK;
  }
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue("--font-mono")
    .trim();
  return value ? `${value}, ${MONO_FALLBACK}` : MONO_FALLBACK;
}

export function buildEmperorTheme(fontFamily = resolveMonoFontFamily()) {
  const axis = {
    axisLine: { lineStyle: { color: workbenchVisualTokens.line } },
    axisTick: { show: false },
    axisLabel: { color: workbenchVisualTokens.inkFaint, fontFamily, fontSize: 10 },
    splitLine: { lineStyle: { color: workbenchVisualTokens.lineSoft } },
    nameTextStyle: { color: workbenchVisualTokens.inkDim, fontFamily },
  };
  return {
    color: scalarSeriesColors,
    backgroundColor: "transparent",
    textStyle: { color: workbenchVisualTokens.inkDim, fontFamily },
    title: {
      textStyle: { color: workbenchVisualTokens.ink },
      subtextStyle: { color: workbenchVisualTokens.inkFaint },
    },
    categoryAxis: axis,
    valueAxis: axis,
    logAxis: axis,
    timeAxis: axis,
    legend: { textStyle: { color: workbenchVisualTokens.inkDim, fontFamily } },
    tooltip: {
      backgroundColor: workbenchVisualTokens.panel,
      borderColor: workbenchVisualTokens.line,
      borderWidth: 1,
      textStyle: { color: workbenchVisualTokens.ink, fontFamily, fontSize: 11 },
      axisPointer: {
        lineStyle: { color: workbenchVisualTokens.violet, opacity: 0.5 },
        crossStyle: { color: workbenchVisualTokens.violet, opacity: 0.5 },
        label: {
          backgroundColor: workbenchVisualTokens.violet,
          color: workbenchVisualTokens.bg,
        },
      },
    },
    dataZoom: {
      borderColor: workbenchVisualTokens.line,
      fillerColor: workbenchVisualTokens.accentFill,
      handleStyle: {
        color: workbenchVisualTokens.violet,
        borderColor: workbenchVisualTokens.violet,
      },
      moveHandleStyle: { color: workbenchVisualTokens.inkFaint },
      textStyle: { color: workbenchVisualTokens.inkFaint, fontFamily },
    },
  };
}

let registered = false;

/** Registers the "emperor" ECharts theme once, on the client. */
export function registerEmperorTheme(): void {
  if (registered) {
    return;
  }
  echarts.registerTheme(EMPEROR_THEME_NAME, buildEmperorTheme());
  registered = true;
}
