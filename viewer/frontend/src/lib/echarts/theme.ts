import { echarts } from "@/lib/echarts/register";
import { scalarSeriesColors } from "@/lib/charts";

export const EMPEROR_THEME_NAME = "emperor";

// Mirrors the CSS tokens in app/globals.css. Canvas text cannot resolve CSS
// custom properties, so colors are inlined and the mono font is resolved at
// registration time from the --font-mono variable (with a system fallback).
const INK = "#ecebf5";
const INK_DIM = "#9c9cb6";
const INK_FAINT = "#62627c";
const LINE = "rgba(255, 255, 255, 0.07)";
const LINE_SOFT = "rgba(255, 255, 255, 0.045)";
const PANEL = "#0c0c15";
const VIOLET = "#a78bfa";

const MONO_FALLBACK =
  "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace";

function resolveMonoFontFamily(): string {
  if (typeof document === "undefined") {
    return MONO_FALLBACK;
  }
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue("--font-mono")
    .trim();
  return value ? `${value}, ${MONO_FALLBACK}` : MONO_FALLBACK;
}

function buildEmperorTheme() {
  const fontFamily = resolveMonoFontFamily();
  const axis = {
    axisLine: { lineStyle: { color: LINE } },
    axisTick: { show: false },
    axisLabel: { color: INK_FAINT, fontFamily, fontSize: 10 },
    splitLine: { lineStyle: { color: LINE_SOFT } },
    nameTextStyle: { color: INK_DIM, fontFamily },
  };
  return {
    color: scalarSeriesColors,
    backgroundColor: "transparent",
    textStyle: { color: INK_DIM, fontFamily },
    title: {
      textStyle: { color: INK },
      subtextStyle: { color: INK_FAINT },
    },
    categoryAxis: axis,
    valueAxis: axis,
    logAxis: axis,
    timeAxis: axis,
    legend: { textStyle: { color: INK_DIM, fontFamily } },
    tooltip: {
      backgroundColor: PANEL,
      borderColor: LINE,
      borderWidth: 1,
      textStyle: { color: INK, fontFamily, fontSize: 11 },
      axisPointer: {
        lineStyle: { color: VIOLET, opacity: 0.5 },
        crossStyle: { color: VIOLET, opacity: 0.5 },
        label: { backgroundColor: VIOLET, color: "#06060c" },
      },
    },
    dataZoom: {
      borderColor: LINE,
      fillerColor: "rgba(167, 139, 250, 0.12)",
      handleStyle: { color: VIOLET, borderColor: VIOLET },
      moveHandleStyle: { color: INK_FAINT },
      textStyle: { color: INK_FAINT, fontFamily },
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
