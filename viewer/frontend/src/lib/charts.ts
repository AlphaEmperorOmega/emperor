import { type LogRun } from "@/lib/api";

/** Compact, locale-aware number formatting for chart axes and legends. */
export function formatNumber(value: number) {
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) {
    return value.toExponential(2);
  }
  return value.toLocaleString("en-US", {
    maximumFractionDigits: 4,
  });
}

/** Palette cycled across overlaid run lines in multi-run charts. */
export const multiRunLineColors = [
  "#67e8f9",
  "#a78bfa",
  "#facc15",
  "#fb7185",
  "#34d399",
];

export function runTimestamp(run: LogRun) {
  return run.timestamp ?? run.version;
}

export function runDisplayName(run: LogRun) {
  return `${run.runName} · ${runTimestamp(run)}`;
}
