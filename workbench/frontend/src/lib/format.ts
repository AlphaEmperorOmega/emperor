export type RunDisplayFields = {
  name?: string | null;
  id?: string | null;
  startTime?: string | null;
};

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) {
    return value.toExponential(2);
  }
  return value.toLocaleString("en-US", { maximumFractionDigits: 4 });
}

export function formatRunTimestamp(startTime?: string | null): string {
  return startTime ?? "unknown";
}

export function formatRunDisplayName(run: RunDisplayFields): string {
  return `${run.name ?? run.id ?? "unknown"} · ${formatRunTimestamp(run.startTime)}`;
}
