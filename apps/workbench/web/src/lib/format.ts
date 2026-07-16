export type RunDisplayFields = {
  name?: string | null;
  id?: string | null;
  startTime?: string | null;
};

const normalNumberFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 4,
});

const significantNumberFormatter = new Intl.NumberFormat(undefined, {
  maximumSignificantDigits: 4,
});

const integerFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

const dateTimeFormatter = new Intl.DateTimeFormat(undefined, {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

function decimalFormatter(
  minimumFractionDigits: number,
  maximumFractionDigits: number,
) {
  return new Intl.NumberFormat(undefined, {
    minimumFractionDigits,
    maximumFractionDigits,
  });
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) {
    return value.toExponential(2);
  }
  return normalNumberFormatter.format(value);
}

export function formatDecimal(
  value: number,
  fractionDigits: number,
  minimumFractionDigits = fractionDigits,
) {
  if (!Number.isFinite(value)) {
    return "0";
  }
  return decimalFormatter(minimumFractionDigits, fractionDigits).format(value);
}

export function formatSignificantNumber(value: number) {
  return Number.isFinite(value) ? significantNumberFormatter.format(value) : "0";
}

export function formatInteger(value: number) {
  return Number.isFinite(value) ? integerFormatter.format(value) : "0";
}

export function formatBytes(sizeBytes: number) {
  if (!Number.isFinite(sizeBytes) || sizeBytes < 0) {
    return "0\u00a0B";
  }
  if (sizeBytes < 1024) {
    return `${decimalFormatter(0, 0).format(sizeBytes)}\u00a0B`;
  }

  const units = ["KB", "MB", "GB"] as const;
  let value = sizeBytes / 1024;
  for (const unit of units) {
    if (value < 1024 || unit === "GB") {
      return `${decimalFormatter(0, value < 10 ? 1 : 0).format(value)}\u00a0${unit}`;
    }
    value /= 1024;
  }
  return `${decimalFormatter(0, 0).format(sizeBytes)}\u00a0B`;
}

export function formatDateTime(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : dateTimeFormatter.format(date);
}

export function formatRunTimestamp(startTime?: string | null): string {
  return startTime ?? "unknown";
}

export function formatRunDisplayName(run: RunDisplayFields): string {
  return `${run.name ?? run.id ?? "unknown"} · ${formatRunTimestamp(run.startTime)}`;
}
