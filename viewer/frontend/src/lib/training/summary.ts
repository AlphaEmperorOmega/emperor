import { type ConfigField, type TrainingJob } from "@/lib/api";
import { type OverrideValues } from "@/lib/config";

/** Picks the most informative metric from a job (accuracy/f1 > loss > first). */
export function metricLabel(job?: TrainingJob) {
  const metrics = job?.metrics ?? {};
  const entries = Object.entries(metrics);
  const preferred =
    entries.find(([key]) => /accuracy|acc|f1/i.test(key)) ??
    entries.find(([key]) => /loss/i.test(key)) ??
    entries[0];
  if (!preferred) {
    return "No metrics";
  }
  const [key, value] = preferred;
  const numberValue = typeof value === "number" ? value.toFixed(4).replace(/0+$/, "") : value;
  return `${key}: ${numberValue}`;
}

/** Resolves override entries to display rows with their human-readable labels. */
export function overrideSummary(fields: ConfigField[], overrides: OverrideValues) {
  const labels = new Map(fields.map((field) => [field.key, field.label]));
  return Object.entries(overrides).map(([key, value]) => ({
    key,
    label: labels.get(key) ?? key,
    value: value || "None",
  }));
}
