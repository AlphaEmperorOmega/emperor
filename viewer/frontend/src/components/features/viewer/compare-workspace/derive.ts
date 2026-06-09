import type {
  ConfigField,
  ConfigValue,
  Dataset,
  InspectResponse,
  MonitorOption,
  Preset,
} from "@/lib/api";
import { modelNameForId, modelTypeForId } from "@/lib/selection";

export type CompareEntry = {
  id: string;
  model: string;
  preset: string;
};

export type CompareEntryData = {
  entry: CompareEntry;
  presets: Preset[];
  datasets: Dataset[];
  monitors: MonitorOption[];
  fields: ConfigField[];
  inspection: InspectResponse | undefined;
  dataset: string;
  isLoading: boolean;
  error: unknown;
};

export type CompareModelOption = {
  id: string;
  label: string;
};

export type StatRow = {
  label: string;
  values: string[];
  changed: boolean;
};

export type ConfigDiffRow = {
  key: string;
  label: string;
  section: string;
  values: string[];
  changed: boolean;
};

export const MAX_COMPARE_TARGETS = 4;

const emptyValues = new Set(["", "—"]);

export function createCompareEntry(
  id: string,
  model = "",
  preset = "",
): CompareEntry {
  return { id, model, preset };
}

export function buildCompareModelOptions(models: string[]): CompareModelOption[] {
  return models.map((model) => ({
    id: model,
    label: `${modelNameForId(model)} · ${modelTypeForId(model)}`,
  }));
}

export function formatInteger(value: number | undefined) {
  if (value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  return new Intl.NumberFormat("en-US").format(value);
}

function configValueText(value: ConfigValue | undefined) {
  if (value === undefined) {
    return "—";
  }
  if (value === null) {
    return "None";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  return String(value);
}

function fieldEffectiveValue(field: ConfigField | undefined) {
  if (!field) {
    return undefined;
  }
  return field.locked ? field.lockedValue ?? field.default : field.default;
}

function uniqueMeaningfulValues(values: string[]) {
  return new Set(values.filter((value) => !emptyValues.has(value)));
}

function hasChangedValues(values: string[]) {
  const meaningfulValues = uniqueMeaningfulValues(values);
  if (meaningfulValues.size > 1) {
    return true;
  }
  return values.some((value) => value === "—") && meaningfulValues.size > 0;
}

function roleNodeCount(
  inspection: InspectResponse | undefined,
  role: "architecture" | "internal" | "runtime",
) {
  return inspection?.nodes.filter((node) => node.graphRole === role).length;
}

export function statRows(entries: CompareEntryData[]): StatRow[] {
  return [
    {
      label: "Parameters",
      values: entries.map((entry) => formatInteger(entry.inspection?.parameterCount)),
    },
    {
      label: "Graph nodes",
      values: entries.map((entry) => formatInteger(entry.inspection?.nodes.length)),
    },
    {
      label: "Graph edges",
      values: entries.map((entry) => formatInteger(entry.inspection?.edges.length)),
    },
    {
      label: "Architecture nodes",
      values: entries.map((entry) =>
        formatInteger(roleNodeCount(entry.inspection, "architecture")),
      ),
    },
    {
      label: "Runtime nodes",
      values: entries.map((entry) =>
        formatInteger(roleNodeCount(entry.inspection, "runtime")),
      ),
    },
    {
      label: "Config fields",
      values: entries.map((entry) => formatInteger(entry.fields.length)),
    },
    {
      label: "Preset-locked fields",
      values: entries.map((entry) =>
        formatInteger(entry.fields.filter((field) => field.locked).length),
      ),
    },
    {
      label: "Datasets",
      values: entries.map((entry) => formatInteger(entry.datasets.length)),
    },
    {
      label: "Monitors",
      values: entries.map((entry) => formatInteger(entry.monitors.length)),
    },
  ].map((row) => ({ ...row, changed: hasChangedValues(row.values) }));
}

export function changedConfigRows(entries: CompareEntryData[]): ConfigDiffRow[] {
  const fieldMeta = new Map<string, { label: string; section: string }>();
  entries.forEach((entry) => {
    entry.fields.forEach((field) => {
      if (!fieldMeta.has(field.key)) {
        fieldMeta.set(field.key, { label: field.label, section: field.section });
      }
    });
  });

  return Array.from(fieldMeta.entries())
    .map(([key, meta]) => {
      const values = entries.map((entry) => {
        const field = entry.fields.find((candidate) => candidate.key === key);
        return configValueText(fieldEffectiveValue(field));
      });
      return {
        key,
        ...meta,
        values,
        changed: hasChangedValues(values),
      };
    })
    .filter((row) => row.changed)
    .sort((left, right) =>
      left.section === right.section
        ? left.label.localeCompare(right.label)
        : left.section.localeCompare(right.section),
    );
}

export function monitorSummary(monitors: MonitorOption[]) {
  if (monitors.length === 0) {
    return "No monitors";
  }
  const labels = monitors
    .slice(0, 3)
    .map((monitor) => monitor.label)
    .join(", ");
  return monitors.length > 3 ? `${labels}, +${monitors.length - 3}` : labels;
}

export function compareHeader(entry: CompareEntryData, index: number) {
  const modelLabel = entry.entry.model ? modelNameForId(entry.entry.model) : "No model";
  const presetLabel = entry.entry.preset || "No preset";
  return `${index + 1}. ${modelLabel} / ${presetLabel}`;
}

export function isReadyEntry(entry: CompareEntryData) {
  return Boolean(entry.entry.model && entry.entry.preset);
}
