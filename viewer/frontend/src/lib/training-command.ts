import {
  type ConfigSection,
  type OverrideValues,
  hasOverride,
  overrideValue,
} from "@/lib/config";
import { type ConfigField } from "@/lib/api";

function shellQuote(value: string) {
  if (value === "") {
    return "''";
  }
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(value)) {
    return value;
  }
  return `'${value.replace(/'/g, `'"'"'`)}'`;
}

function orderedOverrideEntries(sections: ConfigSection[], overrides: OverrideValues) {
  const entries: Array<{ field: ConfigField; value: string }> = [];
  for (const section of sections) {
    for (const field of section.fields) {
      if (hasOverride(overrides, field.key)) {
        entries.push({ field, value: overrideValue(overrides, field.key) ?? "" });
      }
    }
  }
  return entries;
}

function trainingCommandValue(field: ConfigField, value: string) {
  return field.nullable && value === "" ? "None" : value;
}

export function buildTrainingCommand({
  modelType,
  model,
  preset,
  presets,
  sections,
  overrides,
}: {
  modelType: string;
  model: string;
  preset: string;
  presets?: string[];
  sections: ConfigSection[];
  overrides: OverrideValues;
}) {
  const selectedPresets = [
    preset,
    ...(presets ?? []).filter((candidate) => candidate !== preset),
  ].filter((candidate, index, all) => candidate && all.indexOf(candidate) === index);
  const parts = [
    "source",
    "experiment.sh",
    "--model-type",
    shellQuote(modelType),
    "--model",
    shellQuote(model),
  ];
  if (selectedPresets.length > 1) {
    parts.push("--presets", ...selectedPresets.map(shellQuote));
  } else {
    parts.push("--preset", shellQuote(selectedPresets[0] ?? preset));
  }
  const orderedOverrides = orderedOverrideEntries(sections, overrides);

  if (orderedOverrides.length > 0) {
    parts.push("--config");
    for (const { field, value } of orderedOverrides) {
      parts.push(field.flag, shellQuote(trainingCommandValue(field, value)));
    }
  }

  return parts.join(" ");
}
