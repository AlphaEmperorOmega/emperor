import { type ConfigField } from "@/lib/api";
import {
  ADAPTIVE_OPTION_PAIRS,
  configKeyToken,
  defaultConfigFieldValue,
  hasOverride,
  normalizeConfigFieldValue,
  overrideValue,
  overrideValueForConfigField,
  type OverrideValues,
} from "@/lib/config";

export type ConfigSnapshot = {
  id: string;
  name: string;
  modelType: string;
  model: string;
  preset: string;
  overrides: OverrideValues;
  createdAt: string;
};

export type ConfigSnapshotOverrideEntry = {
  key: string;
  label: string;
  value: string;
  displayValue: string;
};

export type ConfigSnapshotGroup = {
  preset: string;
  snapshots: ConfigSnapshot[];
};

export type ConfigSnapshotCreateResult =
  | { ok: true; snapshot: ConfigSnapshot }
  | { ok: false; error: string };

export function configSnapshotOverrideCount(
  snapshot: Pick<ConfigSnapshot, "overrides">,
) {
  return Object.keys(snapshot.overrides).length;
}

export function configSnapshotOverrideCountLabel(count: number) {
  return `${count} override${count === 1 ? "" : "s"}`;
}

function displayValueForField(field: ConfigField, value: string) {
  return field.nullable && value === "" ? "None" : value;
}

export function configSnapshotOverrideEntries(
  fields: ConfigField[],
  overrides: OverrideValues,
) {
  const entries: ConfigSnapshotOverrideEntry[] = [];
  const lockedFields: ConfigField[] = [];

  for (const field of fields) {
    const rawValue = overrideValue(overrides, field.key);
    if (!hasOverride(overrides, field.key) || rawValue === undefined) {
      continue;
    }
    if (field.locked) {
      lockedFields.push(field);
    }
    const normalizedValue = normalizeConfigFieldValue(field, rawValue);
    if (normalizedValue === defaultConfigFieldValue(field)) {
      continue;
    }
    const value = overrideValueForConfigField(field, normalizedValue);
    entries.push({
      key: field.key,
      label: field.label,
      value,
      displayValue: displayValueForField(field, value),
    });
  }

  return { entries, lockedFields };
}

export function overridesFromSnapshotEntries(
  entries: ConfigSnapshotOverrideEntry[],
) {
  return Object.fromEntries(entries.map((entry) => [entry.key, entry.value]));
}

function isEnabledSnapshotValue(value: string | undefined) {
  return ["true", "1", "yes", "on"].includes((value ?? "").trim().toLowerCase());
}

function isPresentSnapshotOption(value: string | undefined) {
  const normalized = (value ?? "").trim().toLowerCase();
  return normalized !== "" && normalized !== "null" && normalized !== "none";
}

function validateAdaptiveOptionRequirements(
  entries: ConfigSnapshotOverrideEntry[],
) {
  const entryValues = new Map(
    entries.map((entry) => [configKeyToken(entry.key), entry.value]),
  );
  for (const { flagKey, optionKey } of ADAPTIVE_OPTION_PAIRS) {
    if (
      isEnabledSnapshotValue(entryValues.get(configKeyToken(flagKey))) &&
      !isPresentSnapshotOption(entryValues.get(configKeyToken(optionKey)))
    ) {
      return {
        ok: false as const,
        error: `Invalid config snapshot overrides: ${optionKey} must be set when ${flagKey} is True.`,
      };
    }
  }
  return { ok: true as const };
}

function snapshotIdentity({
  modelType,
  model,
  preset,
  entries,
}: {
  modelType: string;
  model: string;
  preset: string;
  entries: ConfigSnapshotOverrideEntry[];
}) {
  return [
    modelType,
    model,
    preset,
    ...entries.map((entry) => `${entry.key}=${entry.value}`),
  ].join("\u0000");
}

export function configSnapshotIdentity({
  model,
  modelType,
  preset,
  fields,
  overrides,
}: {
  modelType: string;
  model: string;
  preset: string;
  fields: ConfigField[];
  overrides: OverrideValues;
}) {
  const { entries } = configSnapshotOverrideEntries(fields, overrides);
  return snapshotIdentity({ modelType, model, preset, entries });
}

function normalizeSnapshotName(name: string) {
  return name.trim().toLocaleLowerCase();
}

export function validateConfigSnapshotName({
  modelType,
  model,
  preset,
  name,
  snapshots,
  excludeSnapshotId,
}: {
  modelType: string;
  model: string;
  preset: string;
  name: string;
  snapshots: ConfigSnapshot[];
  excludeSnapshotId?: string;
}) {
  const trimmedName = name.trim();
  if (!trimmedName) {
    return {
      ok: false as const,
      error: "Config snapshot name cannot be empty.",
    };
  }

  const normalizedName = normalizeSnapshotName(trimmedName);
  const duplicate = snapshots.some(
    (snapshot) =>
      snapshot.id !== excludeSnapshotId &&
      snapshot.modelType === modelType &&
      snapshot.model === model &&
      snapshot.preset === preset &&
      normalizeSnapshotName(snapshot.name) === normalizedName,
  );
  if (duplicate) {
    return {
      ok: false as const,
      error: "A snapshot with this name already exists.",
    };
  }

  return { ok: true as const, name: trimmedName };
}

export function validateConfigSnapshotCandidate({
  modelType,
  model,
  preset,
  fields,
  overrides,
  snapshots,
  excludeSnapshotId,
}: {
  modelType: string;
  model: string;
  preset: string;
  fields: ConfigField[];
  overrides: OverrideValues;
  snapshots: ConfigSnapshot[];
  excludeSnapshotId?: string;
}) {
  if (!modelType || !model || !preset) {
    return { ok: false as const, error: "Select a model and preset first." };
  }

  const { entries, lockedFields } = configSnapshotOverrideEntries(fields, overrides);
  if (lockedFields.length > 0) {
    const lockedNames = lockedFields
      .slice(0, 3)
      .map((field) => field.label || field.key)
      .join(", ");
    return {
      ok: false as const,
      error: `Snapshots cannot include preset-locked fields: ${lockedNames}.`,
    };
  }
  if (entries.length === 0) {
    return {
      ok: false as const,
      error: "Change at least one non-default field before adding a snapshot.",
    };
  }

  const adaptiveOptionValidation = validateAdaptiveOptionRequirements(entries);
  if (!adaptiveOptionValidation.ok) {
    return adaptiveOptionValidation;
  }

  const identity = snapshotIdentity({ modelType, model, preset, entries });
  const duplicate = snapshots.some((snapshot) => {
    if (
      snapshot.id === excludeSnapshotId ||
      snapshot.modelType !== modelType ||
      snapshot.model !== model ||
      snapshot.preset !== preset
    ) {
      return false;
    }
    return (
      configSnapshotIdentity({
        model: snapshot.model,
        modelType: snapshot.modelType,
        preset: snapshot.preset,
        fields,
        overrides: snapshot.overrides,
      }) === identity
    );
  });
  if (duplicate) {
    return {
      ok: false as const,
      error: "A snapshot with these config values already exists.",
    };
  }

  return {
    ok: true as const,
    entries,
    overrides: overridesFromSnapshotEntries(entries),
    identity,
  };
}

export function createConfigSnapshot({
  id,
  name,
  model,
  modelType,
  preset,
  fields,
  overrides,
  snapshots,
  createdAt,
}: {
  id: string;
  name: string;
  modelType: string;
  model: string;
  preset: string;
  fields: ConfigField[];
  overrides: OverrideValues;
  snapshots: ConfigSnapshot[];
  createdAt: string;
}): ConfigSnapshotCreateResult {
  const nameValidation = validateConfigSnapshotName({
    model,
    modelType,
    preset,
    name,
    snapshots,
  });
  if (!nameValidation.ok) {
    return nameValidation;
  }

  const validation = validateConfigSnapshotCandidate({
    modelType,
    model,
    preset,
    fields,
    overrides,
    snapshots,
  });
  if (!validation.ok) {
    return validation;
  }

  return {
    ok: true,
    snapshot: {
      id,
      name: nameValidation.name,
      model,
      modelType,
      preset,
      overrides: validation.overrides,
      createdAt,
    },
  };
}

export function selectedConfigSnapshots(
  snapshots: ConfigSnapshot[],
  modelType: string,
  model: string,
  selectedPresets: string[],
) {
  const selectedPresetSet = new Set(selectedPresets);
  return snapshots.filter(
    (snapshot) =>
      snapshot.modelType === modelType &&
      snapshot.model === model &&
      selectedPresetSet.has(snapshot.preset),
  );
}

export function groupConfigSnapshotsByPreset(
  snapshots: ConfigSnapshot[],
  presetOrder: string[],
) {
  const groups = new Map<string, ConfigSnapshot[]>();
  for (const preset of presetOrder) {
    groups.set(preset, []);
  }
  for (const snapshot of snapshots) {
    groups.set(snapshot.preset, [...(groups.get(snapshot.preset) ?? []), snapshot]);
  }
  return Array.from(groups, ([preset, groupedSnapshots]) => ({
    preset,
    snapshots: groupedSnapshots,
  })).filter((group) => group.snapshots.length > 0);
}

function overrideRecordKey(overrides: OverrideValues) {
  return Object.entries(overrides)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => `${key}=${value}`)
    .join("\u0000");
}

export function draftMatchesConfigSnapshot({
  snapshot,
  preset,
  overrides,
}: {
  snapshot: ConfigSnapshot;
  preset: string;
  overrides: OverrideValues;
}) {
  return (
    snapshot.preset === preset &&
    overrideRecordKey(snapshot.overrides) === overrideRecordKey(overrides)
  );
}
