import {
  type ConfigField,
  type ConfigValue,
  type TrainingRun,
  type TrainingRunChange,
  type TrainingRunPlan,
} from "@/lib/api";
import { type OverrideValues } from "@/lib/config";

export type ConfigSnapshot = {
  id: string;
  name: string;
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

function hasOwn(object: Record<string, unknown>, key: string) {
  return Object.prototype.hasOwnProperty.call(object, key);
}

function normalizePrimitive(value: ConfigValue | string) {
  return value === null || value === undefined ? "" : String(value).trim();
}

function normalizeValueForField(field: ConfigField, value: ConfigValue | string) {
  const raw = normalizePrimitive(value);
  if (field.nullable && raw === "") {
    return "null";
  }
  if (field.type === "bool") {
    const lower = raw.toLowerCase();
    if (lower === "true" || lower === "false") {
      return lower;
    }
  }
  if (field.type === "int") {
    const numberValue = Number(raw);
    if (Number.isInteger(numberValue)) {
      return String(numberValue);
    }
  }
  if (field.type === "float") {
    const numberValue = Number(raw);
    if (Number.isFinite(numberValue)) {
      return String(numberValue);
    }
  }
  return raw;
}

function overrideValueForField(field: ConfigField, normalizedValue: string) {
  return field.nullable && normalizedValue === "null" ? "" : normalizedValue;
}

function defaultValueForField(field: ConfigField) {
  return normalizeValueForField(field, field.default);
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
    if (!hasOwn(overrides, field.key)) {
      continue;
    }
    if (field.locked) {
      lockedFields.push(field);
    }
    const normalizedValue = normalizeValueForField(field, overrides[field.key] ?? "");
    if (normalizedValue === defaultValueForField(field)) {
      continue;
    }
    const value = overrideValueForField(field, normalizedValue);
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

function snapshotIdentity({
  model,
  preset,
  entries,
}: {
  model: string;
  preset: string;
  entries: ConfigSnapshotOverrideEntry[];
}) {
  return [
    model,
    preset,
    ...entries.map((entry) => `${entry.key}=${entry.value}`),
  ].join("\u0000");
}

export function configSnapshotIdentity({
  model,
  preset,
  fields,
  overrides,
}: {
  model: string;
  preset: string;
  fields: ConfigField[];
  overrides: OverrideValues;
}) {
  const { entries } = configSnapshotOverrideEntries(fields, overrides);
  return snapshotIdentity({ model, preset, entries });
}

function normalizeSnapshotName(name: string) {
  return name.trim().toLocaleLowerCase();
}

export function validateConfigSnapshotName({
  model,
  preset,
  name,
  snapshots,
  excludeSnapshotId,
}: {
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
  model,
  preset,
  fields,
  overrides,
  snapshots,
  excludeSnapshotId,
}: {
  model: string;
  preset: string;
  fields: ConfigField[];
  overrides: OverrideValues;
  snapshots: ConfigSnapshot[];
  excludeSnapshotId?: string;
}) {
  if (!model || !preset) {
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

  const identity = snapshotIdentity({ model, preset, entries });
  const duplicate = snapshots.some((snapshot) => {
    if (
      snapshot.id === excludeSnapshotId ||
      snapshot.model !== model ||
      snapshot.preset !== preset
    ) {
      return false;
    }
    return (
      configSnapshotIdentity({
        model: snapshot.model,
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
  preset,
  fields,
  overrides,
  snapshots,
  createdAt,
}: {
  id: string;
  name: string;
  model: string;
  preset: string;
  fields: ConfigField[];
  overrides: OverrideValues;
  snapshots: ConfigSnapshot[];
  createdAt: string;
}): ConfigSnapshotCreateResult {
  const nameValidation = validateConfigSnapshotName({
    model,
    preset,
    name,
    snapshots,
  });
  if (!nameValidation.ok) {
    return nameValidation;
  }

  const validation = validateConfigSnapshotCandidate({
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
      preset,
      overrides: validation.overrides,
      createdAt,
    },
  };
}

export function selectedConfigSnapshots(
  snapshots: ConfigSnapshot[],
  model: string,
  selectedPresets: string[],
) {
  const selectedPresetSet = new Set(selectedPresets);
  return snapshots.filter(
    (snapshot) =>
      snapshot.model === model && selectedPresetSet.has(snapshot.preset),
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

function shellQuote(value: string) {
  if (value === "") {
    return "''";
  }
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(value)) {
    return value;
  }
  return `'${value.replace(/'/g, `'"'"'`)}'`;
}

function snapshotCommandFlag(key: string) {
  return `--${key.replace(/_/g, "-")}`;
}

function trainingCommand({
  model,
  preset,
  dataset,
  overrides,
  logFolder,
}: {
  model: string;
  preset: string;
  dataset: string;
  overrides: OverrideValues;
  logFolder: string;
}) {
  const parts = [
    "source",
    "experiment.sh",
    shellQuote(model),
    "--preset",
    shellQuote(preset),
    "--datasets",
    shellQuote(dataset),
  ];
  if (logFolder) {
    parts.push("--logdir", shellQuote(logFolder));
  }
  const overrideEntries = Object.entries(overrides);
  if (overrideEntries.length > 0) {
    parts.push("--config");
    for (const [key, value] of overrideEntries) {
      parts.push(snapshotCommandFlag(key), shellQuote(value || "None"));
    }
  }
  return parts.join(" ");
}

function configValueFromOverride(value: string): ConfigValue {
  return value;
}

function snapshotRunChanges(
  snapshot: ConfigSnapshot,
  fieldsByKey: Map<string, ConfigField>,
) {
  return Object.entries(snapshot.overrides).map<TrainingRunChange>(
    ([key, value]) => ({
      key,
      label: fieldsByKey.get(key)?.label ?? key,
      value: configValueFromOverride(value),
      source: "override",
    }),
  );
}

function epochDefault(fieldsByKey: Map<string, ConfigField>) {
  const field =
    fieldsByKey.get("num_epochs") ??
    Array.from(fieldsByKey.values()).find(
      (candidate) => candidate.configKey.toLowerCase() === "num_epochs",
    );
  const defaultEpochs = Number(field?.default ?? 10);
  return Number.isFinite(defaultEpochs) ? Math.max(0, Math.trunc(defaultEpochs)) : 10;
}

function snapshotEpochs(
  snapshot: ConfigSnapshot,
  fieldsByKey: Map<string, ConfigField>,
) {
  const rawEpochs = snapshot.overrides.num_epochs;
  const epochValue = rawEpochs === undefined ? epochDefault(fieldsByKey) : Number(rawEpochs);
  return Number.isFinite(epochValue) ? Math.max(0, Math.trunc(epochValue)) : 0;
}

function baseRun({
  model,
  preset,
  dataset,
  index,
  fieldsByKey,
  logFolder,
}: {
  model: string;
  preset: string;
  dataset: string;
  index: number;
  fieldsByKey: Map<string, ConfigField>;
  logFolder: string;
}): TrainingRun {
  return {
    id: `preset-${preset}-${dataset}-${index}`,
    index,
    status: "Pending",
    preset,
    dataset,
    changes: [],
    overrides: {},
    command: trainingCommand({
      model,
      preset,
      dataset,
      overrides: {},
      logFolder,
    }),
    totalEpochs: epochDefault(fieldsByKey),
    currentEpoch: 0,
    metrics: {},
    logDir: null,
    error: null,
    errorTraceback: null,
  };
}

function summarizeSnapshotRuns(runs: TrainingRun[]) {
  const totalEpochs = runs.reduce((total, run) => total + run.totalEpochs, 0);
  return {
    totalRuns: runs.length,
    completedRuns: 0,
    runningRuns: 0,
    pendingRuns: runs.length,
    failedRuns: 0,
    cancelledRuns: 0,
    skippedRuns: 0,
    totalEpochs,
    completedEpochs: 0,
    remainingEpochs: totalEpochs,
  };
}

export function buildConfigSnapshotRunPlan({
  model,
  selectedPreset,
  selectedTrainingPresets,
  selectedDatasets,
  snapshots,
  fields,
  logFolder,
}: {
  model: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
  selectedDatasets: string[];
  snapshots: ConfigSnapshot[];
  fields: ConfigField[];
  logFolder: string;
}): TrainingRunPlan | undefined {
  const selectedSnapshots = selectedConfigSnapshots(
    snapshots,
    model,
    selectedTrainingPresets,
  );
  if (selectedSnapshots.length === 0 || selectedDatasets.length === 0) {
    return undefined;
  }

  const fieldsByKey = new Map(fields.map((field) => [field.key, field]));
  const runs: TrainingRun[] = [];
  for (const preset of selectedTrainingPresets) {
    for (const dataset of selectedDatasets) {
      const index = runs.length + 1;
      runs.push(
        baseRun({
          model,
          preset,
          dataset,
          index,
          fieldsByKey,
          logFolder,
        }),
      );
    }
  }
  for (const snapshot of selectedSnapshots) {
    for (const dataset of selectedDatasets) {
      const index = runs.length + 1;
      runs.push({
        id: `snapshot-${snapshot.id}-${dataset}-${index}`,
        index,
        status: "Pending",
        preset: snapshot.preset,
        snapshotId: snapshot.id,
        snapshotName: snapshot.name,
        dataset,
        changes: snapshotRunChanges(snapshot, fieldsByKey),
        overrides: snapshot.overrides,
        command: trainingCommand({
          model,
          preset: snapshot.preset,
          dataset,
          overrides: snapshot.overrides,
          logFolder,
        }),
        totalEpochs: snapshotEpochs(snapshot, fieldsByKey),
        currentEpoch: 0,
        metrics: {},
        logDir: null,
        error: null,
        errorTraceback: null,
      });
    }
  }

  const presets = selectedTrainingPresets.length
    ? selectedTrainingPresets
    : Array.from(new Set(selectedSnapshots.map((snapshot) => snapshot.preset)));
  const primaryPreset = presets.includes(selectedPreset)
    ? selectedPreset
    : presets[0] || selectedPreset;
  return {
    model,
    preset: primaryPreset,
    presets,
    datasets: selectedDatasets,
    overrides: {},
    search: null,
    logFolder,
    isRandomSearch: false,
    runs,
    summary: summarizeSnapshotRuns(runs),
  };
}
