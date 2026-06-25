import {
  type ConfigField,
  type ConfigValue,
  type TrainingRun,
  type TrainingRunChange,
  type TrainingRunPlan,
} from "@/lib/api";
import {
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
  return `--${key.toLowerCase().replace(/_/g, "-")}`;
}

function trainingCommand({
  modelType,
  model,
  preset,
  dataset,
  overrides,
  logFolder,
}: {
  modelType: string;
  model: string;
  preset: string;
  dataset: string;
  overrides: OverrideValues;
  logFolder: string;
}) {
  const parts = [
    "source",
    "experiment.sh",
    "--model-type",
    shellQuote(modelType),
    "--model",
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
  overrides: OverrideValues,
  fieldsByKey: Map<string, ConfigField>,
) {
  return Object.entries(overrides).map<TrainingRunChange>(
    ([key, value]) => ({
      key,
      label: fieldsByKey.get(configKeyToken(key))?.label ?? key,
      value: configValueFromOverride(value),
      source: "override",
    }),
  );
}

function epochDefault(fieldsByKey: Map<string, ConfigField>) {
  const field =
    fieldsByKey.get(configKeyToken("num_epochs")) ??
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
  return epochsFromOverrides(snapshot.overrides, fieldsByKey);
}

function epochsFromOverrides(
  overrides: OverrideValues,
  fieldsByKey: Map<string, ConfigField>,
) {
  const rawEpochs = overrideValue(overrides, "NUM_EPOCHS");
  const epochValue = rawEpochs === undefined ? epochDefault(fieldsByKey) : Number(rawEpochs);
  return Number.isFinite(epochValue) ? Math.max(0, Math.trunc(epochValue)) : 0;
}

function baseRun({
  modelType,
  model,
  preset,
  dataset,
  index,
  fieldsByKey,
  overrides,
  logFolder,
}: {
  modelType: string;
  model: string;
  preset: string;
  dataset: string;
  index: number;
  fieldsByKey: Map<string, ConfigField>;
  overrides: OverrideValues;
  logFolder: string;
}): TrainingRun {
  const runOverrides = { ...overrides };
  return {
    id: `preset-${preset}-${dataset}-${index}`,
    index,
    status: "Pending",
    preset,
    dataset,
    changes: snapshotRunChanges(runOverrides, fieldsByKey),
    overrides: runOverrides,
    command: trainingCommand({
      model,
      modelType,
      preset,
      dataset,
      overrides: runOverrides,
      logFolder,
    }),
    totalEpochs: epochsFromOverrides(runOverrides, fieldsByKey),
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
  modelType,
  model,
  selectedPreset,
  selectedTrainingPresets,
  selectedDatasets,
  snapshots,
  fields,
  presetOverrides = {},
  logFolder,
}: {
  modelType: string;
  model: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
  selectedDatasets: string[];
  snapshots: ConfigSnapshot[];
  fields: ConfigField[];
  presetOverrides?: OverrideValues;
  logFolder: string;
}): TrainingRunPlan | undefined {
  const selectedSnapshots = snapshots.filter(
    (snapshot) => snapshot.modelType === modelType && snapshot.model === model,
  );
  if (
    selectedDatasets.length === 0 ||
    (selectedTrainingPresets.length === 0 && selectedSnapshots.length === 0)
  ) {
    return undefined;
  }

  const fieldsByKey = new Map(
    fields.flatMap((field) => [
      [configKeyToken(field.key), field] as const,
      [configKeyToken(field.configKey), field] as const,
    ]),
  );
  const runs: TrainingRun[] = [];
  for (const preset of selectedTrainingPresets) {
    for (const dataset of selectedDatasets) {
      const index = runs.length + 1;
      runs.push(
        baseRun({
          model,
          modelType,
          preset,
          dataset,
          index,
          fieldsByKey,
          overrides: presetOverrides,
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
        changes: snapshotRunChanges(snapshot.overrides, fieldsByKey),
        overrides: snapshot.overrides,
        command: trainingCommand({
          model,
          modelType,
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

  const presets = Array.from(
    new Set([
      ...selectedTrainingPresets,
      ...selectedSnapshots.map((snapshot) => snapshot.preset),
    ]),
  );
  const primaryPreset = presets.includes(selectedPreset)
    ? selectedPreset
    : presets[0] || selectedPreset;
  return {
    modelType,
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
